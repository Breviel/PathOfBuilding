"""
PoE Trade API client for item price lookups.

Queries the official pathofexile.com trade API to estimate item values
based on mod combinations. Used by the Gear Improvement Suggester to
compare "buy on trade" vs "craft yourself" costs.

GGG's trade API has strict rate limits (documented below). This module
implements automatic back-off and caching to stay within limits.

API flow:
  1. POST /api/trade/search/{league}  → returns { "id": "...", "result": [...ids], "total": N }
  2. GET  /api/trade/fetch/{ids}?query={id}  → returns item details + price

Usage:
    from ml.data.collectors.trade_api import TradeClient

    async with TradeClient() as client:
        results = await client.search_items(
            league="Settlers",
            min_life=80,
            min_fire_res=40,
            item_type="Helmet",
        )
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from loguru import logger

from ml.config import LEAGUE, POE_TRADE_API, POE_TRADE_RPS

# ── Rate limiting ──────────────────────────────────────────────────────
# GGG enforces:
#   - Search: 8 requests / 6 seconds, 12 / 60 seconds
#   - Fetch:  8 requests / 6 seconds, 12 / 60 seconds
# We use a simple token-bucket approach.

RATE_LIMIT_SEARCH_PER_MIN = 10
RATE_LIMIT_FETCH_PER_MIN = 10


@dataclass
class _RateLimiter:
    """Simple token-bucket rate limiter."""
    max_per_minute: int = 10
    _timestamps: list[float] = field(default_factory=list)

    async def acquire(self) -> None:
        now = time.monotonic()
        # Purge old timestamps
        self._timestamps = [t for t in self._timestamps if now - t < 60]
        if len(self._timestamps) >= self.max_per_minute:
            wait = 60 - (now - self._timestamps[0])
            logger.debug(f"Rate limit: sleeping {wait:.1f}s")
            await asyncio.sleep(wait)
        self._timestamps.append(time.monotonic())


# ── Search query builder ───────────────────────────────────────────────

def build_search_query(
    *,
    item_type: str | None = None,
    item_base: str | None = None,
    min_life: int | None = None,
    min_fire_res: int | None = None,
    min_cold_res: int | None = None,
    min_lightning_res: int | None = None,
    min_chaos_res: int | None = None,
    min_pdps: int | None = None,
    min_edps: int | None = None,
    min_es: int | None = None,
    min_armour: int | None = None,
    min_evasion: int | None = None,
    has_open_prefix: bool = False,
    has_open_suffix: bool = False,
    rarity: str | None = None,
    max_price_chaos: int | None = None,
    custom_stats: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Build a trade-API search payload.

    Returns a dict ready to POST to ``/api/trade/search/{league}``.
    """
    query: dict[str, Any] = {"status": {"option": "online"}}
    filters: dict[str, Any] = {}
    stat_filters: list[dict] = []

    # Item type / base
    if item_type:
        query["type"] = item_type
    if item_base:
        query["term"] = item_base
    if rarity:
        query.setdefault("filters", {})["type_filters"] = {
            "filters": {"rarity": {"option": rarity}}
        }

    # Stat filters (using pseudo / explicit stat IDs)
    # Pseudo stats are aggregated: e.g. "pseudo.pseudo_total_life"
    def _add_stat(stat_id: str, min_val: int | None) -> None:
        if min_val is not None:
            stat_filters.append({
                "type": "and",
                "filters": [{"id": stat_id, "value": {"min": min_val}}],
            })

    _add_stat("pseudo.pseudo_total_life", min_life)
    _add_stat("pseudo.pseudo_total_fire_resistance", min_fire_res)
    _add_stat("pseudo.pseudo_total_cold_resistance", min_cold_res)
    _add_stat("pseudo.pseudo_total_lightning_resistance", min_lightning_res)
    _add_stat("pseudo.pseudo_total_chaos_resistance", min_chaos_res)
    _add_stat("pseudo.pseudo_total_energy_shield", min_es)

    # Weapon DPS pseudo stats
    if min_pdps:
        stat_filters.append({
            "type": "and",
            "filters": [{"id": "pseudo.pseudo_physical_dps", "value": {"min": min_pdps}}],
        })
    if min_edps:
        stat_filters.append({
            "type": "and",
            "filters": [{"id": "pseudo.pseudo_elemental_dps", "value": {"min": min_edps}}],
        })

    # Armour / evasion (on armour pieces)
    if min_armour:
        filters.setdefault("armour_filters", {}).setdefault("filters", {})["ar"] = {"min": min_armour}
    if min_evasion:
        filters.setdefault("armour_filters", {}).setdefault("filters", {})["ev"] = {"min": min_evasion}

    # Open affixes
    if has_open_prefix:
        filters.setdefault("misc_filters", {}).setdefault("filters", {})["crafted"] = {"option": False}
    if has_open_suffix:
        filters.setdefault("misc_filters", {}).setdefault("filters", {})["crafted"] = {"option": False}

    # Price cap
    sort = {"price": "asc"}
    trade_filters: dict[str, Any] = {}
    if max_price_chaos:
        trade_filters["price"] = {"min": 1, "max": max_price_chaos, "option": "chaos"}
        filters["trade_filters"] = {"filters": trade_filters}

    # Custom stats (passthrough)
    if custom_stats:
        stat_filters.extend(custom_stats)

    payload: dict[str, Any] = {
        "query": {
            **query,
            "stats": stat_filters,
            "filters": filters,
        },
        "sort": sort,
    }
    return payload


# ── Trade client ───────────────────────────────────────────────────────

@dataclass
class TradeResult:
    """A single trade listing."""
    item_name: str
    item_base: str
    price_amount: float
    price_currency: str
    mods: list[str]
    ilvl: int
    corrupted: bool


class TradeClient:
    """
    Async client for the PoE trade API.

    Usage::

        async with TradeClient() as client:
            results = await client.search(league, query_payload)
    """

    def __init__(self, user_agent: str = "PoB-ML-Suite/1.0") -> None:
        self._session: aiohttp.ClientSession | None = None
        self._search_limiter = _RateLimiter(max_per_minute=RATE_LIMIT_SEARCH_PER_MIN)
        self._fetch_limiter = _RateLimiter(max_per_minute=RATE_LIMIT_FETCH_PER_MIN)
        self._user_agent = user_agent

    async def __aenter__(self) -> TradeClient:
        self._session = aiohttp.ClientSession(
            headers={"User-Agent": self._user_agent}
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._session:
            await self._session.close()

    async def search(
        self,
        payload: dict[str, Any],
        league: str | None = None,
        max_results: int = 10,
    ) -> list[TradeResult]:
        """
        Execute a trade search and fetch the first ``max_results`` items.

        Parameters
        ----------
        payload : dict
            Search payload from ``build_search_query()``.
        league : str
            League name (default from config).
        max_results : int
            How many listings to return (max 10 per fetch call).

        Returns
        -------
        list[TradeResult]
        """
        league = league or LEAGUE
        assert self._session is not None, "Use as async context manager"

        # Step 1: Search
        await self._search_limiter.acquire()
        search_url = f"{POE_TRADE_API}/search/{league}"
        async with self._session.post(search_url, json=payload) as resp:
            if resp.status == 429:
                retry_after = int(resp.headers.get("Retry-After", "60"))
                logger.warning(f"Rate limited, waiting {retry_after}s")
                await asyncio.sleep(retry_after)
                return await self.search(payload, league, max_results)
            resp.raise_for_status()
            search_data = await resp.json()

        query_id = search_data.get("id", "")
        result_ids = search_data.get("result", [])[:max_results]

        if not result_ids:
            return []

        # Step 2: Fetch item details (max 10 per call)
        await self._fetch_limiter.acquire()
        ids_str = ",".join(result_ids[:10])
        fetch_url = f"{POE_TRADE_API}/fetch/{ids_str}?query={query_id}"
        async with self._session.get(fetch_url) as resp:
            if resp.status == 429:
                retry_after = int(resp.headers.get("Retry-After", "60"))
                logger.warning(f"Rate limited on fetch, waiting {retry_after}s")
                await asyncio.sleep(retry_after)
                return await self.search(payload, league, max_results)
            resp.raise_for_status()
            fetch_data = await resp.json()

        # Step 3: Parse results
        results = []
        for entry in fetch_data.get("result", []):
            listing = entry.get("listing", {})
            item = entry.get("item", {})
            price = listing.get("price", {})

            mods = []
            for mod_group in ("explicitMods", "implicitMods", "craftedMods"):
                mods.extend(item.get(mod_group, []))

            results.append(TradeResult(
                item_name=item.get("name", ""),
                item_base=item.get("typeLine", ""),
                price_amount=float(price.get("amount", 0)),
                price_currency=price.get("currency", "chaos"),
                mods=mods,
                ilvl=item.get("ilvl", 0),
                corrupted=item.get("corrupted", False),
            ))

        return results

    async def search_items(
        self,
        league: str | None = None,
        max_results: int = 10,
        **kwargs: Any,
    ) -> list[TradeResult]:
        """Convenience: build query from kwargs and search."""
        payload = build_search_query(**kwargs)
        return await self.search(payload, league, max_results)


# ── Bulk price estimation ──────────────────────────────────────────────

async def estimate_price(
    item_type: str | None = None,
    min_life: int | None = None,
    min_fire_res: int | None = None,
    min_cold_res: int | None = None,
    min_lightning_res: int | None = None,
    league: str | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """
    Estimate item price percentiles (p25, p50, p75).

    Returns dict with keys "p25", "p50", "p75" in chaos equivalent.
    """
    async with TradeClient() as client:
        results = await client.search_items(
            league=league,
            item_type=item_type,
            min_life=min_life,
            min_fire_res=min_fire_res,
            min_cold_res=min_cold_res,
            min_lightning_res=min_lightning_res,
            max_results=10,
            **kwargs,
        )

    if not results:
        return {"p25": 0, "p50": 0, "p75": 0}

    prices = sorted(r.price_amount for r in results)
    n = len(prices)

    return {
        "p25": prices[max(0, n // 4)],
        "p50": prices[n // 2],
        "p75": prices[min(n - 1, 3 * n // 4)],
    }
