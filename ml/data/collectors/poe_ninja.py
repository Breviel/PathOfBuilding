"""
poe.ninja data collector — builds + economy (PoE 1 only).

Collects from the poe.ninja public JSON API (no auth required):

**Builds** (`/api/data/0/getbuildoverview`):
- Passive tree allocations (node hashes / IDs)
- Equipped gear (base types, mods, sockets)
- Skill gem setups (active + support links)
- Character class / ascendancy / level
- Computed DPS, life, ES, depth (from poe.ninja's calculations)

**Economy** (`/api/data/currencyoverview`, `/api/data/itemoverview`):
- Currency chaos-equivalent prices  (Divine, Exalted, …)
- Item prices by type  (Uniques, Maps, Scarabs, Div Cards, …)

API documentation sources:
- https://github.com/Davenads/poeninjaAPI-2025
- https://github.com/JakubRak-gamedev/poeninja_API_guide

Rate limits: 12 requests per 5 minutes.  We enforce this via a token-
bucket semaphore + per-request sleep.

Usage:
    python -m ml.data.collectors.poe_ninja                      # all classes + economy
    python -m ml.data.collectors.poe_ninja builds --class Witch  # single class
    python -m ml.data.collectors.poe_ninja economy               # economy snapshot only
    python -m ml.data.collectors.poe_ninja probe                 # test API connectivity
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
from loguru import logger

from ml.config import (
    DATA_RAW_DIR,
    LEAGUE,
    POE_NINJA_BUILDS_CHARACTER_BASE,
    POE_NINJA_BUILDS_ENDPOINT,
    POE_NINJA_BUILDS_ENDPOINT_ALT,
    POE_NINJA_BUILDS_ENDPOINT_FALLBACK,
    POE_NINJA_CURRENCY_ENDPOINT,
    POE_NINJA_INDEX_STATE_ENDPOINT,
    POE_NINJA_ITEM_ENDPOINT,
    POE_NINJA_ITEM_TYPES,
    POE_NINJA_MAX_REQUESTS,
    POE_NINJA_RPS,
    POE_NINJA_WINDOW_SECONDS,
    TrainingConfig,
)


# ── Rate limiter ───────────────────────────────────────────────────────

class _RateLimiter:
    """
    Token-bucket rate limiter that enforces poe.ninja's 12-req / 5-min
    limit *and* a burst-rate cap.

    Usage::

        rl = _RateLimiter()
        async with rl:
            resp = await session.get(url)
    """

    def __init__(
        self,
        max_tokens: int = POE_NINJA_MAX_REQUESTS,
        window_s: float = POE_NINJA_WINDOW_SECONDS,
        burst_rps: float = POE_NINJA_RPS,
    ) -> None:
        self._sem = asyncio.Semaphore(max_tokens)
        self._window = window_s
        self._burst_delay = 1.0 / burst_rps
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> None:
        await self._sem.acquire()
        async with self._lock:
            await asyncio.sleep(self._burst_delay)

    async def __aexit__(self, *exc: object) -> None:
        # Release the token after the window expires so we
        # don't exceed 12 / 5 min over the long run.
        asyncio.get_event_loop().call_later(
            self._window, self._sem.release,
        )


# ── Shared HTTP helpers ───────────────────────────────────────────────

_USER_AGENT = "PoB-ML-DataCollector/1.0 (github.com/Breviel/PathOfBuilding)"
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "application/json",
}
_TIMEOUT = aiohttp.ClientTimeout(total=30)


async def _get_json(
    session: aiohttp.ClientSession,
    url: str,
    rate_limiter: _RateLimiter,
    params: dict[str, str] | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    """GET JSON with rate limiting, retries and exponential back-off."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            async with rate_limiter:
                async with session.get(url, params=params) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited — waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    resp.raise_for_status()
                    return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(f"Attempt {attempt + 1}/{retries} failed: {exc}. Retrying in {wait}s…")
            await asyncio.sleep(wait)

    raise RuntimeError(f"All {retries} attempts failed for {url}") from last_exc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BUILDS COLLECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# poe.ninja builds API:
#
# BUILDS OVERVIEW (bulk data):
#   GET https://poe.ninja/api/data/0/getbuildoverview
#       ?overview={league_lowercase}&type=exp&language=en
#
#   With poe1 prefix (for PoE1-specific routing):
#   GET https://poe.ninja/poe1/api/data/0/getbuildoverview
#       ?overview={league_lowercase}&type=exp&language=en
#
# Response (JSON):
#   {
#     "classNames":       ["Marauder", "Ranger", ...],
#     "uniqueItems":      [...],
#     "keystoneHashes":   [...],
#     "skills":           [...],
#     "builds": [
#       {
#         "account":     "...",
#         "character":   "...",
#         "class":       3,            ← index into classNames
#         "level":       99,
#         "treeHashes":  [12345, ...], ← allocated passive node IDs
#         "items":       [{...}, ...],
#         "skills":      [{...}, ...],
#         "life":        5200,
#         "es":          0,
#         "dps":         1_500_000,
#         "depth":       300,
#         ...
#       },
#       ...
#     ]
#   }
#
# INDIVIDUAL CHARACTER (requires snapshot version from index-state):
#   GET https://poe.ninja/poe1/api/builds/{version}/character
#       ?account={acct}&name={name}&overview={league_url}&type=0
#
# Per-class filter on overview:
#   &class={ClassName}

# ── Index-state (league / snapshot metadata) ───────────────────────────


async def _get_index_state(
    session: aiohttp.ClientSession,
    rate_limiter: _RateLimiter,
) -> dict[str, Any]:
    """Fetch the poe.ninja index-state to discover leagues and snapshot versions."""
    logger.debug(f"Fetching index-state: {POE_NINJA_INDEX_STATE_ENDPOINT}")
    return await _get_json(session, POE_NINJA_INDEX_STATE_ENDPOINT, rate_limiter)


# ── Builds endpoint candidates ─────────────────────────────────────────
# We try three URL variants in order because poe.ninja has changed its
# routing several times.  The first success is cached for subsequent calls.


def _init_builds_endpoints(
    league: str,
    class_name: str | None = None,
) -> list[tuple[str, dict[str, str]]]:
    """Return (url, params) pairs to try for the builds overview."""
    league_lower = league.lower()
    candidates: list[tuple[str, dict[str, str]]] = []

    base_params: dict[str, str] = {
        "overview": league_lower,
        "type": "exp",
        "language": "en",
    }
    if class_name:
        base_params["class"] = class_name

    # 1. poe1-prefixed: /poe1/api/data/0/getbuildoverview
    candidates.append((POE_NINJA_BUILDS_ENDPOINT, dict(base_params)))

    # 2. Original: /api/data/0/getbuildoverview
    candidates.append((POE_NINJA_BUILDS_ENDPOINT_ALT, dict(base_params)))

    # 3. Fallback (without /0/): /poe1/api/data/getbuildoverview
    candidates.append((POE_NINJA_BUILDS_ENDPOINT_FALLBACK, dict(base_params)))

    return candidates


async def _get_builds_json(
    session: aiohttp.ClientSession,
    rate_limiter: _RateLimiter,
    league: str,
    class_name: str | None = None,
) -> dict[str, Any]:
    """
    Fetch builds overview data, trying multiple endpoint candidates.

    Tries endpoints in order (poe1-prefixed, original, fallback).
    Returns the parsed JSON response from whichever endpoint succeeds.
    Raises RuntimeError if all endpoints fail.
    """
    candidates = _init_builds_endpoints(league, class_name)
    last_exc: Exception | None = None

    for url, params in candidates:
        try:
            logger.debug(f"Trying builds endpoint: {url} params={params}")
            return await _get_json(session, url, rate_limiter, params=params)
        except Exception as exc:
            logger.debug(f"Endpoint {url} failed: {exc}")
            last_exc = exc

    raise RuntimeError(
        f"All builds endpoints failed for league={league!r}, class={class_name!r}. "
        f"Last error: {last_exc}"
    ) from last_exc


def _normalise_build(
    raw: dict[str, Any],
    class_names: list[str],
) -> dict[str, Any]:
    """Flatten a single poe.ninja build into a training-friendly dict."""
    class_idx = raw.get("class", 0)
    class_name = class_names[class_idx] if class_idx < len(class_names) else "Unknown"

    return {
        "account": raw.get("account", ""),
        "character": raw.get("character", ""),
        "class_name": class_name,
        "level": raw.get("level", 0),
        "tree_node_ids": raw.get("treeHashes", []),
        "items": raw.get("items", []),
        "skills": raw.get("skills", []),
        "life": raw.get("life", 0),
        "es": raw.get("es", 0),
        "dps": raw.get("dps", 0),
        "depth": raw.get("depth", 0),
    }


async def collect_builds(
    league: str | None = None,
    classes: list[str] | None = None,
    max_per_class: int | None = None,
    out_dir: Path | None = None,
) -> Path:
    """
    Scrape poe.ninja build data — one JSON file per class.

    Returns path to output directory.
    """
    league = league or LEAGUE
    cfg = TrainingConfig()
    classes = classes or cfg.classes
    max_per_class = max_per_class or cfg.max_builds_per_class
    out_dir = out_dir or (DATA_RAW_DIR / "poe_ninja")
    out_dir.mkdir(parents=True, exist_ok=True)

    rl = _RateLimiter()

    async with aiohttp.ClientSession(headers=_HEADERS, timeout=_TIMEOUT) as session:
        # 1. Overview — shared lookup tables
        logger.info(f"Fetching builds overview for {league}")
        overview = await _get_builds_json(session, rl, league)
        class_names = overview.get("classNames", [])

        lookup_path = out_dir / "lookup_tables.json"
        lookup_data = {
            "classNames": class_names,
            "uniqueItems": overview.get("uniqueItems", []),
            "keystoneHashes": overview.get("keystoneHashes", []),
            "skills": overview.get("skills", []),
        }
        lookup_path.write_text(json.dumps(lookup_data, indent=2))
        logger.info(f"Saved lookup tables → {lookup_path}")

        # 2. Per-class builds
        for cls in classes:
            logger.info(f"Collecting {cls} (max {max_per_class})")
            try:
                data = await _get_builds_json(session, rl, league, cls)
            except Exception as exc:
                logger.warning(f"Failed to fetch {cls}: {exc}")
                continue

            raw_builds = data.get("builds", [])[:max_per_class]
            builds = [_normalise_build(b, class_names) for b in raw_builds]

            out_path = out_dir / f"{cls.lower()}_builds.json"
            out_path.write_text(json.dumps(builds, indent=2))
            logger.info(f"  → {len(builds)} builds saved to {out_path}")

    logger.success(f"Build collection complete → {out_dir}")
    return out_dir


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ECONOMY DATA COLLECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Economy endpoints:
#   Currency:  GET /api/data/currencyoverview?league={L}&type=Currency
#   Fragment:  GET /api/data/currencyoverview?league={L}&type=Fragment
#   Items:     GET /api/data/itemoverview?league={L}&type={TYPE}
#
# Currency response:
#   { "lines": [{"currencyTypeName": "...", "chaosEquivalent": 180.5, ...}],
#     "currencyDetails": [{"id": 22, "name": "...", "icon": "...", "tradeId": "mirror"}] }
#
# Item response:
#   { "lines": [{"name": "...", "chaosValue": 50.0, "exaltedValue": 0.3, "count": 123, ...}] }

@dataclass
class EconomySnapshot:
    """Collected economy prices from a single point in time."""
    league: str
    timestamp: float
    currency: list[dict[str, Any]] = field(default_factory=list)
    currency_details: list[dict[str, Any]] = field(default_factory=list)
    items: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


async def collect_economy(
    league: str | None = None,
    item_types: list[str] | None = None,
    out_dir: Path | None = None,
) -> Path:
    """
    Snapshot poe.ninja economy data (currency + item prices).

    The output is a single JSON file containing the full economy state.
    """
    league = league or LEAGUE
    item_types = item_types or POE_NINJA_ITEM_TYPES
    out_dir = out_dir or (DATA_RAW_DIR / "poe_ninja")
    out_dir.mkdir(parents=True, exist_ok=True)

    rl = _RateLimiter()
    snap = EconomySnapshot(league=league, timestamp=time.time())

    async with aiohttp.ClientSession(headers=_HEADERS, timeout=_TIMEOUT) as session:
        # 1. Currency overview
        for cur_type in ("Currency", "Fragment"):
            logger.info(f"Fetching {cur_type} prices for {league}")
            try:
                data = await _get_json(
                    session, POE_NINJA_CURRENCY_ENDPOINT, rl,
                    params={"league": league, "type": cur_type, "language": "en"},
                )
                snap.currency.extend(data.get("lines", []))
                snap.currency_details.extend(data.get("currencyDetails", []))
                logger.info(f"  → {len(data.get('lines', []))} {cur_type} prices")
            except Exception as exc:
                logger.warning(f"Failed to fetch {cur_type}: {exc}")

        # 2. Item overviews
        pure_item_types = [t for t in item_types if t not in ("Currency", "Fragment")]
        for itype in pure_item_types:
            logger.info(f"Fetching {itype} prices for {league}")
            try:
                data = await _get_json(
                    session, POE_NINJA_ITEM_ENDPOINT, rl,
                    params={"league": league, "type": itype, "language": "en"},
                )
                lines = data.get("lines", [])
                snap.items[itype] = lines
                logger.info(f"  → {len(lines)} {itype} entries")
            except Exception as exc:
                logger.warning(f"Failed to fetch {itype}: {exc}")

    # Save
    out_path = out_dir / "economy_snapshot.json"
    payload = {
        "league": snap.league,
        "timestamp": snap.timestamp,
        "currency": snap.currency,
        "currency_details": snap.currency_details,
        "items": snap.items,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.success(f"Economy snapshot → {out_path} "
                   f"({len(snap.currency)} currencies, {sum(len(v) for v in snap.items.values())} items)")
    return out_dir


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  API PROBE / CONNECTIVITY TEST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def probe_api(league: str | None = None) -> dict[str, Any]:
    """
    Test poe.ninja API connectivity and print response structure.

    This is the *correct* way to probe the API — use the JSON endpoints,
    not HTML scraping.  poe.ninja is an SPA; the HTML page contains only
    a JS bundle, not build data.

    Returns dict with probe results.
    """
    league = league or LEAGUE
    results: dict[str, Any] = {"league": league, "endpoints": {}}

    rl = _RateLimiter()

    async with aiohttp.ClientSession(headers=_HEADERS, timeout=_TIMEOUT) as session:
        # 0. Index state — discover snapshot versions
        logger.info(f"Probing index-state: {POE_NINJA_INDEX_STATE_ENDPOINT}")
        index_state: dict[str, Any] | None = None
        try:
            index_state = await _get_index_state(session, rl)
            build_leagues = index_state.get("buildLeagues", [])
            snapshots = index_state.get("snapshotVersions", [])
            results["endpoints"]["index_state"] = {
                "status": "OK",
                "build_leagues": [bl.get("name") for bl in build_leagues],
                "n_snapshots": len(snapshots),
                "snapshot_versions": [
                    {"url": s.get("url"), "type": s.get("type"), "version": s.get("version")}
                    for s in snapshots
                ],
            }
            logger.info(f"  ✓ {len(build_leagues)} build leagues, {len(snapshots)} snapshots")
            for s in snapshots:
                logger.info(f"    {s.get('url')}/{s.get('type')} → {s.get('version')}")
        except Exception as exc:
            results["endpoints"]["index_state"] = {"status": "FAILED", "error": str(exc)}
            logger.error(f"  ✗ {exc}")

        # 1. Builds overview
        logger.info(f"Probing builds for {league}")
        try:
            data = await _get_builds_json(session, rl, league)
            n_builds = len(data.get("builds", []))
            class_names = data.get("classNames", [])
            sample_keys = list(data.get("builds", [{}])[0].keys()) if n_builds else []
            results["endpoints"]["builds"] = {
                "status": "OK",
                "n_builds": n_builds,
                "classNames": class_names,
                "top_level_keys": list(data.keys()),
                "sample_build_keys": sample_keys,
            }
            logger.info(f"  ✓ {n_builds} builds, classes: {class_names}")
            logger.info(f"  Build record keys: {sample_keys}")
        except Exception as exc:
            results["endpoints"]["builds"] = {"status": "FAILED", "error": str(exc)}
            logger.error(f"  ✗ {exc}")

        # 2. Currency overview
        logger.info(f"Probing currency: {POE_NINJA_CURRENCY_ENDPOINT}")
        try:
            data = await _get_json(
                session, POE_NINJA_CURRENCY_ENDPOINT, rl,
                params={"league": league, "type": "Currency", "language": "en"},
            )
            n_lines = len(data.get("lines", []))
            sample = data["lines"][0] if n_lines else {}
            results["endpoints"]["currency"] = {
                "status": "OK",
                "n_currencies": n_lines,
                "sample_keys": list(sample.keys()),
                "top_3": [
                    {
                        "name": c.get("currencyTypeName", "?"),
                        "chaos": c.get("chaosEquivalent", 0),
                    }
                    for c in data.get("lines", [])[:3]
                ],
            }
            logger.info(f"  ✓ {n_lines} currencies")
            for c in data.get("lines", [])[:3]:
                name = c.get("currencyTypeName", "?")
                chaos = c.get("chaosEquivalent", 0)
                logger.info(f"    {name}: {chaos:.1f} chaos")
        except Exception as exc:
            results["endpoints"]["currency"] = {"status": "FAILED", "error": str(exc)}
            logger.error(f"  ✗ {exc}")

        # 3. Item overview (one type)
        logger.info(f"Probing items: {POE_NINJA_ITEM_ENDPOINT} (DivinationCard)")
        try:
            data = await _get_json(
                session, POE_NINJA_ITEM_ENDPOINT, rl,
                params={"league": league, "type": "DivinationCard", "language": "en"},
            )
            n_lines = len(data.get("lines", []))
            results["endpoints"]["items_divcard"] = {
                "status": "OK",
                "n_items": n_lines,
                "sample_keys": list(data["lines"][0].keys()) if n_lines else [],
            }
            logger.info(f"  ✓ {n_lines} div cards")
        except Exception as exc:
            results["endpoints"]["items_divcard"] = {"status": "FAILED", "error": str(exc)}
            logger.error(f"  ✗ {exc}")

    # Summary
    ok = sum(1 for v in results["endpoints"].values() if v.get("status") == "OK")
    total = len(results["endpoints"])
    logger.info(f"\nProbe complete: {ok}/{total} endpoints OK for league '{league}'")

    if ok == 0:
        logger.warning(
            "All endpoints failed.  Common causes:\n"
            "  1. League name is wrong (case-sensitive: try 'Settlers', 'Standard', etc.)\n"
            "  2. Network is blocked / sandbox environment\n"
            "  3. poe.ninja is down\n"
            "Tip: check available leagues at https://poe.ninja/"
        )

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  COMBINED COLLECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def collect_all(
    league: str | None = None,
    classes: list[str] | None = None,
    max_per_class: int | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Collect both builds and economy data."""
    out_dir = out_dir or (DATA_RAW_DIR / "poe_ninja")
    await collect_builds(league, classes, max_per_class, out_dir)
    await collect_economy(league, out_dir=out_dir)
    return out_dir


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    import typer

    app = typer.Typer(
        help="Collect build + economy data from poe.ninja",
        no_args_is_help=True,
    )

    @app.command()
    def builds(
        league: str = typer.Option(LEAGUE, help="League name (case-sensitive)"),
        cls: str | None = typer.Option(None, "--class", help="Single class to collect"),
        limit: int = typer.Option(10_000, help="Max builds per class"),
    ) -> None:
        """Collect build data (trees, items, skills) from poe.ninja."""
        classes = [cls] if cls else None
        asyncio.run(collect_builds(league=league, classes=classes, max_per_class=limit))

    @app.command()
    def economy(
        league: str = typer.Option(LEAGUE, help="League name"),
    ) -> None:
        """Snapshot economy prices (currency + items) from poe.ninja."""
        asyncio.run(collect_economy(league=league))

    @app.command()
    def probe(
        league: str = typer.Option(LEAGUE, help="League name to test"),
    ) -> None:
        """
        Test API connectivity — the correct way to probe poe.ninja.

        poe.ninja is an SPA; scraping its HTML gives you only a JS bundle.
        Always use the JSON API endpoints documented at:
        https://github.com/Davenads/poeninjaAPI-2025
        """
        results = asyncio.run(probe_api(league=league))
        print(json.dumps(results, indent=2))

    @app.command()
    def all(
        league: str = typer.Option(LEAGUE, help="League name"),
        cls: str | None = typer.Option(None, "--class", help="Single class"),
        limit: int = typer.Option(10_000, help="Max builds per class"),
    ) -> None:
        """Collect everything: builds + economy."""
        classes = [cls] if cls else None
        asyncio.run(collect_all(league=league, classes=classes, max_per_class=limit))

    app()


if __name__ == "__main__":
    main()
