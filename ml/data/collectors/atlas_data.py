"""
Atlas tree data collector — scrapes strategy builds from poe-atlas.com
and atlas tree heatmaps from poe.ninja.

poe-atlas.com provides curated atlas builds organized by:
- Game phase: League Start, Early Mapping, Endgame, Boss Slaying
- Mechanic focus: Harvest, Expedition, Delirium, Blight, Legion, Breach,
  Scarabs, Shrines, Strongboxes, Abyss, Heist, Ritual, Beyond, etc.

Since poe-atlas.com does not expose a public API, this module:
1. Scrapes build pages for atlas tree URLs / exported allocations
2. Parses linked planner URLs (PoE Planner, Maxroll) for node IDs
3. Labels each tree with its strategy category for supervised learning
4. Also fetches popular atlas heatmaps from poe.ninja

Usage:
    python -m ml.data.collectors.atlas_data
    python -m ml.data.collectors.atlas_data --strategy expedition
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
from loguru import logger

from ml.config import (
    ATLAS_STRATEGIES,
    DATA_RAW_DIR,
    LEAGUE,
    MAXROLL_ATLAS_PLANNER_URL,
    POE_ATLAS_COM_URL,
    POE_NINJA_ATLAS_URL,
    POE_VAULT_ATLAS_URL,
)


# ── poe-atlas.com known pages ─────────────────────────────────────────
# These are the guide categories published on poe-atlas.com.
# Each entry maps to a strategy key in config.ATLAS_STRATEGIES.

POE_ATLAS_PAGES: dict[str, str] = {
    "league_start": f"{POE_ATLAS_COM_URL}/league-starter/",
    "early_mapping": f"{POE_ATLAS_COM_URL}/early-mapping/",
    "endgame_general": f"{POE_ATLAS_COM_URL}/best-endgame-atlas/",
    "harvest": f"{POE_ATLAS_COM_URL}/harvest/",
    "expedition": f"{POE_ATLAS_COM_URL}/expedition/",
    "delirium": f"{POE_ATLAS_COM_URL}/delirium/",
    "blight": f"{POE_ATLAS_COM_URL}/blight/",
    "legion": f"{POE_ATLAS_COM_URL}/legion/",
    "breach": f"{POE_ATLAS_COM_URL}/breach/",
    "scarab_shrine_strongbox": f"{POE_ATLAS_COM_URL}/scarabs/",
    "bossing": f"{POE_ATLAS_COM_URL}/bossing/",
    "heist": f"{POE_ATLAS_COM_URL}/heist/",
}

# Additional community atlas planner URLs
MAXROLL_ATLAS_STRATEGIES: dict[str, str] = {
    "maxroll_planner": MAXROLL_ATLAS_PLANNER_URL,
}


# ── Data classes ───────────────────────────────────────────────────────

@dataclass
class AtlasTreeBuild:
    """A single atlas tree build from a guide source."""
    source: str                          # "poe-atlas.com", "poe.ninja", "maxroll"
    strategy: str                        # key from ATLAS_STRATEGIES
    strategy_label: str                  # human-readable label
    game_phase: str                      # "league_start", "early_mapping", "endgame", "bossing"
    allocated_node_ids: list[int] = field(default_factory=list)
    total_points: int = 0
    planner_url: str = ""
    description: str = ""
    reported_chaos_per_hour: float = 0   # if reported by the guide
    mechanics_focus: list[str] = field(default_factory=list)


# ── Planner URL parsing ───────────────────────────────────────────────

def parse_poe_planner_url(url: str) -> list[int]:
    """
    Extract allocated node IDs from a PoE Planner atlas tree URL.

    PoE Planner URLs encode the tree as a hash in the fragment:
        https://poeplanner.com/atlas-tree/AAAA...

    The hash is a base64-encoded list of node IDs (similar to passive tree URLs).
    """
    # Extract the hash portion
    match = re.search(r"atlas-tree/([A-Za-z0-9_\-]+)", url)
    if not match:
        return []

    import base64
    try:
        hash_str = match.group(1)
        # URL-safe base64 → bytes
        b64 = hash_str.replace("-", "+").replace("_", "/")
        b64 += "=" * (-len(b64) % 4)
        data = base64.b64decode(b64)

        # Decode as pairs of big-endian uint16 node IDs
        # (this is the common PoE tree encoding)
        node_ids = []
        for i in range(0, len(data) - 1, 2):
            nid = (data[i] << 8) | data[i + 1]
            if nid > 0:
                node_ids.append(nid)
        return node_ids
    except Exception as e:
        logger.debug(f"Failed to parse planner URL {url}: {e}")
        return []


def parse_maxroll_url(url: str) -> list[int]:
    """
    Extract node IDs from a Maxroll atlas tree planner URL.

    Maxroll encodes the atlas tree in a similar base64 hash format.
    """
    match = re.search(r"poe-atlas-tree/([A-Za-z0-9_\-]+)", url)
    if not match:
        return []
    return parse_poe_planner_url(url)  # same encoding logic


# ── Page scraping ──────────────────────────────────────────────────────

async def _fetch_page(
    session: aiohttp.ClientSession,
    url: str,
) -> str:
    """Fetch a page's HTML content."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 200:
                return await resp.text()
            logger.warning(f"HTTP {resp.status} for {url}")
            return ""
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return ""


def _extract_planner_urls(html: str) -> list[str]:
    """Extract atlas planner URLs from an HTML page."""
    patterns = [
        r'https?://poeplanner\.com/atlas-tree/[A-Za-z0-9_\-]+',
        r'https?://maxroll\.gg/poe/poe-atlas-tree/[A-Za-z0-9_\-]+',
        r'https?://www\.pathofexile\.com/fullscreen-atlas-skill-tree/[A-Za-z0-9_\-]+',
    ]
    urls = []
    for pattern in patterns:
        urls.extend(re.findall(pattern, html))
    return list(set(urls))


def _classify_game_phase(strategy: str) -> str:
    """Map strategy key to game phase."""
    if strategy in ("league_start",):
        return "league_start"
    if strategy in ("early_mapping",):
        return "early_mapping"
    if strategy in ("bossing",):
        return "bossing"
    return "endgame"


async def scrape_poe_atlas_com(
    strategies: list[str] | None = None,
) -> list[AtlasTreeBuild]:
    """
    Scrape atlas tree builds from poe-atlas.com.

    For each strategy page:
    1. Fetch the HTML
    2. Extract any planner URLs
    3. Parse node IDs from the planner URL
    4. Create an AtlasTreeBuild record

    Parameters
    ----------
    strategies : list[str]
        Strategy keys to scrape (default: all known pages).

    Returns
    -------
    list[AtlasTreeBuild]
    """
    if strategies is None:
        strategies = list(POE_ATLAS_PAGES.keys())

    builds: list[AtlasTreeBuild] = []

    async with aiohttp.ClientSession(
        headers={"User-Agent": "PoB-ML-Suite/1.0 (atlas collector)"}
    ) as session:
        for strategy in strategies:
            url = POE_ATLAS_PAGES.get(strategy)
            if not url:
                logger.warning(f"No known page for strategy: {strategy}")
                continue

            logger.info(f"Scraping poe-atlas.com: {strategy} ({url})")
            html = await _fetch_page(session, url)

            if not html:
                # Create a placeholder build with strategy metadata only
                builds.append(AtlasTreeBuild(
                    source="poe-atlas.com",
                    strategy=strategy,
                    strategy_label=strategy.replace("_", " ").title(),
                    game_phase=_classify_game_phase(strategy),
                    description=f"Guide from {url} (HTML not available)",
                    mechanics_focus=list(ATLAS_STRATEGIES.get(strategy, {}).keys()),
                ))
                continue

            planner_urls = _extract_planner_urls(html)
            logger.info(f"  Found {len(planner_urls)} planner URLs")

            if planner_urls:
                for purl in planner_urls:
                    node_ids = parse_poe_planner_url(purl) or parse_maxroll_url(purl)
                    builds.append(AtlasTreeBuild(
                        source="poe-atlas.com",
                        strategy=strategy,
                        strategy_label=strategy.replace("_", " ").title(),
                        game_phase=_classify_game_phase(strategy),
                        allocated_node_ids=node_ids,
                        total_points=len(node_ids),
                        planner_url=purl,
                        mechanics_focus=list(ATLAS_STRATEGIES.get(strategy, {}).keys()),
                    ))
            else:
                # No planner URL found — still record the strategy as a label
                builds.append(AtlasTreeBuild(
                    source="poe-atlas.com",
                    strategy=strategy,
                    strategy_label=strategy.replace("_", " ").title(),
                    game_phase=_classify_game_phase(strategy),
                    description=f"Guide text available, no planner URL extracted",
                    mechanics_focus=list(ATLAS_STRATEGIES.get(strategy, {}).keys()),
                ))

            # Rate limit
            await asyncio.sleep(1.0)

    logger.info(f"Scraped {len(builds)} atlas builds from poe-atlas.com")
    return builds


# ── poe.ninja atlas tree heatmap ──────────────────────────────────────

async def fetch_poe_ninja_atlas_heatmap(
    league: str | None = None,
) -> dict[str, Any]:
    """
    Fetch popular atlas tree allocations from poe.ninja.

    poe.ninja's atlas trees section shows a heatmap of which nodes
    are most commonly allocated. We try their undocumented endpoint.

    Returns raw JSON or empty dict if unavailable.
    """
    league = league or LEAGUE

    # Undocumented endpoint (may change; discovered via browser inspection)
    url = f"https://poe.ninja/api/data/atlas-tree-overview?league={league}&language=en"

    async with aiohttp.ClientSession(
        headers={"User-Agent": "PoB-ML-Suite/1.0"}
    ) as session:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"Fetched poe.ninja atlas heatmap ({len(data)} keys)")
                    return data
                logger.info(f"poe.ninja atlas endpoint returned {resp.status} (may not be available)")
                return {}
        except Exception as e:
            logger.info(f"poe.ninja atlas endpoint not available: {e}")
            return {}


# ── Save collected data ────────────────────────────────────────────────

def save_atlas_builds(
    builds: list[AtlasTreeBuild],
    out_dir: Path | None = None,
) -> Path:
    """Save collected atlas builds to JSON."""
    out_dir = out_dir or (DATA_RAW_DIR / "atlas")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "source": b.source,
            "strategy": b.strategy,
            "strategy_label": b.strategy_label,
            "game_phase": b.game_phase,
            "allocated_node_ids": b.allocated_node_ids,
            "total_points": b.total_points,
            "planner_url": b.planner_url,
            "description": b.description,
            "reported_chaos_per_hour": b.reported_chaos_per_hour,
            "mechanics_focus": b.mechanics_focus,
        }
        for b in builds
    ]

    out_path = out_dir / "atlas_strategy_builds.json"
    out_path.write_text(json.dumps(data, indent=2))
    logger.info(f"Saved {len(data)} atlas builds → {out_path}")
    return out_path


# ── Full collection pipeline ──────────────────────────────────────────

async def collect_all(
    strategies: list[str] | None = None,
    league: str | None = None,
    out_dir: Path | None = None,
) -> Path:
    """
    Collect atlas tree data from all sources:
    1. poe-atlas.com strategy guides
    2. poe.ninja atlas heatmap
    """
    out_dir = out_dir or (DATA_RAW_DIR / "atlas")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. poe-atlas.com
    builds = await scrape_poe_atlas_com(strategies)
    save_atlas_builds(builds, out_dir)

    # 2. poe.ninja heatmap
    heatmap = await fetch_poe_ninja_atlas_heatmap(league)
    if heatmap:
        heatmap_path = out_dir / "poe_ninja_atlas_heatmap.json"
        heatmap_path.write_text(json.dumps(heatmap, indent=2))
        logger.info(f"Saved heatmap → {heatmap_path}")

    # 3. Save strategy definitions (from config) for reference
    strategies_path = out_dir / "strategy_definitions.json"
    strategies_path.write_text(json.dumps(ATLAS_STRATEGIES, indent=2))
    logger.info(f"Saved strategy definitions → {strategies_path}")

    logger.success(f"Atlas data collection complete → {out_dir}")
    return out_dir


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    import typer

    app = typer.Typer(help="Collect atlas tree data from poe-atlas.com and poe.ninja")

    @app.command()
    def collect(
        strategy: str | None = typer.Option(None, help="Single strategy to scrape"),
        league: str = typer.Option(LEAGUE, help="League name"),
    ) -> None:
        strategies = [strategy] if strategy else None
        asyncio.run(collect_all(strategies=strategies, league=league))

    app()


if __name__ == "__main__":
    main()
