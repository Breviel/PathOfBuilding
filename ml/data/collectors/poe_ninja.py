"""
poe.ninja build scraper.

Collects top builds from poe.ninja's public API, extracting:
- Passive tree allocations (node IDs)
- Equipped gear (item base, mods, links)
- Skill gem setups
- Character class / ascendancy
- Computed DPS & EHP estimates (from poe.ninja's own calculations)

Usage:
    python -m ml.data.collectors.poe_ninja            # collect all classes
    python -m ml.data.collectors.poe_ninja --class Witch --limit 5000
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import aiohttp
from loguru import logger

from ml.config import (
    DATA_RAW_DIR,
    LEAGUE,
    POE_NINJA_BUILDS_ENDPOINT,
    POE_NINJA_RPS,
    TrainingConfig,
)

# ── poe.ninja API helpers ──────────────────────────────────────────────

# poe.ninja exposes builds at:
#   GET /api/data/builds?overview={league}&type=exp&language=en
# It returns a JSON blob with:
#   {
#     "classNames": [...],
#     "uniqueItems": [...],       ← shared lookup tables
#     "keystoneHashes": [...],
#     "builds": [
#       {
#         "account": "...",
#         "character": "...",
#         "class": 3,              ← index into classNames
#         "level": 99,
#         "treeHashes": [12345, ...],  ← allocated passive node IDs
#         "items": [{...}, ...],
#         "skills": [{...}, ...],
#         "life": 5200,
#         "es": 0,
#         "dps": 1_500_000,
#         ...
#       },
#       ...
#     ]
#   }

OVERVIEW_URL = f"{POE_NINJA_BUILDS_ENDPOINT}?overview={{league}}&type=exp&language=en"

# poe.ninja also has per-class endpoints for deeper data
CLASS_URL = (
    f"{POE_NINJA_BUILDS_ENDPOINT}?overview={{league}}"
    "&type=exp&language=en&class={{class_name}}"
)


async def _rate_limited_get(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """GET with rate limiting."""
    async with semaphore:
        async with session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.json()
            # Respect rate limit
            await asyncio.sleep(1.0 / POE_NINJA_RPS)
            return data


async def fetch_builds_for_class(
    session: aiohttp.ClientSession,
    class_name: str,
    league: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Fetch builds for a single class."""
    url = CLASS_URL.format(league=league, class_name=class_name)
    logger.info(f"Fetching builds for {class_name} from {url}")
    return await _rate_limited_get(session, url, semaphore)


def _normalise_build(
    raw_build: dict[str, Any],
    class_names: list[str],
) -> dict[str, Any]:
    """
    Flatten a single poe.ninja build record into a training-friendly dict.

    Returns
    -------
    dict with keys:
        account, character, class_name, ascendancy, level,
        tree_node_ids, items, skills, life, es, dps, ...
    """
    class_idx = raw_build.get("class", 0)
    class_name = class_names[class_idx] if class_idx < len(class_names) else "Unknown"

    return {
        "account": raw_build.get("account", ""),
        "character": raw_build.get("character", ""),
        "class_name": class_name,
        "level": raw_build.get("level", 0),
        "tree_node_ids": raw_build.get("treeHashes", []),
        "items": raw_build.get("items", []),
        "skills": raw_build.get("skills", []),
        "life": raw_build.get("life", 0),
        "es": raw_build.get("es", 0),
        "dps": raw_build.get("dps", 0),
        "depth": raw_build.get("depth", 0),
    }


# ── Public API ─────────────────────────────────────────────────────────

async def collect_all(
    league: str | None = None,
    classes: list[str] | None = None,
    max_per_class: int | None = None,
    out_dir: Path | None = None,
) -> Path:
    """
    Scrape poe.ninja and write one JSON file per class under ``out_dir``.

    Parameters
    ----------
    league : str
        League name (default from config).
    classes : list[str]
        Class names to scrape (default: all 7).
    max_per_class : int
        Cap builds per class (default from TrainingConfig).
    out_dir : Path
        Destination directory (default: ``DATA_RAW_DIR / "poe_ninja"``).

    Returns
    -------
    Path to output directory.
    """
    league = league or LEAGUE
    cfg = TrainingConfig()
    classes = classes or cfg.classes
    max_per_class = max_per_class or cfg.max_builds_per_class
    out_dir = out_dir or (DATA_RAW_DIR / "poe_ninja")
    out_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(POE_NINJA_RPS)

    async with aiohttp.ClientSession() as session:
        # 1. Overview (for shared lookup tables)
        overview_url = OVERVIEW_URL.format(league=league)
        logger.info(f"Fetching overview from {overview_url}")
        overview = await _rate_limited_get(session, overview_url, semaphore)
        class_names = overview.get("classNames", [])

        # Save lookup tables
        lookup_path = out_dir / "lookup_tables.json"
        lookup_data = {
            "classNames": class_names,
            "uniqueItems": overview.get("uniqueItems", []),
            "keystoneHashes": overview.get("keystoneHashes", []),
            "skills": overview.get("skills", []),
        }
        lookup_path.write_text(json.dumps(lookup_data, indent=2))
        logger.info(f"Saved lookup tables → {lookup_path}")

        # 2. Per-class collection
        for cls in classes:
            logger.info(f"Collecting {cls} (max {max_per_class})...")
            try:
                data = await fetch_builds_for_class(
                    session, cls, league, semaphore,
                )
            except aiohttp.ClientError as exc:
                logger.warning(f"Failed to fetch {cls}: {exc}")
                continue

            raw_builds = data.get("builds", [])[:max_per_class]
            builds = [_normalise_build(b, class_names) for b in raw_builds]

            out_path = out_dir / f"{cls.lower()}_builds.json"
            out_path.write_text(json.dumps(builds, indent=2))
            logger.info(f"  → {len(builds)} builds saved to {out_path}")

    logger.success(f"Collection complete. Data in {out_dir}")
    return out_dir


# ── CLI entry point ────────────────────────────────────────────────────

def main() -> None:
    import typer

    app = typer.Typer(help="Collect builds from poe.ninja")

    @app.command()
    def collect(
        league: str = typer.Option(LEAGUE, help="League name"),
        cls: str | None = typer.Option(None, "--class", help="Single class to collect"),
        limit: int = typer.Option(10_000, help="Max builds per class"),
    ) -> None:
        classes = [cls] if cls else None
        asyncio.run(collect_all(league=league, classes=classes, max_per_class=limit))

    app()


if __name__ == "__main__":
    main()
