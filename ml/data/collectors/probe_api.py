#!/usr/bin/env python3
"""
Standalone poe.ninja API probe — test connectivity and inspect responses.

Run this script to verify the correct way to interact with poe.ninja:

    python -m ml.data.collectors.probe_api
    python -m ml.data.collectors.probe_api --league Standard
    python -m ml.data.collectors.probe_api --discover-leagues

IMPORTANT: poe.ninja is a Single-Page Application (SPA).  Scraping its
HTML (like with urllib + regex) will only give you a JS bundle, NOT build
data.  The correct approach is to use the **JSON API endpoints**:

  Builds (2-step versioned flow, 2026+):
    Step 1: GET https://poe.ninja/poe1/api/data/index-state
            → returns snapshotVersions with version strings
    Step 2: GET https://poe.ninja/poe1/api/builds/{version}/overview?overview={league_url}&type=exp
            → returns builds overview (classNames, builds, uniqueItems, etc.)

  Currency:  https://poe.ninja/api/data/currencyoverview?league={league}&type=Currency
  Items:     https://poe.ninja/api/data/itemoverview?league={league}&type={type}

No auth required.  Rate limit: 12 requests / 5 minutes.

Documentation:
  https://github.com/Davenads/poeninjaAPI-2025
  https://github.com/JakubRak-gamedev/poeninja_API_guide
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import requests

# ── Configuration ──────────────────────────────────────────────────────

# ⚠ This MUST be the API base, NOT the frontend SPA URL.
#    CORRECT:  https://poe.ninja/api/data        (JSON API — returns data)
#    WRONG:    https://poe.ninja/poe1/mirage      (SPA page — returns JS bundle)
#    WRONG:    https://poe.ninja/poe1/builds/...   (also SPA)
#
# The league name goes in the *query parameter*, not the base path.
POE_NINJA_BASE = "https://poe.ninja/api/data"
POE_OFFICIAL_API = "https://api.pathofexile.com"
# Update this to the current challenge league each cycle.
# Override at runtime with --league or POE_LEAGUE env var.
DEFAULT_LEAGUE = os.environ.get("POE_LEAGUE", "Mirage")

HEADERS = {
    "User-Agent": "PoB-ML-Probe/1.0 (github.com/Breviel/PathOfBuilding)",
    "Accept": "application/json",
}


# ── HTTP helper ────────────────────────────────────────────────────────

def _get(url: str, params: dict[str, str] | None = None, retries: int = 3) -> dict[str, Any]:
    """GET JSON with retries + exponential back-off."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                print(f"  ⚠  Rate limited — waiting {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            wait = 2 ** attempt
            print(f"  ⚠  Attempt {attempt + 1}/{retries} failed: {exc}")
            if attempt < retries - 1:
                print(f"      Retrying in {wait}s…")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


# ── Probes ─────────────────────────────────────────────────────────────

def probe_builds(league: str) -> dict[str, Any]:
    """Test builds API using the 2-step versioned flow.

    Step 1: Fetch index-state to get snapshot version.
    Step 2: Fetch builds overview using the versioned endpoint.
    """
    index_state_url = "https://poe.ninja/poe1/api/data/index-state"
    builds_base = "https://poe.ninja/poe1/api/builds"

    print(f"\n{'═' * 70}")
    print(f"  BUILDS PROBE (versioned 2-step flow)")
    print(f"  League: '{league}'")
    print(f"{'═' * 70}")

    # Step 1: Fetch index-state
    print(f"\n  Step 1: Fetching index-state → {index_state_url}")
    try:
        index_data = _get(index_state_url)
    except Exception as exc:
        print(f"  ✗ index-state failed: {exc}")
        return {"status": "FAILED", "error": f"index-state failed: {exc}"}

    build_leagues = index_data.get("buildLeagues", [])
    snapshots = index_data.get("snapshotVersions", [])
    print(f"  ✓ {len(build_leagues)} build leagues, {len(snapshots)} snapshots")
    for bl in build_leagues:
        print(f"    League: {bl.get('name')} (url: {bl.get('url')})")
    for s in snapshots:
        print(f"    Snapshot: {s.get('url')}/{s.get('type')} → {s.get('version')}")

    # Find the matching snapshot version
    league_lower = league.lower()
    version = None
    league_url = league_lower

    for s in snapshots:
        if s.get("url", "").lower() == league_lower and s.get("type") == "exp":
            version = s.get("version")
            league_url = s.get("url", league_lower)
            break

    # Fallback: try any matching snapshot
    if not version:
        for s in snapshots:
            if s.get("url", "").lower() == league_lower:
                version = s.get("version")
                league_url = s.get("url", league_lower)
                print(f"  ⚠ No 'exp' snapshot; using type={s.get('type')!r}")
                break

    if not version:
        print(f"  ✗ No snapshot found for league '{league}'")
        return {"status": "FAILED", "error": f"No snapshot for league {league!r}"}

    # Step 2: Fetch builds overview
    url = f"{builds_base}/{version}/overview"
    params = {"overview": league_url, "type": "exp"}
    print(f"\n  Step 2: Fetching builds → {url}")
    print(f"    Params: {params}")

    try:
        data = _get(url, params)
        top_keys = list(data.keys())
        builds = data.get("builds", [])
        class_names = data.get("classNames", [])

        print(f"  ✓ SUCCESS (version={version})")
        print(f"  ✓ Response top-level keys: {top_keys}")
        print(f"  ✓ classNames: {class_names}")
        print(f"  ✓ {len(builds)} builds returned")

        if builds:
            sample = builds[0]
            print(f"\n  Sample build record keys:")
            for k, v in sample.items():
                vtype = type(v).__name__
                if isinstance(v, list):
                    vinfo = f"list[{len(v)}]"
                elif isinstance(v, dict):
                    vinfo = f"dict[{len(v)} keys]"
                elif isinstance(v, str):
                    vinfo = repr(v[:60])
                else:
                    vinfo = repr(v)
                print(f"    {k:20s} : {vtype:8s} = {vinfo}")

            # Show tree hashes
            tree = sample.get("treeHashes", [])
            print(f"\n  Passive tree: {len(tree)} node IDs")
            if tree:
                print(f"    First 10: {tree[:10]}")

        return {
            "status": "OK",
            "version": version,
            "endpoint": f"{url}?overview={league_url}&type=exp",
            "n_builds": len(builds),
            "classNames": class_names,
        }

    except Exception as exc:
        print(f"  ✗ Builds overview failed: {exc}")
        return {"status": "FAILED", "error": str(exc)}


def probe_currency(league: str) -> dict[str, Any]:
    """Test currency API."""
    url = f"{POE_NINJA_BASE}/currencyoverview"
    params = {"league": league, "type": "Currency", "language": "en"}

    print(f"\n{'═' * 70}")
    print(f"  CURRENCY PROBE — {url}")
    print(f"  Params: {params}")
    print(f"{'═' * 70}")

    try:
        data = _get(url, params)
        lines = data.get("lines", [])
        details = data.get("currencyDetails", [])

        print(f"  ✓ {len(lines)} currency lines, {len(details)} currency details")

        if lines:
            print(f"\n  Top currencies by chaos value:")
            sorted_lines = sorted(lines, key=lambda x: x.get("chaosEquivalent", 0), reverse=True)
            for c in sorted_lines[:10]:
                name = c.get("currencyTypeName", "?")
                chaos = c.get("chaosEquivalent", 0)
                print(f"    {name:30s} = {chaos:>10.1f} chaos")

            print(f"\n  Sample line keys: {list(lines[0].keys())}")

        return {"status": "OK", "n_currencies": len(lines)}

    except Exception as exc:
        print(f"  ✗ FAILED: {exc}")
        return {"status": "FAILED", "error": str(exc)}


def probe_items(league: str, item_type: str = "DivinationCard") -> dict[str, Any]:
    """Test item overview API."""
    url = f"{POE_NINJA_BASE}/itemoverview"
    params = {"league": league, "type": item_type, "language": "en"}

    print(f"\n{'═' * 70}")
    print(f"  ITEMS PROBE — {url}")
    print(f"  Params: {params}")
    print(f"{'═' * 70}")

    try:
        data = _get(url, params)
        lines = data.get("lines", [])

        print(f"  ✓ {len(lines)} {item_type} entries")

        if lines:
            print(f"\n  Top {item_type} by chaos value:")
            sorted_lines = sorted(lines, key=lambda x: x.get("chaosValue", 0), reverse=True)
            for item in sorted_lines[:10]:
                name = item.get("name", "?")
                chaos = item.get("chaosValue", 0)
                count = item.get("count", 0)
                print(f"    {name:40s} {chaos:>10.1f}c  (listed: {count})")

            print(f"\n  Sample line keys: {list(lines[0].keys())}")

        return {"status": "OK", "n_items": len(lines)}

    except Exception as exc:
        print(f"  ✗ FAILED: {exc}")
        return {"status": "FAILED", "error": str(exc)}


def discover_leagues() -> list[str]:
    """Try to discover valid league names from official GGG API."""
    print(f"\n{'═' * 70}")
    print(f"  LEAGUE DISCOVERY — {POE_OFFICIAL_API}/league")
    print(f"{'═' * 70}")

    try:
        data = _get(f"{POE_OFFICIAL_API}/league", {"realm": "pc", "type": "main"})
        leagues = data if isinstance(data, list) else data.get("leagues", [])

        print(f"  ✓ Found {len(leagues)} leagues")
        for lg in leagues[:15]:
            name = lg.get("name", "?") if isinstance(lg, dict) else str(lg)
            print(f"    - {name}")

        return [lg.get("name", "") if isinstance(lg, dict) else str(lg) for lg in leagues]

    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        print("  Falling back to common league names:")
        common = ["Mirage", "Standard", "Hardcore"]
        for name in common:
            print(f"    - {name}")
        return common


def test_league_name(league: str) -> bool:
    """Quick test: does this league name work with poe.ninja?"""
    try:
        data = _get(
            f"{POE_NINJA_BASE}/currencyoverview",
            {"league": league, "type": "Currency", "language": "en"},
        )
        return len(data.get("lines", [])) > 0
    except Exception:
        return False


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Probe poe.ninja API endpoints",
        epilog=(
            "NOTE: poe.ninja is an SPA. Scraping HTML with urllib gives you a "
            "JS bundle, not data. Always use the JSON API."
        ),
    )
    parser.add_argument("--league", default=DEFAULT_LEAGUE, help="League name (case-sensitive)")
    parser.add_argument("--discover-leagues", action="store_true", help="Discover leagues from GGG API")
    parser.add_argument("--test-league", help="Test if a league name works")
    parser.add_argument("--item-type", default="DivinationCard", help="Item type to probe")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    if args.discover_leagues:
        leagues = discover_leagues()
        if args.json:
            print(json.dumps(leagues, indent=2))
        return

    # Guard: detect if someone edited POE_NINJA_BASE to a frontend URL
    if "/builds/" in POE_NINJA_BASE:
        print(f"\n  ⚠  ERROR: POE_NINJA_BASE looks like a frontend SPA URL, not the API.")
        print(f"     Current: {POE_NINJA_BASE}")
        print(f"     Expected: https://poe.ninja/api/data")
        print(f"\n  Economy endpoints use /api/data/ (no poe1 prefix).")
        print(f"  Builds use the versioned 2-step flow via /poe1/api/builds/{{version}}/")
        print(f"\n  Fix: set POE_NINJA_BASE = \"https://poe.ninja/api/data\"")
        sys.exit(1)

    if args.test_league:
        ok = test_league_name(args.test_league)
        status = "✓ works" if ok else "✗ not found"
        print(f"League '{args.test_league}': {status}")
        sys.exit(0 if ok else 1)

    print(f"\n{'━' * 70}")
    print(f"  poe.ninja API Probe — League: {args.league}")
    print(f"{'━' * 70}")
    print(f"\n  Correct approach: use JSON API, not HTML scraping.")
    print(f"  Economy base: {POE_NINJA_BASE}")
    print(f"  Builds: 2-step versioned flow via /poe1/api/builds/{{version}}/")
    print(f"  Docs: https://github.com/Davenads/poeninjaAPI-2025")

    results = {}
    results["builds"] = probe_builds(args.league)
    time.sleep(1)  # rate limit courtesy
    results["currency"] = probe_currency(args.league)
    time.sleep(1)
    results["items"] = probe_items(args.league, args.item_type)

    # Summary
    ok = sum(1 for v in results.values() if v.get("status") == "OK")
    total = len(results)

    print(f"\n{'━' * 70}")
    print(f"  SUMMARY: {ok}/{total} endpoints OK for league '{args.league}'")
    print(f"{'━' * 70}")

    if ok == 0:
        print(f"\n  All endpoints failed. Possible causes:")
        print(f"    1. Wrong league name (try: Mirage, Standard, …)")
        print(f"    2. Network blocked (sandboxed environment)")
        print(f"    3. poe.ninja is temporarily down")
        print(f"\n  To discover available leagues:")
        print(f"    python -m ml.data.collectors.probe_api --discover-leagues")

    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
