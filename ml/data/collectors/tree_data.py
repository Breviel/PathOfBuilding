"""
GGG passive-tree and atlas-tree data parser.

Reads the tree data that ships with Path of Building (under ``src/TreeData/``)
and converts it into graph structures suitable for GNN training.

The passive tree is a large graph (~1 300 nodes) with:
- Node features (stats, keystone/notable/mastery flags, position)
- Edges (bidirectional connections between adjacent nodes)
- Group membership (visual clusters on the tree)
- Class start positions

This module also exposes helpers to build a ``networkx.DiGraph`` and a
``torch_geometric.data.Data`` object from the raw tree.

Usage:
    python -m ml.data.collectors.tree_data              # writes processed tree
    python -m ml.data.collectors.tree_data --version 3_28
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from loguru import logger

from ml.config import DATA_RAW_DIR, GGG_TREE_DATA_VERSION, ROOT_DIR

# Path of Building ships Lua tree data under src/TreeData/<version>/tree.lua
POB_TREE_DIR = ROOT_DIR.parent / "src" / "TreeData"


# ── Data classes ───────────────────────────────────────────────────────

@dataclass
class TreeNode:
    """A single node on the passive skill tree."""
    node_id: int
    name: str
    group_id: int
    orbit: int
    orbit_index: int
    stats: list[str]
    is_keystone: bool = False
    is_notable: bool = False
    is_mastery: bool = False
    is_jewel_socket: bool = False
    is_ascendancy: bool = False
    ascendancy_name: str = ""
    class_start_index: int = -1  # ≥0 means this is a class-start node
    connections_in: list[int] = field(default_factory=list)
    connections_out: list[int] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0
    mastery_effects: list[dict[str, Any]] = field(default_factory=list)
    recipe: list[str] = field(default_factory=list)  # anointing oils


@dataclass
class PassiveTree:
    """Full passive tree graph."""
    version: str
    nodes: dict[int, TreeNode]
    edges: list[tuple[int, int]]
    class_starts: dict[str, int]  # class_name → start_node_id
    groups: dict[int, dict[str, Any]]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)


# ── Lua → Python parsing ──────────────────────────────────────────────

def _lua_to_json_ish(lua_text: str) -> str:
    """
    Best-effort conversion of a Lua table literal into something
    ``json.loads`` can handle.

    This is NOT a full Lua parser — it handles the subset that
    PoB's tree.lua uses (string keys, numeric keys, booleans,
    nested tables, and numeric/string values).
    """
    text = lua_text

    # Replace Lua comments
    text = re.sub(r"--\[\[.*?\]\]", "", text, flags=re.DOTALL)
    text = re.sub(r"--[^\n]*", "", text)

    # Replace Lua booleans
    text = text.replace("true", "True").replace("false", "False")
    text = text.replace("nil", "None")

    # [123] = { ... }  →  "123": { ... }
    text = re.sub(r"\[(\d+)\]\s*=", r'"\1":', text)

    # ["key"] = val  →  "key": val
    text = re.sub(r'\["([^"]+)"\]\s*=', r'"\1":', text)

    # key = val  →  "key": val  (for bare identifiers)
    text = re.sub(r'(\n\s*)(\w+)\s*=\s*', r'\1"\2": ', text)

    # Trailing commas before }
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*\]", "]", text)

    return text


def _parse_tree_lua(version: str) -> dict[str, Any]:
    """
    Load the PoB tree.lua for a given version and return it as a Python dict.

    Falls back to eval-based parsing since the Lua format is close to Python
    dict syntax after basic substitutions.
    """
    version_dir = POB_TREE_DIR / version
    tree_path = version_dir / "tree.lua"

    if not tree_path.exists():
        # List available versions
        available = [d.name for d in POB_TREE_DIR.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"Tree data not found at {tree_path}. "
            f"Available versions: {available}"
        )

    logger.info(f"Loading tree data from {tree_path}")
    raw = tree_path.read_text(encoding="utf-8", errors="replace")

    # The file usually starts with: return { ... }
    # Extract the table
    match = re.search(r"return\s*\{", raw)
    if match:
        raw = raw[match.start() + len("return"):]

    # Try Python eval (Lua table syntax is very close to Python dict)
    # This works for the simple key=value structure in tree.lua
    try:
        data = eval(raw, {"__builtins__": {}, "True": True, "False": False, "None": None})
        return data
    except Exception as e:
        logger.warning(f"Direct eval failed ({e}), attempting regex-based conversion...")

    # Fallback: regex conversion then eval
    converted = _lua_to_json_ish(raw)
    try:
        data = eval(converted, {"__builtins__": {}, "True": True, "False": False, "None": None})
        return data
    except Exception as e2:
        logger.error(f"Lua parsing failed: {e2}")
        raise


# ── Build graph from parsed data ──────────────────────────────────────

def _extract_nodes_and_edges(
    raw: dict[str, Any],
) -> tuple[dict[int, TreeNode], list[tuple[int, int]], dict[str, int], dict[int, dict]]:
    """
    Walk the parsed tree data and extract nodes, edges, class starts, and groups.
    """
    nodes: dict[int, TreeNode] = {}
    edges: list[tuple[int, int]] = []
    class_starts: dict[str, int] = {}
    groups_raw = raw.get("groups", {})
    groups = {}

    # Class-start mapping
    CLASS_IDS = {
        0: "Scion", 1: "Marauder", 2: "Ranger",
        3: "Witch", 4: "Duelist", 5: "Templar", 6: "Shadow",
    }

    # Process groups
    for gid_str, gdata in groups_raw.items():
        gid = int(gid_str) if isinstance(gid_str, str) else gid_str
        groups[gid] = gdata

    # Process nodes (may be nested under groups or flat)
    nodes_raw = raw.get("nodes", {})
    for nid_str, ndata in nodes_raw.items():
        nid = int(nid_str) if isinstance(nid_str, str) else nid_str
        if not isinstance(ndata, dict):
            continue

        node = TreeNode(
            node_id=nid,
            name=ndata.get("name", ndata.get("dn", "")),
            group_id=int(ndata.get("group", ndata.get("g", 0))),
            orbit=int(ndata.get("orbit", ndata.get("o", 0))),
            orbit_index=int(ndata.get("orbitIndex", ndata.get("oidx", 0))),
            stats=ndata.get("stats", ndata.get("sd", [])),
            is_keystone=bool(ndata.get("isKeystone", ndata.get("ks", False))),
            is_notable=bool(ndata.get("isNotable", ndata.get("not", False))),
            is_mastery=bool(ndata.get("isMastery", False)),
            is_jewel_socket=bool(ndata.get("isJewelSocket", False)),
            is_ascendancy=bool(ndata.get("isAscendancyStart", False) or ndata.get("ascendancyName")),
            ascendancy_name=ndata.get("ascendancyName", ""),
            class_start_index=int(ndata.get("classStartIndex", -1)),
        )

        # Position (may be computed from group + orbit, or stored directly)
        node.x = float(ndata.get("x", 0))
        node.y = float(ndata.get("y", 0))

        # Connections
        out_nodes = ndata.get("out", [])
        in_nodes = ndata.get("in", [])
        node.connections_out = [int(x) for x in out_nodes] if out_nodes else []
        node.connections_in = [int(x) for x in in_nodes] if in_nodes else []

        # Mastery effects
        node.mastery_effects = ndata.get("masteryEffects", [])

        # Recipe (anointing)
        node.recipe = ndata.get("recipe", [])

        nodes[nid] = node

        # Track class starts
        if node.class_start_index >= 0:
            cls_name = CLASS_IDS.get(node.class_start_index, f"Class{node.class_start_index}")
            class_starts[cls_name] = nid

    # Build edge list (bidirectional)
    seen_edges: set[tuple[int, int]] = set()
    for nid, node in nodes.items():
        for neighbor in node.connections_out + node.connections_in:
            if neighbor in nodes:
                edge = (min(nid, neighbor), max(nid, neighbor))
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    edges.append(edge)

    return nodes, edges, class_starts, groups


def load_passive_tree(version: str | None = None) -> PassiveTree:
    """
    Load the passive skill tree for a given version.

    Parameters
    ----------
    version : str
        Tree version directory name (e.g. "3_28"). Defaults to config.

    Returns
    -------
    PassiveTree with all nodes, edges, class starts, and group data.
    """
    version = version or GGG_TREE_DATA_VERSION.replace(".", "_")
    raw = _parse_tree_lua(version)
    nodes, edges, class_starts, groups = _extract_nodes_and_edges(raw)

    tree = PassiveTree(
        version=version,
        nodes=nodes,
        edges=edges,
        class_starts=class_starts,
        groups=groups,
    )
    logger.info(
        f"Loaded tree v{version}: {tree.num_nodes} nodes, "
        f"{tree.num_edges} edges, {len(class_starts)} class starts"
    )
    return tree


# ── Conversion to networkx ─────────────────────────────────────────────

def tree_to_networkx(tree: PassiveTree) -> nx.Graph:
    """Convert a PassiveTree into a networkx undirected graph."""
    G = nx.Graph()

    for nid, node in tree.nodes.items():
        G.add_node(
            nid,
            name=node.name,
            is_keystone=node.is_keystone,
            is_notable=node.is_notable,
            is_mastery=node.is_mastery,
            is_jewel_socket=node.is_jewel_socket,
            is_ascendancy=node.is_ascendancy,
            orbit=node.orbit,
            group_id=node.group_id,
            x=node.x,
            y=node.y,
            num_stats=len(node.stats),
            class_start=node.class_start_index >= 0,
        )

    for u, v in tree.edges:
        G.add_edge(u, v)

    return G


def tree_to_pyg(tree: PassiveTree) -> Any:
    """
    Convert a PassiveTree into a PyTorch Geometric ``Data`` object.

    Returns ``torch_geometric.data.Data`` with:
    - ``x``: node feature matrix [N, F]
    - ``edge_index``: COO edge tensor [2, E]
    - ``node_ids``: original node IDs
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError(
            "torch and torch_geometric are required. "
            "Install with: pip install torch torch-geometric"
        )

    # Build node-ID → contiguous-index mapping
    id_to_idx = {nid: i for i, nid in enumerate(sorted(tree.nodes.keys()))}
    num_nodes = len(id_to_idx)

    # Node features: [is_keystone, is_notable, is_mastery, is_jewel_socket,
    #                  is_ascendancy, orbit, num_stats, x_norm, y_norm]
    FEATURE_DIM = 9
    x = np.zeros((num_nodes, FEATURE_DIM), dtype=np.float32)

    # Compute position bounds for normalisation
    xs = [n.x for n in tree.nodes.values()]
    ys = [n.y for n in tree.nodes.values()]
    x_min, x_max = (min(xs), max(xs)) if xs else (0, 1)
    y_min, y_max = (min(ys), max(ys)) if ys else (0, 1)
    x_range = max(x_max - x_min, 1)
    y_range = max(y_max - y_min, 1)

    node_ids = []
    for nid in sorted(tree.nodes.keys()):
        idx = id_to_idx[nid]
        node = tree.nodes[nid]
        node_ids.append(nid)
        x[idx] = [
            float(node.is_keystone),
            float(node.is_notable),
            float(node.is_mastery),
            float(node.is_jewel_socket),
            float(node.is_ascendancy),
            float(node.orbit),
            float(len(node.stats)),
            (node.x - x_min) / x_range,
            (node.y - y_min) / y_range,
        ]

    # Edge index (COO format, bidirectional)
    src, dst = [], []
    for u, v in tree.edges:
        if u in id_to_idx and v in id_to_idx:
            src.extend([id_to_idx[u], id_to_idx[v]])
            dst.extend([id_to_idx[v], id_to_idx[u]])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x_tensor = torch.from_numpy(x)

    return Data(
        x=x_tensor,
        edge_index=edge_index,
        node_ids=torch.tensor(node_ids, dtype=torch.long),
        num_nodes=num_nodes,
    )


# ── Serialisation ──────────────────────────────────────────────────────

def save_tree_json(tree: PassiveTree, path: Path | None = None) -> Path:
    """Serialise a PassiveTree to JSON for downstream use."""
    path = path or (DATA_RAW_DIR / f"passive_tree_{tree.version}.json")
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "version": tree.version,
        "num_nodes": tree.num_nodes,
        "num_edges": tree.num_edges,
        "class_starts": tree.class_starts,
        "nodes": {
            str(nid): {
                "name": n.name,
                "group_id": n.group_id,
                "orbit": n.orbit,
                "is_keystone": n.is_keystone,
                "is_notable": n.is_notable,
                "is_mastery": n.is_mastery,
                "is_jewel_socket": n.is_jewel_socket,
                "is_ascendancy": n.is_ascendancy,
                "ascendancy_name": n.ascendancy_name,
                "stats": n.stats,
                "x": n.x,
                "y": n.y,
                "connections": sorted(set(n.connections_in + n.connections_out)),
            }
            for nid, n in tree.nodes.items()
        },
        "edges": [[u, v] for u, v in tree.edges],
    }

    path.write_text(json.dumps(data, indent=2))
    logger.info(f"Saved tree JSON → {path}")
    return path


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    import typer

    app = typer.Typer(help="Parse and export PoB passive tree data")

    @app.command()
    def export(
        version: str = typer.Option(
            GGG_TREE_DATA_VERSION.replace(".", "_"),
            help="Tree version directory (e.g. 3_28)",
        ),
    ) -> None:
        tree = load_passive_tree(version)
        save_tree_json(tree)

        # Also build networkx graph and print basic stats
        G = tree_to_networkx(tree)
        logger.info(
            f"NetworkX graph: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges, "
            f"connected components: {nx.number_connected_components(G)}"
        )

    app()


if __name__ == "__main__":
    main()
