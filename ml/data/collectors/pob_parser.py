"""
PoB build-code and XML parser.

Handles two input formats that Path of Building uses:
1. **Build codes** — Base64(Deflate(XML)) strings shared via websites
2. **Raw XML** — direct `.xml` build files

Extracts structured data ready for ML feature engineering:
- Passive tree node allocations
- Equipped items with parsed mods
- Gem/skill setups
- Character metadata (class, ascendancy, level, bandits, pantheon)

Usage:
    from ml.data.collectors.pob_parser import parse_build_code, parse_build_xml

    build = parse_build_code("eNrtfVuT28iR7l...")
    build = parse_build_xml(Path("my_build.xml"))
"""

from __future__ import annotations

import base64
import re
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Data classes ───────────────────────────────────────────────────────

@dataclass
class GemInfo:
    gem_id: str
    skill_id: str
    name: str
    level: int
    quality: int
    enabled: bool


@dataclass
class SkillGroup:
    label: str
    enabled: bool
    slot: str
    gems: list[GemInfo] = field(default_factory=list)


@dataclass
class ItemMod:
    """A single mod line on an item."""
    text: str
    mod_type: str  # "IMPLICIT", "EXPLICIT", "CRAFTED", "ENCHANT", etc.


@dataclass
class ItemInfo:
    """Parsed representation of one equipped item."""
    slot: str  # e.g. "Weapon 1", "Helmet", "Ring 1", etc.
    rarity: str  # "NORMAL", "MAGIC", "RARE", "UNIQUE"
    name: str
    base_type: str
    item_level: int
    quality: int
    sockets: str  # e.g. "R-R-R-G-G-B"
    mods: list[ItemMod] = field(default_factory=list)
    influences: list[str] = field(default_factory=list)


@dataclass
class ParsedBuild:
    """Everything extracted from a single PoB build."""
    # Character
    class_name: str = ""
    ascendancy: str = ""
    level: int = 0
    bandit: str = "None"
    pantheon_major: str = "None"
    pantheon_minor: str = "None"

    # Passive tree
    tree_version: str = ""
    allocated_nodes: list[int] = field(default_factory=list)
    mastery_selections: dict[int, int] = field(default_factory=dict)
    jewel_sockets: dict[int, str] = field(default_factory=dict)

    # Skills
    skill_groups: list[SkillGroup] = field(default_factory=list)
    main_socket_group: int = 0

    # Items
    items: list[ItemInfo] = field(default_factory=list)

    # Raw XML (for anything we haven't parsed yet)
    raw_xml: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict for JSON export."""
        return {
            "class_name": self.class_name,
            "ascendancy": self.ascendancy,
            "level": self.level,
            "bandit": self.bandit,
            "pantheon_major": self.pantheon_major,
            "pantheon_minor": self.pantheon_minor,
            "tree_version": self.tree_version,
            "allocated_nodes": self.allocated_nodes,
            "mastery_selections": self.mastery_selections,
            "jewel_sockets": {str(k): v for k, v in self.jewel_sockets.items()},
            "main_socket_group": self.main_socket_group,
            "skill_groups": [
                {
                    "label": sg.label,
                    "enabled": sg.enabled,
                    "slot": sg.slot,
                    "gems": [
                        {
                            "gem_id": g.gem_id,
                            "skill_id": g.skill_id,
                            "name": g.name,
                            "level": g.level,
                            "quality": g.quality,
                            "enabled": g.enabled,
                        }
                        for g in sg.gems
                    ],
                }
                for sg in self.skill_groups
            ],
            "items": [
                {
                    "slot": it.slot,
                    "rarity": it.rarity,
                    "name": it.name,
                    "base_type": it.base_type,
                    "item_level": it.item_level,
                    "quality": it.quality,
                    "sockets": it.sockets,
                    "mods": [{"text": m.text, "mod_type": m.mod_type} for m in it.mods],
                    "influences": it.influences,
                }
                for it in self.items
            ],
        }


# ── Build-code decoding ───────────────────────────────────────────────

def decode_build_code(code: str) -> str:
    """
    Decode a PoB build code into raw XML.

    Build codes are: URL-safe Base64 → inflate → XML.
    (The encoding is: XML → deflate → Base64 with +→- and /→_)
    """
    # Undo URL-safe replacements
    b64 = code.replace("-", "+").replace("_", "/")
    # Pad if necessary
    b64 += "=" * (-len(b64) % 4)
    compressed = base64.b64decode(b64)
    xml_bytes = zlib.decompress(compressed)
    return xml_bytes.decode("utf-8")


# ── XML parsing ────────────────────────────────────────────────────────

def _parse_tree(tree_elem: ET.Element, build: ParsedBuild) -> None:
    """Parse the <Tree> section."""
    # Each <Spec> child is a tree spec (there can be multiple; we want activeSpec)
    active_spec = int(tree_elem.get("activeSpec", "1")) - 1  # 1-indexed

    specs = list(tree_elem.iter("Spec"))
    if not specs:
        return

    spec = specs[min(active_spec, len(specs) - 1)]
    build.tree_version = spec.get("treeVersion", "")

    # <URL> contains the tree URL with node hashes
    url_elem = spec.find("URL")
    if url_elem is not None and url_elem.text:
        url_text = url_elem.text.strip()
        # URL format: https://www.pathofexile.com/passive-skill-tree/3.28.0/AAAA...
        # The hash at the end encodes node IDs
        # But poe.ninja data gives us node IDs directly — this is a fallback

    # <Sockets> contains jewel-socket assignments
    sockets_elem = spec.find("Sockets")
    if sockets_elem is not None:
        for socket in sockets_elem.iter("Socket"):
            node_id = int(socket.get("nodeId", "0"))
            item_id = socket.get("itemId", "")
            if node_id and item_id:
                build.jewel_sockets[node_id] = item_id

    # Node IDs may be encoded in the URL or stored as a comma-separated list
    # In modern PoB exports, they use <HashList> or embed in URL
    # We also handle direct treeHashes from poe.ninja (see poe_ninja.py)


def _parse_skills(skills_elem: ET.Element, build: ParsedBuild) -> None:
    """Parse the <Skills> section."""
    for sg_elem in skills_elem.iter("SkillGroup"):
        sg = SkillGroup(
            label=sg_elem.get("label", ""),
            enabled=sg_elem.get("enabled", "true").lower() == "true",
            slot=sg_elem.get("slot", ""),
        )
        for gem_elem in sg_elem.iter("Gem"):
            gem = GemInfo(
                gem_id=gem_elem.get("gemId", ""),
                skill_id=gem_elem.get("skillId", ""),
                name=gem_elem.get("nameSpec", ""),
                level=int(gem_elem.get("level", "20")),
                quality=int(gem_elem.get("quality", "0")),
                enabled=gem_elem.get("enabled", "true").lower() == "true",
            )
            sg.gems.append(gem)
        build.skill_groups.append(sg)


def _parse_items(items_elem: ET.Element, build: ParsedBuild) -> None:
    """Parse the <Items> section."""
    # <Items> contains <Item id="1"> blocks and <Slot> assignments
    item_map: dict[str, str] = {}  # id → raw item text

    for item_elem in items_elem.iter("Item"):
        item_id = item_elem.get("id", "")
        raw_text = (item_elem.text or "").strip()
        if item_id and raw_text:
            item_map[item_id] = raw_text

    # <Slot name="Body Armour" itemId="3" />
    for slot_elem in items_elem.iter("Slot"):
        slot_name = slot_elem.get("name", "")
        item_id = slot_elem.get("itemId", "")
        if not slot_name or not item_id or item_id not in item_map:
            continue

        item_info = _parse_item_text(item_map[item_id], slot_name)
        build.items.append(item_info)


def _parse_item_text(raw: str, slot: str) -> ItemInfo:
    """
    Parse a PoB item text block into an ItemInfo.

    Format:
        Rarity: RARE
        My Cool Helmet
        Bone Helmet
        Unique ID: abcdef
        Item Level: 86
        Quality: 20
        Sockets: R-R-R-G
        Implicits: 1
        +2 to Level of all Minion Skill Gems
        {tags:prefix}+92 to maximum Life
        {tags:suffix}+43% to Fire Resistance
        ...
    """
    lines = raw.strip().splitlines()
    info = ItemInfo(slot=slot, rarity="RARE", name="", base_type="",
                    item_level=0, quality=0, sockets="")

    i = 0
    # Rarity line
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Rarity:"):
            info.rarity = line.split(":", 1)[1].strip().upper()
            i += 1
            break
        i += 1

    # Name (1 or 2 lines depending on rarity)
    if i < len(lines):
        info.name = lines[i].strip()
        i += 1
    if i < len(lines) and not lines[i].strip().startswith(("Item Level", "Quality", "Sockets", "Implicits", "{")):
        info.base_type = lines[i].strip()
        i += 1
    else:
        info.base_type = info.name
        info.name = ""

    # Metadata lines
    num_implicits = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Item Level:"):
            info.item_level = int(re.search(r"\d+", line).group())  # type: ignore[union-attr]
        elif line.startswith("Quality:"):
            info.quality = int(re.search(r"\d+", line).group())  # type: ignore[union-attr]
        elif line.startswith("Sockets:"):
            info.sockets = line.split(":", 1)[1].strip()
        elif line.startswith("Implicits:"):
            num_implicits = int(re.search(r"\d+", line).group())  # type: ignore[union-attr]
            i += 1
            break
        elif line.startswith("Shaper Item"):
            info.influences.append("shaper")
        elif line.startswith("Elder Item"):
            info.influences.append("elder")
        elif line.startswith("Crusader Item"):
            info.influences.append("crusader")
        elif line.startswith("Hunter Item"):
            info.influences.append("hunter")
        elif line.startswith("Redeemer Item"):
            info.influences.append("redeemer")
        elif line.startswith("Warlord Item"):
            info.influences.append("warlord")
        i += 1

    # Implicit mods
    for _ in range(num_implicits):
        if i < len(lines):
            mod_text = lines[i].strip()
            info.mods.append(ItemMod(text=mod_text, mod_type="IMPLICIT"))
            i += 1

    # Explicit mods (rest of lines)
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        # PoB tags: {tags:prefix}, {tags:suffix}, {crafted}, {fractured}
        mod_type = "EXPLICIT"
        if "{crafted}" in line:
            mod_type = "CRAFTED"
            line = line.replace("{crafted}", "").strip()
        elif "{fractured}" in line:
            mod_type = "FRACTURED"
            line = line.replace("{fractured}", "").strip()
        # Strip tag markers
        line = re.sub(r"\{[^}]*\}", "", line).strip()
        if line:
            info.mods.append(ItemMod(text=line, mod_type=mod_type))
        i += 1

    return info


def parse_build_xml_string(xml_string: str) -> ParsedBuild:
    """Parse a PoB XML string into a ParsedBuild."""
    build = ParsedBuild(raw_xml=xml_string)

    root = ET.fromstring(xml_string)

    # <Build> element — character metadata
    build_elem = root.find("Build")
    if build_elem is not None:
        build.class_name = build_elem.get("className", "")
        build.ascendancy = build_elem.get("ascendClassName", "")
        build.level = int(build_elem.get("level", "1"))
        build.bandit = build_elem.get("bandit", "None")
        build.pantheon_major = build_elem.get("pantheonMajorGod", "None")
        build.pantheon_minor = build_elem.get("pantheonMinorGod", "None")
        build.main_socket_group = int(build_elem.get("mainSocketGroup", "1"))

    # <Tree> — passive allocations
    tree_elem = root.find("Tree")
    if tree_elem is not None:
        _parse_tree(tree_elem, build)

    # <Skills> — gem setups
    skills_elem = root.find("Skills")
    if skills_elem is not None:
        _parse_skills(skills_elem, build)

    # <Items> — equipped gear
    items_elem = root.find("Items")
    if items_elem is not None:
        _parse_items(items_elem, build)

    return build


# ── Public convenience functions ───────────────────────────────────────

def parse_build_code(code: str) -> ParsedBuild:
    """Decode a PoB build code and parse the resulting XML."""
    xml = decode_build_code(code)
    return parse_build_xml_string(xml)


def parse_build_xml(path: Path) -> ParsedBuild:
    """Parse a PoB XML file on disk."""
    xml = path.read_text(encoding="utf-8")
    return parse_build_xml_string(xml)
