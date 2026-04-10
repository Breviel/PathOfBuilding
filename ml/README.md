# Path of Building — ML Optimization Suite

Three machine-learning systems that plug into the Path of Building ecosystem to
help players squeeze the most out of their passive tree, atlas tree, and gear.

| Model | Goal | Architecture |
|-------|------|--------------|
| **Build Tree Optimizer** | Best passive-point allocation for a given build | Graph Attention Network (GAT) / PPO RL |
| **Atlas Tree Optimizer** | Best atlas-point allocation for a farming strategy | ML return predictor + CP-SAT ILP solver |
| **Gear Improvement Suggester** | Prioritised next-upgrade list for equipped gear | XGBoost slot ranker → LambdaMART mod recommender |

---

## Quick start

```bash
# 1. Install Python dependencies
cd ml/
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Collect training data (poe.ninja + PoB tree JSON)
make collect-data          # or: python -m data.collectors.poe_ninja
                           #     python -m data.collectors.tree_data

# 3. Build features
make features              # or: python -m data.processing.feature_engineering

# 4. Train a model
make train-build-tree      # passive tree GNN
make train-atlas           # atlas tree optimizer
make train-gear            # gear improvement suggester

# 5. Run inference
make serve                 # FastAPI server on :8000
```

---

## Directory layout

```
ml/
├── README.md                          ← you are here
├── requirements.txt                   ← Python deps
├── config.py                          ← centralised settings
├── Makefile                           ← shortcuts
│
├── data/
│   ├── collectors/
│   │   ├── poe_ninja.py               ← scrape builds from poe.ninja
│   │   ├── pob_parser.py              ← parse PoB XML / build codes
│   │   ├── tree_data.py               ← parse GGG passive-tree JSON
│   │   ├── atlas_data.py              ← scrape atlas trees from poe-atlas.com + poe.ninja
│   │   └── trade_api.py               ← PoE Trade API client
│   ├── processing/
│   │   ├── feature_engineering.py     ← build feature vectors
│   │   └── dataset.py                 ← PyTorch / XGBoost datasets
│   └── raw/                           ← downloaded snapshots (gitignored)
│
├── models/
│   ├── build_tree_optimizer.py        ← GAT + RL for passive tree
│   ├── atlas_tree_optimizer.py        ← ML + ILP for atlas tree
│   └── gear_suggester.py             ← XGBoost + LambdaMART for gear
│
├── training/
│   ├── train_build_tree.py
│   ├── train_atlas.py
│   └── train_gear.py
│
└── inference/
    └── predict.py                     ← FastAPI prediction endpoint
```

---

## Data sources

| Source | URL | What we extract |
|--------|-----|----------------|
| **poe.ninja** | `poe.ninja/api/data/builds` | Top builds: tree allocations, gear, DPS |
| **poe.ninja Atlas Trees** | `poe.ninja/builds/atlas` | Popular atlas allocations, heatmaps |
| **GGG Passive Tree** | `pathofexile.com/api/passive-tree` | Node graph: IDs, stats, connections, positions |
| **GGG Atlas Tree** | Datamined / community repos | Atlas nodes, mechanics, spawn-chance bonuses |
| **poe-atlas.com** | `poe-atlas.com` | Curated atlas strategy builds per mechanic & game phase |
| **Maxroll Atlas Planner** | `maxroll.gg/poe/poe-atlas-tree` | Interactive atlas trees, exported allocations |
| **PoE Vault Atlas Strategies** | `poe-vault.com/guides/atlas-passive-skill-tree-strategies` | Atlas strategy tier lists |
| **PoE Trade API** | `pathofexile.com/api/trade` | Item prices by mod combination |
| **Craft of Exile** | `craftofexile.com` (manual) | Crafting cost estimates, mod weights |

All raw data is saved under `data/raw/` and is **gitignored**.

---

## Model details

### 1. Build Tree Optimizer

- **Input**: Class, ascendancy, main skill tags, desired stat weights
- **Graph**: ~1 300 nodes with edges from `in`/`out` connections in `tree.lua`
- **Node features**: `[stat_vector, is_keystone, is_notable, is_mastery, orbit, group]`
- **Output**: Per-node allocation probability (sigmoid)
- **Loss**: BCE on node selection + connectivity penalty + point-budget penalty
- **Constraint enforcement**: Post-hoc BFS from class start to prune disconnected picks

### 2. Atlas Tree Optimizer

- **Input**: Farming strategy (mechanic weights), league economy snapshot
- **Strategies**: Pre-defined archetypes sourced from poe-atlas.com:
  - League Start (map sustain), Scarabs/Shrines/Strongboxes,
    Harvest, Expedition, Heist, Blight, Delirium, Legion, Breach,
    Abyss, Boss/Maven, Ritual+Beyond, Fortress farming
- **Step 1**: Trained regressor predicts per-node return given economy context
- **Step 2**: Google OR-Tools CP-SAT solves knapsack with synergy terms
- **Step 3**: Strategy-specific seed allocations from poe-atlas.com as warm start
- **Output**: ~130-point allocation + estimated chaos/hour breakdown

### 3. Gear Improvement Suggester

- **Slot Impact Predictor** (XGBoost): ranks which slot gives the biggest DPS/EHP lift
- **Mod Recommender** (LambdaMART): ranks candidate mod upgrades per slot
- **Cost Estimator**: queries trade API / Craft of Exile for buy-vs-craft cost
- **Output**: ordered list of `(slot, target_mods, action, cost_estimate)`

---

## Configuration

All tunable knobs live in `config.py`:

- API endpoints and rate limits
- League name (updates each league)
- Model hyperparameters (hidden dims, learning rates, etc.)
- Feature engineering toggles
- Training settings (epochs, batch size, checkpointing)

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0, PyTorch Geometric ≥ 2.4
- XGBoost, LightGBM
- Google OR-Tools (for atlas ILP)
- FastAPI + Uvicorn (serving)
- aiohttp (async data collection)

See `requirements.txt` for pinned versions.
