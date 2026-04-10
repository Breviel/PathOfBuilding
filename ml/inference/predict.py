"""
Unified prediction API — FastAPI server for all three models.

Endpoints:
    POST /api/v1/optimize-build-tree    → passive tree allocation
    POST /api/v1/optimize-atlas         → atlas tree allocation
    POST /api/v1/suggest-gear           → gear improvement suggestions
    POST /api/v1/parse-build            → parse a PoB build code
    GET  /api/v1/health                 → service health check

Usage:
    uvicorn ml.inference.predict:app --host 0.0.0.0 --port 8000
    # or:  make serve
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError("fastapi + pydantic required: pip install fastapi uvicorn pydantic")

from ml.config import CHECKPOINT_DIR

app = FastAPI(
    title="PoB ML Optimization Suite",
    description="ML-powered build tree, atlas tree, and gear optimization for Path of Exile",
    version="0.1.0",
)


# ── Request / Response models ──────────────────────────────────────────

class BuildTreeRequest(BaseModel):
    """Request to optimize a passive build tree."""
    class_name: str = Field(..., description="Character class (e.g. 'Witch')")
    ascendancy: str = Field(..., description="Ascendancy (e.g. 'Occultist')")
    level: int = Field(90, description="Character level")
    main_skill: str = Field("", description="Main skill name")
    stat_weights: dict[str, float] = Field(
        default_factory=lambda: {"dps": 0.6, "ehp": 0.4},
        description="Objective weights",
    )
    locked_nodes: list[int] = Field(
        default_factory=list,
        description="Node IDs to force-include",
    )


class BuildTreeResponse(BaseModel):
    allocated_nodes: list[int]
    total_points: int
    predicted_dps_score: float
    predicted_ehp_score: float
    confidence: float


class AtlasRequest(BaseModel):
    strategy: dict[str, float] | None = Field(
        None,
        description=(
            "Custom farming strategy weights by mechanic (e.g. {'Expedition': 1.0}). "
            "If omitted, strategy_name is used instead."
        ),
    )
    strategy_name: str | None = Field(
        None,
        description=(
            "Pre-defined strategy from poe-atlas.com: league_start, early_mapping, "
            "scarab_shrine_strongbox, harvest, expedition, heist, blight, delirium, "
            "legion, breach, abyss, bossing, ritual_beyond, fortress_farming, "
            "breach_delirium, expedition_shrines_altars"
        ),
    )
    league_day: int = Field(30, description="Day since league start")
    divine_rate: float = Field(180, description="1 Divine = X Chaos")
    max_points: int = Field(132, description="Atlas point budget")


class AtlasResponse(BaseModel):
    selected_nodes: list[int]
    total_return_per_hour: float
    breakdown: dict[str, float]  # per-mechanic return
    solver_status: str
    strategy_used: str


class GearSuggestRequest(BaseModel):
    build_code: str = Field("", description="PoB build code (base64)")
    top_k: int = Field(3, description="Number of slot suggestions to return")


class GearSuggestion(BaseModel):
    slot: str
    predicted_dps_delta: float
    predicted_ehp_delta: float
    confidence: float
    target_mods: list[str]
    recommended_action: str
    estimated_cost_chaos: float


class GearSuggestResponse(BaseModel):
    suggestions: list[GearSuggestion]
    current_dps_estimate: float
    current_ehp_estimate: float


class ParseBuildRequest(BaseModel):
    build_code: str = Field(..., description="PoB build code")


class ParseBuildResponse(BaseModel):
    class_name: str
    ascendancy: str
    level: int
    num_allocated_nodes: int
    num_items: int
    num_skill_groups: int
    skill_names: list[str]


# ── Lazy model loading ─────────────────────────────────────────────────

_models: dict[str, Any] = {}


def _load_model(name: str) -> Any:
    """Lazy-load a model from checkpoint directory."""
    if name in _models:
        return _models[name]

    ckpt = CHECKPOINT_DIR

    if name == "gear_pipeline":
        from ml.models.gear_suggester import GearSuggestionPipeline
        pipeline = GearSuggestionPipeline()
        if (ckpt / "slot_dps.xgb").exists():
            pipeline.load_models(str(ckpt))
            logger.info("Loaded gear suggestion pipeline")
        else:
            logger.warning("Gear models not found — using untrained pipeline")
        _models[name] = pipeline
        return pipeline

    if name == "atlas_predictor":
        from ml.models.atlas_tree_optimizer import AtlasReturnPredictor
        predictor = AtlasReturnPredictor()
        if (ckpt / "atlas_return_predictor.xgb").exists():
            predictor.load(str(ckpt / "atlas_return_predictor.xgb"))
            logger.info("Loaded atlas return predictor")
        else:
            logger.warning("Atlas model not found — using untrained predictor")
        _models[name] = predictor
        return predictor

    raise ValueError(f"Unknown model: {name}")


# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/api/v1/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/api/v1/parse-build", response_model=ParseBuildResponse)
async def parse_build(req: ParseBuildRequest) -> ParseBuildResponse:
    """Parse a PoB build code and return structured data."""
    try:
        from ml.data.collectors.pob_parser import parse_build_code
        build = parse_build_code(req.build_code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse build code: {e}")

    skill_names = []
    for sg in build.skill_groups:
        for gem in sg.gems:
            if gem.name and gem.enabled:
                skill_names.append(gem.name)

    return ParseBuildResponse(
        class_name=build.class_name,
        ascendancy=build.ascendancy,
        level=build.level,
        num_allocated_nodes=len(build.allocated_nodes),
        num_items=len(build.items),
        num_skill_groups=len(build.skill_groups),
        skill_names=skill_names[:10],
    )


@app.post("/api/v1/optimize-build-tree", response_model=BuildTreeResponse)
async def optimize_build_tree(req: BuildTreeRequest) -> BuildTreeResponse:
    """
    Optimize passive tree allocation for the given build.

    Note: This is a placeholder that returns mock data until the model is trained.
    """
    # TODO: Load trained GAT model and run inference
    # For now, return a placeholder
    return BuildTreeResponse(
        allocated_nodes=req.locked_nodes or [1, 2, 3],
        total_points=len(req.locked_nodes) if req.locked_nodes else 3,
        predicted_dps_score=0.0,
        predicted_ehp_score=0.0,
        confidence=0.0,
    )


@app.post("/api/v1/optimize-atlas", response_model=AtlasResponse)
async def optimize_atlas(req: AtlasRequest) -> AtlasResponse:
    """
    Optimize atlas tree allocation for the given farming strategy.

    Supports both custom weights and pre-defined strategy names
    from poe-atlas.com (e.g. "expedition", "breach_delirium").
    """
    from ml.config import ATLAS_STRATEGIES

    # Resolve strategy
    strategy_name = req.strategy_name or "custom"
    if req.strategy:
        strategy_used = "custom"
    elif req.strategy_name and req.strategy_name in ATLAS_STRATEGIES:
        strategy_used = req.strategy_name
    else:
        available = list(ATLAS_STRATEGIES.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Provide 'strategy' weights or a valid 'strategy_name'. Available: {available}",
        )

    return AtlasResponse(
        selected_nodes=[],
        total_return_per_hour=0.0,
        breakdown={},
        solver_status="MODEL_NOT_TRAINED",
        strategy_used=strategy_used,
    )


@app.post("/api/v1/suggest-gear", response_model=GearSuggestResponse)
async def suggest_gear(req: GearSuggestRequest) -> GearSuggestResponse:
    """
    Suggest gear improvements for a build.

    Parses the build code, extracts gear features, and runs the
    slot impact predictor + mod recommender pipeline.
    """
    if not req.build_code:
        raise HTTPException(status_code=400, detail="build_code is required")

    try:
        from ml.data.collectors.pob_parser import parse_build_code
        build = parse_build_code(req.build_code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse build: {e}")

    # Extract features
    from ml.data.processing.feature_engineering import (
        extract_gear_features,
        gear_features_to_array,
    )

    items_data = build.to_dict()["items"]
    gear_feats = extract_gear_features(items_data)
    gear_array = gear_features_to_array(gear_feats)

    build_context = np.array([
        0,  # class_id placeholder
        build.level,
        0,  # dps placeholder
        0,  # ehp placeholder
    ], dtype=np.float32)

    # Run pipeline
    pipeline = _load_model("gear_pipeline")
    suggestions = pipeline.suggest(gear_array, build_context, top_k=req.top_k)

    return GearSuggestResponse(
        suggestions=[
            GearSuggestion(
                slot=s.slot_name,
                predicted_dps_delta=s.predicted_dps_delta,
                predicted_ehp_delta=s.predicted_ehp_delta,
                confidence=s.confidence,
                target_mods=[m.mod_text for m in s.target_mods],
                recommended_action=s.recommended_action or "analyze",
                estimated_cost_chaos=s.trade_price_chaos or s.craft_cost_chaos,
            )
            for s in suggestions
        ],
        current_dps_estimate=0,
        current_ehp_estimate=0,
    )
