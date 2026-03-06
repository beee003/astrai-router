"""
FastAPI server example — expose routing as an API.

Run:
    pip install astrai-router[litellm] fastapi uvicorn
    uvicorn examples.fastapi_server:app --reload
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    raise ImportError("Install fastapi: pip install fastapi uvicorn")

from astrai_router import (
    TaskClassifier,
    EnergyOracle,
    auto_route_request,
    compress_messages,
    configure_storage,
    get_cache_stats,
)

# ── Configure storage on startup ─────────────────────────────────────────
configure_storage("sqlite", path="./router_api.db")

app = FastAPI(title="Astrai Router API", version="0.1.0")
classifier = TaskClassifier()
oracle = EnergyOracle()


class RouteRequest(BaseModel):
    prompt: str
    user_id: str = "anonymous"
    models: List[Dict[str, Any]] = [
        {"model": "gpt-4o", "provider": "openai", "cost_per_1k": 5.0},
        {"model": "claude-sonnet-4-5", "provider": "anthropic", "cost_per_1k": 3.0},
        {"model": "llama-3.3-70b", "provider": "groq", "cost_per_1k": 0.59},
    ]
    quality_vs_cost: float = 0.5
    risk_tolerance: float = 0.3


class CompressRequest(BaseModel):
    messages: List[Dict[str, str]]
    task_type: str = "general"


class EnergyRequest(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int


@app.post("/route")
async def route(req: RouteRequest) -> Dict[str, Any]:
    """Route a prompt to the best model."""
    result = auto_route_request(
        prompt=req.prompt,
        user_id=req.user_id,
        available_models=req.models,
        quality_vs_cost=req.quality_vs_cost,
        risk_tolerance=req.risk_tolerance,
    )
    return result


@app.post("/classify")
async def classify(prompt: str) -> Dict[str, Any]:
    """Classify a prompt into task type."""
    result = classifier.classify(prompt)
    return result.to_dict()


@app.post("/compress")
async def compress(req: CompressRequest) -> Dict[str, Any]:
    """Compress conversation context."""
    compressed, manifest = compress_messages(req.messages, req.task_type)
    return {
        "compressed_messages": compressed,
        "manifest": manifest,
    }


@app.post("/energy")
async def energy(req: EnergyRequest) -> Dict[str, Any]:
    """Estimate energy consumption."""
    est = oracle.estimate_energy(req.model, req.input_tokens, req.output_tokens)
    return {
        "total_joules": est.total_joules,
        "watt_hours": est.watt_hours,
        "co2_grams": est.co2_grams,
        "model_tier": est.model_tier,
    }


@app.get("/cache/stats")
async def cache_stats() -> Dict[str, Any]:
    """Get semantic cache statistics."""
    return get_cache_stats()


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
