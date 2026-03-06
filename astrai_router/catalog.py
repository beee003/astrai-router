import asyncio
import math
import os
import re
import time
from typing import Dict, Any, List, Optional

import httpx


OPENROUTER_MODELS_URL = os.getenv(
    "OPENROUTER_MODELS_URL", "https://openrouter.ai/api/v1/models"
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


MODEL_CATALOG_TTL_SEC = _env_int("MODEL_CATALOG_TTL_SEC", 21600)  # 6 hours

MODEL_CATALOG: Dict[str, Dict[str, Any]] = {}
MODEL_CATALOG_LAST_SYNC: float = 0.0
MODEL_CATALOG_LAST_ERROR: Optional[str] = None

_SYNC_LOCK = asyncio.Lock()


def _parse_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _sanitize_price(value: float) -> float:
    """Clamp invalid/sentinel pricing values to a safe non-negative number."""
    if not math.isfinite(value) or value < 0:
        return 0.0
    return value


def normalize_family(name: str) -> str:
    """
    Normalize model family names to a stable key:
    - lowercase
    - replace "_" with "-"
    - strip YYYY-MM-DD or YYYYMMDD suffixes
    - convert digit-digit to digit.digit (e.g., 4-5 -> 4.5)
    """
    if not name:
        return ""
    s = name.lower().replace("_", "-")
    s = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", s)
    s = re.sub(r"-\d{8}$", "", s)
    s = re.sub(r"(?<=\d)-(?=\d)", ".", s)
    return s


def _is_open_model(entry: Dict[str, Any]) -> bool:
    if entry.get("hugging_face_id"):
        return True
    model_id = (entry.get("id") or "").lower()
    if model_id.endswith(":free"):
        return True
    pricing = entry.get("pricing") or {}
    if not bool(pricing.get("valid", True)):
        return False
    # Support both normalized pricing keys and legacy raw keys.
    prompt_price = pricing.get("prompt_per_token", pricing.get("prompt"))
    completion_price = pricing.get("completion_per_token", pricing.get("completion"))
    return _parse_float(prompt_price) == 0.0 and _parse_float(completion_price) == 0.0


def _build_entry(model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    model_id = model.get("id")
    if not model_id:
        return None

    pricing = model.get("pricing") or {}
    raw_prompt = _parse_float(pricing.get("prompt"))
    raw_completion = _parse_float(pricing.get("completion"))
    prompt = _sanitize_price(raw_prompt)
    completion = _sanitize_price(raw_completion)
    pricing_valid = raw_prompt >= 0 and raw_completion >= 0
    architecture = model.get("architecture") or {}
    family_raw = model_id.split("/", 1)[1] if "/" in model_id else model_id

    entry = {
        "id": model_id,
        "name": model.get("name", model_id),
        "provider": model_id.split("/")[0] if "/" in model_id else "unknown",
        "family": family_raw,
        "family_normalized": normalize_family(family_raw),
        "context_length": model.get("context_length"),
        "pricing": {
            "prompt_per_token": prompt,
            "completion_per_token": completion,
            "prompt_per_1m": round(prompt * 1_000_000, 6),
            "completion_per_1m": round(completion * 1_000_000, 6),
            "valid": pricing_valid,
        },
        "architecture": architecture,
        "supported_parameters": model.get("supported_parameters") or [],
        "input_modalities": model.get("input_modalities") or [],
        "output_modalities": model.get("output_modalities") or [],
        "top_provider": model.get("top_provider"),
        "hugging_face_id": model.get("hugging_face_id"),
    }
    entry["is_open"] = _is_open_model(entry)
    entry["is_free"] = pricing_valid and prompt == 0.0 and completion == 0.0
    return entry


async def sync_openrouter_catalog(force: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Sync model catalog from OpenRouter.
    Returns the current catalog (may be cached if within TTL).
    """
    global MODEL_CATALOG_LAST_SYNC, MODEL_CATALOG_LAST_ERROR, MODEL_CATALOG

    now = time.time()
    if (
        not force
        and MODEL_CATALOG
        and (now - MODEL_CATALOG_LAST_SYNC) < MODEL_CATALOG_TTL_SEC
    ):
        return MODEL_CATALOG

    async with _SYNC_LOCK:
        now = time.time()
        if (
            not force
            and MODEL_CATALOG
            and (now - MODEL_CATALOG_LAST_SYNC) < MODEL_CATALOG_TTL_SEC
        ):
            return MODEL_CATALOG

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(OPENROUTER_MODELS_URL, timeout=15.0)
                response.raise_for_status()
                data = response.json().get("data", [])

            new_catalog: Dict[str, Dict[str, Any]] = {}
            for model in data:
                entry = _build_entry(model)
                if not entry:
                    continue
                new_catalog[entry["id"].lower()] = entry

            if new_catalog:
                MODEL_CATALOG = new_catalog
                MODEL_CATALOG_LAST_SYNC = time.time()
                MODEL_CATALOG_LAST_ERROR = None
        except Exception as exc:
            MODEL_CATALOG_LAST_ERROR = str(exc)

    return MODEL_CATALOG


async def run_model_catalog_sync_loop() -> None:
    """Background loop to keep the catalog fresh."""
    await sync_openrouter_catalog(force=True)
    while True:
        await asyncio.sleep(MODEL_CATALOG_TTL_SEC)
        await sync_openrouter_catalog(force=False)


def get_model_catalog() -> Dict[str, Dict[str, Any]]:
    """Get the current model catalog (may be empty if not synced yet)."""
    return MODEL_CATALOG


def find_models_by_family(
    model_family: str, provider: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Find catalog entries matching a normalized family (optionally by provider)."""
    if not model_family:
        return []
    target = normalize_family(model_family)
    results = []
    for entry in MODEL_CATALOG.values():
        if entry.get("family_normalized") != target:
            continue
        if provider and entry.get("provider") != provider:
            continue
        results.append(entry)
    return results


def get_family_price_per_1m(
    model_family: str, provider: Optional[str] = None
) -> Optional[tuple]:
    """
    Return (input_price, output_price) per 1M tokens from the catalog.
    If multiple entries match, returns the cheapest blended price.
    """
    entries = []
    for entry in find_models_by_family(model_family, provider=provider):
        pricing = entry.get("pricing") or {}
        if not bool(pricing.get("valid", True)):
            continue
        prompt_per_1m = _parse_float(pricing.get("prompt_per_1m"))
        completion_per_1m = _parse_float(pricing.get("completion_per_1m"))
        if not math.isfinite(prompt_per_1m) or not math.isfinite(completion_per_1m):
            continue
        if prompt_per_1m < 0 or completion_per_1m < 0:
            continue
        entries.append(entry)
    if not entries:
        return None

    def blended(entry: Dict[str, Any]) -> float:
        pricing = entry.get("pricing") or {}
        return (
            pricing.get("prompt_per_1m", 0.0) + pricing.get("completion_per_1m", 0.0)
        ) / 2

    best = min(entries, key=blended)
    pricing = best.get("pricing") or {}
    return (pricing.get("prompt_per_1m", 0.0), pricing.get("completion_per_1m", 0.0))
