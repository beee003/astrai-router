"""
Smart Classifier — LLM-based task classification with graceful fallback.

Replaces keyword regex heuristics with a Groq Llama-4-Scout call (~100ms, free tier).
Falls back to existing detect_task_type() + TaskClassifier on timeout/error.

Features:
- Input sanitization (control chars, length truncation)
- In-memory cache with TTL and bounded size
- 2-second timeout on LLM call
- Validation of LLM outputs against known enum values
"""

import asyncio
import hashlib
import json
import re
import time
import logging
import unicodedata
from dataclasses import dataclass
from typing import Dict, Optional
from litellm import acompletion

logger = logging.getLogger(__name__)

# Known valid values for validation
VALID_TASK_TYPES = frozenset(
    {
        "code",
        "research",
        "chat",
        "creative",
        "analysis",
        "math",
        "reasoning",
        "summarization",
        "extraction",
        "translation",
        "unknown",
    }
)
VALID_COMPLEXITIES = frozenset({"low", "medium", "high"})

# Cache config
_CACHE_TTL_SECONDS = 300  # 5 minutes
_CACHE_MAX_ENTRIES = 10_000
_MAX_PROMPT_LENGTH = 500

# Regex to strip control characters (keep \t, \n, \r)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

_SYSTEM_PROMPT = (
    "You are a task classifier. Given a user prompt, respond ONLY with a JSON object "
    "(no markdown, no explanation) with these fields:\n"
    '  "task_type": one of code|research|chat|creative|analysis|math|reasoning|summarization|extraction|translation|unknown\n'
    '  "complexity": one of low|medium|high\n'
    '  "requires_reasoning": true or false\n'
    '  "requires_code": true or false\n'
    '  "requires_creativity": true or false\n'
    '  "requires_math": true or false\n'
)


@dataclass
class SmartClassification:
    task_type: str = "unknown"
    complexity: str = "medium"
    requires_reasoning: bool = False
    requires_code: bool = False
    requires_creativity: bool = False
    requires_math: bool = False
    confidence: float = 0.5
    source: str = "fallback"  # "llm" or "fallback"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_cache: Dict[str, tuple] = {}  # key -> (SmartClassification, expire_ts)
_cache_lock = asyncio.Lock()


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


async def _get_cached(key: str) -> Optional[SmartClassification]:
    async with _cache_lock:
        entry = _cache.get(key)
        if entry and entry[1] > time.time():
            return entry[0]
        if entry:
            del _cache[key]
    return None


async def _set_cached(key: str, value: SmartClassification):
    async with _cache_lock:
        # Evict oldest entries if over limit
        if len(_cache) >= _CACHE_MAX_ENTRIES:
            # Remove ~10% oldest by expiry
            sorted_keys = sorted(_cache, key=lambda k: _cache[k][1])
            for k in sorted_keys[: _CACHE_MAX_ENTRIES // 10]:
                del _cache[k]
        _cache[key] = (value, time.time() + _CACHE_TTL_SECONDS)


# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------
def sanitize_prompt(text: str) -> str:
    """Strip control characters and truncate."""
    text = _CONTROL_CHAR_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    return text[:_MAX_PROMPT_LENGTH].strip()


# ---------------------------------------------------------------------------
# Fallback classifier (existing keyword heuristics)
# ---------------------------------------------------------------------------
def _fallback_classify(prompt: str) -> SmartClassification:
    """Use existing heuristic classifiers as fallback."""
    try:
        from .learning import detect_task_type
        from .classifier import TaskClassifier

        task_type_enum, confidence = detect_task_type(prompt)
        classifier = TaskClassifier()
        hints = classifier.classify(prompt)

        task_type_str = task_type_enum.value if task_type_enum else "unknown"
        # Map to our extended set
        if task_type_str not in VALID_TASK_TYPES:
            task_type_str = "unknown"

        return SmartClassification(
            task_type=task_type_str,
            complexity=hints.complexity if hints else "medium",
            requires_reasoning=getattr(hints, "requires_reasoning", False),
            requires_code=getattr(hints, "requires_code", False),
            requires_creativity=False,
            requires_math=False,
            confidence=confidence,
            source="fallback",
        )
    except Exception as e:
        logger.warning(f"Fallback classifier error: {e}")
        return SmartClassification()


# ---------------------------------------------------------------------------
# Main LLM classifier
# ---------------------------------------------------------------------------
async def smart_classify(prompt: str) -> SmartClassification:
    """
    Classify a prompt using LLM with fallback to keyword heuristics.

    - Sanitizes input (control chars, truncation)
    - Checks in-memory cache (SHA-256 key, 5-min TTL, max 10K entries)
    - Calls Groq Llama-4-Scout (temperature=0, max_tokens=100, timeout=2s)
    - Validates JSON output against known enums
    - Falls back to detect_task_type() + TaskClassifier on any failure
    """
    sanitized = sanitize_prompt(prompt)
    if not sanitized:
        return _fallback_classify(prompt)

    key = _cache_key(sanitized)

    # Check cache
    cached = await _get_cached(key)
    if cached is not None:
        return cached

    # Try LLM classification
    try:
        resp = await acompletion(
            model="groq/meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": sanitized},
            ],
            temperature=0,
            max_tokens=100,
            timeout=2,
        )

        raw = resp.choices[0].message.content or ""
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)

        # Validate task_type
        task_type = data.get("task_type", "unknown")
        if task_type not in VALID_TASK_TYPES:
            raise ValueError(f"Invalid task_type: {task_type}")

        # Validate complexity
        complexity = data.get("complexity", "medium")
        if complexity not in VALID_COMPLEXITIES:
            complexity = "medium"

        result = SmartClassification(
            task_type=task_type,
            complexity=complexity,
            requires_reasoning=bool(data.get("requires_reasoning", False)),
            requires_code=bool(data.get("requires_code", False)),
            requires_creativity=bool(data.get("requires_creativity", False)),
            requires_math=bool(data.get("requires_math", False)),
            confidence=0.9,  # LLM classification is high-confidence
            source="llm",
        )

        await _set_cached(key, result)
        return result

    except json.JSONDecodeError:
        logger.debug("Smart classifier: invalid JSON from LLM, falling back")
    except Exception as e:
        logger.debug(f"Smart classifier: LLM error ({type(e).__name__}), falling back")

    # Fallback
    result = _fallback_classify(sanitized)
    await _set_cached(key, result)
    return result
