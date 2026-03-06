"""
Shadow Mode - Compare actual routing decisions against optimal recommendations.

Shadow mode runs the routing algorithm to determine what it WOULD recommend,
then compares this to what was actually executed. This enables:
1. A/B testing of routing strategies
2. Quality comparison between models
3. Cost savings analysis
4. Continuous improvement of routing decisions

Also includes LLM-as-judge quality sampling for comparing outputs.

PRODUCTION FEATURES:
- Persistent storage via pluggable backend (survives restarts)
- Judge failure tracking (failures, timeouts, heuristic fallbacks)
- Daily aggregation for trend analysis
"""

import os
import random
import asyncio
import json
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from litellm import acompletion
except ImportError:
    acompletion = None

from .storage import get_storage


# ============================================================================
# CONFIGURATION
# ============================================================================

# Sample rate for LLM-as-judge quality comparison (0.0 to 1.0)
# Set to 0 in tests to avoid flaky behavior
DEFAULT_QUALITY_SAMPLE_RATE = float(os.getenv("ASTRAI_QUALITY_SAMPLE_RATE", "0.10"))

# Disable sampling in pytest
if os.getenv("PYTEST_CURRENT_TEST"):
    DEFAULT_QUALITY_SAMPLE_RATE = 0.0

# Model used for quality judging
JUDGE_MODEL = os.getenv("ASTRAI_JUDGE_MODEL", "gpt-4o-mini")


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class ShadowRecommendation:
    """What the router would have recommended."""

    recommended_provider: str
    recommended_model: str
    estimated_cost: float
    estimated_latency_ms: float
    routing_reason: str
    potential_savings_pct: float
    potential_savings_usd: float


@dataclass
class QualityComparison:
    """Result of comparing quality between original and shadow models."""

    original_score: float  # 1-5 scale
    shadow_score: float  # 1-5 scale
    original_better: bool
    shadow_better: bool
    tie: bool
    judge_reasoning: str
    judge_model: str
    judged_at: str
    used_heuristic: bool = False  # True if LLM judge failed


@dataclass
class ShadowResult:
    """Complete shadow mode result."""

    mode: str  # "shadow" or "disabled"
    recommendation: Optional[ShadowRecommendation]
    quality_comparison: Optional[QualityComparison]
    actual_provider: str
    actual_model: str
    actual_cost: float
    would_have_saved: bool
    task_type: str


@dataclass
class QualityAggregate:
    """Aggregated quality comparison statistics."""

    date: str  # YYYY-MM-DD for daily aggregation
    task_type: str
    original_model: str
    shadow_model: str
    original_score_sum: float = 0.0
    shadow_score_sum: float = 0.0
    original_wins: int = 0
    shadow_wins: int = 0
    ties: int = 0
    samples: int = 0
    total_savings_pct: float = 0.0
    # Judge reliability tracking
    judge_successes: int = 0
    judge_failures: int = 0
    judge_timeouts: int = 0
    heuristic_fallbacks: int = 0

    @property
    def avg_original_score(self) -> float:
        return self.original_score_sum / self.samples if self.samples > 0 else 0.0

    @property
    def avg_shadow_score(self) -> float:
        return self.shadow_score_sum / self.samples if self.samples > 0 else 0.0

    @property
    def shadow_win_ratio(self) -> float:
        total = self.original_wins + self.shadow_wins + self.ties
        return self.shadow_wins / total if total > 0 else 0.0

    @property
    def avg_savings_pct(self) -> float:
        return self.total_savings_pct / self.samples if self.samples > 0 else 0.0

    @property
    def judge_success_rate(self) -> float:
        total = self.judge_successes + self.judge_failures + self.judge_timeouts
        return self.judge_successes / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "task_type": self.task_type,
            "original_model": self.original_model,
            "shadow_model": self.shadow_model,
            "avg_original_score": round(self.avg_original_score, 2),
            "avg_shadow_score": round(self.avg_shadow_score, 2),
            "original_wins": self.original_wins,
            "shadow_wins": self.shadow_wins,
            "ties": self.ties,
            "samples": self.samples,
            "shadow_win_ratio": round(self.shadow_win_ratio, 3),
            "avg_savings_pct": round(self.avg_savings_pct, 1),
            # Judge reliability
            "judge_successes": self.judge_successes,
            "judge_failures": self.judge_failures,
            "judge_timeouts": self.judge_timeouts,
            "heuristic_fallbacks": self.heuristic_fallbacks,
            "judge_success_rate": round(self.judge_success_rate, 3),
        }


# ============================================================================
# IN-MEMORY CACHE (syncs to storage backend)
# ============================================================================

# Local cache for fast access (synced to DB periodically)
_quality_aggregates: Dict[str, QualityAggregate] = {}
_aggregate_lock = asyncio.Lock()
_pending_sync = False


def _make_aggregate_key(
    date_str: str, task_type: str, original_model: str, shadow_model: str
) -> str:
    """Create a unique key for aggregation."""
    return f"{date_str}:{task_type}:{original_model}:{shadow_model}"


async def _sync_to_storage(aggregate: QualityAggregate) -> bool:
    """Persist aggregate to storage backend using upsert."""
    storage = get_storage()

    try:
        storage.upsert(
            "quality_comparison_aggregates",
            {
                "date": aggregate.date,
                "task_type": aggregate.task_type,
                "original_model": aggregate.original_model,
                "shadow_model": aggregate.shadow_model,
                "original_score_sum": aggregate.original_score_sum,
                "shadow_score_sum": aggregate.shadow_score_sum,
                "original_wins": aggregate.original_wins,
                "shadow_wins": aggregate.shadow_wins,
                "ties": aggregate.ties,
                "samples": aggregate.samples,
                "total_savings_pct": aggregate.total_savings_pct,
                "judge_successes": aggregate.judge_successes,
                "judge_failures": aggregate.judge_failures,
                "judge_timeouts": aggregate.judge_timeouts,
                "heuristic_fallbacks": aggregate.heuristic_fallbacks,
                "updated_at": datetime.utcnow().isoformat(),
            },
            on_conflict="date,task_type,original_model,shadow_model",
        )
        return True
    except Exception as e:
        print(f"Warning: Failed to sync aggregate to storage: {e}")
        return False


async def _load_from_storage() -> None:
    """Load today's aggregates from storage on startup."""
    storage = get_storage()

    try:
        today = date.today().isoformat()
        rows = storage.get(
            "quality_comparison_aggregates",
            filters={"date": today},
        )

        async with _aggregate_lock:
            for row in rows:
                key = _make_aggregate_key(
                    row["date"],
                    row["task_type"],
                    row["original_model"],
                    row["shadow_model"],
                )
                _quality_aggregates[key] = QualityAggregate(
                    date=row["date"],
                    task_type=row["task_type"],
                    original_model=row["original_model"],
                    shadow_model=row["shadow_model"],
                    original_score_sum=row.get("original_score_sum", 0),
                    shadow_score_sum=row.get("shadow_score_sum", 0),
                    original_wins=row.get("original_wins", 0),
                    shadow_wins=row.get("shadow_wins", 0),
                    ties=row.get("ties", 0),
                    samples=row.get("samples", 0),
                    total_savings_pct=row.get("total_savings_pct", 0),
                    judge_successes=row.get("judge_successes", 0),
                    judge_failures=row.get("judge_failures", 0),
                    judge_timeouts=row.get("judge_timeouts", 0),
                    heuristic_fallbacks=row.get("heuristic_fallbacks", 0),
                )
        print(f"Loaded {len(rows)} quality aggregates from storage")
    except Exception as e:
        print(f"Warning: Failed to load aggregates from storage: {e}")


async def record_quality_comparison(
    task_type: str,
    original_model: str,
    shadow_model: str,
    original_score: float,
    shadow_score: float,
    savings_pct: float,
    used_heuristic: bool = False,
    judge_failed: bool = False,
    judge_timeout: bool = False,
) -> None:
    """Record a quality comparison to the aggregates and persist to storage."""
    today = date.today().isoformat()
    key = _make_aggregate_key(today, task_type, original_model, shadow_model)

    async with _aggregate_lock:
        if key not in _quality_aggregates:
            _quality_aggregates[key] = QualityAggregate(
                date=today,
                task_type=task_type,
                original_model=original_model,
                shadow_model=shadow_model,
            )

        agg = _quality_aggregates[key]
        agg.original_score_sum += original_score
        agg.shadow_score_sum += shadow_score
        agg.samples += 1
        agg.total_savings_pct += savings_pct

        # Determine winner (0.5 tolerance for tie)
        if original_score > shadow_score + 0.5:
            agg.original_wins += 1
        elif shadow_score > original_score + 0.5:
            agg.shadow_wins += 1
        else:
            agg.ties += 1

        # Track judge reliability
        if judge_timeout:
            agg.judge_timeouts += 1
        elif judge_failed:
            agg.judge_failures += 1
        elif used_heuristic:
            agg.heuristic_fallbacks += 1
        else:
            agg.judge_successes += 1

        # Sync to storage in background
        asyncio.create_task(_sync_to_storage(agg))


async def get_quality_aggregates(days: int = 7) -> List[Dict[str, Any]]:
    """Get quality comparison aggregates for the last N days."""
    storage = get_storage()

    try:
        from datetime import timedelta

        cutoff = (date.today() - timedelta(days=days)).isoformat()

        rows = storage.get(
            "quality_comparison_aggregates",
            gte={"date": cutoff},
            order_by="date",
            desc=True,
        )

        if rows:
            # Convert to dict format
            aggregates = []
            for row in rows:
                agg = QualityAggregate(
                    date=row["date"],
                    task_type=row["task_type"],
                    original_model=row["original_model"],
                    shadow_model=row["shadow_model"],
                    original_score_sum=row.get("original_score_sum", 0),
                    shadow_score_sum=row.get("shadow_score_sum", 0),
                    original_wins=row.get("original_wins", 0),
                    shadow_wins=row.get("shadow_wins", 0),
                    ties=row.get("ties", 0),
                    samples=row.get("samples", 0),
                    total_savings_pct=row.get("total_savings_pct", 0),
                    judge_successes=row.get("judge_successes", 0),
                    judge_failures=row.get("judge_failures", 0),
                    judge_timeouts=row.get("judge_timeouts", 0),
                    heuristic_fallbacks=row.get("heuristic_fallbacks", 0),
                )
                aggregates.append(agg.to_dict())
            return aggregates
    except Exception as e:
        print(f"Warning: Failed to fetch from storage, using local cache: {e}")

    # Fall back to local cache
    async with _aggregate_lock:
        return [agg.to_dict() for agg in _quality_aggregates.values()]


async def clear_quality_aggregates() -> None:
    """Clear all aggregates (for testing)."""
    async with _aggregate_lock:
        _quality_aggregates.clear()


# ============================================================================
# LLM-AS-JUDGE
# ============================================================================

JUDGE_PROMPT = """You are an expert judge comparing two AI responses to the same prompt.

USER PROMPT:
{prompt}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

Rate each response on a scale of 1-5 based on:
- Accuracy and correctness
- Completeness
- Clarity and helpfulness
- Following instructions

You MUST respond with ONLY valid JSON in this exact format (no other text):
{{"score_a": <number 1-5>, "score_b": <number 1-5>, "reasoning": "<brief 1-2 sentence explanation>"}}

Be objective. If responses are equal quality, give equal scores."""


async def judge_responses(
    prompt: str,
    response_a: str,
    response_b: str,
    timeout: float = 30.0,
) -> Tuple[float, float, str, bool, bool, bool]:
    """
    Use LLM to judge quality of two responses.

    Returns:
        (score_a, score_b, reasoning, used_heuristic, judge_failed, judge_timeout)
    """
    if acompletion is None:
        scores = _heuristic_judge(response_a, response_b)
        return (
            scores[0],
            scores[1],
            "Heuristic scoring (litellm not available)",
            True,
            True,
            False,
        )

    try:
        judge_prompt = JUDGE_PROMPT.format(
            prompt=prompt[:2000],  # Truncate to avoid token limits
            response_a=response_a[:2000],
            response_b=response_b[:2000],
        )

        response = await asyncio.wait_for(
            acompletion(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.1,
                timeout=timeout,
            ),
            timeout=timeout + 5,  # Extra buffer for network
        )

        content = response.choices[0].message.content

        # Parse JSON response
        # Handle potential markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())

        score_a = float(result.get("score_a", 3))
        score_b = float(result.get("score_b", 3))
        reasoning = result.get("reasoning", "No reasoning provided")

        # Clamp scores to 1-5
        score_a = max(1.0, min(5.0, score_a))
        score_b = max(1.0, min(5.0, score_b))

        return score_a, score_b, reasoning, False, False, False

    except asyncio.TimeoutError:
        print(f"Warning: Judge timed out after {timeout}s")
        scores = _heuristic_judge(response_a, response_b)
        return (
            scores[0],
            scores[1],
            "Heuristic scoring (judge timeout)",
            True,
            False,
            True,
        )

    except json.JSONDecodeError as e:
        print(f"Warning: Judge returned invalid JSON: {e}")
        scores = _heuristic_judge(response_a, response_b)
        return (
            scores[0],
            scores[1],
            "Heuristic scoring (invalid JSON from judge)",
            True,
            True,
            False,
        )

    except Exception as e:
        print(f"Warning: Judge failed: {e}")
        scores = _heuristic_judge(response_a, response_b)
        return (
            scores[0],
            scores[1],
            f"Heuristic scoring (judge error: {type(e).__name__})",
            True,
            True,
            False,
        )


def _heuristic_judge(response_a: str, response_b: str) -> Tuple[float, float]:
    """Fallback heuristic judging when LLM judge fails."""

    def score_response(text: str) -> float:
        score = 3.0  # Base score

        # Length bonus (reasonable length is good)
        length = len(text)
        if 100 < length < 2000:
            score += 0.5
        elif length < 50:
            score -= 0.5

        # Structure bonus (has paragraphs, lists, etc.)
        if "\n\n" in text or "\n- " in text or "\n1." in text:
            score += 0.3

        # Code block bonus for technical content
        if "```" in text:
            score += 0.2

        # Refusal penalty
        refusal_phrases = ["i cannot", "i'm unable", "i can't", "as an ai"]
        if any(phrase in text.lower() for phrase in refusal_phrases):
            score -= 1.0

        return max(1.0, min(5.0, score))

    score_a = score_response(response_a)
    score_b = score_response(response_b)

    return score_a, score_b


# ============================================================================
# SHADOW MODE CORE
# ============================================================================


class ShadowModeEngine:
    """
    Shadow mode engine for comparing routing decisions.

    Usage:
        shadow = ShadowModeEngine(inference_router)
        result = await shadow.get_recommendation(
            model_requested="gpt-4o",
            actual_provider="openai",
            actual_model="gpt-4o",
            actual_cost=0.01,
            strategy=ExecutionStrategy.BALANCED,
        )
    """

    def __init__(self, inference_router):
        """
        Initialize shadow mode engine.

        Args:
            inference_router: The INFERENCE_ROUTER instance
        """
        self.router = inference_router
        self.sample_rate = DEFAULT_QUALITY_SAMPLE_RATE

    async def get_recommendation(
        self,
        model_requested: str,
        actual_provider: str,
        actual_model: str,
        actual_cost: float,
        strategy,  # ExecutionStrategy
        excluded_venues: List[str] = None,
        max_latency_ms: Optional[float] = None,
        quality_tier: str = "target",
        preferred_providers: List[str] = None,
    ) -> ShadowRecommendation:
        """
        Get what the router WOULD recommend for this request.

        This runs the routing algorithm fresh to determine the optimal
        choice, ignoring what was actually selected.
        """
        # Get optimal route using CHEAPEST strategy (to show max savings)
        try:
            from .execution import ExecutionStrategy
        except ImportError:
            ExecutionStrategy = None

        cheapest_strategy = (
            ExecutionStrategy.CHEAPEST if ExecutionStrategy else strategy
        )

        selected_quote, routing_reason, alternatives = await self.router.route(
            model_requested=model_requested,
            strategy=cheapest_strategy,  # Always show cheapest option
            excluded_venues=excluded_venues or [],
            max_latency_ms=max_latency_ms,
            quality_tier=quality_tier,
            preferred_providers=preferred_providers,
        )

        if not selected_quote:
            return ShadowRecommendation(
                recommended_provider=actual_provider,
                recommended_model=actual_model,
                estimated_cost=actual_cost,
                estimated_latency_ms=0,
                routing_reason="No alternative available",
                potential_savings_pct=0,
                potential_savings_usd=0,
            )

        # Estimate cost for recommended model (rough estimate based on bid price)
        estimated_cost = selected_quote.bid_price * 0.001  # Rough conversion

        # Calculate savings
        if actual_cost > 0:
            savings_usd = actual_cost - estimated_cost
            savings_pct = (savings_usd / actual_cost) * 100 if actual_cost > 0 else 0
        else:
            savings_usd = 0
            savings_pct = 0

        return ShadowRecommendation(
            recommended_provider=selected_quote.provider_id,
            recommended_model=selected_quote.model_id,
            estimated_cost=max(0, estimated_cost),
            estimated_latency_ms=selected_quote.latency_ms or 0,
            routing_reason=routing_reason,
            potential_savings_pct=max(0, savings_pct),
            potential_savings_usd=max(0, savings_usd),
        )

    def should_sample_quality(self) -> bool:
        """Determine if we should run quality comparison for this request."""
        if self.sample_rate <= 0:
            return False
        return random.random() < self.sample_rate

    async def run_shadow_model(
        self,
        messages: List[Dict[str, Any]],
        shadow_model: str,
        temperature: float = 0.7,
        timeout: float = 60.0,
    ) -> Optional[str]:
        """
        Run the shadow-recommended model to get its response.

        Returns the response content or None if failed.
        """
        if acompletion is None:
            print("Warning: Shadow model execution requires litellm")
            return None

        try:
            response = await acompletion(
                model=shadow_model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Warning: Shadow model execution failed: {e}")
            return None

    async def compare_quality(
        self,
        prompt: str,
        original_response: str,
        shadow_response: str,
        original_model: str,
        shadow_model: str,
        task_type: str,
        savings_pct: float,
    ) -> QualityComparison:
        """
        Compare quality between original and shadow responses.

        Uses LLM-as-judge with fallback to heuristics.
        """
        # Run judge (now returns additional tracking info)
        (
            original_score,
            shadow_score,
            reasoning,
            used_heuristic,
            judge_failed,
            judge_timeout,
        ) = await judge_responses(
            prompt=prompt,
            response_a=original_response,
            response_b=shadow_response,
        )

        # Record to aggregates with judge reliability tracking
        await record_quality_comparison(
            task_type=task_type,
            original_model=original_model,
            shadow_model=shadow_model,
            original_score=original_score,
            shadow_score=shadow_score,
            savings_pct=savings_pct,
            used_heuristic=used_heuristic,
            judge_failed=judge_failed,
            judge_timeout=judge_timeout,
        )

        # Determine winner
        original_better = original_score > shadow_score + 0.5
        shadow_better = shadow_score > original_score + 0.5
        tie = not original_better and not shadow_better

        return QualityComparison(
            original_score=original_score,
            shadow_score=shadow_score,
            original_better=original_better,
            shadow_better=shadow_better,
            tie=tie,
            judge_reasoning=reasoning,
            judge_model=JUDGE_MODEL if not used_heuristic else "heuristic",
            judged_at=datetime.utcnow().isoformat(),
            used_heuristic=used_heuristic,
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_shadow_result(
    recommendation: Optional[ShadowRecommendation],
    quality_comparison: Optional[QualityComparison],
    actual_provider: str,
    actual_model: str,
    actual_cost: float,
    task_type: str,
) -> Dict[str, Any]:
    """Create the _astrai_shadow dict for API response."""
    if not recommendation:
        return {
            "mode": "disabled",
            "message": "Shadow mode not available",
        }

    result = {
        "mode": "shadow",
        "recommendation": {
            "provider": recommendation.recommended_provider,
            "model": recommendation.recommended_model,
            "estimated_cost": round(recommendation.estimated_cost, 6),
            "estimated_latency_ms": round(recommendation.estimated_latency_ms, 1),
            "routing_reason": recommendation.routing_reason,
        },
        "comparison": {
            "actual_provider": actual_provider,
            "actual_model": actual_model,
            "actual_cost": round(actual_cost, 6),
            "would_have_saved": recommendation.potential_savings_pct > 5,
            "potential_savings_pct": round(recommendation.potential_savings_pct, 1),
            "potential_savings_usd": round(recommendation.potential_savings_usd, 6),
        },
        "task_type": task_type,
    }

    if quality_comparison:
        result["quality"] = {
            "original_score": quality_comparison.original_score,
            "shadow_score": quality_comparison.shadow_score,
            "winner": "original"
            if quality_comparison.original_better
            else ("shadow" if quality_comparison.shadow_better else "tie"),
            "judge_reasoning": quality_comparison.judge_reasoning,
            "judge_model": quality_comparison.judge_model,
            "used_heuristic": quality_comparison.used_heuristic,
        }

    return result


# Global shadow engine instance (initialized lazily)
_shadow_engine: Optional[ShadowModeEngine] = None
_initialized = False


async def init_shadow_mode() -> None:
    """Initialize shadow mode by loading aggregates from storage."""
    global _initialized
    if not _initialized:
        await _load_from_storage()
        _initialized = True


def get_shadow_engine(inference_router) -> ShadowModeEngine:
    """Get or create the shadow mode engine."""
    global _shadow_engine
    if _shadow_engine is None:
        _shadow_engine = ShadowModeEngine(inference_router)
        # Schedule initialization
        asyncio.create_task(init_shadow_mode())
    return _shadow_engine
