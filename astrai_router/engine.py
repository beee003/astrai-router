"""
Astrai Routing Engine - Privacy-preserving intelligent routing.

Combines multiple mathematical models for optimal routing:
1. GlimpRouter (entropy-based tier selection)
2. Smart Order Router (venue scoring with weights)
3. ARBITRAGE (advantage-aware switching recommendations)
4. Thompson Sampling (self-learning from feedback)

This engine receives ONLY metadata from the SDK (no content) and returns
routing recommendations. Content never touches Astrai servers in local mode.
"""

# NOTE: Thompson priors and quality matrix are now stored via the pluggable
# storage backend (MemoryStorage / SQLiteStorage / PostgresStorage).
# Local disk storage (~/.astrai/routing_intelligence/) is no longer used for
# production persistence. This ensures learning survives deploys and
# works across multiple machines.

import random
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .storage import get_storage

logger = logging.getLogger(__name__)


class RoutingTier(Enum):
    """Quality tiers for model routing."""

    DRAFT = "draft"  # Fast, cheap, good for simple tasks
    BALANCED = "balanced"  # Middle ground
    ULTRA = "ultra"  # Highest quality, expensive


@dataclass
class ThompsonPrior:
    """Beta distribution prior for Thompson Sampling."""

    alpha: float = 1.0  # Successes + 1
    beta: float = 1.0  # Failures + 1

    def sample(self) -> float:
        """Sample from Beta(alpha, beta) distribution."""
        return random.betavariate(self.alpha, self.beta)

    def update_success(self):
        """Update prior after successful outcome."""
        self.alpha += 1

    def update_failure(self):
        """Update prior after failed outcome."""
        self.beta += 1

    @property
    def mean(self) -> float:
        """Expected success rate."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def confidence(self) -> float:
        """Confidence based on sample size."""
        total = self.alpha + self.beta - 2  # Subtract initial priors
        return min(1.0, total / 100)  # 100 samples = full confidence


@dataclass
class VenueStats:
    """Statistics for a venue (provider/model combination)."""

    provider: str
    model: str
    tier: RoutingTier
    cost_per_mtok_input: float
    cost_per_mtok_output: float
    latency_p50_ms: float
    latency_p99_ms: float
    success_rate: float = 1.0
    quality_score: float = 1.0  # 0-1, based on task-specific performance
    uptime: float = 1.0


@dataclass
class RoutingDecision:
    """Result of routing decision."""

    provider: str
    model: str
    tier: str
    score: float
    reasoning: str
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    thompson_sample: float = 0.0
    entropy_tier: str = ""
    arbitrage_delta: float = 0.0


@dataclass
class RoutingConstraints:
    """User-specified routing constraints."""

    max_cost_per_request: Optional[float] = None
    max_latency_ms: Optional[float] = None
    required_tier: Optional[str] = None
    excluded_providers: List[str] = field(default_factory=list)
    preferred_providers: List[str] = field(default_factory=list)
    region: Optional[str] = None


class RoutingEngine:
    """
    Privacy-preserving routing engine.

    Receives metadata-only requests and applies mathematical models
    to recommend the optimal venue without seeing content.
    """

    # Default venue registry (would be loaded from market data in production)
    DEFAULT_VENUES: List[VenueStats] = [
        # Draft tier (fast, cheap)
        VenueStats(
            "groq",
            "llama-3.3-70b",
            RoutingTier.DRAFT,
            0.0,
            0.0,
            150,
            500,
            0.98,
            0.85,
            0.99,
        ),
        VenueStats(
            "cerebras",
            "llama-3.3-70b",
            RoutingTier.DRAFT,
            0.0,
            0.0,
            100,
            300,
            0.95,
            0.82,
            0.97,
        ),
        VenueStats(
            "groq",
            "llama-3.1-8b",
            RoutingTier.DRAFT,
            0.0,
            0.0,
            80,
            200,
            0.99,
            0.75,
            0.99,
        ),
        VenueStats(
            "deepinfra",
            "llama-3.3-70b",
            RoutingTier.DRAFT,
            0.23,
            0.40,
            200,
            600,
            0.97,
            0.85,
            0.98,
        ),
        # Balanced tier
        VenueStats(
            "openai",
            "gpt-4o-mini",
            RoutingTier.BALANCED,
            0.15,
            0.60,
            300,
            1000,
            0.99,
            0.90,
            0.999,
        ),
        VenueStats(
            "anthropic",
            "claude-3.5-haiku",
            RoutingTier.BALANCED,
            0.80,
            4.00,
            350,
            1200,
            0.98,
            0.88,
            0.998,
        ),
        VenueStats(
            "together",
            "llama-3.3-70b",
            RoutingTier.BALANCED,
            0.40,
            0.80,
            250,
            800,
            0.96,
            0.86,
            0.97,
        ),
        VenueStats(
            "deepseek",
            "deepseek-chat-v3.1",
            RoutingTier.BALANCED,
            0.14,
            0.28,
            400,
            1500,
            0.95,
            0.89,
            0.96,
        ),
        # Ultra tier (highest quality)
        VenueStats(
            "openai",
            "gpt-4o",
            RoutingTier.ULTRA,
            2.50,
            10.00,
            500,
            2000,
            0.99,
            0.95,
            0.999,
        ),
        VenueStats(
            "openai",
            "o1",
            RoutingTier.ULTRA,
            15.00,
            60.00,
            5000,
            30000,
            0.98,
            0.99,
            0.999,
        ),
        VenueStats(
            "anthropic",
            "claude-3.5-sonnet",
            RoutingTier.ULTRA,
            3.00,
            15.00,
            600,
            2500,
            0.99,
            0.96,
            0.998,
        ),
        VenueStats(
            "anthropic",
            "claude-opus-4.5",
            RoutingTier.ULTRA,
            15.00,
            75.00,
            800,
            5000,
            0.98,
            0.99,
            0.997,
        ),
    ]

    # Task-to-quality mapping (which models excel at which tasks)
    TASK_QUALITY_MAP: Dict[str, Dict[str, float]] = {
        "coding": {
            "claude-3.5-sonnet": 0.98,
            "claude-opus-4.5": 0.99,
            "gpt-4o": 0.95,
            "deepseek-chat-v3.1": 0.92,
            "llama-3.3-70b": 0.85,
            "gpt-4o-mini": 0.80,
        },
        "reasoning": {
            "o1": 0.99,
            "claude-opus-4.5": 0.97,
            "gpt-4o": 0.93,
            "claude-3.5-sonnet": 0.92,
        },
        "creative": {
            "claude-opus-4.5": 0.98,
            "claude-3.5-sonnet": 0.95,
            "gpt-4o": 0.92,
        },
        "classification": {
            "llama-3.1-8b": 0.90,
            "llama-3.3-70b": 0.92,
            "gpt-4o-mini": 0.93,
        },
        "extraction": {
            "llama-3.3-70b": 0.92,
            "gpt-4o-mini": 0.91,
            "claude-3.5-haiku": 0.90,
        },
        "chat": {
            "llama-3.1-8b": 0.88,
            "gpt-4o-mini": 0.90,
            "claude-3.5-haiku": 0.89,
        },
    }

    def __init__(
        self,
        venues: Optional[List[VenueStats]] = None,
        # Smart Order Router weights
        w_cost: float = 0.3,
        w_latency: float = 0.2,
        w_quality: float = 0.35,
        w_reliability: float = 0.15,
        # GlimpRouter thresholds
        entropy_low: float = 0.6,
        entropy_high: float = 1.8,
        # ARBITRAGE threshold
        arbitrage_threshold: float = 0.1,
    ):
        self.venues = venues or self.DEFAULT_VENUES
        self.w_cost = w_cost
        self.w_latency = w_latency
        self.w_quality = w_quality
        self.w_reliability = w_reliability
        self.entropy_low = entropy_low
        self.entropy_high = entropy_high
        self.arbitrage_threshold = arbitrage_threshold

        # Thompson Sampling priors: (task_type, venue_id) -> ThompsonPrior
        self._thompson_priors: Dict[Tuple[str, str], ThompsonPrior] = {}
        self._priors_dirty = False
        self._last_persist = 0.0
        self._persist_interval = 60.0  # persist at most every 60s

        # Load persisted priors on startup
        self._load_priors()

    def _get_venue_id(self, venue: VenueStats) -> str:
        """Get unique venue identifier."""
        return f"{venue.provider}/{venue.model}"

    def _get_thompson_prior(self, task_type: str, venue: VenueStats) -> ThompsonPrior:
        """Get or create Thompson Sampling prior for task/venue pair."""
        key = (task_type, self._get_venue_id(venue))
        if key not in self._thompson_priors:
            self._thompson_priors[key] = ThompsonPrior()
        return self._thompson_priors[key]

    def recommend(
        self,
        task_type: str,
        complexity: str,
        entropy: float,
        tokens: int,
        constraints: Optional[RoutingConstraints] = None,
    ) -> RoutingDecision:
        """
        Recommend optimal venue based on metadata.

        This is the main entry point. No content is required.

        Args:
            task_type: Classified task type (coding, writing, etc.)
            complexity: Complexity level (low, medium, high)
            entropy: Shannon entropy of request
            tokens: Estimated token count
            constraints: User-specified constraints

        Returns:
            RoutingDecision with recommended venue and reasoning
        """
        constraints = constraints or RoutingConstraints()

        # 1. GlimpRouter: Determine tier from entropy
        entropy_tier = self._glimp_router(entropy)

        # 2. Filter venues by constraints and tier
        feasible = self._filter_venues(entropy_tier, complexity, constraints)

        if not feasible:
            # Fallback: relax constraints
            feasible = self._filter_venues(None, None, constraints)

        if not feasible:
            # Ultimate fallback
            feasible = self.venues[:3]

        # 3. Smart Order Router: Score each venue
        scored = []
        for venue in feasible:
            score = self._score_venue(venue, task_type, tokens, constraints)
            thompson_sample = self._get_thompson_prior(task_type, venue).sample()
            # Combine deterministic score with Thompson exploration
            combined_score = 0.7 * score + 0.3 * thompson_sample
            scored.append((venue, score, thompson_sample, combined_score))

        # Sort by combined score (descending)
        scored.sort(key=lambda x: x[3], reverse=True)

        # 4. Select best venue
        best_venue, best_score, best_thompson, best_combined = scored[0]

        # 5. ARBITRAGE: Check if switching is recommended
        arbitrage_delta = 0.0
        if len(scored) > 1:
            _, second_score, _, _ = scored[1]
            arbitrage_delta = best_combined - second_score

        # Build alternatives
        alternatives = []
        for venue, score, ts, cs in scored[1:4]:
            alternatives.append(
                {
                    "provider": venue.provider,
                    "model": venue.model,
                    "score": round(score, 4),
                    "tier": venue.tier.value,
                }
            )

        # Build reasoning
        reasons = []
        reasons.append(f"Task: {task_type} ({complexity} complexity)")
        reasons.append(f"Entropy: {entropy:.2f} -> {entropy_tier} tier")
        reasons.append(f"Score: {best_score:.3f} (thompson: {best_thompson:.3f})")
        if arbitrage_delta > self.arbitrage_threshold:
            reasons.append(f"Strong recommendation (delta: {arbitrage_delta:.3f})")

        return RoutingDecision(
            provider=best_venue.provider,
            model=best_venue.model,
            tier=best_venue.tier.value,
            score=round(best_combined, 4),
            reasoning=" | ".join(reasons),
            alternatives=alternatives,
            thompson_sample=round(best_thompson, 4),
            entropy_tier=entropy_tier,
            arbitrage_delta=round(arbitrage_delta, 4),
        )

    def _glimp_router(self, entropy: float) -> str:
        """
        GlimpRouter: Entropy-based tier selection.

        Lower entropy = simpler, more repetitive content -> draft tier
        Higher entropy = more complex, diverse content -> ultra tier
        """
        if entropy < self.entropy_low:
            return "draft"
        elif entropy > self.entropy_high:
            return "ultra"
        else:
            return "balanced"

    def _filter_venues(
        self,
        tier: Optional[str],
        complexity: Optional[str],
        constraints: RoutingConstraints,
    ) -> List[VenueStats]:
        """Filter venues by tier, complexity, and constraints."""
        result = []

        for venue in self.venues:
            # Tier filter
            if tier and venue.tier.value != tier:
                # Allow upgrade for high complexity
                if complexity == "high" and venue.tier == RoutingTier.ULTRA:
                    pass  # Allow
                # Allow downgrade for low complexity
                elif complexity == "low" and venue.tier == RoutingTier.DRAFT:
                    pass  # Allow
                else:
                    continue

            # Excluded providers
            if venue.provider in constraints.excluded_providers:
                continue

            # Max cost filter (estimate for 1000 tokens)
            if constraints.max_cost_per_request:
                est_cost = (
                    venue.cost_per_mtok_input + venue.cost_per_mtok_output
                ) * 0.001
                if est_cost > constraints.max_cost_per_request:
                    continue

            # Max latency filter
            if constraints.max_latency_ms:
                if venue.latency_p50_ms > constraints.max_latency_ms:
                    continue

            # Required tier override
            if constraints.required_tier:
                if venue.tier.value != constraints.required_tier:
                    continue

            result.append(venue)

        # Boost preferred providers to front
        if constraints.preferred_providers:
            preferred = [
                v for v in result if v.provider in constraints.preferred_providers
            ]
            others = [
                v for v in result if v.provider not in constraints.preferred_providers
            ]
            result = preferred + others

        return result

    def _score_venue(
        self,
        venue: VenueStats,
        task_type: str,
        tokens: int,
        constraints: RoutingConstraints,
    ) -> float:
        """
        Smart Order Router: Score venue with weighted factors.

        score = w_cost * (1 - normalized_cost)
              + w_latency * (1 - normalized_latency)
              + w_quality * task_quality_match
              + w_reliability * uptime_score
        """
        # Normalize cost (0 = expensive, 1 = cheap)
        max_cost = 20.0  # $/1M tokens (o1 level)
        avg_cost = (venue.cost_per_mtok_input + venue.cost_per_mtok_output) / 2
        normalized_cost = 1 - min(1, avg_cost / max_cost)

        # Normalize latency (0 = slow, 1 = fast)
        max_latency = 5000  # ms
        normalized_latency = 1 - min(1, venue.latency_p50_ms / max_latency)

        # Task-specific quality
        task_qualities = self.TASK_QUALITY_MAP.get(task_type, {})
        quality = task_qualities.get(venue.model, venue.quality_score)

        # Reliability (uptime)
        reliability = venue.uptime

        # Weighted sum
        score = (
            self.w_cost * normalized_cost
            + self.w_latency * normalized_latency
            + self.w_quality * quality
            + self.w_reliability * reliability
        )

        return score

    def record_feedback(
        self,
        task_type: str,
        provider: str,
        model: str,
        success: bool,
        latency_ms: float,
        cost_usd: float,
    ):
        """
        Record outcome for Thompson Sampling learning.

        Called after each request completes (without content).
        Updates the Beta distribution prior for the task/venue pair
        and periodically persists to disk.
        """
        venue_id = f"{provider}/{model}"
        key = (task_type, venue_id)

        if key not in self._thompson_priors:
            self._thompson_priors[key] = ThompsonPrior()

        prior = self._thompson_priors[key]

        if success:
            prior.update_success()
        else:
            prior.update_failure()

        self._priors_dirty = True

        # Persist periodically (not every call — would be too slow)
        now = time.time()
        if now - self._last_persist > self._persist_interval:
            self._persist_priors()

    def get_thompson_sample_for_venue(self, task_type: str, venue_id: str) -> float:
        """
        Sample from the Thompson prior for a specific task/venue pair.

        Returns a value in [0, 1] where higher = historically better.
        Returns 0.5 (neutral) if no prior exists for this pair.
        """
        key = (task_type, venue_id)
        if key not in self._thompson_priors:
            return 0.5  # Neutral — no data yet
        return self._thompson_priors[key].sample()

    def get_thompson_stats(self, task_type: str) -> Dict[str, Dict[str, Any]]:
        """Get Thompson Sampling statistics for a task type."""
        result = {}
        for (t, venue_id), prior in self._thompson_priors.items():
            if t == task_type:
                result[venue_id] = {
                    "alpha": prior.alpha,
                    "beta": prior.beta,
                    "mean": round(prior.mean, 4),
                    "confidence": round(prior.confidence, 4),
                }
        return result

    # ------ Persistence ------

    def _load_priors(self):
        """Load persisted Thompson Sampling priors from storage backend."""
        try:
            storage = get_storage()
            data = storage.get("thompson_priors", limit=5000)
            count = 0
            for entry in data:
                key = (entry["task_type"], entry["venue_id"])
                prior = ThompsonPrior(
                    alpha=float(entry.get("alpha", 1.0) or 1.0),
                    beta=float(entry.get("beta", 1.0) or 1.0),
                )
                self._thompson_priors[key] = prior
                count += 1
            if count:
                logger.info(f"Loaded {count} Thompson priors from storage")
        except Exception as e:
            logger.warning(f"Failed to load Thompson priors: {e}")

    def _persist_priors(self):
        """Persist Thompson Sampling priors to storage backend."""
        if not self._priors_dirty:
            return
        try:
            storage = get_storage()
            for (task_type, venue_id), prior in self._thompson_priors.items():
                storage.upsert(
                    "thompson_priors",
                    {
                        "task_type": task_type,
                        "venue_id": venue_id,
                        "alpha": prior.alpha,
                        "beta": prior.beta,
                    },
                    on_conflict="task_type,venue_id",
                )
            self._priors_dirty = False
            self._last_persist = time.time()
            logger.debug(
                f"Persisted {len(self._thompson_priors)} Thompson priors to storage"
            )
        except Exception as e:
            self._last_persist = time.time()
            logger.warning(f"Failed to persist Thompson priors: {e}")

    def flush_priors(self):
        """Force-flush priors to storage (call on shutdown)."""
        self._persist_priors()

    def calculate_arbitrage(
        self,
        current_provider: str,
        current_model: str,
        task_type: str,
        complexity: str,
    ) -> Dict[str, Any]:
        """
        ARBITRAGE: Calculate expected quality improvement from switching.

        delta = E[quality_optimal] - E[quality_current]
        If delta > threshold: recommend switch

        Returns recommendation with expected delta and reasoning.
        """
        # Find current venue
        current_venue = None
        for v in self.venues:
            if v.provider == current_provider and v.model == current_model:
                current_venue = v
                break

        if not current_venue:
            return {"switch_recommended": False, "reason": "Current venue not found"}

        # Get task quality for current
        task_qualities = self.TASK_QUALITY_MAP.get(task_type, {})
        current_quality = task_qualities.get(current_model, current_venue.quality_score)

        # Find optimal for this task
        best_quality = current_quality
        best_venue = current_venue

        for venue in self.venues:
            quality = task_qualities.get(venue.model, venue.quality_score)
            if quality > best_quality:
                best_quality = quality
                best_venue = venue

        delta = best_quality - current_quality

        return {
            "switch_recommended": delta > self.arbitrage_threshold,
            "delta": round(delta, 4),
            "current": {
                "provider": current_provider,
                "model": current_model,
                "expected_quality": round(current_quality, 4),
            },
            "recommended": {
                "provider": best_venue.provider,
                "model": best_venue.model,
                "expected_quality": round(best_quality, 4),
            },
            "reason": f"Quality delta: {delta:.2%}"
            if delta > 0
            else "Current is optimal",
        }


# Global singleton for the routing engine
_routing_engine: Optional[RoutingEngine] = None


def get_routing_engine() -> RoutingEngine:
    """Get or create the global routing engine instance."""
    global _routing_engine
    if _routing_engine is None:
        _routing_engine = RoutingEngine()
    return _routing_engine
