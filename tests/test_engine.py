"""
Tests for astrai_router.engine — Privacy-preserving intelligent routing engine.

Covers:
1. RoutingTier enum values
2. ThompsonPrior sampling and updates
3. RoutingEngine initialization
4. Basic routing recommendation generation
5. Constraint filtering
6. Arbitrage calculation
7. Feedback recording
"""

import pytest
from unittest.mock import patch, MagicMock

from astrai_router.engine import (
    RoutingEngine,
    RoutingTier,
    ThompsonPrior,
    VenueStats,
    RoutingDecision,
    RoutingConstraints,
)


# =============================================================================
# ROUTING TIER TESTS
# =============================================================================


class TestRoutingTier:
    """Test RoutingTier enum values."""

    def test_draft_tier(self):
        assert RoutingTier.DRAFT.value == "draft"

    def test_balanced_tier(self):
        assert RoutingTier.BALANCED.value == "balanced"

    def test_ultra_tier(self):
        assert RoutingTier.ULTRA.value == "ultra"

    def test_all_tiers_present(self):
        tier_values = {t.value for t in RoutingTier}
        assert tier_values == {"draft", "balanced", "ultra"}


# =============================================================================
# THOMPSON PRIOR TESTS
# =============================================================================


class TestThompsonPrior:
    """Test ThompsonPrior Beta distribution for Thompson Sampling."""

    def test_default_prior(self):
        """Default prior should be Beta(1, 1) — uniform."""
        prior = ThompsonPrior()
        assert prior.alpha == 1.0
        assert prior.beta == 1.0

    def test_mean_uniform(self):
        """Uniform prior should have mean 0.5."""
        prior = ThompsonPrior()
        assert prior.mean == 0.5

    def test_sample_in_range(self):
        """Samples from Beta distribution should be in [0, 1]."""
        prior = ThompsonPrior(alpha=5.0, beta=2.0)
        for _ in range(100):
            sample = prior.sample()
            assert 0.0 <= sample <= 1.0

    def test_update_success(self):
        """update_success should increment alpha."""
        prior = ThompsonPrior()
        prior.update_success()
        assert prior.alpha == 2.0
        assert prior.beta == 1.0

    def test_update_failure(self):
        """update_failure should increment beta."""
        prior = ThompsonPrior()
        prior.update_failure()
        assert prior.alpha == 1.0
        assert prior.beta == 2.0

    def test_mean_after_successes(self):
        """Mean should increase after successes."""
        prior = ThompsonPrior()
        for _ in range(10):
            prior.update_success()
        # alpha=11, beta=1, mean = 11/12 ~ 0.917
        assert prior.mean > 0.9

    def test_mean_after_failures(self):
        """Mean should decrease after failures."""
        prior = ThompsonPrior()
        for _ in range(10):
            prior.update_failure()
        # alpha=1, beta=11, mean = 1/12 ~ 0.083
        assert prior.mean < 0.1

    def test_confidence_zero_initially(self):
        """Confidence should be 0 with no samples (only priors)."""
        prior = ThompsonPrior()
        assert prior.confidence == 0.0

    def test_confidence_scales_with_samples(self):
        """Confidence should increase with more updates."""
        prior = ThompsonPrior()
        for _ in range(50):
            prior.update_success()
        # 50 samples / 100 = 0.5
        assert prior.confidence == 0.5

    def test_confidence_caps_at_one(self):
        """Confidence should cap at 1.0."""
        prior = ThompsonPrior()
        for _ in range(200):
            prior.update_success()
        assert prior.confidence == 1.0

    def test_custom_prior_values(self):
        """Custom alpha/beta should be respected."""
        prior = ThompsonPrior(alpha=10.0, beta=3.0)
        assert prior.alpha == 10.0
        assert prior.beta == 3.0
        # mean = 10/13 ~ 0.769
        assert abs(prior.mean - 10.0 / 13.0) < 0.001


# =============================================================================
# ROUTING ENGINE INITIALIZATION TESTS
# =============================================================================


class TestRoutingEngineInit:
    """Test RoutingEngine initialization."""

    @patch("astrai_router.engine.get_storage")
    def test_default_init(self, mock_storage):
        """Engine should initialize with default venues and weights."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        engine = RoutingEngine()
        assert len(engine.venues) > 0
        assert engine.w_cost == 0.3
        assert engine.w_latency == 0.2
        assert engine.w_quality == 0.35
        assert engine.w_reliability == 0.15

    @patch("astrai_router.engine.get_storage")
    def test_custom_weights(self, mock_storage):
        """Engine should accept custom weights."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        engine = RoutingEngine(
            w_cost=0.5, w_latency=0.1, w_quality=0.3, w_reliability=0.1
        )
        assert engine.w_cost == 0.5
        assert engine.w_latency == 0.1
        assert engine.w_quality == 0.3
        assert engine.w_reliability == 0.1

    @patch("astrai_router.engine.get_storage")
    def test_custom_venues(self, mock_storage):
        """Engine should accept custom venue list."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        custom_venues = [
            VenueStats(
                "test-provider",
                "test-model",
                RoutingTier.DRAFT,
                0.0,
                0.0,
                100,
                300,
                0.99,
                0.85,
                0.99,
            ),
        ]
        engine = RoutingEngine(venues=custom_venues)
        assert len(engine.venues) == 1
        assert engine.venues[0].provider == "test-provider"

    @patch("astrai_router.engine.get_storage")
    def test_default_venues_have_all_tiers(self, mock_storage):
        """Default venues should include all three tiers."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        engine = RoutingEngine()
        tiers = {v.tier for v in engine.venues}
        assert RoutingTier.DRAFT in tiers
        assert RoutingTier.BALANCED in tiers
        assert RoutingTier.ULTRA in tiers

    @patch("astrai_router.engine.get_storage")
    def test_entropy_thresholds(self, mock_storage):
        """Engine should accept custom entropy thresholds."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        engine = RoutingEngine(entropy_low=0.4, entropy_high=2.0)
        assert engine.entropy_low == 0.4
        assert engine.entropy_high == 2.0


# =============================================================================
# ROUTING RECOMMENDATION TESTS
# =============================================================================


class TestRoutingRecommendation:
    """Test routing recommendation generation."""

    @patch("astrai_router.engine.get_storage")
    def test_basic_recommendation(self, mock_storage):
        """Engine should return a RoutingDecision."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        engine = RoutingEngine()
        decision = engine.recommend(
            task_type="coding",
            complexity="medium",
            entropy=1.0,
            tokens=500,
        )
        assert isinstance(decision, RoutingDecision)
        assert decision.provider != ""
        assert decision.model != ""
        assert decision.tier in {"draft", "balanced", "ultra"}
        assert decision.score > 0

    @patch("astrai_router.engine.get_storage")
    def test_recommendation_has_reasoning(self, mock_storage):
        """Decision should include human-readable reasoning."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        engine = RoutingEngine()
        decision = engine.recommend(
            task_type="coding",
            complexity="high",
            entropy=2.0,
            tokens=1000,
        )
        assert len(decision.reasoning) > 0
        assert "Task:" in decision.reasoning

    @patch("astrai_router.engine.get_storage")
    def test_recommendation_has_alternatives(self, mock_storage):
        """Decision should include alternative venues."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        engine = RoutingEngine()
        decision = engine.recommend(
            task_type="chat",
            complexity="low",
            entropy=0.3,
            tokens=100,
        )
        # There should be alternatives (multiple venues available)
        assert isinstance(decision.alternatives, list)

    @patch("astrai_router.engine.get_storage")
    def test_low_entropy_routes_to_draft(self, mock_storage):
        """Low entropy input should tend toward draft tier."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        engine = RoutingEngine()
        decision = engine.recommend(
            task_type="classification",
            complexity="low",
            entropy=0.3,  # Below entropy_low threshold
            tokens=50,
        )
        assert decision.entropy_tier == "draft"

    @patch("astrai_router.engine.get_storage")
    def test_high_entropy_routes_to_ultra(self, mock_storage):
        """High entropy input should tend toward ultra tier."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        engine = RoutingEngine()
        decision = engine.recommend(
            task_type="reasoning",
            complexity="high",
            entropy=2.5,  # Above entropy_high threshold
            tokens=2000,
        )
        assert decision.entropy_tier == "ultra"

    @patch("astrai_router.engine.get_storage")
    def test_medium_entropy_routes_to_balanced(self, mock_storage):
        """Medium entropy input should route to balanced tier."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        engine = RoutingEngine()
        decision = engine.recommend(
            task_type="chat",
            complexity="medium",
            entropy=1.0,  # Between low and high
            tokens=300,
        )
        assert decision.entropy_tier == "balanced"


# =============================================================================
# CONSTRAINT TESTS
# =============================================================================


class TestConstraints:
    """Test routing with user-specified constraints."""

    @patch("astrai_router.engine.get_storage")
    def test_excluded_providers(self, mock_storage):
        """Excluded providers should not appear in recommendation."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        engine = RoutingEngine()
        constraints = RoutingConstraints(excluded_providers=["openai", "anthropic"])
        decision = engine.recommend(
            task_type="chat",
            complexity="low",
            entropy=0.5,
            tokens=100,
            constraints=constraints,
        )
        assert decision.provider not in ["openai", "anthropic"]

    @patch("astrai_router.engine.get_storage")
    def test_required_tier(self, mock_storage):
        """Required tier should force venue selection to that tier."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        engine = RoutingEngine()
        constraints = RoutingConstraints(required_tier="ultra")
        decision = engine.recommend(
            task_type="coding",
            complexity="low",
            entropy=0.3,  # Would normally route to draft
            tokens=100,
            constraints=constraints,
        )
        assert decision.tier == "ultra"

    @patch("astrai_router.engine.get_storage")
    def test_max_latency_filter(self, mock_storage):
        """Venues exceeding max latency should be filtered out."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        engine = RoutingEngine()
        constraints = RoutingConstraints(max_latency_ms=200)
        decision = engine.recommend(
            task_type="chat",
            complexity="low",
            entropy=0.5,
            tokens=100,
            constraints=constraints,
        )
        # All venues with p50 > 200ms should be excluded
        # Fast providers (groq, cerebras, deepinfra) have p50 <= 200
        assert decision.provider in ["groq", "cerebras", "deepinfra"]


# =============================================================================
# FEEDBACK / THOMPSON LEARNING TESTS
# =============================================================================


class TestFeedback:
    """Test feedback recording and Thompson Sampling updates."""

    @patch("astrai_router.engine.get_storage")
    def test_record_feedback_success(self, mock_storage):
        """Recording success should update Thompson prior."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        engine = RoutingEngine()
        engine.record_feedback(
            task_type="coding",
            provider="openai",
            model="gpt-4o",
            success=True,
            latency_ms=300,
            cost_usd=0.01,
        )
        key = ("coding", "openai/gpt-4o")
        assert key in engine._thompson_priors
        assert engine._thompson_priors[key].alpha == 2.0  # 1 (initial) + 1

    @patch("astrai_router.engine.get_storage")
    def test_record_feedback_failure(self, mock_storage):
        """Recording failure should update Thompson prior."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        engine = RoutingEngine()
        engine.record_feedback(
            task_type="coding",
            provider="openai",
            model="gpt-4o",
            success=False,
            latency_ms=5000,
            cost_usd=0.01,
        )
        key = ("coding", "openai/gpt-4o")
        assert engine._thompson_priors[key].beta == 2.0  # 1 (initial) + 1

    @patch("astrai_router.engine.get_storage")
    def test_thompson_sample_for_unknown_venue(self, mock_storage):
        """Unknown venue should return neutral sample (0.5)."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        engine = RoutingEngine()
        sample = engine.get_thompson_sample_for_venue("coding", "unknown/model-xyz")
        assert sample == 0.5

    @patch("astrai_router.engine.get_storage")
    def test_thompson_stats_empty(self, mock_storage):
        """Stats for unknown task type should be empty."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        engine = RoutingEngine()
        stats = engine.get_thompson_stats("nonexistent_task")
        assert stats == {}

    @patch("astrai_router.engine.get_storage")
    def test_thompson_stats_populated(self, mock_storage):
        """Stats should reflect recorded feedback."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        engine = RoutingEngine()
        for _ in range(5):
            engine.record_feedback("coding", "openai", "gpt-4o", True, 300, 0.01)
        for _ in range(2):
            engine.record_feedback("coding", "openai", "gpt-4o", False, 5000, 0.01)

        stats = engine.get_thompson_stats("coding")
        assert "openai/gpt-4o" in stats
        entry = stats["openai/gpt-4o"]
        assert entry["alpha"] == 6.0  # 1 initial + 5 successes
        assert entry["beta"] == 3.0  # 1 initial + 2 failures


# =============================================================================
# ARBITRAGE TESTS
# =============================================================================


class TestArbitrage:
    """Test ARBITRAGE quality improvement calculations."""

    @patch("astrai_router.engine.get_storage")
    def test_arbitrage_no_switch_for_optimal(self, mock_storage):
        """No switch recommended when using optimal model."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        engine = RoutingEngine()
        result = engine.calculate_arbitrage(
            current_provider="anthropic",
            current_model="claude-opus-4.5",
            task_type="coding",
            complexity="high",
        )
        # claude-opus-4.5 is top or near-top for coding (0.99)
        assert result["delta"] <= engine.arbitrage_threshold

    @patch("astrai_router.engine.get_storage")
    def test_arbitrage_recommends_switch_from_weak_model(self, mock_storage):
        """Should recommend switch from a weaker model for the task."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        engine = RoutingEngine()
        result = engine.calculate_arbitrage(
            current_provider="openai",
            current_model="gpt-4o-mini",
            task_type="coding",
            complexity="high",
        )
        # gpt-4o-mini has 0.80 for coding, while top is 0.99
        assert result["switch_recommended"] is True
        assert result["delta"] > 0

    @patch("astrai_router.engine.get_storage")
    def test_arbitrage_unknown_venue(self, mock_storage):
        """Unknown venue should not crash."""
        mock_storage.return_value = MagicMock(get=MagicMock(return_value=[]))
        engine = RoutingEngine()
        result = engine.calculate_arbitrage(
            current_provider="unknown",
            current_model="unknown-model",
            task_type="coding",
            complexity="medium",
        )
        assert result["switch_recommended"] is False
        assert "not found" in result["reason"]


# =============================================================================
# VENUE STATS TESTS
# =============================================================================


class TestVenueStats:
    """Test VenueStats dataclass."""

    def test_venue_creation(self):
        venue = VenueStats(
            provider="test",
            model="test-model",
            tier=RoutingTier.DRAFT,
            cost_per_mtok_input=0.1,
            cost_per_mtok_output=0.2,
            latency_p50_ms=100,
            latency_p99_ms=500,
        )
        assert venue.provider == "test"
        assert venue.tier == RoutingTier.DRAFT
        assert venue.success_rate == 1.0  # default
        assert venue.quality_score == 1.0  # default


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
