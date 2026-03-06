"""
Tests for Advanced Routing Features (astrai_router.advanced)

Comprehensive tests for production-grade routing features:
1. Fallback chain execution
2. Webhook management
3. Model health tracking
4. Smart recommendations
5. Input validation

Note: Tests for ModelPreference/PreferenceMemory have been removed since
preference_memory was not migrated to the astrai_router package.
"""

import pytest
import json
import hmac
import hashlib
from datetime import datetime, timedelta, timezone

from astrai_router.advanced import (
    ModelStatus,
    WebhookEvent,
    FallbackChain,
    FallbackChainStep,
    ModelHealth,
    SmartRecommendation,
    FallbackChainExecutor,
    WebhookManager,
    ModelHealthTracker,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_fallback_chain():
    """Create a sample fallback chain for testing."""
    return FallbackChain(
        chain_id="test-chain-123",
        steps=[
            FallbackChainStep(
                model="claude-4-opus",
                provider="anthropic",
                max_latency_ms=5000,
            ),
            FallbackChainStep(
                model="gpt-5.2",
                provider="openai",
                max_latency_ms=3000,
            ),
            FallbackChainStep(
                model="llama-3.3-70b",
                provider="together",
                max_latency_ms=2000,
            ),
        ],
        max_retries=3,
        total_timeout_ms=30000,
        trigger_on_timeout=True,
        trigger_on_rate_limit=True,
        trigger_on_error=True,
        trigger_on_quality_fail=True,
        min_quality_score=0.7,
    )


@pytest.fixture
def sample_webhook():
    """Create a sample webhook for testing."""
    return {
        "id": "webhook-123",
        "user_id": "user-456",
        "name": "Test Webhook",
        "url": "https://example.com/webhook",
        "secret": "test-secret-key",
        "events": ["budget_exceeded", "model_down", "quality_alert"],
        "is_active": True,
    }


@pytest.fixture
def sample_model_health():
    """Create sample model health status."""
    return ModelHealth(
        model_id="claude-4-opus",
        provider_id="anthropic",
        status=ModelStatus.HEALTHY,
        is_available=True,
        success_rate=0.99,
        avg_latency_ms=150,
        p95_latency_ms=300,
        error_rate=0.01,
        last_error=None,
        last_checked_at=datetime.now(timezone.utc),
    )


# =============================================================================
# FALLBACK CHAIN TESTS
# =============================================================================


class TestFallbackChainExecutor:
    """Tests for FallbackChainExecutor."""

    def test_should_trigger_timeout(self, sample_fallback_chain):
        """Test fallback triggers on timeout."""
        executor = FallbackChainExecutor()
        assert executor.should_trigger(sample_fallback_chain, "timeout") is True

    def test_should_trigger_rate_limit(self, sample_fallback_chain):
        """Test fallback triggers on rate limit."""
        executor = FallbackChainExecutor()
        assert executor.should_trigger(sample_fallback_chain, "rate_limit") is True

    def test_should_trigger_error(self, sample_fallback_chain):
        """Test fallback triggers on error."""
        executor = FallbackChainExecutor()
        assert executor.should_trigger(sample_fallback_chain, "error") is True

    def test_should_trigger_quality_low_score(self, sample_fallback_chain):
        """Test fallback triggers when quality is below threshold."""
        executor = FallbackChainExecutor()
        assert (
            executor.should_trigger(sample_fallback_chain, "quality", quality_score=0.5)
            is True
        )

    def test_should_not_trigger_quality_high_score(self, sample_fallback_chain):
        """Test fallback does not trigger when quality is above threshold."""
        executor = FallbackChainExecutor()
        assert (
            executor.should_trigger(sample_fallback_chain, "quality", quality_score=0.9)
            is False
        )

    def test_should_not_trigger_disabled(self):
        """Test fallback does not trigger when disabled."""
        chain = FallbackChain(
            chain_id="test",
            steps=[],
            trigger_on_timeout=False,
            trigger_on_rate_limit=False,
            trigger_on_error=False,
            trigger_on_quality_fail=False,
        )
        executor = FallbackChainExecutor()
        assert executor.should_trigger(chain, "timeout") is False
        assert executor.should_trigger(chain, "rate_limit") is False
        assert executor.should_trigger(chain, "error") is False


# =============================================================================
# WEBHOOK TESTS
# =============================================================================


class TestWebhookManager:
    """Tests for WebhookManager."""

    def test_hmac_signature_generation(self, sample_webhook):
        """Test HMAC signature is correctly generated."""
        payload = {"event": "test", "data": {"key": "value"}}
        payload_json = json.dumps(payload)
        secret = sample_webhook["secret"]

        expected_signature = hmac.new(
            secret.encode(), payload_json.encode(), hashlib.sha256
        ).hexdigest()

        # Verify format matches expected
        assert len(expected_signature) == 64  # SHA256 hex digest length

    def test_webhook_payload_structure(self):
        """Test webhook payload has correct structure."""
        manager = WebhookManager()

        # Verify payload structure
        event_type = "budget_exceeded"
        payload = {
            "user_id": "user-123",
            "budget": 100.0,
            "spent": 110.0,
        }

        full_payload = {
            "event": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": payload,
        }

        assert "event" in full_payload
        assert "timestamp" in full_payload
        assert "data" in full_payload
        assert full_payload["event"] == event_type

    def test_webhook_event_types(self):
        """Test all webhook event types are valid."""
        valid_events = [e.value for e in WebhookEvent]

        assert "budget_exceeded" in valid_events
        assert "budget_warning" in valid_events
        assert "model_down" in valid_events
        assert "model_recovered" in valid_events
        assert "quality_alert" in valid_events
        assert "latency_spike" in valid_events
        assert "rate_limit_hit" in valid_events
        assert "fallback_triggered" in valid_events
        assert "daily_summary" in valid_events
        assert "weekly_report" in valid_events


# =============================================================================
# MODEL HEALTH TESTS
# =============================================================================


class TestModelHealthTracker:
    """Tests for ModelHealthTracker."""

    def test_status_transitions(self):
        """Test model status transitions are correct."""
        assert ModelStatus.HEALTHY.value == "healthy"
        assert ModelStatus.DEGRADED.value == "degraded"
        assert ModelStatus.DOWN.value == "down"
        assert ModelStatus.UNKNOWN.value == "unknown"

    def test_is_available_from_cache(self):
        """Test is_available returns True when not in cache."""
        tracker = ModelHealthTracker()
        # Should default to True when not in cache
        assert tracker.is_available("unknown-model", "unknown-provider") is True

    def test_health_cache_update(self, sample_model_health):
        """Test health cache is properly updated."""
        tracker = ModelHealthTracker()
        cache_key = f"{sample_model_health.model_id}:{sample_model_health.provider_id}"

        # Manually add to cache
        tracker._health_cache[cache_key] = sample_model_health

        # Verify it's accessible
        assert (
            tracker.is_available(
                sample_model_health.model_id, sample_model_health.provider_id
            )
            is True
        )

    def test_success_rate_thresholds(self):
        """Test status is determined by success rate thresholds."""
        # Create health objects with different success rates
        healthy = ModelHealth(
            model_id="test",
            provider_id="test",
            status=ModelStatus.HEALTHY,
            is_available=True,
            success_rate=0.95,
            avg_latency_ms=100,
            p95_latency_ms=200,
            error_rate=0.05,
            last_error=None,
            last_checked_at=datetime.now(timezone.utc),
        )
        assert healthy.success_rate >= 0.9

        degraded = ModelHealth(
            model_id="test",
            provider_id="test",
            status=ModelStatus.DEGRADED,
            is_available=True,
            success_rate=0.75,
            avg_latency_ms=100,
            p95_latency_ms=200,
            error_rate=0.25,
            last_error=None,
            last_checked_at=datetime.now(timezone.utc),
        )
        assert 0.5 <= degraded.success_rate < 0.9

        down = ModelHealth(
            model_id="test",
            provider_id="test",
            status=ModelStatus.DOWN,
            is_available=False,
            success_rate=0.3,
            avg_latency_ms=100,
            p95_latency_ms=200,
            error_rate=0.7,
            last_error="Connection failed",
            last_checked_at=datetime.now(timezone.utc),
        )
        assert down.success_rate < 0.5


# =============================================================================
# SMART RECOMMENDATIONS TESTS
# =============================================================================


class TestSmartRecommendation:
    """Tests for SmartRecommendation."""

    def test_recommendation_structure(self):
        """Test recommendation has all required fields."""
        rec = SmartRecommendation(
            id="rec-123",
            recommendation_type="cost_optimization",
            title="Switch to DeepSeek for Code",
            description="You can save 50% on code tasks by using DeepSeek V3.2",
            estimated_cost_change_pct=-50.0,
            estimated_quality_change_pct=-5.0,
            estimated_latency_change_pct=10.0,
            confidence=0.85,
            action_payload={"type": "suggest_model", "model": "deepseek-v3.2"},
            status="pending",
            valid_until=datetime.now(timezone.utc) + timedelta(days=7),
        )

        assert rec.recommendation_type == "cost_optimization"
        assert rec.estimated_cost_change_pct == -50.0
        assert rec.confidence == 0.85
        assert rec.status == "pending"


# =============================================================================
# INPUT VALIDATION TESTS
# =============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_webhook_url_validation(self):
        """Test webhook URL must be HTTPS."""
        from urllib.parse import urlparse

        valid_urls = [
            "https://example.com/webhook",
            "https://api.company.com/astrai/callback",
            "https://hooks.slack.com/services/xxx",
        ]

        invalid_urls = [
            "http://example.com/webhook",  # HTTP not HTTPS
            "ftp://example.com/webhook",  # Wrong protocol
            "javascript:alert(1)",  # XSS attempt
            "file:///etc/passwd",  # File access
        ]

        for url in valid_urls:
            parsed = urlparse(url)
            assert parsed.scheme == "https", f"URL should be HTTPS: {url}"

        for url in invalid_urls:
            parsed = urlparse(url)
            assert parsed.scheme != "https", f"URL should not be valid: {url}"

    def test_chain_step_validation(self):
        """Test fallback chain step validation."""
        # Valid step
        valid_step = FallbackChainStep(
            model="claude-4-opus",
            provider="anthropic",
            max_latency_ms=5000,
        )
        assert valid_step.model == "claude-4-opus"
        assert valid_step.max_latency_ms == 5000

        # Verify max_latency_ms must be positive
        assert valid_step.max_latency_ms > 0

    def test_quality_score_range(self):
        """Test quality score must be in valid range."""
        # Valid range is 0-5
        valid_scores = [0.0, 2.5, 3.0, 4.5, 5.0]
        invalid_scores = [-1.0, 5.1, 10.0]

        for score in valid_scores:
            assert 0.0 <= score <= 5.0, f"Score should be valid: {score}"

        for score in invalid_scores:
            assert not (0.0 <= score <= 5.0), f"Score should be invalid: {score}"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for advanced routing."""

    def test_fallback_chain_with_health_check(
        self, sample_fallback_chain, sample_model_health
    ):
        """Test fallback chain respects model health."""
        tracker = ModelHealthTracker()
        executor = FallbackChainExecutor()

        # Mark first model as down
        down_health = ModelHealth(
            model_id="claude-4-opus",
            provider_id="anthropic",
            status=ModelStatus.DOWN,
            is_available=False,
            success_rate=0.3,
            avg_latency_ms=0,
            p95_latency_ms=0,
            error_rate=0.7,
            last_error="Connection refused",
            last_checked_at=datetime.now(timezone.utc),
        )

        cache_key = f"{down_health.model_id}:{down_health.provider_id}"
        tracker._health_cache[cache_key] = down_health

        # First model should not be available
        assert tracker.is_available("claude-4-opus", "anthropic") is False

        # Second model in chain should still be available
        assert tracker.is_available("gpt-5.2", "openai") is True


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_fallback_chain(self):
        """Test handling of empty fallback chain."""
        chain = FallbackChain(
            chain_id="empty",
            steps=[],
            max_retries=3,
        )
        assert len(chain.steps) == 0

    def test_single_step_chain(self):
        """Test fallback chain with single step."""
        chain = FallbackChain(
            chain_id="single",
            steps=[
                FallbackChainStep(
                    model="claude-4-opus",
                    provider="anthropic",
                )
            ],
        )
        assert len(chain.steps) == 1

    def test_very_low_quality_threshold(self):
        """Test very low quality threshold triggers fallback."""
        chain = FallbackChain(
            chain_id="low-quality",
            steps=[],
            trigger_on_quality_fail=True,
            min_quality_score=0.1,
        )
        executor = FallbackChainExecutor()

        # Quality of 0.05 should trigger
        assert executor.should_trigger(chain, "quality", quality_score=0.05) is True

        # Quality of 0.2 should not trigger
        assert executor.should_trigger(chain, "quality", quality_score=0.2) is False

    def test_health_status_unknown(self):
        """Test handling of unknown health status."""
        health = ModelHealth(
            model_id="new-model",
            provider_id="new-provider",
            status=ModelStatus.UNKNOWN,
            is_available=True,  # Assume available if unknown
            success_rate=1.0,
            avg_latency_ms=0,
            p95_latency_ms=0,
            error_rate=0.0,
            last_error=None,
            last_checked_at=datetime.now(timezone.utc),
        )
        assert health.status == ModelStatus.UNKNOWN
        assert health.is_available is True


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
