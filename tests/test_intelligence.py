"""
Tests for astrai_router.intelligence — Routing Intelligence Store.

Covers:
1. RoutingIntelligenceStore — SQLite CRUD for routing patterns
2. RoutingIntelligenceManager — per-user store management
3. Workflow pattern recording and recommendation
4. End-to-end: record outcomes -> build history -> get recommendation

Note: Tests that depended on main._infer_tier_from_model, main._resolve_auto_tier,
and main._step_intelligence_savings from the old monolith have been removed since
those functions are not part of the astrai_router package.
"""

import asyncio
import tempfile
import shutil

import pytest

from astrai_router.intelligence import (
    RoutingIntelligenceStore,
    RoutingIntelligenceManager,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def tmp_data_dir():
    """Temporary directory for per-test SQLite stores."""
    d = tempfile.mkdtemp(prefix="routing_intel_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(tmp_data_dir):
    """Fresh RoutingIntelligenceStore for a test user."""
    return RoutingIntelligenceStore("test-user-001", tmp_data_dir)


@pytest.fixture
def manager(tmp_data_dir):
    """Fresh RoutingIntelligenceManager pointing at temp dir."""
    return RoutingIntelligenceManager(data_dir=tmp_data_dir)


def run(coro):
    """Helper to run async in sync tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ============================================================================
# 1. RoutingIntelligenceStore TESTS
# ============================================================================


class TestRoutingIntelligenceStore:
    """Test the per-user SQLite store for routing intelligence."""

    def test_store_creates_db(self, store, tmp_data_dir):
        """Store should create SQLite DB on init."""
        import os

        db_files = [f for f in os.listdir(tmp_data_dir) if f.endswith(".db")]
        assert len(db_files) == 1
        assert "routing_" in db_files[0]

    def test_record_outcome_basic(self, store):
        """Recording an outcome should not raise."""
        run(
            store.record_outcome(
                task_type="code",
                model="gpt-5.2",
                provider="openai",
                success=True,
                quality_score=4.0,
                latency_ms=500,
                cost_usd=0.01,
            )
        )

    def test_record_outcome_with_workflow(self, store):
        """Recording with workflow_id + step_id should create a pattern."""
        run(
            store.record_outcome(
                task_type="code",
                model="gpt-5.2",
                provider="openai",
                success=True,
                quality_score=4.5,
                latency_ms=300,
                cost_usd=0.005,
                workflow_id="wf-test-001",
                step_id="step-compile",
            )
        )
        pattern = run(store.get_workflow_recommendation("wf-test-001", "step-compile"))
        assert pattern is not None
        assert pattern.best_model == "gpt-5.2"
        assert pattern.sample_count == 1

    def test_workflow_pattern_accumulates(self, store):
        """Multiple outcomes for same step should accumulate."""
        for i in range(5):
            run(
                store.record_outcome(
                    task_type="code",
                    model="gpt-5.2",
                    provider="openai",
                    success=True,
                    quality_score=4.0 + (i * 0.1),
                    latency_ms=300,
                    cost_usd=0.005,
                    workflow_id="wf-001",
                    step_id="step-build",
                )
            )
        pattern = run(store.get_workflow_recommendation("wf-001", "step-build"))
        assert pattern is not None
        assert pattern.sample_count == 5
        assert pattern.avg_quality_score > 4.0

    def test_best_model_updates_on_higher_quality(self, store):
        """Best model should update when a better-quality model is recorded."""
        # First: mediocre model
        run(
            store.record_outcome(
                task_type="code",
                model="llama-3.3-70b",
                provider="groq",
                success=True,
                quality_score=3.0,
                latency_ms=200,
                cost_usd=0.001,
                workflow_id="wf-002",
                step_id="step-a",
            )
        )
        # Second: better model
        run(
            store.record_outcome(
                task_type="code",
                model="gpt-5.2",
                provider="openai",
                success=True,
                quality_score=4.8,
                latency_ms=400,
                cost_usd=0.01,
                workflow_id="wf-002",
                step_id="step-a",
            )
        )
        pattern = run(store.get_workflow_recommendation("wf-002", "step-a"))
        assert pattern.best_model == "gpt-5.2"

    def test_get_best_model_for_task(self, store):
        """After enough samples, get_best_model_for_task should return a result."""
        for i in range(6):
            run(
                store.record_outcome(
                    task_type="research",
                    model="deepseek-v3.2",
                    provider="deepinfra",
                    success=True,
                    quality_score=4.2,
                    latency_ms=600,
                    cost_usd=0.008,
                )
            )
        result = run(store.get_best_model_for_task("research"))
        assert result is not None
        model_name, success_rate = result
        assert model_name == "deepseek-v3.2"
        assert success_rate == 1.0

    def test_get_best_model_below_min_samples(self, store):
        """Below min_samples, should return None."""
        run(
            store.record_outcome(
                task_type="creative",
                model="gpt-5.2",
                provider="openai",
                success=True,
                quality_score=4.0,
                latency_ms=400,
                cost_usd=0.01,
            )
        )
        result = run(store.get_best_model_for_task("creative", min_samples=5))
        assert result is None

    def test_no_recommendation_for_unknown_step(self, store):
        """Unknown workflow/step should return None."""
        result = run(store.get_workflow_recommendation("unknown-wf", "unknown-step"))
        assert result is None


# ============================================================================
# 2. get_routing_context INTEGRATION TESTS
# ============================================================================


class TestGetRoutingContext:
    """Test the full routing context assembly."""

    def test_empty_context_for_new_user(self, store):
        """New user with no history should get minimal context."""
        ctx = run(store.get_routing_context("code"))
        assert ctx["task_type"] == "code"
        assert ctx["has_history"] is False
        assert "workflow_recommendation" not in ctx

    def test_context_with_model_history(self, store):
        """After recording enough outcomes, should get model recommendation."""
        for _ in range(6):
            run(
                store.record_outcome(
                    task_type="code",
                    model="gpt-5.2",
                    provider="openai",
                    success=True,
                    quality_score=4.5,
                    latency_ms=300,
                    cost_usd=0.01,
                )
            )
        ctx = run(store.get_routing_context("code"))
        assert ctx["has_history"] is True
        assert ctx["recommended_model"] == "gpt-5.2"

    def test_context_with_workflow_recommendation(self, store):
        """With enough workflow samples, should include workflow_recommendation."""
        for _ in range(5):
            run(
                store.record_outcome(
                    task_type="code",
                    model="deepseek-v3.2",
                    provider="deepinfra",
                    success=True,
                    quality_score=4.0,
                    latency_ms=400,
                    cost_usd=0.005,
                    workflow_id="wf-ctx-test",
                    step_id="step-gen",
                )
            )
        ctx = run(
            store.get_routing_context(
                "code", workflow_id="wf-ctx-test", step_id="step-gen"
            )
        )
        assert "workflow_recommendation" in ctx
        rec = ctx["workflow_recommendation"]
        assert rec["model"] == "deepseek-v3.2"
        assert rec["confidence"] > 0
        assert rec["sample_count"] == 5

    def test_context_workflow_below_threshold(self, store):
        """Below 3 samples, no workflow_recommendation."""
        run(
            store.record_outcome(
                task_type="chat",
                model="llama-4-scout",
                provider="groq",
                success=True,
                quality_score=3.0,
                latency_ms=100,
                cost_usd=0.0001,
                workflow_id="wf-few",
                step_id="step-few",
            )
        )
        ctx = run(
            store.get_routing_context("chat", workflow_id="wf-few", step_id="step-few")
        )
        assert "workflow_recommendation" not in ctx

    def test_confidence_scaling(self, store):
        """Confidence should scale with sample count (min 3 to appear, capped at 1.0)."""
        for _ in range(10):
            run(
                store.record_outcome(
                    task_type="code",
                    model="gpt-5.2",
                    provider="openai",
                    success=True,
                    quality_score=4.0,
                    latency_ms=300,
                    cost_usd=0.01,
                    workflow_id="wf-conf",
                    step_id="step-conf",
                )
            )
        ctx = run(
            store.get_routing_context(
                "code", workflow_id="wf-conf", step_id="step-conf"
            )
        )
        rec = ctx["workflow_recommendation"]
        assert rec["confidence"] == 0.5  # 10 / 20

    def test_confidence_caps_at_one(self, store):
        """Confidence should cap at 1.0 even with many samples."""
        for _ in range(25):
            run(
                store.record_outcome(
                    task_type="code",
                    model="gpt-5.2",
                    provider="openai",
                    success=True,
                    quality_score=4.0,
                    latency_ms=300,
                    cost_usd=0.01,
                    workflow_id="wf-cap",
                    step_id="step-cap",
                )
            )
        ctx = run(
            store.get_routing_context("code", workflow_id="wf-cap", step_id="step-cap")
        )
        assert ctx["workflow_recommendation"]["confidence"] == 1.0


# ============================================================================
# 3. RoutingIntelligenceManager TESTS
# ============================================================================


class TestRoutingIntelligenceManager:
    """Test the global manager that handles per-user stores."""

    def test_manager_creates_store_per_user(self, manager):
        """Different user IDs should get different stores."""
        store_a = manager.get_store("user-a")
        store_b = manager.get_store("user-b")
        assert store_a.user_id != store_b.user_id

    def test_manager_reuses_store(self, manager):
        """Same user ID should return the same store instance."""
        store1 = manager.get_store("user-x")
        store2 = manager.get_store("user-x")
        assert store1 is store2

    def test_manager_record_and_context(self, manager):
        """Manager convenience functions should work end to end."""
        for _ in range(5):
            run(
                manager.record_outcome(
                    user_id="user-mgr",
                    task_type="code",
                    model="gpt-5.2",
                    provider="openai",
                    success=True,
                    quality_score=4.2,
                    latency_ms=350,
                    cost_usd=0.008,
                    workflow_id="wf-mgr",
                    step_id="step-mgr",
                )
            )
        ctx = run(
            manager.get_routing_context(
                user_id="user-mgr",
                task_type="code",
                workflow_id="wf-mgr",
                step_id="step-mgr",
            )
        )
        assert "workflow_recommendation" in ctx
        assert ctx["workflow_recommendation"]["model"] == "gpt-5.2"


# ============================================================================
# 4. END-TO-END FLOW TESTS
# ============================================================================


class TestEndToEndStepLearning:
    """Simulate the full flow: record outcomes -> build history -> use in routing."""

    def test_full_workflow_learning_cycle(self, store):
        """
        1. First request: no history -> classification
        2. Record outcome
        3. After 6 outcomes: history available -> step-learned
        """
        # Step 1: No history yet
        ctx = run(
            store.get_routing_context(
                "code", workflow_id="wf-e2e", step_id="step-compile"
            )
        )
        assert "workflow_recommendation" not in ctx

        # Step 2: Record 6 outcomes (need confidence >= 0.3 -> 6/20 = 0.3, sample_count >= 3)
        for i in range(6):
            run(
                store.record_outcome(
                    task_type="code",
                    model="deepseek-v3.2",
                    provider="deepinfra",
                    success=True,
                    quality_score=4.0,
                    latency_ms=300,
                    cost_usd=0.003,
                    workflow_id="wf-e2e",
                    step_id="step-compile",
                )
            )

        # Step 3: Now history is available
        ctx = run(
            store.get_routing_context(
                "code", workflow_id="wf-e2e", step_id="step-compile"
            )
        )
        assert "workflow_recommendation" in ctx
        rec = ctx["workflow_recommendation"]
        assert rec["model"] == "deepseek-v3.2"
        assert rec["sample_count"] == 6
        assert rec["confidence"] >= 0.3  # 6/20 = 0.3

    def test_different_steps_learn_independently(self, store):
        """Two steps in the same workflow should learn different models."""
        # Step A: code generation -> ultra model
        for _ in range(5):
            run(
                store.record_outcome(
                    task_type="code",
                    model="gpt-5.2",
                    provider="openai",
                    success=True,
                    quality_score=4.5,
                    latency_ms=500,
                    cost_usd=0.01,
                    workflow_id="wf-multi",
                    step_id="step-codegen",
                )
            )

        # Step B: formatting -> draft model
        for _ in range(5):
            run(
                store.record_outcome(
                    task_type="chat",
                    model="llama-4-scout",
                    provider="groq",
                    success=True,
                    quality_score=3.5,
                    latency_ms=50,
                    cost_usd=0.0001,
                    workflow_id="wf-multi",
                    step_id="step-format",
                )
            )

        # Verify they're independent
        ctx_a = run(
            store.get_routing_context(
                "code", workflow_id="wf-multi", step_id="step-codegen"
            )
        )
        ctx_b = run(
            store.get_routing_context(
                "chat", workflow_id="wf-multi", step_id="step-format"
            )
        )

        assert ctx_a["workflow_recommendation"]["model"] == "gpt-5.2"
        assert ctx_b["workflow_recommendation"]["model"] == "llama-4-scout"

    def test_failure_signals_affect_learning(self, store):
        """Failures should lower the success rate for a model."""
        # 3 successes
        for _ in range(3):
            run(
                store.record_outcome(
                    task_type="code",
                    model="gpt-5.2",
                    provider="openai",
                    success=True,
                    quality_score=4.0,
                    latency_ms=300,
                    cost_usd=0.01,
                )
            )
        # 3 failures
        for _ in range(3):
            run(
                store.record_outcome(
                    task_type="code",
                    model="gpt-5.2",
                    provider="openai",
                    success=False,
                    quality_score=1.0,
                    latency_ms=300,
                    cost_usd=0.01,
                )
            )
        result = run(store.get_best_model_for_task("code"))
        assert result is not None
        _, success_rate = result
        assert success_rate == 0.5  # 3 success / 6 total


# ============================================================================
# 5. WORKFLOW SAVINGS TRACKING
# ============================================================================


class TestWorkflowSavingsTracking:
    """Test that step learning actually produces savings over classification."""

    def test_draft_step_saves_vs_ultra_classification(self, store):
        """
        A step that learns to use draft models instead of ultra
        should show cost savings potential.
        """
        for _ in range(10):
            run(
                store.record_outcome(
                    task_type="chat",
                    model="llama-4-scout",
                    provider="groq",
                    success=True,
                    quality_score=3.8,
                    latency_ms=50,
                    cost_usd=0.0001,
                    workflow_id="wf-savings",
                    step_id="step-format",
                )
            )

        pattern = run(store.get_workflow_recommendation("wf-savings", "step-format"))
        assert pattern is not None
        assert pattern.best_model == "llama-4-scout"

        # The avg quality should still be good even with draft model
        assert pattern.avg_quality_score >= 3.5

    def test_multi_step_workflow_diverse_tiers(self, store):
        """
        A multi-step workflow should learn different models for different steps.
        Step 1 (code gen): ultra model
        Step 2 (format): draft model
        Step 3 (review): balanced model
        """
        steps = [
            ("step-codegen", "code", "gpt-5.2", "openai", 4.5, 0.01),
            ("step-format", "chat", "llama-4-scout", "groq", 3.5, 0.0001),
            ("step-review", "research", "llama-3.3-70b", "deepinfra", 4.0, 0.003),
        ]
        for step_id, task_type, model, provider, quality, cost in steps:
            for _ in range(6):
                run(
                    store.record_outcome(
                        task_type=task_type,
                        model=model,
                        provider=provider,
                        success=True,
                        quality_score=quality,
                        latency_ms=300,
                        cost_usd=cost,
                        workflow_id="wf-diverse",
                        step_id=step_id,
                    )
                )

        # Verify each step learned the right model
        for step_id, _, expected_model, _, _, _ in steps:
            pattern = run(store.get_workflow_recommendation("wf-diverse", step_id))
            assert pattern.best_model == expected_model, (
                f"Step {step_id}: expected {expected_model}, got {pattern.best_model}"
            )

        # Verify recommendations exist and are distinct
        ctx_codegen = run(
            store.get_routing_context("code", "wf-diverse", "step-codegen")
        )
        ctx_format = run(store.get_routing_context("chat", "wf-diverse", "step-format"))
        ctx_review = run(
            store.get_routing_context("research", "wf-diverse", "step-review")
        )

        assert ctx_codegen["workflow_recommendation"]["model"] == "gpt-5.2"
        assert ctx_format["workflow_recommendation"]["model"] == "llama-4-scout"
        assert ctx_review["workflow_recommendation"]["model"] == "llama-3.3-70b"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
