"""Tests for astrai_router.classifier — health classification and task classification."""

from datetime import datetime, timedelta, timezone
from typing import Optional

from astrai_router.classifier import (
    RoutingHealthClassifier,
    RoutingHealthStatus,
    TaskClassifier,
    ProviderHealthTracker,
    BreakerState,
)


def _make_record(
    status: str = "success",
    latency_ms: float = 200.0,
    fallback_count: int = 0,
    cost_est: float = 1.0,
    cost_actual: float = 1.0,
    timestamp: Optional[datetime] = None,
):
    return {
        "final_status": status,
        "latency_ms": latency_ms,
        "fallback_count": fallback_count,
        "cost_usd_est": cost_est,
        "cost_usd_actual": cost_actual,
        "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
    }


def test_routing_health_unknown_when_insufficient_samples():
    classifier = RoutingHealthClassifier()
    records = [_make_record() for _ in range(5)]
    report = classifier.classify(records)
    assert report.status == RoutingHealthStatus.UNKNOWN
    assert "insufficient_samples" in report.reasons


def test_routing_health_healthy_defaults():
    classifier = RoutingHealthClassifier()
    records = [_make_record() for _ in range(30)]
    report = classifier.classify(records)
    assert report.status == RoutingHealthStatus.HEALTHY
    assert report.reasons == []


def test_routing_health_degraded_on_latency():
    classifier = RoutingHealthClassifier()
    records = [_make_record(latency_ms=1500.0) for _ in range(30)]
    report = classifier.classify(records)
    assert report.status == RoutingHealthStatus.DEGRADED
    assert "latency_above_target" in report.reasons


def test_routing_health_down_on_low_success_rate():
    classifier = RoutingHealthClassifier()
    records = []
    for _ in range(10):
        records.append(_make_record(status="success"))
    for _ in range(20):
        records.append(_make_record(status="degraded"))
    report = classifier.classify(records)
    assert report.status == RoutingHealthStatus.DOWN
    assert "low_success_rate" in report.reasons


def test_routing_health_window_filtering():
    classifier = RoutingHealthClassifier()
    now = datetime.now(timezone.utc)
    recent = [_make_record(timestamp=now - timedelta(minutes=5)) for _ in range(25)]
    old = [
        _make_record(status="degraded", timestamp=now - timedelta(minutes=120))
        for _ in range(25)
    ]
    report = classifier.classify(recent + old, window_minutes=30, now=now)
    assert report.status == RoutingHealthStatus.HEALTHY


def test_task_classifier_basic_signals():
    clf = TaskClassifier()
    res = clf.classify("Summarize in JSON the risks and mitigations.")
    assert res.task_type == "summarize"
    assert res.needs_json is True
    assert res.risk_level in {"low", "medium", "high"}


def test_provider_health_tracker_scores_and_breaker():
    tracker = ProviderHealthTracker(
        halflife_sec=600,
        min_score=0.55,
        half_open_score=0.65,
        cooldown_sec=60,
        ttl_sec=1,
    )
    provider = "groq"
    for _ in range(20):
        tracker.record(provider, success=True, latency_ms=200, is_timeout=False)
    snap_good = tracker.snapshot(provider)
    assert snap_good.score > 0.8
    # Inject failures/timeouts to drop score
    for _ in range(15):
        tracker.record(provider, success=False, latency_ms=2500, is_timeout=True)
    snap_bad = tracker.snapshot(provider)
    assert snap_bad.score < 0.55
    assert snap_bad.breaker_state == BreakerState.OPEN
