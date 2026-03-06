"""
Routing Health Classifier - Classifies requests and monitors provider health.

Provides:
- TaskClassifier: Classifies prompts into task types for routing optimization
- RoutingHealthClassifier: Analyzes routing telemetry for health monitoring
- ProviderHealthTracker: Tracks provider availability and performance
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from enum import Enum
import re


@dataclass
class TaskClassification:
    """Result of task classification."""
    task_type: str = "general"
    complexity: str = "medium"  # low, medium, high
    domain: Optional[str] = None
    requires_reasoning: bool = False
    requires_creativity: bool = False
    requires_code: bool = False
    requires_math: bool = False
    estimated_tokens: int = 500
    confidence: float = 0.5
    needs_json: bool = False
    risk_level: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "complexity": self.complexity,
            "domain": self.domain,
            "requires_reasoning": self.requires_reasoning,
            "requires_creativity": self.requires_creativity,
            "requires_code": self.requires_code,
            "requires_math": self.requires_math,
            "estimated_tokens": self.estimated_tokens,
            "confidence": self.confidence,
            "needs_json": self.needs_json,
            "risk_level": self.risk_level,
        }


class TaskClassifier:
    """
    Classifies prompts into task types for intelligent routing.

    Uses keyword matching and heuristics to determine:
    - Task type (coding, writing, analysis, chat, etc.)
    - Complexity level
    - Special requirements (reasoning, creativity, math)
    """

    TASK_PATTERNS = {
        "coding": [
            r"\b(code|program|function|class|implement|debug|fix|refactor)\b",
            r"\b(python|javascript|typescript|java|rust|go|sql)\b",
            r"\b(api|endpoint|database|query|algorithm)\b",
        ],
        "writing": [
            r"\b(write|draft|compose|create|generate)\b.*(article|essay|blog|story|email)",
            r"\b(summarize|paraphrase|rewrite|edit)\b",
        ],
        "analysis": [
            r"\b(analyze|evaluate|compare|assess|review)\b",
            r"\b(data|metrics|statistics|trends|insights)\b",
        ],
        "math": [
            r"\b(calculate|compute|solve|equation|formula)\b",
            r"\b(math|algebra|calculus|statistics|probability)\b",
            r"\d+[\+\-\*\/\^]\d+",
        ],
        "reasoning": [
            r"\b(explain|why|how|reason|logic|think through)\b",
            r"\b(step by step|chain of thought|reasoning)\b",
        ],
        "creative": [
            r"\b(creative|imaginative|brainstorm|ideas)\b",
            r"\b(story|poem|fiction|narrative)\b",
        ],
        "chat": [
            r"\b(hello|hi|hey|thanks|thank you)\b",
            r"^.{0,50}$",  # Short messages
        ],
    }

    COMPLEXITY_INDICATORS = {
        "high": [
            r"\b(complex|advanced|sophisticated|comprehensive)\b",
            r"\b(architect|design|system|infrastructure)\b",
            r".{1000,}",  # Long prompts
        ],
        "low": [
            r"\b(simple|basic|quick|brief)\b",
            r"^.{0,100}$",  # Very short
        ],
    }

    def classify(self, prompt: str) -> TaskClassification:
        """Classify a prompt into task type and requirements."""
        if not prompt:
            return TaskClassification()

        prompt_lower = prompt.lower()
        result = TaskClassification()

        # Quick JSON summarization heuristic
        if "summarize" in prompt_lower and "json" in prompt_lower:
            result.task_type = "summarize"
            result.needs_json = True
            result.risk_level = "medium"
            return result

        # Detect task type
        task_scores: Dict[str, int] = {}
        for task_type, patterns in self.TASK_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, prompt_lower, re.IGNORECASE))
            if score > 0:
                task_scores[task_type] = score

        if task_scores:
            result.task_type = max(task_scores, key=task_scores.get)
            result.confidence = min(0.9, 0.5 + task_scores[result.task_type] * 0.1)

        # Detect complexity
        for complexity, patterns in self.COMPLEXITY_INDICATORS.items():
            if any(re.search(p, prompt, re.IGNORECASE) for p in patterns):
                result.complexity = complexity
                break

        # Detect special requirements
        result.requires_code = result.task_type == "coding" or bool(
            re.search(r"```|def |function |class ", prompt)
        )
        result.requires_math = result.task_type == "math" or bool(
            re.search(r"\b(calculate|compute|equation|\d+[\+\-\*\/])\b", prompt_lower)
        )
        result.requires_reasoning = result.task_type == "reasoning" or bool(
            re.search(r"\b(why|explain|step by step|reason)\b", prompt_lower)
        )
        result.requires_creativity = result.task_type == "creative" or bool(
            re.search(r"\b(creative|brainstorm|imagine|story)\b", prompt_lower)
        )

        # Estimate tokens (rough heuristic)
        word_count = len(prompt.split())
        if result.complexity == "high":
            result.estimated_tokens = max(1000, word_count * 10)
        elif result.complexity == "low":
            result.estimated_tokens = max(100, word_count * 3)
        else:
            result.estimated_tokens = max(300, word_count * 5)

        return result


@dataclass
class RoutingHealthThresholds:
    """Thresholds for routing health classification."""
    error_rate_warning: float = 0.05  # 5%
    error_rate_critical: float = 0.15  # 15%
    latency_p50_warning_ms: int = 500
    latency_p50_critical_ms: int = 2000
    latency_p99_warning_ms: int = 5000
    latency_p99_critical_ms: int = 15000
    min_samples: int = 10


class RoutingHealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


@dataclass
class HealthReport:
    """Health report for routing system."""
    status: "RoutingHealthStatus" = None  # set after enum definition
    error_rate: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0
    sample_count: int = 0
    providers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    window_minutes: int = 60
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "error_rate": self.error_rate,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "sample_count": self.sample_count,
            "providers": self.providers,
            "alerts": self.alerts,
            "window_minutes": self.window_minutes,
        }


class RoutingHealthClassifier:
    """
    Analyzes routing telemetry to assess system health.

    Monitors:
    - Overall error rates
    - Latency percentiles
    - Provider-specific health
    """

    def __init__(self, thresholds: Optional[RoutingHealthThresholds] = None):
        self.thresholds = thresholds or RoutingHealthThresholds()

    def classify(
        self,
        telemetry: List[Dict[str, Any]],
        window_minutes: int = 60,
        now: Optional[datetime] = None,
    ) -> HealthReport:
        """Classify routing health from telemetry data."""
        report = HealthReport(window_minutes=window_minutes)
        now = now or datetime.now(timezone.utc)

        if not telemetry:
            report.status = RoutingHealthStatus.UNKNOWN
            report.reasons.append("no_telemetry")
            return report

        # Window filtering
        window_cutoff = now - timedelta(minutes=window_minutes)
        telemetry = [
            t for t in telemetry
            if datetime.fromisoformat(t.get("timestamp").replace("Z", "+00:00")) >= window_cutoff
        ]

        report.sample_count = len(telemetry)

        # Calculate error rate
        errors = sum(
            1
            for t in telemetry
            if t.get("error")
            or t.get("status_code", 200) >= 400
            or t.get("status", "success") != "success"
            or t.get("final_status", "success") != "success"
        )
        report.error_rate = errors / len(telemetry) if telemetry else 0

        # Calculate latencies
        latencies = [t.get("latency_ms", 0) for t in telemetry if t.get("latency_ms")]
        if latencies:
            latencies.sort()
            report.latency_p50_ms = latencies[len(latencies) // 2]
            report.latency_p99_ms = latencies[int(len(latencies) * 0.99)]

        # Aggregate by provider
        provider_stats: Dict[str, Dict[str, Any]] = {}
        for t in telemetry:
            provider = t.get("provider", "unknown")
            if provider not in provider_stats:
                provider_stats[provider] = {
                    "requests": 0,
                    "errors": 0,
                    "latencies": [],
                }
            provider_stats[provider]["requests"] += 1
            if t.get("error") or t.get("status_code", 200) >= 400:
                provider_stats[provider]["errors"] += 1
            if t.get("latency_ms"):
                provider_stats[provider]["latencies"].append(t["latency_ms"])

        for provider, stats in provider_stats.items():
            error_rate = stats["errors"] / stats["requests"] if stats["requests"] else 0
            avg_latency = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0
            report.providers[provider] = {
                "requests": stats["requests"],
                "error_rate": error_rate,
                "avg_latency_ms": avg_latency,
                "status": "healthy" if error_rate < self.thresholds.error_rate_warning else "degraded",
            }

        # Determine overall status
        if report.sample_count < self.thresholds.min_samples:
            report.status = RoutingHealthStatus.UNKNOWN
            report.reasons.append("insufficient_samples")
        elif report.error_rate >= self.thresholds.error_rate_critical:
            report.status = RoutingHealthStatus.DOWN
            report.reasons.append("critical_error_rate")
            report.reasons.append("low_success_rate")
        elif report.latency_p99_ms >= self.thresholds.latency_p99_critical_ms:
            report.status = RoutingHealthStatus.DOWN
            report.reasons.append("latency_p99_critical")
        elif report.error_rate >= self.thresholds.error_rate_warning:
            report.status = RoutingHealthStatus.DEGRADED
            report.reasons.append("error_rate_warning")
        elif report.latency_p50_ms >= self.thresholds.latency_p50_warning_ms:
            report.status = RoutingHealthStatus.DEGRADED
            report.reasons.append("latency_above_target")
        else:
            report.status = RoutingHealthStatus.HEALTHY
            report.reasons = []

        return report


class ProviderHealthTracker:
    """
    Tracks real-time health of LLM providers.

    Maintains:
    - Success/failure counts with sliding window
    - Latency statistics
    - Circuit breaker state
    """

    class BreakerState(Enum):
        CLOSED = "closed"
        HALF_OPEN = "half_open"
        OPEN = "open"

    BreakerState = BreakerState  # alias for external import

    @dataclass
    class ProviderHealthSnapshot:
        status: str
        score: float
        breaker_state: "ProviderHealthTracker.BreakerState"
        success_rate: float
        latency_p50_ms: float
        latency_p95_ms: float
        samples: int

        def to_dict(self) -> dict:
            return {
                "status": self.status,
                "score": self.score,
                "breaker_state": self.breaker_state.value if self.breaker_state else "closed",
                "success_rate": self.success_rate,
                "latency_p50_ms": self.latency_p50_ms,
                "latency_p95_ms": self.latency_p95_ms,
                "samples": self.samples,
            }

    def __init__(
        self,
        halflife_sec: int = 300,
        min_score: float = 0.5,
        half_open_score: float = 0.65,
        cooldown_sec: int = 60,
        ttl_sec: int = 900,
    ):
        self.halflife_sec = halflife_sec
        self.min_score = min_score
        self.half_open_score = half_open_score
        self.cooldown_sec = cooldown_sec
        self.ttl_sec = ttl_sec
        self._providers: Dict[str, Dict[str, Any]] = {}

    def _decay_weight(self, age_sec: float) -> float:
        return 0.5 ** (age_sec / self.halflife_sec)

    @classmethod
    def from_env(cls):
        """Factory to keep backward compatibility with older imports."""
        return cls()

    def record_success(self, provider: str, latency_ms: float):
        """Record a successful request."""
        self._ensure_provider(provider)
        now = datetime.now(timezone.utc)
        self._providers[provider]["successes"].append((now, latency_ms))
        self._providers[provider]["last_success"] = now
        self._cleanup_old_records(provider)
        self._update_circuit(provider)

    def record_failure(self, provider: str, error: str):
        """Record a failed request."""
        self._ensure_provider(provider)
        now = datetime.now(timezone.utc)
        self._providers[provider]["failures"].append((now, error))
        self._providers[provider]["last_failure"] = now
        self._cleanup_old_records(provider)
        self._update_circuit(provider)

    def record(self, provider: str, success: bool, latency_ms: float = 0.0, is_timeout: bool = False):
        """Unified record method used in tests."""
        if success:
            self.record_success(provider, latency_ms)
        else:
            self.record_failure(provider, "timeout" if is_timeout else "error")

    def is_healthy(self, provider: str) -> bool:
        """Check if provider is healthy (circuit closed)."""
        snap = self.snapshot(provider)
        return snap.breaker_state != self.BreakerState.OPEN

    def get_stats(self, provider: str) -> Dict[str, Any]:
        """Get health statistics for a provider."""
        snap = self.snapshot(provider)
        return {
            "status": snap.status,
            "requests": snap.samples,
            "success_rate": snap.success_rate,
            "avg_latency_ms": snap.latency_p50_ms,
            "circuit_open": snap.breaker_state == self.BreakerState.OPEN,
        }

    def snapshot(self, provider: str) -> "ProviderHealthSnapshot":
        self._cleanup_old_records(provider)
        data = self._providers.get(provider)
        if not data:
            return self.ProviderHealthSnapshot(
                status="unknown",
                score=1.0,
                breaker_state=self.BreakerState.CLOSED,
                success_rate=1.0,
                latency_p50_ms=0.0,
                latency_p95_ms=0.0,
                samples=0,
            )

        now = datetime.now(timezone.utc)
        success_weights = [self._decay_weight((now - t).total_seconds()) for t, _ in data["successes"]]
        failure_weights = [
            self._decay_weight((now - t).total_seconds()) * (2.0 if err == "timeout" else 1.0)
            for t, err in data["failures"]
        ]
        w_success = sum(success_weights)
        w_failure = sum(failure_weights)
        score = w_success / (w_success + w_failure + 1e-9)

        latencies = sorted([lat for _, lat in data["successes"]])
        def percentile(vals: List[float], pct: float) -> float:
            if not vals:
                return 0.0
            k = max(0, min(len(vals) - 1, int(len(vals) * pct)))
            return vals[k]
        p50 = percentile(latencies, 0.5)
        p95 = percentile(latencies, 0.95)

        breaker_state = (
            self.BreakerState.OPEN if score < self.min_score else
            self.BreakerState.HALF_OPEN if score < self.half_open_score else
            self.BreakerState.CLOSED
        )
        status = "healthy" if breaker_state == self.BreakerState.CLOSED else "unhealthy"

        return self.ProviderHealthSnapshot(
            status=status,
            score=score,
            breaker_state=breaker_state,
            success_rate=w_success / (w_success + w_failure + 1e-9),
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            samples=len(data["successes"]) + len(data["failures"]),
        )

    def _ensure_provider(self, provider: str):
        if provider not in self._providers:
            self._providers[provider] = {
                "successes": [],
                "failures": [],
                "circuit_open": False,
                "circuit_open_until": None,
                "last_success": None,
                "last_failure": None,
            }

    def _cleanup_old_records(self, provider: str):
        if provider not in self._providers:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.ttl_sec)
        p = self._providers[provider]
        p["successes"] = [(t, lat) for t, lat in p["successes"] if t > cutoff]
        p["failures"] = [(t, err) for t, err in p["failures"] if t > cutoff]

    def _update_circuit(self, provider: str):
        if provider not in self._providers:
            return
        snap = self.snapshot(provider)
        p = self._providers[provider]
        if snap.breaker_state == self.BreakerState.OPEN:
            p["circuit_open"] = True
            p["circuit_open_until"] = datetime.now(timezone.utc) + timedelta(seconds=self.cooldown_sec)
        elif snap.breaker_state == self.BreakerState.CLOSED:
            p["circuit_open"] = False


# Re-export for legacy imports
BreakerState = ProviderHealthTracker.BreakerState
