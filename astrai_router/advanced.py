"""
Advanced Routing Features (Production-Grade)

Premium features for enterprise LLM routing:
1. User-defined fallback chains
2. Webhook notifications
3. Model health tracking
4. Smart routing recommendations
5. Per-request model overrides
"""

import asyncio
import hashlib
import hmac
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import httpx

from .storage import get_storage

# Import validation utilities
try:
    from .validation import (
        validate_webhook_url,
        validate_webhook_events,
        validate_webhook_secret,
        validate_webhook_name,
        validate_fallback_chain,
        validate_chain_name,
        validate_chain_settings,
        check_webhook_rate_limit,
        ValidationError,
    )

    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================


class ModelStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


class WebhookEvent(str, Enum):
    BUDGET_EXCEEDED = "budget_exceeded"
    BUDGET_WARNING = "budget_warning"
    MODEL_DOWN = "model_down"
    MODEL_RECOVERED = "model_recovered"
    QUALITY_ALERT = "quality_alert"
    LATENCY_SPIKE = "latency_spike"
    RATE_LIMIT_HIT = "rate_limit_hit"
    FALLBACK_TRIGGERED = "fallback_triggered"
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_REPORT = "weekly_report"


@dataclass
class FallbackChainStep:
    """A single step in a fallback chain."""

    model: str
    provider: str
    max_latency_ms: int = 5000
    max_cost_usd: float = None


@dataclass
class FallbackChain:
    """User-defined fallback chain configuration."""

    chain_id: str
    steps: List[FallbackChainStep]
    max_retries: int = 3
    total_timeout_ms: int = 30000
    trigger_on_timeout: bool = True
    trigger_on_rate_limit: bool = True
    trigger_on_error: bool = True
    trigger_on_quality_fail: bool = False
    min_quality_score: float = 0.7


@dataclass
class ModelHealth:
    """Real-time health status for a model."""

    model_id: str
    provider_id: str
    status: ModelStatus
    is_available: bool
    success_rate: float
    avg_latency_ms: int
    p95_latency_ms: int
    error_rate: float
    last_error: Optional[str]
    last_checked_at: datetime


@dataclass
class SmartRecommendation:
    """AI-generated routing recommendation."""

    id: str
    recommendation_type: str
    title: str
    description: str
    estimated_cost_change_pct: Optional[float]
    estimated_quality_change_pct: Optional[float]
    estimated_latency_change_pct: Optional[float]
    confidence: float
    action_payload: Optional[Dict[str, Any]]
    status: str
    valid_until: datetime


# =============================================================================
# FALLBACK CHAIN EXECUTOR
# =============================================================================


class FallbackChainExecutor:
    """
    Executes user-defined fallback chains.

    When the primary model fails (timeout, rate limit, error, or quality),
    automatically tries the next model in the chain.
    """

    def __init__(self):
        self._chain_cache: Dict[str, FallbackChain] = {}
        self._cache_ttl = 60  # 1 minute

    async def get_chain(
        self, user_id: str, task_type: Optional[str] = None
    ) -> Optional[FallbackChain]:
        """Get user's fallback chain for a task type."""
        storage = get_storage()

        try:
            # Query fallback_chains table for this user/task_type
            filters = {"user_id": user_id}
            if task_type:
                filters["task_type"] = task_type

            rows = storage.get(
                "fallback_chains",
                filters=filters,
                limit=1,
            )

            if not rows:
                return None

            row = rows[0]
            chain_data = row.get("chain", [])
            if isinstance(chain_data, str):
                chain_data = json.loads(chain_data)

            steps = [
                FallbackChainStep(
                    model=s["model"],
                    provider=s["provider"],
                    max_latency_ms=s.get("max_latency_ms", 5000),
                    max_cost_usd=s.get("max_cost_usd"),
                )
                for s in chain_data
            ]

            return FallbackChain(
                chain_id=row["chain_id"],
                steps=steps,
                max_retries=row.get("max_retries", 3),
                total_timeout_ms=row.get("total_timeout_ms", 30000),
                trigger_on_timeout=row.get("trigger_on_timeout", True),
                trigger_on_rate_limit=row.get("trigger_on_rate_limit", True),
                trigger_on_error=row.get("trigger_on_error", True),
                trigger_on_quality_fail=row.get("trigger_on_quality_fail", False),
                min_quality_score=float(row.get("min_quality_score") or 0.7),
            )

        except Exception as e:
            print(f"Warning: Error fetching fallback chain: {e}")
            return None

    def should_trigger(
        self,
        chain: FallbackChain,
        error_type: str,
        quality_score: Optional[float] = None,
    ) -> bool:
        """Check if fallback should be triggered based on error type."""
        if error_type == "timeout" and chain.trigger_on_timeout:
            return True
        if error_type == "rate_limit" and chain.trigger_on_rate_limit:
            return True
        if error_type == "error" and chain.trigger_on_error:
            return True
        if error_type == "quality" and chain.trigger_on_quality_fail:
            if quality_score is not None and quality_score < chain.min_quality_score:
                return True
        return False

    async def record_trigger(self, chain_id: str, succeeded: bool):
        """Record that a fallback chain was triggered."""
        storage = get_storage()

        try:
            storage.insert(
                "fallback_triggers",
                {
                    "chain_id": chain_id,
                    "succeeded": succeeded,
                    "triggered_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as e:
            print(f"Warning: Error recording fallback trigger: {e}")


# =============================================================================
# WEBHOOK MANAGER
# =============================================================================


class WebhookManager:
    """
    Manages webhook notifications.

    Sends alerts to user-configured endpoints when events occur.
    """

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=10.0)

    async def get_webhooks(
        self, user_id: str, event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get user's active webhooks, optionally filtered by event type."""
        storage = get_storage()

        try:
            rows = storage.get(
                "user_webhooks",
                filters={"user_id": user_id, "is_active": True},
            )

            webhooks = []
            for row in rows:
                events = row.get("events", [])
                if isinstance(events, str):
                    events = json.loads(events)
                if event_type is None or event_type in events:
                    webhooks.append(row)

            return webhooks

        except Exception as e:
            print(f"Warning: Error fetching webhooks: {e}")
            return []

    async def send_webhook(
        self, webhook: Dict[str, Any], event_type: str, payload: Dict[str, Any]
    ) -> bool:
        """Send a webhook notification."""
        url = webhook.get("url")
        secret = webhook.get("secret")
        webhook_id = webhook.get("id", "unknown")

        if not url:
            return False

        # Validate URL (security check)
        if VALIDATION_AVAILABLE:
            is_valid, error = validate_webhook_url(url)
            if not is_valid:
                print(f"Warning: Webhook URL validation failed: {error}")
                return False

            # Check rate limit
            is_allowed, error = check_webhook_rate_limit(webhook_id)
            if not is_allowed:
                print(f"Warning: Webhook rate limited: {error}")
                return False

        # Build payload
        full_payload = {
            "event": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": payload,
        }
        payload_json = json.dumps(full_payload)

        # Build headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Astrai-Webhook/1.0",
            "X-Astrai-Event": event_type,
        }

        # Add HMAC signature if secret is configured
        if secret:
            signature = hmac.new(
                secret.encode(), payload_json.encode(), hashlib.sha256
            ).hexdigest()
            headers["X-Astrai-Signature"] = f"sha256={signature}"

        try:
            start_time = time.monotonic()
            response = await self._client.post(
                url, content=payload_json, headers=headers
            )
            response_time_ms = int((time.monotonic() - start_time) * 1000)

            success = 200 <= response.status_code < 300

            # Log delivery
            await self._log_delivery(
                webhook_id=webhook["id"],
                user_id=webhook["user_id"],
                event_type=event_type,
                payload=full_payload,
                status_code=response.status_code,
                response_body=response.text[:1000] if response.text else None,
                success=success,
                response_time_ms=response_time_ms,
            )

            return success

        except Exception as e:
            print(f"Warning: Webhook delivery failed: {e}")
            await self._log_delivery(
                webhook_id=webhook["id"],
                user_id=webhook["user_id"],
                event_type=event_type,
                payload=full_payload,
                status_code=None,
                response_body=str(e),
                success=False,
                response_time_ms=None,
            )
            return False

    async def _log_delivery(
        self,
        webhook_id: str,
        user_id: str,
        event_type: str,
        payload: Dict[str, Any],
        status_code: Optional[int],
        response_body: Optional[str],
        success: bool,
        response_time_ms: Optional[int],
    ):
        """Log webhook delivery attempt."""
        storage = get_storage()

        try:
            # Insert delivery log
            storage.insert(
                "webhook_deliveries",
                {
                    "webhook_id": webhook_id,
                    "user_id": user_id,
                    "event_type": event_type,
                    "payload": payload,
                    "status_code": status_code,
                    "response_body": response_body,
                    "success": success,
                    "response_time_ms": response_time_ms,
                },
            )

            # Update webhook stats: fetch current stats, increment, and update
            current = storage.get(
                "user_webhooks",
                filters={"id": webhook_id},
                limit=1,
            )
            if current:
                row = current[0]
                storage.update(
                    "user_webhooks",
                    {
                        "total_sent": (row.get("total_sent", 0) or 0) + 1,
                        "total_failed": (row.get("total_failed", 0) or 0)
                        + (0 if success else 1),
                        "last_sent_at": datetime.now(timezone.utc).isoformat(),
                        "last_error": None if success else response_body,
                    },
                    filters={"id": webhook_id},
                )

        except Exception as e:
            print(f"Warning: Error logging webhook delivery: {e}")

    async def notify(self, user_id: str, event_type: str, payload: Dict[str, Any]):
        """Send notification to all user webhooks subscribed to this event."""
        webhooks = await self.get_webhooks(user_id, event_type)

        for webhook in webhooks:
            # Fire and forget - don't block on webhook delivery
            asyncio.create_task(self.send_webhook(webhook, event_type, payload))


# =============================================================================
# MODEL HEALTH TRACKER
# =============================================================================


class ModelHealthTracker:
    """
    Tracks real-time health of all models.

    Monitors success rates, latency, and errors to determine
    which models are healthy, degraded, or down.
    """

    def __init__(self):
        self._health_cache: Dict[str, ModelHealth] = {}
        self._cache_ttl = 30  # 30 seconds

    async def record_request(
        self,
        model_id: str,
        provider_id: str,
        success: bool,
        latency_ms: int,
        error: Optional[str] = None,
    ):
        """Record a request outcome for health tracking."""
        storage = get_storage()

        try:
            # Upsert model health status
            existing = storage.get(
                "model_health_status",
                filters={"model_id": model_id, "provider_id": provider_id},
                limit=1,
            )

            now = datetime.now(timezone.utc).isoformat()

            if existing:
                row = existing[0]
                old_success_rate = float(row.get("success_rate", 1.0) or 1.0)
                old_avg_latency = int(row.get("avg_latency_ms", 100) or 100)
                old_error_rate = float(row.get("error_rate", 0.0) or 0.0)

                new_success_rate = old_success_rate * 0.9 + (0.1 if success else 0.0)
                new_avg_latency = int(old_avg_latency * 0.9 + latency_ms * 0.1)
                new_error_rate = old_error_rate * 0.9 + (0.0 if success else 0.1)

                # Determine status
                if new_success_rate < 0.5:
                    status = ModelStatus.DOWN.value
                    is_available = False
                elif new_success_rate < 0.9:
                    status = ModelStatus.DEGRADED.value
                    is_available = True
                else:
                    status = ModelStatus.HEALTHY.value
                    is_available = True

                storage.update(
                    "model_health_status",
                    {
                        "success_rate": new_success_rate,
                        "avg_latency_ms": new_avg_latency,
                        "error_rate": new_error_rate,
                        "status": status,
                        "is_available": is_available,
                        "last_error": error if not success else row.get("last_error"),
                        "last_checked_at": now,
                    },
                    filters={"model_id": model_id, "provider_id": provider_id},
                )
            else:
                # First record for this model
                status = (
                    ModelStatus.HEALTHY.value if success else ModelStatus.DEGRADED.value
                )
                storage.insert(
                    "model_health_status",
                    {
                        "model_id": model_id,
                        "provider_id": provider_id,
                        "success_rate": 1.0 if success else 0.0,
                        "avg_latency_ms": latency_ms,
                        "p95_latency_ms": latency_ms,
                        "error_rate": 0.0 if success else 1.0,
                        "status": status,
                        "is_available": True,
                        "last_error": error,
                        "last_checked_at": now,
                    },
                )

            # Update cache
            cache_key = f"{model_id}:{provider_id}"
            if cache_key in self._health_cache:
                health = self._health_cache[cache_key]
                # Quick in-memory update
                if success:
                    health.success_rate = health.success_rate * 0.9 + 0.1
                    health.avg_latency_ms = int(
                        health.avg_latency_ms * 0.9 + latency_ms * 0.1
                    )
                else:
                    health.success_rate = health.success_rate * 0.9
                    health.last_error = error

                # Update status
                if health.success_rate < 0.5:
                    health.status = ModelStatus.DOWN
                    health.is_available = False
                elif health.success_rate < 0.9:
                    health.status = ModelStatus.DEGRADED
                    health.is_available = True
                else:
                    health.status = ModelStatus.HEALTHY
                    health.is_available = True

        except Exception as e:
            print(f"Warning: Error recording model health: {e}")

    async def get_health(
        self, model_id: str, provider_id: str
    ) -> Optional[ModelHealth]:
        """Get health status for a specific model."""
        cache_key = f"{model_id}:{provider_id}"

        # Check cache first
        if cache_key in self._health_cache:
            return self._health_cache[cache_key]

        storage = get_storage()

        try:
            rows = storage.get(
                "model_health_status",
                filters={"model_id": model_id, "provider_id": provider_id},
                limit=1,
            )

            if not rows:
                return None

            row = rows[0]
            health = ModelHealth(
                model_id=row["model_id"],
                provider_id=row["provider_id"],
                status=ModelStatus(row["status"]),
                is_available=row["is_available"],
                success_rate=float(row["success_rate"] or 1.0),
                avg_latency_ms=int(row["avg_latency_ms"] or 100),
                p95_latency_ms=int(row["p95_latency_ms"] or 200),
                error_rate=float(row["error_rate"] or 0.0),
                last_error=row.get("last_error"),
                last_checked_at=datetime.fromisoformat(
                    row["last_checked_at"].replace("Z", "+00:00")
                )
                if row.get("last_checked_at")
                else datetime.now(timezone.utc),
            )

            self._health_cache[cache_key] = health
            return health

        except Exception as e:
            print(f"Warning: Error fetching model health: {e}")
            return None

    async def get_all_health(self) -> List[ModelHealth]:
        """Get health status for all models."""
        storage = get_storage()

        try:
            rows = storage.get(
                "model_health_status",
                order_by="status",
                desc=False,
            )

            health_list = []
            for row in rows:
                health_list.append(
                    ModelHealth(
                        model_id=row["model_id"],
                        provider_id=row["provider_id"],
                        status=ModelStatus(row["status"]),
                        is_available=row["is_available"],
                        success_rate=float(row["success_rate"] or 1.0),
                        avg_latency_ms=int(row["avg_latency_ms"] or 100),
                        p95_latency_ms=int(row["p95_latency_ms"] or 200),
                        error_rate=float(row["error_rate"] or 0.0),
                        last_error=row.get("last_error"),
                        last_checked_at=datetime.fromisoformat(
                            row["last_checked_at"].replace("Z", "+00:00")
                        )
                        if row.get("last_checked_at")
                        else datetime.now(timezone.utc),
                    )
                )

            return health_list

        except Exception as e:
            print(f"Warning: Error fetching all model health: {e}")
            return []

    def is_available(self, model_id: str, provider_id: str) -> bool:
        """Quick check if a model is available (from cache)."""
        cache_key = f"{model_id}:{provider_id}"
        if cache_key in self._health_cache:
            return self._health_cache[cache_key].is_available
        return True  # Assume available if not in cache


# =============================================================================
# SMART RECOMMENDATIONS
# =============================================================================


class SmartRecommendationEngine:
    """
    Generates AI-powered routing recommendations.

    Analyzes user's usage patterns and suggests optimizations.
    """

    async def get_recommendations(
        self, user_id: str, limit: int = 10
    ) -> List[SmartRecommendation]:
        """Get pending recommendations for a user."""
        storage = get_storage()

        try:
            now_iso = datetime.now(timezone.utc).isoformat()

            rows = storage.get(
                "smart_recommendations",
                filters={"user_id": user_id, "status": "pending"},
                order_by="created_at",
                desc=True,
                limit=limit,
            )

            recommendations = []
            for row in rows:
                # Filter out expired recommendations
                valid_until = row.get("valid_until", "")
                if valid_until and valid_until < now_iso:
                    continue

                recommendations.append(
                    SmartRecommendation(
                        id=row["id"],
                        recommendation_type=row["recommendation_type"],
                        title=row["title"],
                        description=row["description"],
                        estimated_cost_change_pct=float(
                            row["estimated_cost_change_pct"]
                        )
                        if row.get("estimated_cost_change_pct")
                        else None,
                        estimated_quality_change_pct=float(
                            row["estimated_quality_change_pct"]
                        )
                        if row.get("estimated_quality_change_pct")
                        else None,
                        estimated_latency_change_pct=float(
                            row["estimated_latency_change_pct"]
                        )
                        if row.get("estimated_latency_change_pct")
                        else None,
                        confidence=float(row.get("confidence") or 0.8),
                        action_payload=row.get("action_payload"),
                        status=row["status"],
                        valid_until=datetime.fromisoformat(
                            row["valid_until"].replace("Z", "+00:00")
                        )
                        if row.get("valid_until")
                        else datetime.now(timezone.utc),
                    )
                )

            return recommendations

        except Exception as e:
            print(f"Warning: Error fetching recommendations: {e}")
            return []

    async def accept_recommendation(self, recommendation_id: str) -> bool:
        """Accept a recommendation and apply its action."""
        storage = get_storage()

        try:
            storage.update(
                "smart_recommendations",
                {
                    "status": "accepted",
                    "accepted_at": datetime.now(timezone.utc).isoformat(),
                },
                filters={"id": recommendation_id},
            )
            return True

        except Exception as e:
            print(f"Warning: Error accepting recommendation: {e}")
            return False

    async def reject_recommendation(self, recommendation_id: str) -> bool:
        """Reject a recommendation."""
        storage = get_storage()

        try:
            storage.update(
                "smart_recommendations",
                {
                    "status": "rejected",
                    "rejected_at": datetime.now(timezone.utc).isoformat(),
                },
                filters={"id": recommendation_id},
            )
            return True

        except Exception as e:
            print(f"Warning: Error rejecting recommendation: {e}")
            return False

    async def generate_recommendations(self, user_id: str) -> List[str]:
        """
        Analyze user's usage and generate recommendations.

        Returns list of recommendation IDs created.
        """
        # This would typically run as a background job
        # For now, generate some basic recommendations based on usage patterns
        recommendations_created = []
        storage = get_storage()

        try:
            # Get user's recent routing history
            history = storage.get(
                "user_routing_history",
                filters={"user_id": user_id},
                order_by="created_at",
                desc=True,
                limit=100,
            )

            if not history or len(history) < 10:
                return recommendations_created

            # Analyze for cost optimization opportunities
            # (In production, this would be much more sophisticated)

            # Example: If user is using expensive models for simple tasks
            code_requests = [h for h in history if h.get("task_type") == "code"]
            if len(code_requests) > 5:
                expensive_code = [
                    h
                    for h in code_requests
                    if "gpt-5" in h.get("selected_model", "")
                    or "claude-4" in h.get("selected_model", "")
                ]
                if len(expensive_code) / len(code_requests) > 0.7:
                    # Suggest trying a cheaper model
                    rec = storage.insert(
                        "smart_recommendations",
                        {
                            "user_id": user_id,
                            "recommendation_type": "cost_optimization",
                            "title": "Try DeepSeek for Code Tasks",
                            "description": "You're using premium models for 70%+ of code tasks. DeepSeek V3.2 offers similar quality at 90% lower cost for most coding tasks.",
                            "estimated_cost_change_pct": -50.0,
                            "estimated_quality_change_pct": -5.0,
                            "estimated_latency_change_pct": None,
                            "confidence": 0.8,
                            "action_payload": json.dumps(
                                {
                                    "type": "suggest_model",
                                    "task_type": "code",
                                    "model": "deepseek-v3.2",
                                    "provider": "deepseek",
                                }
                            ),
                            "status": "pending",
                            "valid_until": (
                                datetime.now(timezone.utc) + timedelta(days=7)
                            ).isoformat(),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    if rec:
                        recommendations_created.append(rec.get("id", ""))

            return recommendations_created

        except Exception as e:
            print(f"Warning: Error generating recommendations: {e}")
            return recommendations_created


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

FALLBACK_EXECUTOR = FallbackChainExecutor()
WEBHOOK_MANAGER = WebhookManager()
MODEL_HEALTH_TRACKER = ModelHealthTracker()
RECOMMENDATION_ENGINE = SmartRecommendationEngine()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def get_fallback_chain(
    user_id: str, task_type: str = None
) -> Optional[FallbackChain]:
    """Get user's fallback chain for a task type."""
    return await FALLBACK_EXECUTOR.get_chain(user_id, task_type)


async def notify_user(user_id: str, event_type: str, payload: Dict[str, Any]):
    """Send webhook notification to user."""
    await WEBHOOK_MANAGER.notify(user_id, event_type, payload)


async def record_model_health(
    model_id: str,
    provider_id: str,
    success: bool,
    latency_ms: int,
    error: Optional[str] = None,
):
    """Record model health data."""
    await MODEL_HEALTH_TRACKER.record_request(
        model_id, provider_id, success, latency_ms, error
    )


async def get_model_health_status() -> List[Dict[str, Any]]:
    """Get health status for all models."""
    health_list = await MODEL_HEALTH_TRACKER.get_all_health()
    return [
        {
            "model_id": h.model_id,
            "provider_id": h.provider_id,
            "status": h.status.value,
            "is_available": h.is_available,
            "success_rate": h.success_rate,
            "avg_latency_ms": h.avg_latency_ms,
            "p95_latency_ms": h.p95_latency_ms,
            "error_rate": h.error_rate,
            "last_error": h.last_error,
        }
        for h in health_list
    ]


async def get_user_recommendations(user_id: str) -> List[Dict[str, Any]]:
    """Get recommendations for a user."""
    recommendations = await RECOMMENDATION_ENGINE.get_recommendations(user_id)
    return [
        {
            "id": r.id,
            "type": r.recommendation_type,
            "title": r.title,
            "description": r.description,
            "cost_change_pct": r.estimated_cost_change_pct,
            "quality_change_pct": r.estimated_quality_change_pct,
            "latency_change_pct": r.estimated_latency_change_pct,
            "confidence": r.confidence,
            "action": r.action_payload,
            "valid_until": r.valid_until.isoformat(),
        }
        for r in recommendations
    ]
