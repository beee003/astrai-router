"""
Routing Intelligence Memory for Astrai
Stores PATTERNS, not CONTENT. Privacy-preserving by design.

What we store:
- Which models work best for which task types
- User routing preferences (latency, cost, quality)
- Provider success rates per user
- Workflow step patterns

What we DON'T store:
- Prompts or responses
- User content of any kind
- Conversation history
"""

import sqlite3
import json
import time
import hashlib
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# ============================================================================
# DATA STRUCTURES (No content, only patterns)
# ============================================================================

@dataclass
class ModelPerformance:
    """Performance stats for a model on a task type."""
    model: str
    task_type: str
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0
    total_cost_usd: float = 0
    avg_quality_score: float = 0
    sample_count: int = 0
    last_used: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.sample_count if self.sample_count > 0 else 0

    @property
    def avg_cost_usd(self) -> float:
        return self.total_cost_usd / self.sample_count if self.sample_count > 0 else 0


@dataclass
class UserRoutingProfile:
    """Aggregated routing preferences for a user."""
    user_id: str
    # Inferred preferences (from behavior)
    preferred_latency_tier: str = "balanced"  # fastest, balanced, relaxed
    preferred_cost_tier: str = "balanced"     # cheapest, balanced, premium
    preferred_quality_tier: str = "target"    # draft, target, ultra
    # Behavioral patterns
    total_requests: int = 0
    avg_latency_tolerance_ms: float = 500
    escalation_rate: float = 0.1  # How often quality gates fail
    # Task distribution
    task_distribution: Dict[str, float] = field(default_factory=dict)  # {task: percentage}
    # Provider affinities (learned)
    provider_success_rates: Dict[str, float] = field(default_factory=dict)


@dataclass
class WorkflowPattern:
    """Pattern for a workflow step (no content, just structure)."""
    workflow_id: str
    step_id: str
    task_type: str
    # Best performing configs for this step
    best_model: Optional[str] = None
    best_provider: Optional[str] = None
    best_strategy: str = "balanced"
    # Performance stats
    avg_latency_ms: float = 0
    avg_quality_score: float = 0
    sample_count: int = 0


# ============================================================================
# ROUTING INTELLIGENCE STORE
# ============================================================================

class RoutingIntelligenceStore:
    """
    Stores routing patterns and preferences WITHOUT user content.

    This is the "Personal Market Brain" - learns what works for each user
    without ever seeing their actual prompts or responses.
    """

    def __init__(self, user_id: str, data_dir: str = "data/routing_intelligence"):
        self.user_id = user_id
        self.data_dir = Path(data_dir)
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            self.data_dir = Path("/tmp/astrai_data/routing_intelligence")
            self.data_dir.mkdir(parents=True, exist_ok=True)

        # Per-user database (only routing patterns)
        self.db_path = str(self.data_dir / f"routing_{user_id[:16]}.db")
        self._init_db()

    def _init_db(self):
        """Initialize routing intelligence tables."""
        conn = sqlite3.connect(self.db_path)

        # Model performance per task type
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                model TEXT NOT NULL,
                task_type TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                total_latency_ms REAL DEFAULT 0,
                total_cost_usd REAL DEFAULT 0,
                total_quality_score REAL DEFAULT 0,
                sample_count INTEGER DEFAULT 0,
                last_used REAL NOT NULL,
                PRIMARY KEY (model, task_type)
            )
        """)

        # Provider performance
        conn.execute("""
            CREATE TABLE IF NOT EXISTS provider_performance (
                provider TEXT NOT NULL,
                task_type TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                total_latency_ms REAL DEFAULT 0,
                sample_count INTEGER DEFAULT 0,
                last_used REAL NOT NULL,
                PRIMARY KEY (provider, task_type)
            )
        """)

        # User routing profile (aggregated preferences)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS routing_profile (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        # Workflow step patterns
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_patterns (
                workflow_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                task_type TEXT,
                best_model TEXT,
                best_provider TEXT,
                best_strategy TEXT DEFAULT 'balanced',
                avg_latency_ms REAL DEFAULT 0,
                avg_quality_score REAL DEFAULT 0,
                sample_count INTEGER DEFAULT 0,
                updated_at REAL NOT NULL,
                PRIMARY KEY (workflow_id, step_id)
            )
        """)

        # Task type distribution (what kinds of tasks this user does)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_distribution (
                task_type TEXT PRIMARY KEY,
                count INTEGER DEFAULT 0,
                last_seen REAL NOT NULL
            )
        """)

        # Indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model_perf_task ON model_performance(task_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_provider_perf_task ON provider_performance(task_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_workflow_patterns ON workflow_patterns(workflow_id)")

        conn.commit()
        conn.close()

    # ========================================================================
    # RECORD ROUTING OUTCOMES (Called after each request)
    # ========================================================================

    async def record_outcome(
        self,
        task_type: str,
        model: str,
        provider: str,
        success: bool,
        quality_score: float,
        latency_ms: float,
        cost_usd: float,
        workflow_id: Optional[str] = None,
        step_id: Optional[str] = None,
        strategy: str = "balanced",
    ):
        """
        Record a routing outcome. NO CONTENT STORED.

        Only records:
        - Which model/provider was used
        - Task type (code, analysis, etc.)
        - Success/failure
        - Performance metrics
        """
        now = time.time()
        conn = sqlite3.connect(self.db_path)

        # Update model performance
        conn.execute("""
            INSERT INTO model_performance
                (model, task_type, success_count, failure_count, total_latency_ms,
                 total_cost_usd, total_quality_score, sample_count, last_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(model, task_type) DO UPDATE SET
                success_count = success_count + ?,
                failure_count = failure_count + ?,
                total_latency_ms = total_latency_ms + ?,
                total_cost_usd = total_cost_usd + ?,
                total_quality_score = total_quality_score + ?,
                sample_count = sample_count + 1,
                last_used = ?
        """, (
            model, task_type,
            1 if success else 0,
            0 if success else 1,
            latency_ms, cost_usd, quality_score, now,
            # ON CONFLICT values
            1 if success else 0,
            0 if success else 1,
            latency_ms, cost_usd, quality_score, now
        ))

        # Update provider performance
        conn.execute("""
            INSERT INTO provider_performance
                (provider, task_type, success_count, failure_count, total_latency_ms, sample_count, last_used)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(provider, task_type) DO UPDATE SET
                success_count = success_count + ?,
                failure_count = failure_count + ?,
                total_latency_ms = total_latency_ms + ?,
                sample_count = sample_count + 1,
                last_used = ?
        """, (
            provider, task_type,
            1 if success else 0,
            0 if success else 1,
            latency_ms, now,
            1 if success else 0,
            0 if success else 1,
            latency_ms, now
        ))

        # Update task distribution
        conn.execute("""
            INSERT INTO task_distribution (task_type, count, last_seen)
            VALUES (?, 1, ?)
            ON CONFLICT(task_type) DO UPDATE SET
                count = count + 1,
                last_seen = ?
        """, (task_type, now, now))

        # Update workflow pattern if provided
        if workflow_id and step_id:
            await self._update_workflow_pattern(
                conn, workflow_id, step_id, task_type,
                model, provider, strategy,
                latency_ms, quality_score, now
            )

        conn.commit()
        conn.close()

    async def _update_workflow_pattern(
        self,
        conn: sqlite3.Connection,
        workflow_id: str,
        step_id: str,
        task_type: str,
        model: str,
        provider: str,
        strategy: str,
        latency_ms: float,
        quality_score: float,
        now: float,
    ):
        """Update workflow step pattern with new outcome."""
        # Get existing pattern
        cursor = conn.execute("""
            SELECT best_model, best_provider, avg_quality_score, sample_count
            FROM workflow_patterns
            WHERE workflow_id = ? AND step_id = ?
        """, (workflow_id, step_id))
        row = cursor.fetchone()

        if row:
            old_best_model, old_best_provider, old_avg_quality, old_count = row
            new_count = old_count + 1
            new_avg_quality = (old_avg_quality * old_count + quality_score) / new_count

            # Update best model if this one performed better
            # (Simple heuristic: use most recent high-quality result)
            new_best_model = model if quality_score > old_avg_quality else old_best_model
            new_best_provider = provider if quality_score > old_avg_quality else old_best_provider

            conn.execute("""
                UPDATE workflow_patterns SET
                    task_type = ?,
                    best_model = ?,
                    best_provider = ?,
                    best_strategy = ?,
                    avg_latency_ms = (avg_latency_ms * ? + ?) / ?,
                    avg_quality_score = ?,
                    sample_count = ?,
                    updated_at = ?
                WHERE workflow_id = ? AND step_id = ?
            """, (
                task_type, new_best_model, new_best_provider, strategy,
                old_count, latency_ms, new_count,
                new_avg_quality, new_count, now,
                workflow_id, step_id
            ))
        else:
            # Insert new pattern
            conn.execute("""
                INSERT INTO workflow_patterns
                    (workflow_id, step_id, task_type, best_model, best_provider,
                     best_strategy, avg_latency_ms, avg_quality_score, sample_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """, (
                workflow_id, step_id, task_type, model, provider,
                strategy, latency_ms, quality_score, now
            ))

    # ========================================================================
    # QUERY ROUTING INTELLIGENCE
    # ========================================================================

    async def get_best_model_for_task(
        self,
        task_type: str,
        min_samples: int = 5,
    ) -> Optional[Tuple[str, float]]:
        """
        Get the best performing model for a task type.

        Returns: (model_name, success_rate) or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT model,
                   CAST(success_count AS REAL) / (success_count + failure_count) as success_rate,
                   sample_count
            FROM model_performance
            WHERE task_type = ? AND sample_count >= ?
            ORDER BY success_rate DESC, sample_count DESC
            LIMIT 1
        """, (task_type, min_samples))
        row = cursor.fetchone()
        conn.close()

        if row:
            return (row[0], row[1])
        return None

    async def get_best_provider_for_task(
        self,
        task_type: str,
        min_samples: int = 5,
    ) -> Optional[Tuple[str, float]]:
        """Get the best performing provider for a task type."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT provider,
                   CAST(success_count AS REAL) / (success_count + failure_count) as success_rate,
                   sample_count
            FROM provider_performance
            WHERE task_type = ? AND sample_count >= ?
            ORDER BY success_rate DESC, sample_count DESC
            LIMIT 1
        """, (task_type, min_samples))
        row = cursor.fetchone()
        conn.close()

        if row:
            return (row[0], row[1])
        return None

    async def get_workflow_recommendation(
        self,
        workflow_id: str,
        step_id: str,
    ) -> Optional[WorkflowPattern]:
        """Get routing recommendation for a workflow step."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT workflow_id, step_id, task_type, best_model, best_provider,
                   best_strategy, avg_latency_ms, avg_quality_score, sample_count
            FROM workflow_patterns
            WHERE workflow_id = ? AND step_id = ?
        """, (workflow_id, step_id))
        row = cursor.fetchone()
        conn.close()

        if row:
            return WorkflowPattern(
                workflow_id=row[0],
                step_id=row[1],
                task_type=row[2],
                best_model=row[3],
                best_provider=row[4],
                best_strategy=row[5],
                avg_latency_ms=row[6],
                avg_quality_score=row[7],
                sample_count=row[8],
            )
        return None

    async def get_model_performance_matrix(
        self,
        min_samples: int = 3,
    ) -> Dict[str, Dict[str, ModelPerformance]]:
        """
        Get full performance matrix: {task_type: {model: performance}}.

        This is what powers the "which model for which task" intelligence.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT model, task_type, success_count, failure_count,
                   total_latency_ms, total_cost_usd, total_quality_score,
                   sample_count, last_used
            FROM model_performance
            WHERE sample_count >= ?
            ORDER BY task_type, success_count DESC
        """, (min_samples,))
        rows = cursor.fetchall()
        conn.close()

        matrix: Dict[str, Dict[str, ModelPerformance]] = {}
        for row in rows:
            task_type = row[1]
            if task_type not in matrix:
                matrix[task_type] = {}

            matrix[task_type][row[0]] = ModelPerformance(
                model=row[0],
                task_type=row[1],
                success_count=row[2],
                failure_count=row[3],
                total_latency_ms=row[4],
                total_cost_usd=row[5],
                avg_quality_score=row[6] / row[7] if row[7] > 0 else 0,
                sample_count=row[7],
                last_used=row[8],
            )

        return matrix

    async def get_user_profile(self) -> UserRoutingProfile:
        """
        Get aggregated user routing profile.

        Infers preferences from behavior patterns.
        """
        conn = sqlite3.connect(self.db_path)

        # Get task distribution
        cursor = conn.execute("""
            SELECT task_type, count FROM task_distribution
            ORDER BY count DESC
        """)
        task_rows = cursor.fetchall()
        total_tasks = sum(row[1] for row in task_rows)
        task_distribution = {
            row[0]: row[1] / total_tasks if total_tasks > 0 else 0
            for row in task_rows
        }

        # Get provider success rates
        cursor = conn.execute("""
            SELECT provider,
                   CAST(SUM(success_count) AS REAL) / SUM(success_count + failure_count) as success_rate
            FROM provider_performance
            GROUP BY provider
            HAVING SUM(success_count + failure_count) >= 5
        """)
        provider_success = {row[0]: row[1] for row in cursor.fetchall()}

        # Get average latency tolerance (from successful requests)
        cursor = conn.execute("""
            SELECT AVG(total_latency_ms / sample_count)
            FROM model_performance
            WHERE sample_count > 0
        """)
        avg_latency = cursor.fetchone()[0] or 500

        # Get escalation rate (failure rate across all)
        cursor = conn.execute("""
            SELECT
                CAST(SUM(failure_count) AS REAL) / SUM(success_count + failure_count)
            FROM model_performance
        """)
        escalation_rate = cursor.fetchone()[0] or 0.1

        conn.close()

        # Infer preferences from behavior
        preferred_latency = "fastest" if avg_latency < 200 else ("relaxed" if avg_latency > 1000 else "balanced")
        preferred_quality = "ultra" if escalation_rate < 0.05 else ("draft" if escalation_rate > 0.2 else "target")

        return UserRoutingProfile(
            user_id=self.user_id,
            preferred_latency_tier=preferred_latency,
            preferred_quality_tier=preferred_quality,
            total_requests=total_tasks,
            avg_latency_tolerance_ms=avg_latency,
            escalation_rate=escalation_rate,
            task_distribution=task_distribution,
            provider_success_rates=provider_success,
        )

    async def get_routing_context(
        self,
        task_type: str,
        workflow_id: Optional[str] = None,
        step_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get routing context for a request.

        Returns intelligence that can inform routing decisions,
        WITHOUT any user content.
        """
        context = {
            "task_type": task_type,
            "has_history": False,
        }

        # Best model for this task
        best_model = await self.get_best_model_for_task(task_type)
        if best_model:
            context["recommended_model"] = best_model[0]
            context["model_success_rate"] = best_model[1]
            context["has_history"] = True

        # Best provider for this task
        best_provider = await self.get_best_provider_for_task(task_type)
        if best_provider:
            context["recommended_provider"] = best_provider[0]
            context["provider_success_rate"] = best_provider[1]

        # Workflow-specific recommendation
        if workflow_id and step_id:
            pattern = await self.get_workflow_recommendation(workflow_id, step_id)
            if pattern and pattern.sample_count >= 3:
                context["workflow_recommendation"] = {
                    "model": pattern.best_model,
                    "provider": pattern.best_provider,
                    "strategy": pattern.best_strategy,
                    "avg_quality": pattern.avg_quality_score,
                    "confidence": min(1.0, pattern.sample_count / 20),
                    "sample_count": pattern.sample_count,
                }

        # User profile summary
        profile = await self.get_user_profile()
        context["user_profile"] = {
            "preferred_latency": profile.preferred_latency_tier,
            "preferred_quality": profile.preferred_quality_tier,
            "escalation_rate": profile.escalation_rate,
            "top_task_types": list(profile.task_distribution.keys())[:3],
        }

        return context

    # ========================================================================
    # EXPORT (For user transparency)
    # ========================================================================

    async def export_intelligence(self) -> Dict[str, Any]:
        """
        Export all routing intelligence for this user.

        This is what users see - their routing patterns, not content.
        """
        matrix = await self.get_model_performance_matrix(min_samples=1)
        profile = await self.get_user_profile()

        # Get workflow patterns
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT workflow_id, step_id, task_type, best_model,
                   best_provider, avg_quality_score, sample_count
            FROM workflow_patterns
            WHERE sample_count >= 1
            ORDER BY sample_count DESC
            LIMIT 50
        """)
        workflow_patterns = [
            {
                "workflow": row[0],
                "step": row[1],
                "task_type": row[2],
                "best_model": row[3],
                "best_provider": row[4],
                "avg_quality": row[5],
                "samples": row[6],
            }
            for row in cursor.fetchall()
        ]
        conn.close()

        return {
            "user_id": self.user_id[:8] + "...",
            "exported_at": datetime.now().isoformat(),
            "profile": {
                "preferred_latency": profile.preferred_latency_tier,
                "preferred_quality": profile.preferred_quality_tier,
                "escalation_rate": round(profile.escalation_rate, 3),
                "total_requests": profile.total_requests,
                "task_distribution": profile.task_distribution,
                "provider_success_rates": profile.provider_success_rates,
            },
            "model_performance": {
                task_type: {
                    model: {
                        "success_rate": round(perf.success_rate, 3),
                        "avg_latency_ms": round(perf.avg_latency_ms, 1),
                        "avg_cost_usd": round(perf.avg_cost_usd, 6),
                        "samples": perf.sample_count,
                    }
                    for model, perf in models.items()
                }
                for task_type, models in matrix.items()
            },
            "workflow_patterns": workflow_patterns,
        }


# ============================================================================
# GLOBAL MANAGER
# ============================================================================

class RoutingIntelligenceManager:
    """Global manager for per-user routing intelligence."""

    def __init__(self, data_dir: str = "data/routing_intelligence"):
        self.data_dir = data_dir
        self._stores: Dict[str, RoutingIntelligenceStore] = {}

    def get_store(self, user_id: str) -> RoutingIntelligenceStore:
        """Get or create store for a user."""
        if user_id not in self._stores:
            self._stores[user_id] = RoutingIntelligenceStore(user_id, self.data_dir)
        return self._stores[user_id]

    async def record_outcome(
        self,
        user_id: str,
        task_type: str,
        model: str,
        provider: str,
        success: bool,
        quality_score: float,
        latency_ms: float,
        cost_usd: float,
        workflow_id: Optional[str] = None,
        step_id: Optional[str] = None,
        strategy: str = "balanced",
    ):
        """Record routing outcome for a user."""
        store = self.get_store(user_id)
        await store.record_outcome(
            task_type=task_type,
            model=model,
            provider=provider,
            success=success,
            quality_score=quality_score,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            workflow_id=workflow_id,
            step_id=step_id,
            strategy=strategy,
        )

    async def get_routing_context(
        self,
        user_id: str,
        task_type: str,
        workflow_id: Optional[str] = None,
        step_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get routing context for a request."""
        store = self.get_store(user_id)
        return await store.get_routing_context(task_type, workflow_id, step_id)


# Global instance
ROUTING_INTELLIGENCE = RoutingIntelligenceManager()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def record_routing_outcome(
    user_id: str,
    task_type: str,
    model: str,
    provider: str,
    success: bool,
    quality_score: float,
    latency_ms: float,
    cost_usd: float,
    workflow_id: Optional[str] = None,
    step_id: Optional[str] = None,
):
    """Record a routing outcome (called after each request)."""
    await ROUTING_INTELLIGENCE.record_outcome(
        user_id=user_id,
        task_type=task_type,
        model=model,
        provider=provider,
        success=success,
        quality_score=quality_score,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        workflow_id=workflow_id,
        step_id=step_id,
    )


async def get_routing_context(
    user_id: str,
    task_type: str,
    workflow_id: Optional[str] = None,
    step_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Get routing intelligence for a request."""
    return await ROUTING_INTELLIGENCE.get_routing_context(
        user_id=user_id,
        task_type=task_type,
        workflow_id=workflow_id,
        step_id=step_id,
    )


async def export_user_intelligence(user_id: str) -> Dict[str, Any]:
    """Export all routing intelligence for a user."""
    store = ROUTING_INTELLIGENCE.get_store(user_id)
    return await store.export_intelligence()
