"""
Astrai Router — Open-source intelligent LLM router.

Combines Thompson Sampling, Berkeley ARBITRAGE, trading-style best execution,
energy-aware routing, and privacy-preserving intelligence.

Quick start:
    from astrai_router import RoutingEngine, TaskClassifier, EnergyOracle

    classifier = TaskClassifier()
    result = classifier.classify("Write a Python function to sort a list")

    engine = RoutingEngine()
    recommendation = engine.route("coding", available_venues=["openai", "anthropic", "groq"])
"""

__version__ = "0.1.0"

# ── Storage (configure before using stateful components) ──────────────────
from .storage import (
    StorageBackend,
    MemoryStorage,
    SQLiteStorage,
    configure_storage,
    get_storage,
)

# ── Classification ────────────────────────────────────────────────────────
from .classifier import (
    TaskClassifier,
    TaskClassification,
    RoutingHealthClassifier,
    RoutingHealthStatus,
    HealthReport,
    ProviderHealthTracker,
    BreakerState,
)

# ── Core Routing Engine ───────────────────────────────────────────────────
from .engine import (
    RoutingEngine,
    RoutingTier,
    ThompsonPrior,
)

# ── Energy-Aware Routing ─────────────────────────────────────────────────
from .energy import (
    EnergyOracle,
    EnergyEstimate,
    EfficiencyGain,
    ENERGY_ORACLE,
    estimate_request_energy,
    get_energy_savings,
)

# ── Context Compression ──────────────────────────────────────────────────
from .compression import compress_messages

# ── Validation ────────────────────────────────────────────────────────────
from .validation import (
    ValidationError,
    validate_webhook_url,
    validate_model,
    validate_provider,
    validate_fallback_chain,
    RateLimiter,
)

# ── Intelligence (privacy-preserving) ────────────────────────────────────
from .intelligence import (
    RoutingIntelligenceStore,
    RoutingIntelligenceManager,
    ROUTING_INTELLIGENCE,
    record_routing_outcome,
    get_routing_context,
)

# ── Auto-Learning ────────────────────────────────────────────────────────
from .learning import (
    AutoLearningEngine,
    TaskType,
    detect_task_type,
    auto_route_request,
    record_request_outcome,
    get_default_eval_contract,
    AUTO_LEARNING_ENGINE,
)

# ── Semantic Cache ───────────────────────────────────────────────────────
from .cache import (
    SemanticCache,
    SEMANTIC_CACHE,
    get_cached_response,
    cache_response,
    get_cache_stats,
)

# ── Model Configuration ──────────────────────────────────────────────────
from .models import (
    MODEL_PROVIDERS,
    MODEL_CONFIG,
    classify_task,
    get_model_pair,
    get_model_cost,
    calculate_savings,
    score_response,
)

# ── Telemetry ────────────────────────────────────────────────────────────
from .telemetry import (
    RoutingTelemetryRecord,
    store_routing_telemetry,
    query_routing_telemetry,
)

# ── Lazy imports for heavier modules (avoid import-time overhead) ────────


def get_arbitrage_router():
    """Get the ARBITRAGE router (lazy import)."""
    from .arbitrage import ArbitrageRouter

    return ArbitrageRouter


def get_unified_router():
    """Get the unified router (lazy import)."""
    from .unified import route_request

    return route_request


def get_shadow_engine():
    """Get the shadow mode engine (lazy import)."""
    from .shadow import ShadowModeEngine

    return ShadowModeEngine


def get_production_router():
    """Get the production router (lazy import)."""
    from .production import ProductionRouter

    return ProductionRouter


__all__ = [
    # Version
    "__version__",
    # Storage
    "StorageBackend",
    "MemoryStorage",
    "SQLiteStorage",
    "configure_storage",
    "get_storage",
    # Classification
    "TaskClassifier",
    "TaskClassification",
    "RoutingHealthClassifier",
    "RoutingHealthStatus",
    "HealthReport",
    "ProviderHealthTracker",
    "BreakerState",
    # Engine
    "RoutingEngine",
    "RoutingTier",
    "ThompsonPrior",
    # Energy
    "EnergyOracle",
    "EnergyEstimate",
    "EfficiencyGain",
    "ENERGY_ORACLE",
    "estimate_request_energy",
    "get_energy_savings",
    # Compression
    "compress_messages",
    # Validation
    "ValidationError",
    "validate_webhook_url",
    "validate_model",
    "validate_provider",
    "validate_fallback_chain",
    "RateLimiter",
    # Intelligence
    "RoutingIntelligenceStore",
    "RoutingIntelligenceManager",
    "ROUTING_INTELLIGENCE",
    "record_routing_outcome",
    "get_routing_context",
    # Learning
    "AutoLearningEngine",
    "TaskType",
    "detect_task_type",
    "auto_route_request",
    "record_request_outcome",
    "get_default_eval_contract",
    "AUTO_LEARNING_ENGINE",
    # Cache
    "SemanticCache",
    "SEMANTIC_CACHE",
    "get_cached_response",
    "cache_response",
    "get_cache_stats",
    # Models
    "MODEL_PROVIDERS",
    "MODEL_CONFIG",
    "classify_task",
    "get_model_pair",
    "get_model_cost",
    "calculate_savings",
    "score_response",
    # Telemetry
    "RoutingTelemetryRecord",
    "store_routing_telemetry",
    "query_routing_telemetry",
    # Lazy accessors
    "get_arbitrage_router",
    "get_unified_router",
    "get_shadow_engine",
    "get_production_router",
]
