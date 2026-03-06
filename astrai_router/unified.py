# unified.py
# Astrai Unified Router - Multi-Tier Quality Prediction
#
# Instead of binary "draft vs target", this router predicts quality scores
# for ALL available models and picks the one with best utility-per-cost.
#
# The Moat: We're the only ones with cross-provider quality data.
# OpenAI doesn't know when Groq fails. Groq doesn't know when they match GPT-5.2.
# Only WE have the matrix of (prompt_features, model, actual_quality).

import os
import pickle
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .energy import ENERGY_ORACLE

try:
    from .learning import get_quality_learner
except ImportError:
    get_quality_learner = None

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model tiers based on capability and cost."""

    DRAFT_8B = "draft_8b"  # Fast, cheap, good for simple tasks
    DRAFT_70B = "draft_70b"  # Mid-tier, good balance
    TARGET_70B = "target_70b"  # High quality, moderate cost
    ULTRA_400B = "ultra_400b"  # Frontier quality, high cost


@dataclass
class ModelInfo:
    """Information about a model."""

    name: str
    tier: ModelTier
    provider: str
    cost_per_mtok: float
    avg_latency_ms: float
    energy_tier: str  # For energy oracle
    capabilities: List[str]  # code, reasoning, creative, etc.
    supports_tools: bool = False  # Function/tool calling support
    supports_vision: bool = False  # Vision/image support
    max_context: int = 128000  # Max context window
    agent_score: float = 0.5  # How good for agentic workflows (0-1)


# Complete model registry with ALL available providers
# Agent scores: 0.3=poor, 0.5=basic, 0.7=good, 0.9=excellent for agentic tasks
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    # ============ DRAFT TIER (8B) - Ultra Fast & Cheap ============
    "groq/llama-3.1-8b-instant": ModelInfo(
        name="groq/llama-3.1-8b-instant",
        tier=ModelTier.DRAFT_8B,
        provider="groq",
        cost_per_mtok=0.05,
        avg_latency_ms=80,
        energy_tier="draft_8b",
        capabilities=["general", "code", "writing"],
        supports_tools=True,
        supports_vision=False,
        max_context=131072,
        agent_score=0.4,  # Fast but limited reasoning for agents
    ),
    "groq/llama-4-maverick-17b": ModelInfo(
        name="groq/llama-4-maverick-17b",
        tier=ModelTier.DRAFT_8B,
        provider="groq",
        cost_per_mtok=0.20,
        avg_latency_ms=100,
        energy_tier="draft_8b",
        capabilities=["general", "code", "reasoning"],
        supports_tools=True,
        supports_vision=False,
        max_context=131072,
        agent_score=0.55,  # Better reasoning for agents
    ),
    "openai/gpt-4o-mini": ModelInfo(
        name="openai/gpt-4o-mini",
        tier=ModelTier.DRAFT_8B,
        provider="openai",
        cost_per_mtok=0.15,
        avg_latency_ms=200,
        energy_tier="draft_8b",
        capabilities=["general", "code", "writing", "reasoning"],
        supports_tools=True,
        supports_vision=True,
        max_context=128000,
        agent_score=0.7,  # Good tool calling, vision support
    ),
    "deepseek/deepseek-chat": ModelInfo(
        name="deepseek/deepseek-chat",
        tier=ModelTier.DRAFT_8B,
        provider="deepseek",
        cost_per_mtok=0.27,
        avg_latency_ms=150,
        energy_tier="draft_8b",
        capabilities=["general", "code", "reasoning"],
        supports_tools=True,
        supports_vision=False,
        max_context=64000,
        agent_score=0.5,
    ),
    "groq/mixtral-8x7b-32768": ModelInfo(
        name="groq/mixtral-8x7b-32768",
        tier=ModelTier.DRAFT_8B,
        provider="groq",
        cost_per_mtok=0.24,
        avg_latency_ms=120,
        energy_tier="draft_8b",
        capabilities=["general", "code", "writing"],
        supports_tools=True,
        supports_vision=False,
        max_context=32768,
        agent_score=0.45,
    ),
    # ============ DRAFT TIER (70B) - Mid-Tier Balance ============
    "groq/llama-3.3-70b-versatile": ModelInfo(
        name="groq/llama-3.3-70b-versatile",
        tier=ModelTier.DRAFT_70B,
        provider="groq",
        cost_per_mtok=0.59,
        avg_latency_ms=300,
        energy_tier="target_70b",
        capabilities=["general", "code", "writing", "reasoning", "analysis"],
        supports_tools=True,
        supports_vision=False,
        max_context=131072,
        agent_score=0.7,  # Good balance for agents
    ),
    "groq/llama-3.1-70b-versatile": ModelInfo(
        name="groq/llama-3.1-70b-versatile",
        tier=ModelTier.DRAFT_70B,
        provider="groq",
        cost_per_mtok=0.59,
        avg_latency_ms=300,
        energy_tier="target_70b",
        capabilities=["general", "code", "writing", "reasoning"],
        supports_tools=True,
        supports_vision=False,
        max_context=131072,
        agent_score=0.65,
    ),
    # ============ TARGET TIER (70B) - High Quality ============
    "openai/gpt-4o": ModelInfo(
        name="openai/gpt-4o",
        tier=ModelTier.TARGET_70B,
        provider="openai",
        cost_per_mtok=5.00,
        avg_latency_ms=800,
        energy_tier="target_70b",
        capabilities=["general", "code", "writing", "reasoning", "analysis", "vision"],
        supports_tools=True,
        supports_vision=True,
        max_context=128000,
        agent_score=0.85,  # Excellent for agents
    ),
    "anthropic/claude-3-5-sonnet-latest": ModelInfo(
        name="anthropic/claude-3-5-sonnet-latest",
        tier=ModelTier.TARGET_70B,
        provider="anthropic",
        cost_per_mtok=3.00,
        avg_latency_ms=700,
        energy_tier="target_70b",
        capabilities=["general", "code", "writing", "reasoning", "analysis"],
        supports_tools=True,
        supports_vision=True,
        max_context=200000,
        agent_score=0.9,  # Best for agentic coding tasks
    ),
    "openai/o1-mini": ModelInfo(
        name="openai/o1-mini",
        tier=ModelTier.TARGET_70B,
        provider="openai",
        cost_per_mtok=3.00,
        avg_latency_ms=2000,
        energy_tier="target_70b",
        capabilities=["reasoning", "code", "math"],
        supports_tools=False,  # o1 doesn't support tools yet
        supports_vision=False,
        max_context=128000,
        agent_score=0.5,  # Good reasoning but no tools
    ),
    "openai/o3-mini": ModelInfo(
        name="openai/o3-mini",
        tier=ModelTier.TARGET_70B,
        provider="openai",
        cost_per_mtok=4.40,
        avg_latency_ms=2500,
        energy_tier="target_70b",
        capabilities=["reasoning", "code", "math", "analysis"],
        supports_tools=True,
        supports_vision=False,
        max_context=200000,
        agent_score=0.8,  # Great reasoning + tools
    ),
    "deepseek/deepseek-reasoner": ModelInfo(
        name="deepseek/deepseek-reasoner",
        tier=ModelTier.TARGET_70B,
        provider="deepseek",
        cost_per_mtok=2.19,
        avg_latency_ms=1500,
        energy_tier="target_70b",
        capabilities=["reasoning", "math", "code"],
        supports_tools=True,
        supports_vision=False,
        max_context=64000,
        agent_score=0.6,
    ),
    "openai/gpt-5": ModelInfo(
        name="openai/gpt-5",
        tier=ModelTier.TARGET_70B,
        provider="openai",
        cost_per_mtok=5.625,
        avg_latency_ms=900,
        energy_tier="target_70b",
        capabilities=["general", "code", "writing", "reasoning", "analysis"],
        supports_tools=True,
        supports_vision=True,
        max_context=128000,
        agent_score=0.85,
    ),
    # ============ ULTRA TIER (400B+) - Frontier Quality ============
    "openai/gpt-5.2": ModelInfo(
        name="openai/gpt-5.2",
        tier=ModelTier.ULTRA_400B,
        provider="openai",
        cost_per_mtok=7.875,
        avg_latency_ms=1200,
        energy_tier="ultra_400b",
        capabilities=[
            "general",
            "code",
            "writing",
            "reasoning",
            "analysis",
            "vision",
            "agentic",
        ],
        supports_tools=True,
        supports_vision=True,
        max_context=256000,
        agent_score=0.92,  # Excellent for complex agents
    ),
    "openai/gpt-5.2-pro": ModelInfo(
        name="openai/gpt-5.2-pro",
        tier=ModelTier.ULTRA_400B,
        provider="openai",
        cost_per_mtok=94.50,
        avg_latency_ms=3000,
        energy_tier="ultra_400b",
        capabilities=[
            "general",
            "code",
            "writing",
            "reasoning",
            "analysis",
            "vision",
            "agentic",
        ],
        supports_tools=True,
        supports_vision=True,
        max_context=1000000,  # 1M context
        agent_score=0.98,  # Best for long-horizon agentic tasks
    ),
    "anthropic/claude-opus-4-5-20251101": ModelInfo(
        name="anthropic/claude-opus-4-5-20251101",
        tier=ModelTier.ULTRA_400B,
        provider="anthropic",
        cost_per_mtok=15.00,
        avg_latency_ms=1500,
        energy_tier="ultra_400b",
        capabilities=[
            "general",
            "code",
            "writing",
            "reasoning",
            "analysis",
            "creative",
            "agentic",
        ],
        supports_tools=True,
        supports_vision=True,
        max_context=200000,
        agent_score=0.95,  # Excellent for agentic coding
    ),
    "anthropic/claude-4.5-sonnet": ModelInfo(
        name="anthropic/claude-4.5-sonnet",
        tier=ModelTier.ULTRA_400B,
        provider="anthropic",
        cost_per_mtok=9.00,
        avg_latency_ms=1000,
        energy_tier="ultra_400b",
        capabilities=["general", "code", "writing", "reasoning", "analysis", "agentic"],
        supports_tools=True,
        supports_vision=True,
        max_context=200000,
        agent_score=0.88,
    ),
    "together/llama-3.1-405b": ModelInfo(
        name="together/llama-3.1-405b",
        tier=ModelTier.ULTRA_400B,
        provider="together",
        cost_per_mtok=3.50,
        avg_latency_ms=2000,
        energy_tier="ultra_400b",
        capabilities=["general", "code", "writing", "reasoning"],
        supports_tools=True,
        supports_vision=False,
        max_context=131072,
        agent_score=0.7,
    ),
}


@dataclass
class RoutingDecision:
    """Result of routing decision."""

    selected_model: str
    model_info: ModelInfo
    predicted_quality: float
    utility_score: float
    cost_estimate: float
    energy_estimate: float
    all_scores: Dict[str, Dict[str, float]]
    reasoning: str


@dataclass
class PromptFeatures:
    """Extracted features from a prompt for ML prediction."""

    length: int
    word_count: int
    has_code_keywords: bool
    has_reasoning_keywords: bool
    has_creative_keywords: bool
    has_analysis_keywords: bool
    question_count: int
    complexity_score: float
    task_type: str


class UnifiedRouter:
    """
    Unified Router that predicts quality for ALL model tiers.

    Instead of binary "draft vs target", we:
    1. Extract features from the prompt
    2. Predict quality score for each model tier
    3. Calculate utility = quality / cost (with energy penalty)
    4. Pick the model with highest utility

    This is the MOAT - we're the only ones with cross-provider quality data.
    """

    # Task classification keywords
    CODE_KEYWORDS = [
        "code",
        "python",
        "javascript",
        "react",
        "debug",
        "css",
        "html",
        "function",
        "class",
        "import",
        "api",
        "database",
        "sql",
        "typescript",
        "rust",
        "golang",
        "java",
        "c++",
        "programming",
        "algorithm",
        "bug",
    ]

    REASONING_KEYWORDS = [
        "calculate",
        "math",
        "solve",
        "equation",
        "proof",
        "logic",
        "reason",
        "deduce",
        "theorem",
        "probability",
        "why",
        "explain",
        "step by step",
        "derive",
        "prove",
        "analyze",
    ]

    CREATIVE_KEYWORDS = [
        "write",
        "story",
        "poem",
        "creative",
        "imagine",
        "fiction",
        "narrative",
        "character",
        "plot",
        "dialogue",
        "script",
    ]

    ANALYSIS_KEYWORDS = [
        "analyze",
        "compare",
        "evaluate",
        "research",
        "study",
        "data",
        "statistics",
        "trend",
        "report",
        "insight",
        "summarize",
        "review",
    ]

    def __init__(self, models_dir: str = "models"):
        """
        Initialize the Unified Router.

        Args:
            models_dir: Directory containing trained predictor models
        """
        self.models_dir = models_dir
        self.predictors: Dict[str, Any] = {}
        self._load_predictors()
        self.quality_learner = get_quality_learner() if get_quality_learner else None

        # Quality priors (used when no trained model available)
        # Based on empirical observations
        self.quality_priors = {
            ModelTier.DRAFT_8B: {
                "general": 0.70,
                "code": 0.60,
                "writing": 0.65,
                "reasoning": 0.45,
                "analysis": 0.55,
            },
            ModelTier.DRAFT_70B: {
                "general": 0.82,
                "code": 0.75,
                "writing": 0.80,
                "reasoning": 0.65,
                "analysis": 0.75,
            },
            ModelTier.TARGET_70B: {
                "general": 0.90,
                "code": 0.88,
                "writing": 0.90,
                "reasoning": 0.85,
                "analysis": 0.88,
            },
            ModelTier.ULTRA_400B: {
                "general": 0.96,
                "code": 0.95,
                "writing": 0.95,
                "reasoning": 0.94,
                "analysis": 0.95,
            },
        }

    def _load_predictors(self):
        """Load trained predictor models if available."""
        for tier in ModelTier:
            model_path = os.path.join(self.models_dir, f"{tier.value}_predictor.pkl")
            if os.path.exists(model_path):
                try:
                    with open(model_path, "rb") as f:
                        self.predictors[tier.value] = pickle.load(f)
                    print(f"Loaded predictor for {tier.value}")
                except Exception as e:
                    print(f"Warning: Failed to load predictor for {tier.value}: {e}")

    def extract_features(self, prompt: str) -> PromptFeatures:
        """Extract features from a prompt for ML prediction."""
        prompt_lower = prompt.lower()
        words = prompt.split()

        # Keyword detection
        has_code = any(kw in prompt_lower for kw in self.CODE_KEYWORDS)
        has_reasoning = any(kw in prompt_lower for kw in self.REASONING_KEYWORDS)
        has_creative = any(kw in prompt_lower for kw in self.CREATIVE_KEYWORDS)
        has_analysis = any(kw in prompt_lower for kw in self.ANALYSIS_KEYWORDS)

        # Question count
        question_count = prompt.count("?")

        # Complexity score (heuristic)
        complexity = 0.0
        complexity += min(len(prompt) / 1000, 1.0) * 0.3  # Length
        complexity += min(len(words) / 200, 1.0) * 0.2  # Word count
        complexity += 0.2 if has_reasoning else 0
        complexity += 0.15 if has_code else 0
        complexity += 0.1 if question_count > 2 else 0
        complexity += 0.05 if has_analysis else 0

        # Task type classification
        if has_code:
            task_type = "code"
        elif has_reasoning:
            task_type = "reasoning"
        elif has_creative:
            task_type = "writing"
        elif has_analysis:
            task_type = "analysis"
        else:
            task_type = "general"

        return PromptFeatures(
            length=len(prompt),
            word_count=len(words),
            has_code_keywords=has_code,
            has_reasoning_keywords=has_reasoning,
            has_creative_keywords=has_creative,
            has_analysis_keywords=has_analysis,
            question_count=question_count,
            complexity_score=round(complexity, 4),
            task_type=task_type,
        )

    def predict_quality(
        self,
        prompt: str,
        model_tier: ModelTier,
        features: Optional[PromptFeatures] = None,
    ) -> float:
        """
        Predict quality score (0-1) for a specific model tier.

        Uses trained ML model if available, otherwise falls back to priors.
        """
        if features is None:
            features = self.extract_features(prompt)

        # Try ML predictor first
        if model_tier.value in self.predictors:
            try:
                feature_vector = self._features_to_vector(features)
                predictor = self.predictors[model_tier.value]
                quality = predictor.predict_proba([feature_vector])[0][1]
                return round(quality, 4)
            except Exception as e:
                print(f"Warning: ML prediction failed for {model_tier.value}: {e}")

        # Fall back to priors with adjustments
        base_quality = self.quality_priors[model_tier].get(features.task_type, 0.7)

        # Adjust based on complexity
        if features.complexity_score > 0.6:
            # Complex prompts: lower quality for draft, higher for ultra
            if model_tier == ModelTier.DRAFT_8B:
                base_quality *= 0.8
            elif model_tier == ModelTier.ULTRA_400B:
                base_quality = min(base_quality * 1.05, 1.0)

        # Adjust based on prompt length
        if features.length < 50:
            # Short prompts: draft models do fine
            if model_tier == ModelTier.DRAFT_8B:
                base_quality = min(base_quality * 1.1, 0.95)

        return round(base_quality, 4)

    def _features_to_vector(self, features: PromptFeatures) -> List[float]:
        """Convert PromptFeatures to a feature vector for ML."""
        return [
            features.length / 1000,  # Normalized length
            features.word_count / 200,  # Normalized word count
            float(features.has_code_keywords),
            float(features.has_reasoning_keywords),
            float(features.has_creative_keywords),
            float(features.has_analysis_keywords),
            features.question_count / 5,  # Normalized question count
            features.complexity_score,
        ]

    def calculate_utility(
        self,
        quality: float,
        cost_per_mtok: float,
        energy_joules: float,
        green_factor: float = 0.01,
        latency_ms: float = 0,
        latency_factor: float = 0.0001,
    ) -> float:
        """
        Calculate utility score for a model.

        Utility = (Quality - Energy_Penalty - Latency_Penalty) / Cost

        Args:
            quality: Predicted quality (0-1)
            cost_per_mtok: Cost per million tokens
            energy_joules: Estimated energy consumption
            green_factor: Energy penalty weight (lambda)
            latency_ms: Expected latency
            latency_factor: Latency penalty weight
        """
        # Energy penalty
        energy_penalty = green_factor * energy_joules

        # Latency penalty (optional)
        latency_penalty = latency_factor * latency_ms

        # Adjusted quality
        adjusted_quality = quality - energy_penalty - latency_penalty

        # Utility = Quality / Cost (higher is better)
        # Add small epsilon to avoid division by zero
        utility = adjusted_quality / (cost_per_mtok + 0.001)

        return round(utility, 4)

    def route(
        self,
        prompt: str,
        budget: Optional[float] = None,
        green_factor: float = 0.01,
        required_capabilities: Optional[List[str]] = None,
        exclude_providers: Optional[List[str]] = None,
        min_quality: float = 0.0,
        prefer_latency: bool = False,
    ) -> RoutingDecision:
        """
        Route a request to the best model based on utility-per-cost.

        Args:
            prompt: The user's prompt
            budget: Maximum cost per million tokens (optional)
            green_factor: Energy penalty weight (0 = ignore, 0.1 = aggressive)
            required_capabilities: List of required capabilities (e.g., ["code", "reasoning"])
            exclude_providers: Providers to exclude (e.g., ["openai"])
            min_quality: Minimum acceptable quality score
            prefer_latency: If True, add latency penalty to utility

        Returns:
            RoutingDecision with selected model and reasoning
        """
        features = self.extract_features(prompt)

        # Estimate tokens for energy calculation
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = 500  # Assume average output

        all_scores: Dict[str, Dict[str, float]] = {}
        best_model = None
        best_utility = -float("inf")
        best_info = None

        for model_name, model_info in MODEL_REGISTRY.items():
            # Filter by budget
            if budget and model_info.cost_per_mtok > budget:
                continue

            # Filter by capabilities
            if required_capabilities:
                if not all(
                    cap in model_info.capabilities for cap in required_capabilities
                ):
                    continue

            # Filter by provider
            if exclude_providers and model_info.provider in exclude_providers:
                continue

            # Predict quality
            quality = self.predict_quality(prompt, model_info.tier, features)

            # Filter by minimum quality
            if quality < min_quality:
                continue

            # Calculate energy
            energy = ENERGY_ORACLE.estimate_joules(
                model_info.energy_tier, int(input_tokens), int(output_tokens)
            )

            # Calculate utility
            latency_factor = 0.0001 if prefer_latency else 0
            utility = self.calculate_utility(
                quality=quality,
                cost_per_mtok=model_info.cost_per_mtok,
                energy_joules=energy,
                green_factor=green_factor,
                latency_ms=model_info.avg_latency_ms,
                latency_factor=latency_factor,
            )

            # Store scores
            all_scores[model_name] = {
                "quality": quality,
                "utility": utility,
                "cost": model_info.cost_per_mtok,
                "energy": energy,
                "latency": model_info.avg_latency_ms,
                "tier": model_info.tier.value,
            }

            # Track best
            if utility > best_utility:
                best_utility = utility
                best_model = model_name
                best_info = model_info

        # Blend Thompson-like utility with quality learner recommendations.
        if all_scores and self.quality_learner is not None:
            models = list(all_scores.keys())
            quality_pick = self.quality_learner.get_strong_recommendation(
                features.task_type, models
            )
            classifier_pick = self.quality_learner.classifier_prediction(
                {
                    "task_type_code": {
                        "general": 0,
                        "code": 1,
                        "reasoning": 2,
                        "analysis": 3,
                        "writing": 4,
                    }.get(features.task_type, 0),
                    "prompt_length": features.length,
                    "has_code": 1 if features.has_code_keywords else 0,
                    "has_math": 1 if features.has_reasoning_keywords else 0,
                    "language_code": 0,
                    "complexity_score": features.complexity_score,
                    "time_of_day": 0,
                    "user_tier": 0,
                }
            )

            classifier_weight = (
                0.3 if self.quality_learner.training_samples <= 1000 else 0.6
            )
            thompson_weight = (
                0.7 if self.quality_learner.training_samples <= 1000 else 0.4
            )

            for model_name in models:
                base = all_scores[model_name]["utility"]
                quality_boost = 0.0
                if quality_pick and model_name == quality_pick.get("model"):
                    quality_boost = 0.2
                classifier_boost = (
                    classifier_weight if classifier_pick == model_name else 0.0
                )
                ensemble = (base * thompson_weight) + quality_boost + classifier_boost
                all_scores[model_name]["ensemble"] = ensemble

            ensemble_best = max(
                models,
                key=lambda m: all_scores[m].get("ensemble", all_scores[m]["utility"]),
            )
            if ensemble_best != best_model:
                logger.info(
                    "quality_learner overrode thompson_sampling: chose %s over %s for %s tasks",
                    ensemble_best,
                    best_model,
                    features.task_type,
                )
                best_model = ensemble_best
                best_info = MODEL_REGISTRY[best_model]
                best_utility = all_scores[best_model].get(
                    "ensemble", all_scores[best_model]["utility"]
                )

        if best_model is None:
            # Fallback to default
            best_model = "groq/llama-3.1-8b-instant"
            best_info = MODEL_REGISTRY[best_model]
            best_utility = 0

        # Calculate estimates for selected model
        energy_estimate = ENERGY_ORACLE.estimate_joules(
            best_info.energy_tier, int(input_tokens), int(output_tokens)
        )
        cost_estimate = (
            best_info.cost_per_mtok * (input_tokens + output_tokens) / 1_000_000
        )

        # Build reasoning
        reasoning = self._build_reasoning(
            features, best_model, best_info, all_scores, green_factor
        )

        return RoutingDecision(
            selected_model=best_model,
            model_info=best_info,
            predicted_quality=all_scores.get(best_model, {}).get("quality", 0.7),
            utility_score=best_utility,
            cost_estimate=cost_estimate,
            energy_estimate=energy_estimate,
            all_scores=all_scores,
            reasoning=reasoning,
        )

    def _build_reasoning(
        self,
        features: PromptFeatures,
        selected_model: str,
        model_info: ModelInfo,
        all_scores: Dict[str, Dict[str, float]],
        green_factor: float,
    ) -> str:
        """Build human-readable reasoning for the routing decision."""
        lines = [
            f"Task Type: {features.task_type}",
            f"Complexity: {features.complexity_score:.2f}",
            f"Green Factor: {green_factor}",
            "",
            f"Selected: {selected_model}",
            f"  Tier: {model_info.tier.value}",
            f"  Cost: ${model_info.cost_per_mtok}/1M tokens",
            f"  Quality: {all_scores.get(selected_model, {}).get('quality', 0):.2f}",
            f"  Utility: {all_scores.get(selected_model, {}).get('utility', 0):.4f}",
            "",
            "Alternatives considered:",
        ]

        # Sort by utility
        sorted_models = sorted(
            all_scores.items(), key=lambda x: x[1]["utility"], reverse=True
        )[:5]  # Top 5

        for model, scores in sorted_models:
            if model != selected_model:
                lines.append(
                    f"  {model}: Q={scores['quality']:.2f}, U={scores['utility']:.4f}, ${scores['cost']}/1M"
                )

        return "\n".join(lines)

    def get_models_by_tier(self, tier: ModelTier) -> List[str]:
        """Get all models in a specific tier."""
        return [name for name, info in MODEL_REGISTRY.items() if info.tier == tier]

    def get_cheapest_model(self, min_quality: float = 0.5) -> str:
        """Get the cheapest model that meets minimum quality."""
        cheapest = None
        lowest_cost = float("inf")

        for name, info in MODEL_REGISTRY.items():
            # Use general quality prior
            quality = self.quality_priors[info.tier].get("general", 0.7)
            if quality >= min_quality and info.cost_per_mtok < lowest_cost:
                cheapest = name
                lowest_cost = info.cost_per_mtok

        return cheapest or "groq/llama-3.1-8b-instant"

    def route_for_agent(
        self,
        prompt: str,
        session_context: Optional[Dict] = None,
        step_number: int = 1,
        total_steps_estimate: int = 10,
        requires_tools: bool = True,
        requires_vision: bool = False,
        min_context_window: int = 32000,
        budget_remaining: Optional[float] = None,
        green_factor: float = 0.01,
    ) -> RoutingDecision:
        """
        Route a request optimized for agentic workflows.

        Agents have unique requirements:
        1. Multi-step consistency - prefer same model across steps
        2. Tool calling - must support function calling
        3. Long context - accumulates context over time
        4. Cost accumulation - many small calls add up
        5. Reliability - agents need consistent, predictable outputs

        Args:
            prompt: The current step's prompt
            session_context: Dict with session state (model_used, total_cost, etc.)
            step_number: Current step in the workflow (1-indexed)
            total_steps_estimate: Estimated total steps in workflow
            requires_tools: Whether tool/function calling is needed
            requires_vision: Whether vision/image support is needed
            min_context_window: Minimum context window size needed
            budget_remaining: Remaining budget for this session
            green_factor: Energy penalty weight

        Returns:
            RoutingDecision optimized for agentic use
        """
        features = self.extract_features(prompt)

        # Estimate tokens
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = 500

        # Calculate per-step budget if total budget given
        if budget_remaining and total_steps_estimate > step_number:
            steps_remaining = total_steps_estimate - step_number + 1
            per_step_budget = budget_remaining / steps_remaining
        else:
            per_step_budget = None

        all_scores: Dict[str, Dict[str, float]] = {}
        best_model = None
        best_utility = -float("inf")
        best_info = None

        # If we have session context with a previously used model, prefer it
        preferred_model = session_context.get("model_used") if session_context else None

        for model_name, model_info in MODEL_REGISTRY.items():
            # Filter: Must support tools if required
            if requires_tools and not model_info.supports_tools:
                continue

            # Filter: Must support vision if required
            if requires_vision and not model_info.supports_vision:
                continue

            # Filter: Must have sufficient context window
            if model_info.max_context < min_context_window:
                continue

            # Filter by per-step budget
            if per_step_budget:
                estimated_cost = (
                    model_info.cost_per_mtok
                    * (input_tokens + output_tokens)
                    / 1_000_000
                )
                if estimated_cost > per_step_budget:
                    continue

            # Predict quality
            quality = self.predict_quality(prompt, model_info.tier, features)

            # Calculate energy
            energy = ENERGY_ORACLE.estimate_joules(
                model_info.energy_tier, int(input_tokens), int(output_tokens)
            )

            # Agent-specific utility calculation
            # Utility = (Quality x AgentScore - Energy_Penalty) / Cost
            agent_adjusted_quality = quality * model_info.agent_score

            # Bonus for model consistency (same model across steps)
            consistency_bonus = 0.1 if model_name == preferred_model else 0

            # Bonus for early steps (can afford better models early)
            early_step_bonus = 0.05 if step_number <= 3 else 0

            # Penalty for very expensive models in long workflows
            long_workflow_penalty = 0
            if total_steps_estimate > 20 and model_info.cost_per_mtok > 5:
                long_workflow_penalty = 0.1

            adjusted_quality = (
                agent_adjusted_quality
                + consistency_bonus
                + early_step_bonus
                - long_workflow_penalty
            )

            utility = self.calculate_utility(
                quality=adjusted_quality,
                cost_per_mtok=model_info.cost_per_mtok,
                energy_joules=energy,
                green_factor=green_factor,
                latency_ms=model_info.avg_latency_ms,
                latency_factor=0.0001,  # Agents care about latency
            )

            all_scores[model_name] = {
                "quality": quality,
                "agent_score": model_info.agent_score,
                "utility": utility,
                "cost": model_info.cost_per_mtok,
                "energy": energy,
                "latency": model_info.avg_latency_ms,
                "tier": model_info.tier.value,
                "supports_tools": model_info.supports_tools,
                "max_context": model_info.max_context,
            }

            if utility > best_utility:
                best_utility = utility
                best_model = model_name
                best_info = model_info

        # Fallback
        if best_model is None:
            best_model = "openai/gpt-4o-mini"  # Safe default for agents
            best_info = MODEL_REGISTRY[best_model]
            best_utility = 0

        # Calculate estimates
        energy_estimate = ENERGY_ORACLE.estimate_joules(
            best_info.energy_tier, int(input_tokens), int(output_tokens)
        )
        cost_estimate = (
            best_info.cost_per_mtok * (input_tokens + output_tokens) / 1_000_000
        )

        # Build agent-specific reasoning
        reasoning = self._build_agent_reasoning(
            features,
            best_model,
            best_info,
            all_scores,
            step_number,
            total_steps_estimate,
            requires_tools,
        )

        return RoutingDecision(
            selected_model=best_model,
            model_info=best_info,
            predicted_quality=all_scores.get(best_model, {}).get("quality", 0.7),
            utility_score=best_utility,
            cost_estimate=cost_estimate,
            energy_estimate=energy_estimate,
            all_scores=all_scores,
            reasoning=reasoning,
        )

    def _build_agent_reasoning(
        self,
        features: PromptFeatures,
        selected_model: str,
        model_info: ModelInfo,
        all_scores: Dict[str, Dict[str, float]],
        step_number: int,
        total_steps: int,
        requires_tools: bool,
    ) -> str:
        """Build reasoning for agent routing decision."""
        lines = [
            f"AGENT ROUTING (Step {step_number}/{total_steps})",
            f"Task Type: {features.task_type}",
            f"Requires Tools: {requires_tools}",
            "",
            f"Selected: {selected_model}",
            f"  Agent Score: {model_info.agent_score:.2f}",
            f"  Supports Tools: {model_info.supports_tools}",
            f"  Max Context: {model_info.max_context:,} tokens",
            f"  Cost: ${model_info.cost_per_mtok}/1M tokens",
            "",
            "Top Agent-Optimized Models:",
        ]

        # Sort by agent_score x utility
        sorted_models = sorted(
            all_scores.items(),
            key=lambda x: x[1].get("agent_score", 0) * x[1]["utility"],
            reverse=True,
        )[:5]

        for model, scores in sorted_models:
            lines.append(
                f"  {model}: Agent={scores.get('agent_score', 0):.2f}, "
                f"Tools={scores.get('supports_tools', False)}, "
                f"${scores['cost']}/1M"
            )

        return "\n".join(lines)

    def get_best_agent_models(self, top_n: int = 5) -> List[Dict]:
        """Get the best models for agentic workflows."""
        models = []
        for name, info in MODEL_REGISTRY.items():
            if info.supports_tools:  # Must support tools for agents
                models.append(
                    {
                        "name": name,
                        "agent_score": info.agent_score,
                        "tier": info.tier.value,
                        "cost": info.cost_per_mtok,
                        "max_context": info.max_context,
                        "supports_vision": info.supports_vision,
                    }
                )

        # Sort by agent_score
        models.sort(key=lambda x: x["agent_score"], reverse=True)
        return models[:top_n]


# Global instance
UNIFIED_ROUTER = UnifiedRouter()


def route_request(
    prompt: str, budget: Optional[float] = None, green_factor: float = 0.01
) -> RoutingDecision:
    """
    Route a request using the Unified Router.

    This is the main entry point for multi-tier routing.
    """
    return UNIFIED_ROUTER.route(prompt=prompt, budget=budget, green_factor=green_factor)
