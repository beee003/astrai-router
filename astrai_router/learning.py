"""
Auto-Learning Engine for Astrai

This is the core intelligence layer that:
1. Auto-detects task types from prompts
2. Auto-applies default eval contracts (no user config needed)
3. Captures implicit signals (regenerations, timing, accepts)
4. Learns hierarchically (global → workflow → step → user)
5. Explores safely via Thompson Sampling

Philosophy: "Users don't define quality. Astrai learns what quality means by observing behavior."
"""

import re
import os
import time
import hashlib
import random
import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


# =============================================================================
# TASK TYPE DETECTION
# =============================================================================

class TaskType(str, Enum):
    """Auto-detected task types with associated quality expectations."""
    CODE = "code"
    RESEARCH = "research"
    CHAT = "chat"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    EXTRACTION = "extraction"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    FUNCTION_CALL = "function_call"
    UNKNOWN = "unknown"


# Task detection patterns
TASK_PATTERNS = {
    TaskType.CODE: [
        r'\b(write|create|implement|code|function|class|method|debug|fix|refactor)\b.*\b(python|javascript|typescript|java|rust|go|c\+\+|sql|html|css)\b',
        r'\b(def|class|function|const|let|var|import|from|return)\s+\w+',
        r'```(\w+)?\n',  # Code blocks
        r'\b(api|endpoint|database|query|algorithm|data structure)\b',
        r'\b(bug|error|exception|traceback|stack trace)\b',
    ],
    TaskType.RESEARCH: [
        r'\b(research|investigate|find out|what is|explain|how does|why does|compare|contrast)\b',
        r'\b(study|paper|article|source|reference|citation)\b',
        r'\b(according to|based on|evidence|data shows)\b',
    ],
    TaskType.ANALYSIS: [
        r'\b(analyze|analysis|evaluate|assess|review|examine|interpret)\b',
        r'\b(pros and cons|strengths and weaknesses|advantages|disadvantages)\b',
        r'\b(data|metrics|statistics|trends|patterns)\b',
        r'\b(stock|market|financial|investment|portfolio)\b',
    ],
    TaskType.CREATIVE: [
        r'\b(write|create|compose|generate)\b.*\b(story|poem|essay|article|blog|script|lyrics)\b',
        r'\b(creative|imaginative|fictional|narrative)\b',
        r'\b(character|plot|setting|theme|tone)\b',
    ],
    TaskType.SUMMARIZATION: [
        r'\b(summarize|summary|tldr|brief|condense|shorten)\b',
        r'\b(key points|main points|highlights|takeaways)\b',
    ],
    TaskType.EXTRACTION: [
        r'\b(extract|pull out|identify|find|list)\b.*\b(from|in)\b',
        r'\b(entities|names|dates|numbers|keywords)\b',
        r'\b(parse|structure|format)\b',
    ],
    TaskType.TRANSLATION: [
        r'\b(translate|translation|convert)\b.*\b(to|into|from)\b',
        r'\b(language|english|spanish|french|german|chinese|japanese)\b',
    ],
    TaskType.FUNCTION_CALL: [
        r'\b(call|invoke|execute|run)\b.*\b(function|api|tool)\b',
        r'"function"|"name"|"arguments"',  # JSON function call patterns
        r'\b(tool_calls|function_call)\b',
    ],
    TaskType.CHAT: [
        r'^(hi|hello|hey|thanks|thank you|please|help)\b',
        r'\?$',  # Questions
        r'\b(can you|could you|would you|will you)\b',
    ],
}

# Minimum confidence threshold for task classification
# If confidence is below this, return UNKNOWN to avoid noisy stats
TASK_CONFIDENCE_THRESHOLD = float(__import__('os').getenv("ASTRAI_TASK_CONFIDENCE_THRESHOLD", "0.4"))


def detect_task_type(prompt: str) -> Tuple[TaskType, float]:
    """
    Auto-detect task type from prompt with confidence score.

    Uses keyword matching with confidence threshold to ensure stable classification.
    If confidence is below TASK_CONFIDENCE_THRESHOLD, returns UNKNOWN to avoid
    polluting aggregation stats with ambiguous classifications.

    Returns:
        Tuple of (TaskType, confidence 0-1)
    """
    if not prompt or len(prompt.strip()) < 10:
        return TaskType.UNKNOWN, 0.0

    prompt_lower = prompt.lower()
    scores = defaultdict(float)
    match_counts = defaultdict(int)

    for task_type, patterns in TASK_PATTERNS.items():
        for pattern in patterns:
            try:
                matches = re.findall(pattern, prompt_lower, re.IGNORECASE)
                if matches:
                    match_counts[task_type] += 1
                    # Score based on number of matches and pattern diversity
                    scores[task_type] += len(matches) * 0.15
            except re.error:
                # Skip invalid patterns
                continue

    if not scores:
        return TaskType.UNKNOWN, 0.0

    # Normalize scores: consider both total score and pattern diversity
    for task_type in scores:
        # Bonus for matching multiple different patterns (more confident)
        diversity_bonus = min(0.3, match_counts[task_type] * 0.1)
        scores[task_type] += diversity_bonus

    # Get highest scoring task type
    best_type = max(scores, key=scores.get)
    raw_confidence = scores[best_type]

    # Calculate relative confidence (how much better than second best)
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1:
        margin = sorted_scores[0] - sorted_scores[1]
        # Higher margin = more confident
        relative_confidence = min(1.0, raw_confidence * (1 + margin * 0.5))
    else:
        relative_confidence = min(1.0, raw_confidence)

    # Apply confidence threshold
    if relative_confidence < TASK_CONFIDENCE_THRESHOLD:
        return TaskType.UNKNOWN, relative_confidence

    return best_type, relative_confidence


def detect_task_type_stable(prompt: str) -> Tuple[TaskType, float, bool]:
    """
    Wrapper around detect_task_type that also returns stability flag.

    Returns:
        Tuple of (TaskType, confidence, is_stable)
        is_stable = True if confidence >= threshold
    """
    task_type, confidence = detect_task_type(prompt)
    is_stable = confidence >= TASK_CONFIDENCE_THRESHOLD and task_type != TaskType.UNKNOWN
    return task_type, confidence, is_stable


# =============================================================================
# DEFAULT EVAL CONTRACTS PER TASK TYPE
# =============================================================================

DEFAULT_EVALS: Dict[TaskType, Dict[str, Any]] = {
    TaskType.CODE: {
        "name": "Auto: Code Quality",
        "automatic_checks": [
            {"type": "not_empty"},
            {"type": "no_refusal"},
            {"type": "min_length", "value": 50},
            {"type": "valid_code"},
        ],
        "quality_weights": {
            "correctness": 0.4,
            "completeness": 0.3,
            "efficiency": 0.2,
            "readability": 0.1,
        },
        "failure_signals": ["syntax error", "undefined", "traceback", "exception"],
    },
    TaskType.RESEARCH: {
        "name": "Auto: Research Quality",
        "automatic_checks": [
            {"type": "not_empty"},
            {"type": "no_refusal"},
            {"type": "min_length", "value": 200},
            {"type": "has_citations"},
            {"type": "no_hallucination_markers"},
        ],
        "quality_weights": {
            "accuracy": 0.4,
            "depth": 0.3,
            "citations": 0.2,
            "clarity": 0.1,
        },
    },
    TaskType.ANALYSIS: {
        "name": "Auto: Analysis Quality",
        "automatic_checks": [
            {"type": "not_empty"},
            {"type": "no_refusal"},
            {"type": "min_length", "value": 150},
            {"type": "professional_tone"},
        ],
        "quality_weights": {
            "insight": 0.4,
            "accuracy": 0.3,
            "structure": 0.2,
            "actionability": 0.1,
        },
    },
    TaskType.CREATIVE: {
        "name": "Auto: Creative Quality",
        "automatic_checks": [
            {"type": "not_empty"},
            {"type": "no_refusal"},
            {"type": "min_length", "value": 100},
        ],
        "quality_weights": {
            "creativity": 0.4,
            "coherence": 0.3,
            "engagement": 0.2,
            "style": 0.1,
        },
    },
    TaskType.SUMMARIZATION: {
        "name": "Auto: Summary Quality",
        "automatic_checks": [
            {"type": "not_empty"},
            {"type": "no_refusal"},
            {"type": "min_length", "value": 50},
            {"type": "max_length", "value": 500},
        ],
        "quality_weights": {
            "accuracy": 0.4,
            "conciseness": 0.3,
            "completeness": 0.2,
            "clarity": 0.1,
        },
    },
    TaskType.EXTRACTION: {
        "name": "Auto: Extraction Quality",
        "automatic_checks": [
            {"type": "not_empty"},
            {"type": "no_refusal"},
        ],
        "quality_weights": {
            "precision": 0.5,
            "recall": 0.3,
            "format": 0.2,
        },
    },
    TaskType.FUNCTION_CALL: {
        "name": "Auto: Function Call Quality",
        "automatic_checks": [
            {"type": "not_empty"},
            {"type": "valid_json"},
        ],
        "quality_weights": {
            "correctness": 0.6,
            "completeness": 0.3,
            "efficiency": 0.1,
        },
    },
    TaskType.TRANSLATION: {
        "name": "Auto: Translation Quality",
        "automatic_checks": [
            {"type": "not_empty"},
            {"type": "no_refusal"},
            {"type": "min_length", "value": 10},
        ],
        "quality_weights": {
            "accuracy": 0.5,
            "fluency": 0.3,
            "style": 0.2,
        },
    },
    TaskType.CHAT: {
        "name": "Auto: Chat Quality",
        "automatic_checks": [
            {"type": "not_empty"},
            {"type": "no_refusal"},
        ],
        "quality_weights": {
            "helpfulness": 0.4,
            "accuracy": 0.3,
            "clarity": 0.2,
            "tone": 0.1,
        },
    },
    TaskType.UNKNOWN: {
        "name": "Auto: General Quality",
        "automatic_checks": [
            {"type": "not_empty"},
            {"type": "no_refusal"},
            {"type": "min_length", "value": 20},
        ],
        "quality_weights": {
            "relevance": 0.4,
            "accuracy": 0.3,
            "clarity": 0.2,
            "completeness": 0.1,
        },
    },
}


# =============================================================================
# IMPLICIT FEEDBACK SIGNALS
# =============================================================================

@dataclass
class ImplicitSignal:
    """Captures implicit user feedback signals."""
    user_id: str
    request_hash: str
    timestamp: float
    signal_type: str  # regenerate, accept, edit, escalate, timeout
    signal_value: float  # -1 to 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImplicitFeedbackTracker:
    """
    Tracks implicit user feedback without explicit labeling.

    Signals captured:
    - Regeneration (immediate retry) → negative
    - Quick accept (< 5s to next action) → positive
    - Long dwell time (reading) → neutral-positive
    - Edit/modification → mildly negative
    - Escalation request → very negative
    - Copy/share → positive
    """

    def __init__(self):
        # Recent requests per user for detecting regenerations
        self.recent_requests: Dict[str, List[Dict]] = defaultdict(list)
        # Accumulated signals per (user, model, task_type)
        self.signals: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
        # Request-response timing
        self.response_times: Dict[str, float] = {}

    def record_request(
        self,
        user_id: str,
        request_hash: str,
        prompt: str,
        model: str,
        task_type: str,
        timestamp: float = None
    ):
        """Record a new request for tracking."""
        timestamp = timestamp or time.time()

        request_data = {
            "hash": request_hash,
            "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()[:16],
            "model": model,
            "task_type": task_type,
            "timestamp": timestamp,
        }

        # Check if this is a regeneration (similar prompt within 60s)
        recent = self.recent_requests[user_id]
        for prev in recent[-5:]:  # Check last 5 requests
            if (timestamp - prev["timestamp"]) < 60:  # Within 60 seconds
                # Check if prompts are similar (same hash = exact same, or check similarity)
                if prev["prompt_hash"] == request_data["prompt_hash"]:
                    # This is a regeneration - negative signal for previous model
                    self._record_signal(
                        user_id, prev["model"], prev["task_type"],
                        signal_value=-0.5,  # Regeneration is negative
                        signal_type="regenerate"
                    )
                    break

        # Add to recent requests
        self.recent_requests[user_id].append(request_data)

        # Keep only last 20 requests per user
        if len(self.recent_requests[user_id]) > 20:
            self.recent_requests[user_id] = self.recent_requests[user_id][-20:]

        # Record response time for timing analysis
        self.response_times[request_hash] = timestamp

    def record_response_received(
        self,
        user_id: str,
        request_hash: str,
        model: str,
        task_type: str,
        latency_ms: float
    ):
        """Record when response was received (for timing analysis)."""
        self.response_times[request_hash] = time.time()

    def record_next_action(
        self,
        user_id: str,
        previous_request_hash: str,
        model: str,
        task_type: str,
        action_type: str,  # "new_request", "copy", "share", "edit"
        time_since_response_ms: float
    ):
        """Record user's next action after receiving a response."""

        if action_type == "copy" or action_type == "share":
            # User copied/shared = positive signal
            self._record_signal(user_id, model, task_type, 0.8, "accept")
        elif action_type == "edit":
            # User edited = mild negative (needed modification)
            self._record_signal(user_id, model, task_type, -0.2, "edit")
        elif action_type == "new_request":
            if time_since_response_ms < 5000:
                # Very quick follow-up might be regeneration or refinement
                self._record_signal(user_id, model, task_type, -0.3, "quick_followup")
            elif time_since_response_ms > 30000:
                # Long dwell time = user probably read and used it
                self._record_signal(user_id, model, task_type, 0.3, "dwell")

    def _record_signal(
        self,
        user_id: str,
        model: str,
        task_type: str,
        signal_value: float,
        signal_type: str
    ):
        """Record a feedback signal."""
        key = (user_id, model, task_type)
        self.signals[key].append(signal_value)

        # Keep only last 100 signals per key
        if len(self.signals[key]) > 100:
            self.signals[key] = self.signals[key][-100:]

    def get_implicit_score(
        self,
        user_id: str,
        model: str,
        task_type: str
    ) -> Tuple[float, int]:
        """
        Get aggregated implicit feedback score for a model.

        Returns:
            Tuple of (score -1 to 1, sample_count)
        """
        key = (user_id, model, task_type)
        signals = self.signals.get(key, [])

        if not signals:
            return 0.0, 0

        # Weighted average with recency bias
        weights = [1.0 + 0.1 * i for i in range(len(signals))]
        weighted_sum = sum(s * w for s, w in zip(signals, weights))
        total_weight = sum(weights)

        return weighted_sum / total_weight, len(signals)


# =============================================================================
# HIERARCHICAL MODEL SCORING
# =============================================================================

@dataclass
class ModelScore:
    """Hierarchical model score with uncertainty."""
    model: str
    provider: str

    # Scores at different levels (0-5 scale)
    global_score: float = 3.0
    task_type_score: float = 3.0
    workflow_score: float = 3.0
    step_score: float = 3.0
    user_score: float = 3.0

    # Sample counts (for uncertainty)
    global_samples: int = 0
    task_type_samples: int = 0
    workflow_samples: int = 0
    step_samples: int = 0
    user_samples: int = 0

    # Cost and latency
    cost_per_1k_tokens: float = 0.0
    avg_latency_ms: float = 0.0

    # Implicit feedback adjustment
    implicit_adjustment: float = 0.0

    def combined_score(self, weights: Dict[str, float] = None) -> float:
        """
        Calculate combined score with hierarchical weighting.

        More specific levels get higher weight when they have enough samples.
        """
        weights = weights or {
            "global": 0.1,
            "task_type": 0.2,
            "workflow": 0.25,
            "step": 0.3,
            "user": 0.15,
        }

        # Adjust weights based on sample counts (more samples = more trust)
        def confidence_weight(samples: int, base_weight: float) -> float:
            # Sigmoid-like confidence scaling
            confidence = min(1.0, samples / 50)  # Full confidence at 50 samples
            return base_weight * confidence

        adjusted_weights = {
            "global": confidence_weight(max(self.global_samples, 10), weights["global"]),
            "task_type": confidence_weight(self.task_type_samples, weights["task_type"]),
            "workflow": confidence_weight(self.workflow_samples, weights["workflow"]),
            "step": confidence_weight(self.step_samples, weights["step"]),
            "user": confidence_weight(self.user_samples, weights["user"]),
        }

        total_weight = sum(adjusted_weights.values())
        if total_weight == 0:
            return self.global_score

        # Normalize weights
        for k in adjusted_weights:
            adjusted_weights[k] /= total_weight

        score = (
            self.global_score * adjusted_weights["global"] +
            self.task_type_score * adjusted_weights["task_type"] +
            self.workflow_score * adjusted_weights["workflow"] +
            self.step_score * adjusted_weights["step"] +
            self.user_score * adjusted_weights["user"] +
            self.implicit_adjustment
        )

        return max(0.0, min(5.0, score))

    def uncertainty(self) -> float:
        """Calculate uncertainty based on sample counts."""
        total_samples = (
            self.global_samples +
            self.task_type_samples +
            self.workflow_samples +
            self.step_samples +
            self.user_samples
        )
        # Higher uncertainty with fewer samples
        return 1.0 / (1.0 + math.sqrt(total_samples / 10))


# =============================================================================
# THOMPSON SAMPLING FOR EXPLORATION
# =============================================================================

class ThompsonSamplingSelector:
    """
    Thompson Sampling for model selection with exploration.

    Balances exploitation (use best known model) with exploration
    (try alternatives to discover better options).
    """

    def __init__(self, exploration_rate: float = 0.1):
        """
        Args:
            exploration_rate: Minimum exploration probability (0-1)
        """
        self.exploration_rate = exploration_rate
        # Beta distribution parameters per (context, model)
        self.alphas: Dict[str, float] = defaultdict(lambda: 1.0)  # Successes
        self.betas: Dict[str, float] = defaultdict(lambda: 1.0)   # Failures
        self._dirty = False
        self._last_persist = 0.0
        self._persist_interval = 60.0

        # Load persisted state
        self._load()

    def select_model(
        self,
        candidates: List[ModelScore],
        context_key: str,
        prefer_exploration: bool = False
    ) -> Tuple[ModelScore, str]:
        """
        Select a model using Thompson Sampling.

        Args:
            candidates: List of ModelScore objects
            context_key: Key for this decision context (e.g., task_type)
            prefer_exploration: If True, increase exploration rate

        Returns:
            Tuple of (selected ModelScore, reason)
        """
        if not candidates:
            raise ValueError("No candidates provided")

        if len(candidates) == 1:
            return candidates[0], "only_option"

        # Calculate Thompson samples for each candidate
        samples = []
        for model_score in candidates:
            key = f"{context_key}:{model_score.model}"
            alpha = self.alphas[key]
            beta = self.betas[key]

            # Add uncertainty bonus for under-explored models
            uncertainty_bonus = model_score.uncertainty() * 0.5

            # Sample from Beta distribution
            sampled_value = random.betavariate(alpha, beta)

            # Combine with quality score
            combined = model_score.combined_score() / 5.0  # Normalize to 0-1

            # Thompson sample with quality prior
            final_sample = 0.6 * combined + 0.3 * sampled_value + 0.1 * uncertainty_bonus

            samples.append((final_sample, model_score))

        # Sort by sampled value
        samples.sort(key=lambda x: x[0], reverse=True)

        # Forced exploration with probability exploration_rate
        exploration_rate = self.exploration_rate * (2 if prefer_exploration else 1)
        if random.random() < exploration_rate and len(samples) > 1:
            # Pick a random non-best option
            idx = random.randint(1, min(3, len(samples) - 1))
            return samples[idx][1], "exploration"

        return samples[0][1], "exploitation"

    def record_outcome(
        self,
        context_key: str,
        model: str,
        success: bool,
        quality_score: float = None
    ):
        """Record outcome to update beliefs and periodically persist."""
        key = f"{context_key}:{model}"

        if success:
            # Success increases alpha
            increment = 1.0 + (quality_score / 5.0 if quality_score else 0.5)
            self.alphas[key] += increment
        else:
            # Failure increases beta
            self.betas[key] += 1.0

        # Decay old observations (recency bias)
        decay = 0.99
        self.alphas[key] *= decay
        self.betas[key] *= decay

        # Ensure minimum values
        self.alphas[key] = max(1.0, self.alphas[key])
        self.betas[key] = max(1.0, self.betas[key])

        self._dirty = True
        now = time.time()
        if now - self._last_persist > self._persist_interval:
            self._persist()

    # ------ Persistence ------

    @staticmethod
    def _priors_path() -> str:
        data_dir = os.getenv("ASTRAI_DATA_DIR", "./data")
        try:
            os.makedirs(data_dir, exist_ok=True)
        except PermissionError:
            data_dir = "/tmp/astrai_data"
            os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, "auto_learning_priors.json")

    def _load(self):
        path = self._priors_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            for key, vals in data.items():
                self.alphas[key] = vals["alpha"]
                self.betas[key] = vals["beta"]
            logger.info(f"Loaded {len(data)} auto-learning priors from {path}")
        except Exception as e:
            logger.warning(f"Failed to load auto-learning priors: {e}")

    def _persist(self):
        if not self._dirty:
            return
        path = self._priors_path()
        try:
            data = {}
            for key in set(list(self.alphas.keys()) + list(self.betas.keys())):
                data[key] = {"alpha": self.alphas[key], "beta": self.betas[key]}
            with open(path, "w") as f:
                json.dump(data, f)
            self._dirty = False
            self._last_persist = time.time()
        except Exception as e:
            logger.warning(f"Failed to persist auto-learning priors: {e}")

    def flush(self):
        """Force-flush to disk (call on shutdown)."""
        self._persist()


# =============================================================================
# MAIN AUTO-LEARNING ENGINE
# =============================================================================

class AutoLearningEngine:
    """
    Main engine that orchestrates auto-learning.

    This is the "brain" that:
    1. Auto-detects task types
    2. Applies default evals
    3. Tracks implicit feedback
    4. Scores models hierarchically
    5. Selects models with exploration
    """

    def __init__(self):
        self.feedback_tracker = ImplicitFeedbackTracker()
        self.model_selector = ThompsonSamplingSelector(exploration_rate=0.1)

        # Cache for model scores
        self.score_cache: Dict[str, ModelScore] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps: Dict[str, float] = {}

    def process_request(
        self,
        prompt: str,
        user_id: str,
        available_models: List[Dict[str, Any]],
        workflow_id: str = None,
        step_id: str = None,
        # User preferences for auto-learning
        auto_optimize: bool = True,
        quality_vs_cost: float = 0.5,  # 0 = cheaper, 1 = higher quality
        risk_tolerance: float = 0.3,   # 0 = conservative, 1 = explore more
    ) -> Dict[str, Any]:
        """
        Process a request and return routing decision.

        Args:
            prompt: The user's prompt
            user_id: User identifier
            available_models: List of available models with metadata
            workflow_id: Optional workflow context
            step_id: Optional step context
            auto_optimize: If False, skip learning-based routing
            quality_vs_cost: 0.0 = prefer cheaper, 1.0 = prefer higher quality
            risk_tolerance: 0.0 = conservative, 1.0 = explore more

        Returns:
            Dict with:
                - task_type: detected task type
                - task_confidence: detection confidence
                - eval_contract: auto-generated eval contract
                - recommended_model: selected model
                - selection_reason: why this model was selected
                - explore_mode: whether this is exploration
        """
        # 1. Detect task type
        task_type, task_confidence = detect_task_type(prompt)

        # 2. Get default eval contract
        eval_contract = DEFAULT_EVALS.get(task_type, DEFAULT_EVALS[TaskType.UNKNOWN])

        # 3. Build model scores with quality_vs_cost weighting
        model_scores = []
        for model_info in available_models:
            score = self._build_model_score(
                model=model_info["model"],
                provider=model_info.get("provider", "unknown"),
                task_type=task_type,
                user_id=user_id,
                workflow_id=workflow_id,
                step_id=step_id,
                cost=model_info.get("cost_per_1k", 0),
                latency=model_info.get("avg_latency_ms", 100),
                quality_vs_cost=quality_vs_cost,  # Pass user preference
            )
            model_scores.append(score)

        # 4. Select model (with or without auto-optimization)
        context_key = f"{task_type.value}:{workflow_id or 'global'}:{step_id or 'global'}"

        if auto_optimize:
            # Use Thompson Sampling with user's risk tolerance
            prefer_exploration = task_confidence < 0.7 or risk_tolerance > 0.5
            # Adjust exploration rate based on risk tolerance
            original_rate = self.model_selector.exploration_rate
            self.model_selector.exploration_rate = 0.05 + (risk_tolerance * 0.15)  # 5-20%

            selected, reason = self.model_selector.select_model(
                model_scores, context_key, prefer_exploration
            )

            # Restore original rate
            self.model_selector.exploration_rate = original_rate
        else:
            # No auto-optimization: just pick highest score
            model_scores.sort(key=lambda x: x.combined_score(), reverse=True)
            selected = model_scores[0] if model_scores else None
            reason = "manual_no_optimize"

        # 5. Generate request hash for tracking
        request_hash = hashlib.md5(f"{prompt}:{time.time()}".encode()).hexdigest()[:16]

        # 6. Record request for implicit feedback tracking
        self.feedback_tracker.record_request(
            user_id=user_id,
            request_hash=request_hash,
            prompt=prompt,
            model=selected.model,
            task_type=task_type.value,
        )

        return {
            "task_type": task_type.value,
            "task_confidence": round(task_confidence, 3),
            "eval_contract": eval_contract,
            "recommended_model": selected.model,
            "recommended_provider": selected.provider,
            "selection_reason": reason,
            "explore_mode": reason == "exploration",
            "request_hash": request_hash,
            "model_score": round(selected.combined_score(), 3),
            "model_uncertainty": round(selected.uncertainty(), 3),
        }

    def record_outcome(
        self,
        request_hash: str,
        user_id: str,
        model: str,
        task_type: str,
        eval_passed: bool,
        eval_score: float,
        workflow_id: str = None,
        step_id: str = None,
    ):
        """Record outcome for learning."""
        context_key = f"{task_type}:{workflow_id or 'global'}:{step_id or 'global'}"

        self.model_selector.record_outcome(
            context_key=context_key,
            model=model,
            success=eval_passed,
            quality_score=eval_score,
        )

    def _build_model_score(
        self,
        model: str,
        provider: str,
        task_type: TaskType,
        user_id: str,
        workflow_id: str = None,
        step_id: str = None,
        cost: float = 0,
        latency: float = 100,
        quality_vs_cost: float = 0.5,  # User preference
    ) -> ModelScore:
        """
        Build hierarchical model score with user preference weighting.

        Args:
            quality_vs_cost: 0.0 = heavily prefer cheaper models,
                           1.0 = heavily prefer higher quality models
        """
        # Get implicit feedback adjustment
        implicit_score, implicit_samples = self.feedback_tracker.get_implicit_score(
            user_id, model, task_type.value
        )

        # Calculate cost penalty based on user preference
        # Higher quality_vs_cost = less penalty for expensive models
        max_cost = 10.0  # $10/1k tokens as reference max
        cost_factor = min(cost / max_cost, 1.0) if max_cost > 0 else 0
        cost_penalty = cost_factor * (1.0 - quality_vs_cost)  # 0 to 1

        # Create score object
        # In production, these would come from database queries
        score = ModelScore(
            model=model,
            provider=provider,
            global_score=3.5,  # Default prior
            task_type_score=3.5,
            workflow_score=3.5,
            step_score=3.5,
            user_score=3.5,
            global_samples=100,  # Assume some global data
            task_type_samples=0,
            workflow_samples=0,
            step_samples=0,
            user_samples=implicit_samples,
            cost_per_1k_tokens=cost,
            avg_latency_ms=latency,
            implicit_adjustment=implicit_score * 0.5 - cost_penalty,  # Apply cost penalty
        )

        return score

    def get_learning_stats(self, user_id: str = None) -> Dict[str, Any]:
        """Get statistics about the learning system."""
        return {
            "total_tracked_users": len(self.feedback_tracker.recent_requests),
            "total_signal_keys": len(self.feedback_tracker.signals),
            "model_selector_contexts": len(self.model_selector.alphas),
            "exploration_rate": self.model_selector.exploration_rate,
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

AUTO_LEARNING_ENGINE = AutoLearningEngine()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def auto_route_request(
    prompt: str,
    user_id: str,
    available_models: List[Dict[str, Any]],
    workflow_id: str = None,
    step_id: str = None,
    # User preferences (from RoutingPreferences)
    auto_optimize: bool = True,
    quality_vs_cost: float = 0.5,
    risk_tolerance: float = 0.3,
) -> Dict[str, Any]:
    """
    Convenience function for auto-routing a request.

    This is the main entry point for the auto-learning system.

    Args:
        prompt: User's prompt
        user_id: User identifier
        available_models: List of available models
        workflow_id: Optional workflow context
        step_id: Optional step context
        auto_optimize: Enable automatic learning-based routing
        quality_vs_cost: 0.0 = prefer cheaper, 1.0 = prefer higher quality
        risk_tolerance: 0.0 = conservative, 1.0 = explore more
    """
    return AUTO_LEARNING_ENGINE.process_request(
        prompt=prompt,
        user_id=user_id,
        available_models=available_models,
        workflow_id=workflow_id,
        step_id=step_id,
        auto_optimize=auto_optimize,
        quality_vs_cost=quality_vs_cost,
        risk_tolerance=risk_tolerance,
    )


def record_request_outcome(
    request_hash: str,
    user_id: str,
    model: str,
    task_type: str,
    eval_passed: bool,
    eval_score: float,
    workflow_id: str = None,
    step_id: str = None,
):
    """Record outcome for a completed request."""
    AUTO_LEARNING_ENGINE.record_outcome(
        request_hash=request_hash,
        user_id=user_id,
        model=model,
        task_type=task_type,
        eval_passed=eval_passed,
        eval_score=eval_score,
        workflow_id=workflow_id,
        step_id=step_id,
    )


def get_default_eval_contract(task_type: str) -> Dict[str, Any]:
    """Get default eval contract for a task type."""
    try:
        tt = TaskType(task_type)
    except ValueError:
        tt = TaskType.UNKNOWN
    return DEFAULT_EVALS.get(tt, DEFAULT_EVALS[TaskType.UNKNOWN])
