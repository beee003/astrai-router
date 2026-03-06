"""
ARBITRAGE Router - Learning-Based Model Routing

Implements the Berkeley ARBITRAGE paper approach:
1. Generate Mode: Parallel inference on draft + target models, collect training data
2. Router Mode: Use trained classifier to route requests intelligently

Draft Model: Fast, cheap (e.g., Llama-3-70b via Fireworks)
Target Model: High quality, expensive (e.g., Claude-3-Opus)
"""

import os
import time
import asyncio
import pickle
import logging
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from litellm import acompletion

from .storage import get_storage

logger = logging.getLogger(__name__)

# Configuration
ARBITRAGE_MODE = os.getenv(
    "ARBITRAGE_MODE", "disabled"
)  # "disabled", "generate", "router"
DRAFT_MODEL = os.getenv(
    "ARBITRAGE_DRAFT_MODEL", "groq/llama-4-maverick-17b-128e-instruct"
)  # Llama 4 via Groq
TARGET_MODEL = os.getenv(
    "ARBITRAGE_TARGET_MODEL", "openai/gpt-5.2"
)  # GPT-5.2 high quality target
ROUTER_MODEL_PATH = os.getenv("ARBITRAGE_ROUTER_PATH", "arbitrage_router_v1.pkl")
ROUTER_THRESHOLD = float(os.getenv("ARBITRAGE_ROUTER_THRESHOLD", "0.5"))

# Cost estimates per 1M tokens (2026 pricing)
MODEL_COSTS = {
    # Draft Models (Cheap, Fast)
    "groq/llama-4-maverick-17b-128e-instruct": 0.375,  # Llama 4 Maverick
    "groq/llama-4-scout-17b-16e-instruct": 0.19,  # Llama 4 Scout
    "groq/llama-3.3-70b-versatile": 0.59,
    "deepseek/deepseek-chat": 0.27,
    # Target Models (High Quality)
    "openai/gpt-5.2": 7.875,  # GPT-5.2
    "openai/gpt-5.2-pro": 94.50,  # GPT-5.2 Pro
    "openai/gpt-5": 5.625,
    "openai/gpt-4o": 5.00,
    "openai/gpt-4o-mini": 0.15,
    "anthropic/claude-4.5-opus": 15.00,
    "anthropic/claude-4.5-sonnet": 9.00,
    "anthropic/claude-3-5-sonnet-latest": 3.00,
}


class ArbitrageMode(Enum):
    DISABLED = "disabled"
    GENERATE = "generate"  # Parallel inference, collect training data
    ROUTER = "router"  # Use trained model to route


@dataclass
class ArbitrageResult:
    """Result from arbitrage routing decision"""

    chosen_model: str
    chosen_response: str
    chosen_latency_ms: float
    chosen_cost_usd: float
    routing_reason: str
    # For logging
    draft_response: Optional[str] = None
    draft_latency_ms: Optional[float] = None
    draft_cost_usd: Optional[float] = None
    target_response: Optional[str] = None
    target_latency_ms: Optional[float] = None
    target_cost_usd: Optional[float] = None
    router_probability: Optional[float] = None


class ArbitrageRouter:
    """
    Learning-based router that decides between draft and target models.
    """

    def __init__(self):
        self.mode = ArbitrageMode(ARBITRAGE_MODE)
        self.draft_model = DRAFT_MODEL
        self.target_model = TARGET_MODEL
        self.router_model = None
        self.router_threshold = ROUTER_THRESHOLD

        # Load router model if in router mode
        if self.mode == ArbitrageMode.ROUTER:
            self._load_router_model()

    def _load_router_model(self):
        """Load the trained classifier from disk"""
        try:
            if os.path.exists(ROUTER_MODEL_PATH):
                with open(ROUTER_MODEL_PATH, "rb") as f:
                    self.router_model = pickle.load(f)
                logger.info(f"ARBITRAGE: Loaded router model from {ROUTER_MODEL_PATH}")
            else:
                logger.warning(
                    f"ARBITRAGE: Router model not found at {ROUTER_MODEL_PATH}, falling back to target model"
                )
                self.mode = ArbitrageMode.DISABLED
        except Exception as e:
            logger.error(f"ARBITRAGE: Failed to load router model: {e}")
            self.mode = ArbitrageMode.DISABLED

    def _estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Estimate cost in USD for a model call"""
        cost_per_1m = MODEL_COSTS.get(model, 1.0)
        total_tokens = input_tokens + output_tokens
        return (total_tokens / 1_000_000) * cost_per_1m

    def _extract_features(self, prompt: str) -> Dict[str, Any]:
        """
        Extract features from prompt for router classification.
        These features help predict if draft model is sufficient.
        """
        # Basic features
        prompt_length = len(prompt)
        word_count = len(prompt.split())

        # Code detection
        code_keywords = [
            "python",
            "javascript",
            "function",
            "def ",
            "class ",
            "import ",
            "return",
            "const ",
            "let ",
            "var ",
            "```",
            "async",
            "await",
            "try:",
            "except:",
            "if __name__",
        ]
        num_code_keywords = sum(
            1 for kw in code_keywords if kw.lower() in prompt.lower()
        )
        has_code = num_code_keywords > 2 or "```" in prompt

        # Question detection
        is_question = prompt.strip().endswith("?")

        # Complexity indicators
        has_multiple_questions = prompt.count("?") > 1
        has_numbered_list = any(
            f"{i}." in prompt or f"{i})" in prompt for i in range(1, 10)
        )

        # Task type detection
        task_keywords = {
            "summarize": ["summarize", "summary", "tldr", "brief"],
            "explain": ["explain", "what is", "how does", "why"],
            "creative": ["write", "create", "generate", "story", "poem"],
            "analysis": ["analyze", "compare", "evaluate", "assess"],
            "code": ["code", "implement", "function", "debug", "fix"],
        }

        detected_tasks = []
        for task, keywords in task_keywords.items():
            if any(kw in prompt.lower() for kw in keywords):
                detected_tasks.append(task)

        return {
            "prompt_length": prompt_length,
            "word_count": word_count,
            "num_code_keywords": num_code_keywords,
            "has_code": has_code,
            "is_question": is_question,
            "has_multiple_questions": has_multiple_questions,
            "has_numbered_list": has_numbered_list,
            "detected_tasks": detected_tasks,
            "is_simple_question": is_question and word_count < 20 and not has_code,
        }

    def _features_to_vector(self, features: Dict[str, Any]) -> List[float]:
        """Convert feature dict to numeric vector for classifier"""
        return [
            features["prompt_length"],
            features["word_count"],
            features["num_code_keywords"],
            1.0 if features["has_code"] else 0.0,
            1.0 if features["is_question"] else 0.0,
            1.0 if features["has_multiple_questions"] else 0.0,
            1.0 if features["has_numbered_list"] else 0.0,
            1.0 if features["is_simple_question"] else 0.0,
            1.0 if "code" in features["detected_tasks"] else 0.0,
            1.0 if "creative" in features["detected_tasks"] else 0.0,
            1.0 if "analysis" in features["detected_tasks"] else 0.0,
        ]

    async def _call_model(
        self, model: str, messages: List[Dict], temperature: float = 0.7
    ) -> Tuple[str, float, int, int]:
        """
        Call a model and return (response, latency_ms, input_tokens, output_tokens)
        """
        start = time.monotonic()

        response = await acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=False,
        )

        latency_ms = (time.monotonic() - start) * 1000
        content = response.choices[0].message.content
        input_tokens = (
            response.usage.prompt_tokens
            if hasattr(response, "usage")
            else len(str(messages)) // 4
        )
        output_tokens = (
            response.usage.completion_tokens
            if hasattr(response, "usage")
            else len(content) // 4
        )

        return content, latency_ms, input_tokens, output_tokens

    async def route(
        self,
        messages: List[Dict],
        user_id: str,
        temperature: float = 0.7,
        request_id: Optional[str] = None,
    ) -> ArbitrageResult:
        """
        Main routing function. Behavior depends on mode:

        - DISABLED: Just call target model directly
        - GENERATE: Call both models in parallel, return target, log both
        - ROUTER: Use classifier to decide which model to call
        """
        prompt = " ".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )
        features = self._extract_features(prompt)

        # === DISABLED MODE ===
        if self.mode == ArbitrageMode.DISABLED:
            content, latency_ms, input_tokens, output_tokens = await self._call_model(
                self.target_model, messages, temperature
            )
            cost = self._estimate_cost(self.target_model, input_tokens, output_tokens)

            return ArbitrageResult(
                chosen_model=self.target_model,
                chosen_response=content,
                chosen_latency_ms=latency_ms,
                chosen_cost_usd=cost,
                routing_reason="ARBITRAGE disabled - using target model",
            )

        # === GENERATE MODE (Parallel Inference) ===
        if self.mode == ArbitrageMode.GENERATE:
            # Call both models in parallel
            draft_task = self._call_model(self.draft_model, messages, temperature)
            target_task = self._call_model(self.target_model, messages, temperature)

            try:
                (
                    (draft_content, draft_latency, draft_in, draft_out),
                    (target_content, target_latency, target_in, target_out),
                ) = await asyncio.gather(draft_task, target_task)
            except Exception as e:
                # If parallel fails, fall back to target only
                logger.warning(f"ARBITRAGE parallel inference failed: {e}")
                (
                    content,
                    latency_ms,
                    input_tokens,
                    output_tokens,
                ) = await self._call_model(self.target_model, messages, temperature)
                cost = self._estimate_cost(
                    self.target_model, input_tokens, output_tokens
                )
                return ArbitrageResult(
                    chosen_model=self.target_model,
                    chosen_response=content,
                    chosen_latency_ms=latency_ms,
                    chosen_cost_usd=cost,
                    routing_reason=f"ARBITRAGE parallel failed: {e}",
                )

            draft_cost = self._estimate_cost(self.draft_model, draft_in, draft_out)
            target_cost = self._estimate_cost(self.target_model, target_in, target_out)

            # Log training data (async, non-blocking)
            asyncio.create_task(
                self._log_training_data(
                    user_id=user_id,
                    prompt=prompt,
                    prompt_length=features["prompt_length"],
                    draft_model=self.draft_model,
                    draft_response=draft_content,
                    draft_latency_ms=int(draft_latency),
                    draft_cost_usd=draft_cost,
                    target_model=self.target_model,
                    target_response=target_content,
                    target_latency_ms=int(target_latency),
                    target_cost_usd=target_cost,
                    final_choice="target",
                    request_id=request_id,
                )
            )

            # Always return target in generate mode
            return ArbitrageResult(
                chosen_model=self.target_model,
                chosen_response=target_content,
                chosen_latency_ms=target_latency,
                chosen_cost_usd=target_cost,
                routing_reason="ARBITRAGE generate mode - collected training data",
                draft_response=draft_content,
                draft_latency_ms=draft_latency,
                draft_cost_usd=draft_cost,
                target_response=target_content,
                target_latency_ms=target_latency,
                target_cost_usd=target_cost,
            )

        # === ROUTER MODE (Learned Routing) ===
        if self.mode == ArbitrageMode.ROUTER and self.router_model is not None:
            # Get prediction from classifier
            feature_vector = [self._features_to_vector(features)]

            try:
                prob_target_needed = self.router_model.predict_proba(feature_vector)[0][
                    1
                ]
            except Exception as e:
                logger.warning(f"ARBITRAGE router prediction failed: {e}")
                prob_target_needed = 1.0  # Default to target on error

            # Make routing decision
            use_target = prob_target_needed > self.router_threshold
            chosen_model = self.target_model if use_target else self.draft_model

            # Call the chosen model
            content, latency_ms, input_tokens, output_tokens = await self._call_model(
                chosen_model, messages, temperature
            )
            cost = self._estimate_cost(chosen_model, input_tokens, output_tokens)

            # Log the routing decision
            asyncio.create_task(
                self._log_training_data(
                    user_id=user_id,
                    prompt=prompt,
                    prompt_length=features["prompt_length"],
                    draft_model=self.draft_model,
                    draft_response=content if not use_target else None,
                    draft_latency_ms=int(latency_ms) if not use_target else None,
                    draft_cost_usd=cost if not use_target else None,
                    target_model=self.target_model,
                    target_response=content if use_target else None,
                    target_latency_ms=int(latency_ms) if use_target else None,
                    target_cost_usd=cost if use_target else None,
                    final_choice="target" if use_target else "draft",
                    request_id=request_id,
                    strategy="router",
                )
            )

            return ArbitrageResult(
                chosen_model=chosen_model,
                chosen_response=content,
                chosen_latency_ms=latency_ms,
                chosen_cost_usd=cost,
                routing_reason=f"ARBITRAGE router: P(target)={prob_target_needed:.2f}, chose {'target' if use_target else 'draft'}",
                router_probability=prob_target_needed,
            )

        # Fallback: just use target
        content, latency_ms, input_tokens, output_tokens = await self._call_model(
            self.target_model, messages, temperature
        )
        cost = self._estimate_cost(self.target_model, input_tokens, output_tokens)

        return ArbitrageResult(
            chosen_model=self.target_model,
            chosen_response=content,
            chosen_latency_ms=latency_ms,
            chosen_cost_usd=cost,
            routing_reason="ARBITRAGE fallback - using target model",
        )

    async def _log_training_data(
        self,
        user_id: str,
        prompt: str,
        prompt_length: int,
        draft_model: str,
        draft_response: Optional[str],
        draft_latency_ms: Optional[int],
        draft_cost_usd: Optional[float],
        target_model: str,
        target_response: Optional[str],
        target_latency_ms: Optional[int],
        target_cost_usd: Optional[float],
        final_choice: str,
        request_id: Optional[str] = None,
        strategy: str = "generate",
    ):
        """Log training data to storage backend for future model training"""
        try:
            storage = get_storage()
            storage.insert(
                "arbitrage_training_data",
                {
                    "user_id": user_id,
                    "prompt": prompt[:10000],  # Truncate very long prompts
                    "prompt_length": prompt_length,
                    "draft_model": draft_model,
                    "draft_response": draft_response[:10000]
                    if draft_response
                    else None,
                    "draft_latency_ms": draft_latency_ms,
                    "draft_cost_usd": draft_cost_usd,
                    "target_model": target_model,
                    "target_response": target_response[:10000]
                    if target_response
                    else None,
                    "target_latency_ms": target_latency_ms,
                    "target_cost_usd": target_cost_usd,
                    "final_choice": final_choice,
                    "request_id": request_id,
                    "strategy": strategy,
                    "was_regenerated": False,  # Will be updated if user regenerates
                },
            )
            logger.debug(f"ARBITRAGE: Logged training data (choice={final_choice})")
        except Exception as e:
            logger.warning(f"ARBITRAGE: Failed to log training data: {e}")

    async def mark_regenerated(self, request_id: str):
        """Mark a request as regenerated (user wasn't satisfied with response)"""
        if not request_id:
            return

        try:
            storage = get_storage()
            storage.update(
                "arbitrage_training_data",
                {"was_regenerated": True},
                {"request_id": request_id},
            )
            logger.debug(f"ARBITRAGE: Marked request {request_id} as regenerated")
        except Exception as e:
            logger.warning(f"ARBITRAGE: Failed to mark regenerated: {e}")


# Global instance
ARBITRAGE_ROUTER = ArbitrageRouter()


def get_arbitrage_stats() -> Dict[str, Any]:
    """Get current arbitrage router statistics"""
    return {
        "mode": ARBITRAGE_ROUTER.mode.value,
        "draft_model": ARBITRAGE_ROUTER.draft_model,
        "target_model": ARBITRAGE_ROUTER.target_model,
        "router_loaded": ARBITRAGE_ROUTER.router_model is not None,
        "router_threshold": ARBITRAGE_ROUTER.router_threshold,
    }
