# production.py
# Astrai Market Maker - Production Router
# Uses trained ML models to intelligently route requests
# Falls back to robust heuristics when ML classifiers are not available.

import os
import time
import logging
from typing import Dict, Optional, Tuple, List

try:
    import joblib
except ImportError:
    joblib = None

try:
    from litellm import acompletion
except ImportError:
    acompletion = None

from .storage import get_storage
from .models import (
    classify_task,
    get_model_pair,
    get_model_cost,
    score_response,
    MODEL_CONFIG,
)

logger = logging.getLogger(__name__)

# Router threshold - higher = more requests go to draft
ROUTER_THRESHOLD = float(os.getenv("ROUTER_THRESHOLD", "0.5"))

# Load trained routers at startup
ROUTERS = {}
_ml_router_available = False
if joblib is not None:
    for task_type in MODEL_CONFIG.keys():
        path = f"training/models/router_{task_type}.pkl"
        if os.path.exists(path):
            try:
                ROUTERS[task_type] = joblib.load(path)
                _ml_router_available = True
                logger.info(f"Loaded ML router for task type: {task_type}")
            except Exception as e:
                logger.warning(f"Failed to load ML router for {task_type}: {e}")

if not _ml_router_available:
    logger.info(
        "No ML routers found -- using heuristic escalation (this is normal for fresh deployments)"
    )


def extract_features(prompt: str, draft_score: float = 0.0) -> List[float]:
    """Extract features from prompt for router prediction."""
    prompt_lower = prompt.lower()

    # Length features
    prompt_length = len(prompt)
    word_count = len(prompt.split())

    # Code indicators
    code_keywords = [
        "code",
        "python",
        "function",
        "class",
        "import",
        "```",
        "debug",
        "error",
        "api",
        "database",
    ]
    num_code_keywords = sum(1 for kw in code_keywords if kw in prompt_lower)
    has_code = 1.0 if num_code_keywords > 2 or "```" in prompt else 0.0

    # Complexity indicators
    complexity_keywords = [
        "explain",
        "analyze",
        "compare",
        "evaluate",
        "design",
        "implement",
        "optimize",
        "complex",
    ]
    num_complexity = sum(1 for kw in complexity_keywords if kw in prompt_lower)

    # Question indicators
    is_question = 1.0 if prompt.strip().endswith("?") else 0.0
    num_questions = prompt.count("?")

    # Task type indicators
    is_simple = 1.0 if is_question and word_count < 20 and not has_code else 0.0
    is_creative = (
        1.0 if any(kw in prompt_lower for kw in ["write", "create", "story"]) else 0.0
    )
    is_math = (
        1.0
        if any(kw in prompt_lower for kw in ["calculate", "solve", "equation"])
        else 0.0
    )

    return [
        prompt_length,
        word_count,
        num_code_keywords,
        has_code,
        num_complexity,
        is_question,
        num_questions,
        is_simple,
        is_creative,
        is_math,
        draft_score,
    ]


async def call_model(
    model: str, messages: list, temperature: float = 0.7
) -> Tuple[str, float, int]:
    """Call a single model and return response, latency, and token count."""
    if acompletion is None:
        raise RuntimeError("litellm is required for model execution")

    start = time.monotonic()
    try:
        response = await acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=False,
        )
        latency_ms = (time.monotonic() - start) * 1000
        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if hasattr(response, "usage") else 0
        return content, latency_ms, tokens
    except Exception as e:
        latency_ms = (time.monotonic() - start) * 1000
        return f"Error: {str(e)}", latency_ms, 0


def predict_escalation(prompt: str, task_type: str, draft_score: float = 0.0) -> float:
    """
    Predict the probability that we need to escalate to the target model.
    Returns a probability between 0 and 1.
    """
    # If we have a trained router, use it
    if task_type in ROUTERS:
        try:
            features = [extract_features(prompt, draft_score)]
            prob = ROUTERS[task_type].predict_proba(features)[0][1]
            return prob
        except Exception as e:
            print(f"Warning: Router prediction failed: {e}")

    # Heuristic-based prediction (robust fallback for when ML models aren't trained yet)
    features = extract_features(prompt, draft_score)
    (
        prompt_length,
        word_count,
        num_code,
        has_code,
        num_complexity,
        is_question,
        num_questions,
        is_simple,
        is_creative,
        is_math,
        ds,
    ) = features

    prob = 0.3  # Base probability of needing target

    # --- Features that REDUCE escalation (draft is sufficient) ---
    if is_simple:
        prob -= 0.2  # Short simple questions handled well by draft
    if word_count < 15 and not has_code:
        prob -= 0.1  # Very short non-code prompts

    # --- Features that INCREASE escalation (target likely needed) ---
    if has_code or num_code > 3:
        prob += 0.3  # Code generation/review needs stronger model
    if num_complexity > 2:
        prob += 0.25  # Multiple complexity keywords = hard task
    elif num_complexity > 0:
        prob += 0.15  # Some complexity
    if is_math:
        prob += 0.25  # Math/proofs need reasoning
    if word_count > 300:
        prob += 0.2  # Very long prompts = complex context
    elif word_count > 150:
        prob += 0.1
    if num_questions > 2:
        prob += 0.15  # Many questions = multi-part task
    elif num_questions > 1:
        prob += 0.1

    # --- Draft quality signal: if draft already scored poorly, escalate ---
    if ds > 0 and ds < 0.4:
        prob += 0.3  # Draft response was weak, definitely escalate
    elif ds > 0 and ds < 0.6:
        prob += 0.15  # Draft was mediocre
    elif ds >= 0.8:
        prob -= 0.15  # Draft was good, probably fine

    return max(0.0, min(1.0, prob))


async def execute_with_router(
    prompt: str,
    messages: list,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    temperature: float = 0.7,
    force_target: bool = False,
) -> Dict:
    """
    Execute inference with intelligent routing.

    1. Classify the task
    2. Get draft response
    3. Predict if escalation is needed
    4. Route to draft or target based on prediction
    5. Log results
    """
    # 1. Classify task and get model pair
    task_type = classify_task(prompt)
    draft_model, target_model = get_model_pair(task_type)

    # 2. Get draft response first
    draft_response, draft_latency, draft_tokens = await call_model(
        draft_model, messages, temperature
    )

    # 3. Score draft response
    draft_score = score_response(prompt, draft_response)

    # 4. Predict escalation probability
    escalation_prob = predict_escalation(prompt, task_type, draft_score)

    # 5. Route decision
    if force_target or escalation_prob > ROUTER_THRESHOLD:
        # Escalate to target model
        target_response, target_latency, target_tokens = await call_model(
            target_model, messages, temperature
        )
        final_response = target_response
        model_used = target_model
        was_escalated = True
        total_latency = draft_latency + target_latency
        total_tokens = draft_tokens + target_tokens
    else:
        # Use draft response
        final_response = draft_response
        model_used = draft_model
        was_escalated = False
        total_latency = draft_latency
        total_tokens = draft_tokens

    # 6. Calculate costs and savings
    draft_cost = (draft_tokens / 1_000_000) * get_model_cost(draft_model)
    target_cost_would_be = (total_tokens / 1_000_000) * get_model_cost(target_model)
    actual_cost = (total_tokens / 1_000_000) * get_model_cost(model_used)

    if was_escalated:
        # We paid for both draft and target
        actual_cost = draft_cost + (
            (target_tokens / 1_000_000) * get_model_cost(target_model)
        )
        cost_saved = 0.0
    else:
        # We only paid for draft
        cost_saved = target_cost_would_be - actual_cost

    # 7. Log to production_logs via storage backend
    storage = get_storage()
    try:
        log_record = {
            "user_id": user_id,
            "task_type": task_type,
            "prompt_length": len(prompt),
            "model_used": model_used,
            "was_escalated": was_escalated,
            "escalation_probability": escalation_prob,
            "cost_usd": actual_cost,
            "cost_saved_usd": cost_saved,
            "retail_price_usd": target_cost_would_be,
            "latency_ms": int(total_latency),
            "tokens_used": total_tokens,
            "request_id": request_id,
            "strategy": "router",
        }
        storage.insert("production_logs", log_record)
    except Exception as e:
        print(f"Warning: Router: Failed to log production data: {e}")

    # 8. Return result
    return {
        "response": final_response,
        "model_used": model_used,
        "task_type": task_type,
        "routing": {
            "was_escalated": was_escalated,
            "escalation_probability": round(escalation_prob, 3),
            "threshold": ROUTER_THRESHOLD,
            "draft_model": draft_model,
            "target_model": target_model,
            "draft_score": draft_score,
        },
        "costs": {
            "actual_cost_usd": round(actual_cost, 6),
            "would_have_cost_usd": round(target_cost_would_be, 6),
            "saved_usd": round(cost_saved, 6),
        },
        "latency_ms": int(total_latency),
        "tokens_used": total_tokens,
    }
