# market_maker_config.py
# Astrai Market Maker - Model Configuration (2026)

import os

# Model Providers with costs per 1M tokens
MODEL_PROVIDERS = {
    # OpenAI Models (2026)
    "openai/gpt-5.2": {"cost_per_mtok": 7.875, "type": "target"},
    "openai/gpt-5.2-pro": {"cost_per_mtok": 94.50, "type": "target"},
    "openai/gpt-5": {"cost_per_mtok": 5.625, "type": "target"},
    "openai/gpt-4o": {"cost_per_mtok": 5.00, "type": "target"},
    "openai/gpt-4o-mini": {"cost_per_mtok": 0.15, "type": "draft"},
    
    # Anthropic Models (2026)
    "anthropic/claude-opus-4-6": {"cost_per_mtok": 15.00, "type": "target"},
    "anthropic/claude-4.5-opus": {"cost_per_mtok": 15.00, "type": "target"},
    "anthropic/claude-sonnet-4-5": {"cost_per_mtok": 9.00, "type": "target"},
    "anthropic/claude-4.5-sonnet": {"cost_per_mtok": 9.00, "type": "target"},
    "anthropic/claude-3-5-sonnet-latest": {"cost_per_mtok": 3.00, "type": "target"},
    
    # Groq Models (Fast & Cheap)
    "groq/llama-3.3-70b-versatile": {"cost_per_mtok": 0.59, "type": "draft"},
    "groq/llama-3.1-70b-versatile": {"cost_per_mtok": 0.59, "type": "draft"},
    "groq/llama-3.1-8b-instant": {"cost_per_mtok": 0.05, "type": "draft"},
    "groq/mixtral-8x7b-32768": {"cost_per_mtok": 0.24, "type": "draft"},
    
    # Groq Llama 4 (2026)
    "groq/llama-4-maverick-17b-128e-instruct": {"cost_per_mtok": 0.375, "type": "draft"},
    "groq/llama-4-scout-17b-16e-instruct": {"cost_per_mtok": 0.19, "type": "draft"},

    # DeepSeek
    "deepseek/deepseek-chat": {"cost_per_mtok": 0.27, "type": "draft"},
    "deepseek/deepseek-reasoner": {"cost_per_mtok": 2.19, "type": "target"},

    # Google Gemini (2026)
    "google/gemini-2.5-flash": {"cost_per_mtok": 0.15, "type": "draft"},
    "google/gemini-2.5-pro": {"cost_per_mtok": 1.25, "type": "target"},

    # Moonshot Kimi (2026)
    "moonshot/kimi-2.5": {"cost_per_mtok": 0.50, "type": "draft"},

    # Together AI
    "together/llama-3.1-405b": {"cost_per_mtok": 3.50, "type": "target"},
}

# Dynamic Draft/Target pairs based on task type
MODEL_CONFIG = {
    "code": {
        "draft": "groq/llama-3.1-8b-instant",  # Fastest: ~50-100ms
        "target": "openai/gpt-5.2",  # Latest GPT for code
        "description": "Code generation, debugging, refactoring"
    },
    "writing": {
        "draft": "groq/llama-3.1-8b-instant",
        "target": "anthropic/claude-opus-4-5-20251101",  # Best for writing
        "description": "Creative writing, marketing, emails"
    },
    "reasoning": {
        "draft": "groq/llama-3.1-8b-instant",
        "target": "openai/o3-mini",  # Best for reasoning
        "description": "Math, logic, complex reasoning"
    },
    "general": {
        "draft": "groq/llama-3.1-8b-instant",  # Ultra-fast for simple queries
        "target": "openai/gpt-5.2",
        "description": "General queries, Q&A"
    },
    "analysis": {
        "draft": "groq/llama-3.1-8b-instant",
        "target": "anthropic/claude-opus-4-5-20251101",
        "description": "Data analysis, research"
    }
}

# Task classification keywords
TASK_KEYWORDS = {
    "code": ["code", "python", "javascript", "react", "debug", "css", "html", "function", 
             "class", "import", "api", "database", "sql", "typescript", "rust", "golang"],
    "writing": ["write", "summarize", "email", "blog", "post", "marketing", "story", 
                "article", "essay", "creative", "poem", "script"],
    "reasoning": ["calculate", "math", "solve", "equation", "proof", "logic", "reason",
                  "analyze", "deduce", "theorem", "probability"],
    "analysis": ["analyze", "compare", "evaluate", "research", "study", "data", 
                 "statistics", "trend", "report", "insight"]
}


def classify_task(prompt: str) -> str:
    """Classify the task type based on prompt content."""
    prompt_lower = prompt.lower()
    
    # Check each task type
    for task_type, keywords in TASK_KEYWORDS.items():
        if any(kw in prompt_lower for kw in keywords):
            return task_type
    
    return "general"


def get_model_pair(task_type: str) -> tuple:
    """Get the draft and target model for a task type."""
    config = MODEL_CONFIG.get(task_type, MODEL_CONFIG["general"])
    return config["draft"], config["target"]


def get_model_cost(model: str) -> float:
    """Get cost per 1M tokens for a model. Falls back to OpenRouter catalog."""
    cost = MODEL_PROVIDERS.get(model, {}).get("cost_per_mtok")
    if cost is not None:
        return cost
    # Dynamic fallback: check OpenRouter model catalog
    try:
        from .catalog import get_family_price_per_1m, normalize_family
        family = model.split("/", 1)[1] if "/" in model else model
        result = get_family_price_per_1m(normalize_family(family))
        if result:
            return round((result[0] + result[1]) / 2, 4)
    except Exception:
        pass
    return 1.0


def calculate_savings(draft_model: str, target_model: str, tokens: int, used_draft: bool) -> float:
    """Calculate cost savings from using draft model."""
    draft_cost = get_model_cost(draft_model)
    target_cost = get_model_cost(target_model)
    
    if used_draft:
        actual_cost = (tokens / 1_000_000) * draft_cost
        would_have_cost = (tokens / 1_000_000) * target_cost
        return would_have_cost - actual_cost
    return 0.0


def score_response(prompt: str, response: str) -> float:
    """
    Simple heuristic-based response quality scorer.
    Higher score = better quality.
    """
    if not response:
        return 0.0
    
    score = 0.0
    
    # Length score (normalized)
    score += min(len(response) / 2000.0, 1.0) * 0.3
    
    # Code block bonus
    if "```" in response:
        score += 0.25
    
    # Structured reasoning bonus
    if any(marker in response for marker in ["Step 1", "First,", "1.", "Conclusion", "Therefore"]):
        score += 0.2
    
    # Completeness indicators
    if response.strip().endswith(('.', '!', '?', '```')):
        score += 0.1
    
    # Relevance (simple keyword overlap)
    prompt_words = set(prompt.lower().split())
    response_words = set(response.lower().split())
    overlap = len(prompt_words & response_words) / max(len(prompt_words), 1)
    score += overlap * 0.15
    
    return round(min(score, 1.0), 4)
