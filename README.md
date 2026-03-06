# Astrai Router

**Open-source intelligent LLM router** with Thompson Sampling, Berkeley ARBITRAGE, trading-style best execution, energy-aware routing, and privacy-preserving intelligence.

> Competitors (OpenRouter, Martian, Unify) are closed-source or VC-funded. This is the same production system that powers [Astrai](https://astrai-compute.fly.dev) — now MIT-licensed.

## Features

| Feature | Description |
|---------|-------------|
| **Thompson Sampling** | Self-learning model selection with Beta distribution priors |
| **ARBITRAGE** | Berkeley-paper advantage-aware switching between models |
| **Best Execution** | Trading-style venue scoring (latency, cost, quality, fill rate) |
| **Energy Oracle** | Research-based energy estimation (Joules, Wh, CO2) per request |
| **Task Classification** | Auto-detect task type (code, research, analysis, creative, etc.) |
| **Semantic Cache** | Embedding-based similarity matching — 50-90% savings on repeated queries |
| **Context Compression** | System dedup, whitespace normalization, old-turn summarization |
| **Shadow Mode** | Compare routing decisions with LLM-as-judge quality scoring |
| **Privacy-Preserving** | Stores patterns, not content. GDPR-compatible by design |
| **Auto-Learning** | Hierarchical scoring (global → workflow → step → user) with implicit feedback |
| **Pluggable Storage** | Memory (default), SQLite, or Postgres — no vendor lock-in |

## Install

```bash
pip install astrai-router

# With ML features (scikit-learn for quality learning)
pip install astrai-router[ml]

# With LiteLLM for multi-provider inference
pip install astrai-router[litellm]

# Everything
pip install astrai-router[all]
```

## Quick Start

### Classify a prompt

```python
from astrai_router import TaskClassifier

classifier = TaskClassifier()
result = classifier.classify("Write a Python function to sort a list")
print(result.task_type)       # "coding"
print(result.complexity)      # "medium"
print(result.requires_code)   # True
```

### Route a request

```python
from astrai_router import RoutingEngine, RoutingTier

engine = RoutingEngine()

# Get routing recommendation
recommendation = engine.route(
    task_type="coding",
    complexity="high",
    tier=RoutingTier.BALANCED,
)
print(recommendation)
```

### Estimate energy consumption

```python
from astrai_router import EnergyOracle

oracle = EnergyOracle()
estimate = oracle.estimate_energy(
    model_name="openai/gpt-4o",
    input_tokens=1000,
    output_tokens=500,
)
print(f"Energy: {estimate.total_joules:.2f}J")
print(f"CO2: {estimate.co2_grams:.4f}g")
```

### Compress conversation context

```python
from astrai_router import compress_messages

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "system", "content": "You are a helpful assistant."},  # duplicate
    {"role": "user", "content": "Hello " * 500},
    {"role": "assistant", "content": "Hi there! How can I help you today? " * 100},
    {"role": "user", "content": "What is Python?"},
]

compressed, manifest = compress_messages(messages, task_type="code")
if manifest:
    print(f"Compressed {manifest['original_tokens']} → {manifest['compressed_tokens']} tokens")
    print(f"Ratio: {manifest['compression_ratio']}x")
```

### Auto-learning routing

```python
from astrai_router import auto_route_request

result = auto_route_request(
    prompt="Analyze the Q3 earnings report",
    user_id="user-123",
    available_models=[
        {"model": "gpt-4o", "provider": "openai", "cost_per_1k": 5.0},
        {"model": "claude-sonnet-4-5", "provider": "anthropic", "cost_per_1k": 3.0},
        {"model": "llama-3.3-70b", "provider": "groq", "cost_per_1k": 0.59},
    ],
)
print(f"Task: {result['task_type']}")
print(f"Model: {result['recommended_model']}")
print(f"Reason: {result['selection_reason']}")
```

### Configure persistent storage

```python
from astrai_router import configure_storage

# SQLite (persists to disk)
configure_storage("sqlite", path="./router.db")

# Postgres (production)
configure_storage("postgres", dsn="postgresql://user:pass@host/db")
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Your Application                │
├─────────────────────────────────────────────────┤
│              astrai_router (this package)         │
│                                                   │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Classify  │→│  Route    │→│  Execute       │  │
│  │ (task     │  │ (Thompson│  │  (best exec,   │  │
│  │  type)    │  │  + ARBI- │  │  fallback      │  │
│  │           │  │  TRAGE)  │  │  chains)       │  │
│  └──────────┘  └──────────┘  └───────────────┘  │
│       │              │              │             │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Energy   │  │ Shadow   │  │  Telemetry     │  │
│  │ Oracle   │  │ Mode     │  │  + Learning    │  │
│  └──────────┘  └──────────┘  └───────────────┘  │
│                      │                            │
│              ┌───────────────┐                    │
│              │   Storage     │                    │
│              │ (Memory/SQL/  │                    │
│              │  Postgres)    │                    │
│              └───────────────┘                    │
└─────────────────────────────────────────────────┘
```

## Modules

| Module | Description | Lines |
|--------|-------------|-------|
| `engine` | Core routing with Thompson Sampling, GlimpRouter, venue scoring | ~600 |
| `classifier` | Task classification and provider health tracking | ~500 |
| `intelligence` | Privacy-preserving routing memory (SQLite) | ~770 |
| `unified` | Unified router combining all strategies | ~1000 |
| `arbitrage` | Berkeley ARBITRAGE paper implementation | ~430 |
| `execution` | Trading-style best execution engine | ~680 |
| `production` | Production router with logging | ~265 |
| `advanced` | Webhooks, fallback chains, smart recommendations | ~830 |
| `telemetry` | Request telemetry and analytics | ~480 |
| `smart_classify` | LLM-powered task classification | ~220 |
| `energy` | Energy consumption estimation | ~420 |
| `models` | Model configuration and pricing | ~170 |
| `learning` | Auto-learning with Thompson Sampling | ~1020 |
| `catalog` | OpenRouter model catalog sync | ~210 |
| `quality` | Quality learning with ML | ~300 |
| `cache` | Semantic caching with embeddings | ~480 |
| `compression` | Context compression for token savings | ~290 |
| `shadow` | Shadow mode quality comparison | ~730 |
| `validation` | Input validation and rate limiting | ~540 |
| `storage` | Pluggable storage (Memory/SQLite/Postgres) | ~430 |

## Research Papers

This router implements ideas from:
- [ARBITRAGE: Advantage-Aware Routing](https://arxiv.org/abs/2502.04343) (Berkeley, 2025)
- [From Prompts to Power](https://arxiv.org/abs/2312.11617) — Energy-aware inference
- Thompson Sampling for online learning (1933, rediscovered for LLM routing)
- Market microstructure theory (best execution, venue scoring)

## License

MIT — see [LICENSE](LICENSE).
