"""
Basic routing example — classify a prompt and get energy estimates.

No API keys required. Pure local computation.
"""

from astrai_router import (
    TaskClassifier,
    EnergyOracle,
    compress_messages,
    auto_route_request,
    configure_storage,
)


def main():
    # ── 1. Classify a prompt ─────────────────────────────────────────────
    classifier = TaskClassifier()
    prompts = [
        "Write a Python function to calculate Fibonacci numbers",
        "Summarize the key points of this article about climate change",
        "What is 2 + 2?",
        "Create a marketing email for our new product launch",
    ]

    print("=== Task Classification ===")
    for prompt in prompts:
        result = classifier.classify(prompt)
        print(
            f"  {prompt[:50]:50s} → {result.task_type:12s} (complexity={result.complexity})"
        )

    # ── 2. Energy estimation ─────────────────────────────────────────────
    oracle = EnergyOracle()
    models = [
        "groq/llama-3.1-8b-instant",
        "openai/gpt-4o",
        "anthropic/claude-opus-4-6",
    ]

    print("\n=== Energy Estimates (1000 input + 500 output tokens) ===")
    for model in models:
        est = oracle.estimate_energy(model, input_tokens=1000, output_tokens=500)
        print(
            f"  {model:40s} → {est.total_joules:8.3f}J  {est.co2_grams:.6f}g CO2  (tier={est.model_tier})"
        )

    # ── 3. Energy savings from draft vs ultra ────────────────────────────
    gain = oracle.get_efficiency_alpha(input_tokens=2000, output_tokens=1000)
    print("\n=== Draft→Ultra Savings (2000in + 1000out) ===")
    print(f"  Joules saved: {gain.joules_saved:.2f}")
    print(f"  Efficiency:   {gain.efficiency_gain_pct:.1f}%")
    print(f"  CO2 saved:    {gain.co2_grams_saved:.6f}g")

    # ── 4. Context compression ───────────────────────────────────────────
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "system", "content": "You are a helpful coding assistant."},  # dupe
        {"role": "user", "content": "Write a function. " * 200},
        {
            "role": "assistant",
            "content": "Here's the function:\n```python\ndef foo():\n    pass\n```\n"
            * 50,
        },
        {"role": "user", "content": "Now add error handling"},
    ]

    compressed, manifest = compress_messages(messages, task_type="code")
    print("\n=== Context Compression ===")
    if manifest:
        print(
            f"  {manifest['original_tokens']} → {manifest['compressed_tokens']} tokens ({manifest['compression_ratio']}x)"
        )
        print(f"  Techniques: {manifest['techniques_applied']}")
    else:
        print("  (no compression needed)")

    # ── 5. Auto-learning route ───────────────────────────────────────────
    result = auto_route_request(
        prompt="Analyze the performance of NVIDIA stock over the past quarter",
        user_id="demo-user-001",
        available_models=[
            {
                "model": "gpt-4o",
                "provider": "openai",
                "cost_per_1k": 5.0,
                "avg_latency_ms": 800,
            },
            {
                "model": "claude-sonnet-4-5",
                "provider": "anthropic",
                "cost_per_1k": 3.0,
                "avg_latency_ms": 1200,
            },
            {
                "model": "llama-3.3-70b",
                "provider": "groq",
                "cost_per_1k": 0.59,
                "avg_latency_ms": 200,
            },
        ],
    )
    print("\n=== Auto-Learning Route ===")
    print(
        f"  Task:       {result['task_type']} (confidence={result['task_confidence']})"
    )
    print(f"  Model:      {result['recommended_model']}")
    print(f"  Reason:     {result['selection_reason']}")
    print(f"  Exploring:  {result['explore_mode']}")

    # ── 6. Storage demo ──────────────────────────────────────────────────
    print("\n=== Storage (SQLite) ===")
    configure_storage("sqlite", path="/tmp/astrai_demo.db")
    from astrai_router import get_storage

    storage = get_storage()
    storage.insert("demo_table", {"key": "hello", "value": "world"})
    rows = storage.get("demo_table", filters={"key": "hello"})
    print(f"  Inserted and retrieved: {rows}")


if __name__ == "__main__":
    main()
