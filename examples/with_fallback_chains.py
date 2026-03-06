"""
Fallback chain example — configure multi-step routing with automatic failover.

Demonstrates:
- Provider health tracking with circuit breakers
- Automatic fallback when providers fail
- Quality scoring and learning
"""

from astrai_router import (
    ProviderHealthTracker,
    EnergyOracle,
    configure_storage,
)


def main():
    # ── Configure persistent storage ─────────────────────────────────────
    configure_storage("sqlite", path="/tmp/fallback_demo.db")

    # ── Set up provider health tracking ──────────────────────────────────
    tracker = ProviderHealthTracker(
        halflife_sec=300,
        min_score=0.5,
        half_open_score=0.65,
        cooldown_sec=60,
    )

    # Simulate some provider history
    print("=== Simulating Provider Health ===")
    providers = ["openai", "anthropic", "groq", "together"]

    # OpenAI: healthy
    for _ in range(20):
        tracker.record_success("openai", latency_ms=350.0)

    # Anthropic: mostly healthy, one failure
    for _ in range(15):
        tracker.record_success("anthropic", latency_ms=500.0)
    tracker.record_failure("anthropic", "timeout")

    # Groq: very fast
    for _ in range(25):
        tracker.record_success("groq", latency_ms=80.0)

    # Together: having issues
    for _ in range(5):
        tracker.record_success("together", latency_ms=1200.0)
    for _ in range(8):
        tracker.record_failure("together", "503_error")

    # ── Check health of each provider ────────────────────────────────────
    print("\nProvider Health Status:")
    for provider in providers:
        snap = tracker.snapshot(provider)
        print(
            f"  {provider:12s} → {snap.status:10s} "
            f"score={snap.score:.3f} "
            f"p50={snap.latency_p50_ms:.0f}ms "
            f"breaker={snap.breaker_state.value}"
        )

    # ── Build fallback chain based on health ─────────────────────────────
    print("\n=== Fallback Chain (auto-generated) ===")
    healthy_providers = [
        (p, tracker.snapshot(p)) for p in providers if tracker.is_healthy(p)
    ]

    # Sort by score (best first)
    healthy_providers.sort(key=lambda x: x[1].score, reverse=True)

    for i, (provider, snap) in enumerate(healthy_providers):
        print(
            f"  Step {i + 1}: {provider} (score={snap.score:.3f}, p50={snap.latency_p50_ms:.0f}ms)"
        )

    unhealthy = [p for p in providers if not tracker.is_healthy(p)]
    if unhealthy:
        print(f"\n  Excluded (circuit open): {', '.join(unhealthy)}")

    # ── Energy comparison across the chain ───────────────────────────────
    oracle = EnergyOracle()
    print("\n=== Energy per Provider (1000 in + 500 out tokens) ===")

    model_map = {
        "openai": "openai/gpt-4o",
        "anthropic": "anthropic/claude-sonnet-4-5-20250929",
        "groq": "groq/llama-3.1-8b-instant",
        "together": "together/llama-3.1-405b",
    }

    for provider, _ in healthy_providers:
        model = model_map.get(provider, "openai/gpt-4o")
        est = oracle.estimate_energy(model, 1000, 500)
        energy_score = oracle.get_energy_score(model)
        print(
            f"  {provider:12s} ({model:40s}) → {est.total_joules:.3f}J  "
            f"energy_score={energy_score:.3f}"
        )


if __name__ == "__main__":
    main()
