# execution.py
# Astrai Best Execution Engine - Trading-Style AI Inference Routing
#
# Implements the ARBITRAGE paper's approach:
# 1. Glimpse Signal: Early exit if query is too complex for draft model
# 2. Speculative Step: Try cheap model first
# 3. Arbitrage Verification: Verify draft quality before "selling" to user

import time
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import litellm
except ImportError:
    litellm = None

from .models import get_model_cost
from .energy import ENERGY_ORACLE
from .unified import MODEL_REGISTRY, route_request

# Entropy threshold - if first token entropy > this, skip draft model
ENTROPY_THRESHOLD = 2.5  # Higher = more uncertain, needs target model

# Verification threshold - confidence required to accept draft (lower = accept more drafts)
VERIFICATION_THRESHOLD = 0.5

# Cost constants (per 1M tokens)
DRAFT_COST = 0.05  # Groq Llama 3.1 8B
TARGET_COST = 7.88  # GPT-5.2


# Default green factor for energy-aware routing
DEFAULT_GREEN_FACTOR = 0.01  # Higher = more aggressive energy savings


@dataclass
class TradeResult:
    """Result of an execution trade."""

    response: str
    model_used: str
    actual_cost: float
    market_value: float  # What we charge user
    spread_captured: float
    execution_path: str  # "glimpse_exit" | "draft_accepted" | "target_executed"
    latency_ms: float
    tokens_used: int
    input_tokens: int = 0
    output_tokens: int = 0
    verification_score: Optional[float] = None
    energy_joules: Optional[float] = None
    energy_saved_joules: Optional[float] = None
    co2_grams: Optional[float] = None


@dataclass
class BlotterEntry:
    """Trade log entry for the blotter."""

    timestamp: float
    request_id: str
    task_type: str
    execution_path: str
    draft_model: str
    target_model: str
    actual_cost: float
    market_value: float
    spread_captured: float
    spread_bps: float  # Basis points
    latency_ms: float
    tokens: int
    energy_joules: float = 0.0
    energy_saved_joules: float = 0.0
    co2_grams: float = 0.0


class Blotter:
    """Trade blotter for logging execution economics."""

    def __init__(self):
        self.trades: List[BlotterEntry] = []
        self.total_spread_captured = 0.0
        self.total_volume = 0.0
        self.total_energy_joules = 0.0
        self.total_energy_saved_joules = 0.0
        self.total_co2_grams = 0.0

    def log_trade(
        self,
        request_id: str,
        task_type: str,
        execution_path: str,
        draft_model: str,
        target_model: str,
        actual_cost: float,
        market_value: float,
        latency_ms: float,
        tokens: int,
        energy_joules: float = 0.0,
        energy_saved_joules: float = 0.0,
        co2_grams: float = 0.0,
    ):
        spread = market_value - actual_cost
        spread_bps = (spread / market_value * 10000) if market_value > 0 else 0

        entry = BlotterEntry(
            timestamp=time.time(),
            request_id=request_id,
            task_type=task_type,
            execution_path=execution_path,
            draft_model=draft_model,
            target_model=target_model,
            actual_cost=actual_cost,
            market_value=market_value,
            spread_captured=spread,
            spread_bps=spread_bps,
            latency_ms=latency_ms,
            tokens=tokens,
            energy_joules=energy_joules,
            energy_saved_joules=energy_saved_joules,
            co2_grams=co2_grams,
        )

        self.trades.append(entry)
        self.total_spread_captured += spread
        self.total_volume += market_value
        self.total_energy_joules += energy_joules
        self.total_energy_saved_joules += energy_saved_joules
        self.total_co2_grams += co2_grams

        print(
            f"TRADE: {execution_path} | Cost: ${actual_cost:.6f} | Value: ${market_value:.6f} | Spread: {spread_bps:.0f}bps | Energy: {energy_joules:.2f}J"
        )

        return entry

    def get_stats(self) -> Dict:
        if not self.trades:
            return {
                "trades": 0,
                "total_spread": 0,
                "avg_spread_bps": 0,
                "total_energy_joules": 0,
            }

        return {
            "trades": len(self.trades),
            "total_spread": self.total_spread_captured,
            "total_volume": self.total_volume,
            "avg_spread_bps": sum(t.spread_bps for t in self.trades) / len(self.trades),
            "draft_rate": sum(
                1 for t in self.trades if t.execution_path == "draft_accepted"
            )
            / len(self.trades),
            "glimpse_exit_rate": sum(
                1 for t in self.trades if t.execution_path == "glimpse_exit"
            )
            / len(self.trades),
            "total_energy_joules": self.total_energy_joules,
            "total_energy_saved_joules": self.total_energy_saved_joules,
            "total_co2_grams": self.total_co2_grams,
            "energy_efficiency_pct": (
                self.total_energy_saved_joules
                / (self.total_energy_joules + self.total_energy_saved_joules)
                * 100
            )
            if (self.total_energy_joules + self.total_energy_saved_joules) > 0
            else 0,
        }


class Oracle:
    """
    Oracle for verifying draft model responses.
    Uses semantic similarity to determine if draft is "good enough".
    """

    def __init__(self):
        self.verification_cache = {}

    async def verify(
        self, prompt: str, draft_response: str, task_type: str
    ) -> Tuple[bool, float]:
        """
        Verify if draft response is acceptable quality.

        Returns: (is_verified, confidence_score)
        """
        # Quick heuristics first - but allow short responses for simple queries
        response_len = len(draft_response.strip())
        prompt_len = len(prompt.strip())

        # Empty response is always bad
        if response_len == 0:
            return False, 0.0

        # Check for error indicators
        error_indicators = [
            "i don't know",
            "i cannot",
            "i'm not sure",
            "as an ai",
            "i apologize",
        ]
        if any(indicator in draft_response.lower() for indicator in error_indicators):
            return False, 0.3

        # For short prompts, short responses are OK
        # "What is 2+2?" -> "4" is perfectly fine
        if prompt_len < 50 and response_len >= 1:
            # Simple query, any non-error response is good
            return True, 0.8

        # Task-specific verification
        if task_type == "code":
            # Code should have structure if asked for code
            code_indicators = [
                "def ",
                "class ",
                "function",
                "return",
                "import",
                "{",
                "}",
                "=",
                "print",
            ]
            has_code = any(ind in draft_response for ind in code_indicators)
            if (
                "code" in prompt.lower()
                or "function" in prompt.lower()
                or "write" in prompt.lower()
            ):
                if not has_code and response_len < 100:
                    return False, 0.4

        # Length-based confidence relative to prompt complexity
        expected_length = min(prompt_len * 3, 500)  # Expect ~3x prompt length
        length_score = (
            min(response_len / expected_length, 1.0) if expected_length > 0 else 0.5
        )

        # Structure score (has paragraphs, lists, etc.)
        structure_score = 0.6  # Base score
        if "\n" in draft_response:
            structure_score += 0.15
        if any(c in draft_response for c in ["1.", "2.", "-", "*", "\u2022"]):
            structure_score += 0.15
        if "```" in draft_response:
            structure_score += 0.1

        # Combined score
        confidence = length_score * 0.3 + structure_score * 0.7

        is_verified = confidence >= VERIFICATION_THRESHOLD

        return is_verified, confidence

    async def get_target_verification(
        self, prompt: str, draft_response: str, target_model: str
    ) -> Tuple[bool, float]:
        """
        Use target model to verify draft response (expensive but accurate).
        Only used for training data collection.
        """
        if litellm is None:
            return True, 0.5

        verification_prompt = f"""Rate the following response to the query on a scale of 1-10.
Only output a single number.

Query: {prompt[:500]}

Response: {draft_response[:1000]}

Rating (1-10):"""

        try:
            result = await litellm.acompletion(
                model=target_model,
                messages=[{"role": "user", "content": verification_prompt}],
                max_tokens=5,
                temperature=0,
            )

            rating_text = result.choices[0].message.content.strip()
            rating = float(rating_text.split()[0])
            confidence = rating / 10.0

            return confidence >= 0.7, confidence

        except Exception as e:
            print(f"Warning: Target verification failed: {e}")
            return True, 0.5  # Default to accepting


class BestExecutionEngine:
    """
    Best Execution Engine for AI Inference.

    Implements trading-style execution:
    1. Glimpse Signal - Early exit for complex queries
    2. Speculative Step - Try cheap model first
    3. Arbitrage Verification - Verify before accepting draft
    """

    def __init__(self):
        self.blotter = Blotter()
        self.oracle = Oracle()
        self.request_count = 0

    def calculate_entropy(self, logprobs: List[Dict]) -> float:
        """
        Calculate entropy from logprobs.
        High entropy = model is uncertain = need target model.
        """
        if not logprobs:
            return 0.0

        try:
            # Get probabilities from logprobs
            probs = []
            for lp in logprobs:
                if hasattr(lp, "top_logprobs") and lp.top_logprobs:
                    for token_lp in lp.top_logprobs:
                        prob = math.exp(token_lp.logprob)
                        probs.append(prob)

            if not probs:
                return 0.0

            # Normalize
            total = sum(probs)
            probs = [p / total for p in probs]

            # Calculate entropy: -sum(p * log(p))
            entropy = -sum(p * math.log(p + 1e-10) for p in probs)

            return entropy

        except Exception as e:
            print(f"Warning: Entropy calculation failed: {e}")
            return 0.0

    async def glimpse_signal(self, prompt: str, draft_model: str) -> Tuple[bool, float]:
        """
        Glimpse Signal: Check first token entropy to decide if draft model is viable.

        Returns: (should_try_draft, entropy)
        """
        if litellm is None:
            return True, 0.0

        try:
            # Get first token with logprobs
            result = await litellm.acompletion(
                model=draft_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0,
                logprobs=True,
                top_logprobs=5,
            )

            # Calculate entropy from logprobs
            logprobs = result.choices[0].logprobs
            if logprobs and hasattr(logprobs, "content"):
                entropy = self.calculate_entropy(logprobs.content)
            else:
                entropy = 0.0

            should_try_draft = entropy < ENTROPY_THRESHOLD

            print(
                f"GLIMPSE: entropy={entropy:.2f}, threshold={ENTROPY_THRESHOLD}, try_draft={should_try_draft}"
            )

            return should_try_draft, entropy

        except Exception as e:
            print(f"Warning: Glimpse signal failed: {e}")
            # Default to trying draft model
            return True, 0.0

    async def execute_draft(
        self,
        prompt: str,
        messages: List[Dict],
        draft_model: str,
        temperature: float = 0.7,
    ) -> Tuple[str, int, float]:
        """
        Execute on draft model (cheap/fast).

        Returns: (response, tokens, latency_ms)
        """
        if litellm is None:
            raise RuntimeError("litellm is required for model execution")

        start = time.time()

        result = await litellm.acompletion(
            model=draft_model, messages=messages, temperature=temperature
        )

        latency_ms = (time.time() - start) * 1000
        response = result.choices[0].message.content
        tokens = result.usage.total_tokens if result.usage else 0

        return response, tokens, latency_ms

    async def execute_target(
        self,
        prompt: str,
        messages: List[Dict],
        target_model: str,
        temperature: float = 0.7,
    ) -> Tuple[str, int, float]:
        """
        Execute on target model (expensive/high-quality).

        Returns: (response, tokens, latency_ms)
        """
        if litellm is None:
            raise RuntimeError("litellm is required for model execution")

        start = time.time()

        result = await litellm.acompletion(
            model=target_model, messages=messages, temperature=temperature
        )

        latency_ms = (time.time() - start) * 1000
        response = result.choices[0].message.content
        tokens = result.usage.total_tokens if result.usage else 0

        return response, tokens, latency_ms

    async def execute(
        self,
        prompt: str,
        messages: List[Dict],
        user_id: str,
        request_id: str,
        temperature: float = 0.7,
        skip_glimpse: bool = False,
        green_factor: float = DEFAULT_GREEN_FACTOR,
    ) -> TradeResult:
        """
        Execute request with Best Execution logic.

        1. Glimpse Signal - Check if draft model is viable
        2. Speculative Step - Try draft model
        3. Arbitrage Verification - Verify draft quality
        4. Target Execution - Fall back to target if needed

        Energy-aware routing: Uses green_factor (lambda) to penalize high-energy models.
        Formula: Utility = Predicted_Accuracy - (lambda x Estimated_Joules)
        """
        self.request_count += 1
        start_time = time.time()

        # Use UnifiedRouter for multi-tier model selection
        routing_decision = route_request(prompt, green_factor=green_factor)

        # Get the optimal model from unified router
        optimal_model = routing_decision.selected_model
        optimal_tier = routing_decision.model_info.tier.value
        predicted_quality = routing_decision.predicted_quality

        # Classify task for logging
        task_type = (
            routing_decision.model_info.capabilities[0]
            if routing_decision.model_info.capabilities
            else "general"
        )

        # Determine draft and target based on routing decision
        # If optimal is draft tier, use it as draft; otherwise use cheapest draft
        if optimal_tier in ["draft_8b", "draft_70b"]:
            draft_model = optimal_model
            # Find best target model from scores
            target_candidates = [
                (m, s)
                for m, s in routing_decision.all_scores.items()
                if MODEL_REGISTRY.get(m)
                and MODEL_REGISTRY[m].tier.value in ["target_70b", "ultra_400b"]
            ]
            if target_candidates:
                target_model = max(target_candidates, key=lambda x: x[1]["quality"])[0]
            else:
                target_model = "openai/gpt-5.2"
        else:
            # Optimal is already a target/ultra model
            draft_model = "groq/llama-3.1-8b-instant"
            target_model = optimal_model

        # Calculate market value (what we charge user)
        # Use target model pricing as the "market rate"
        target_cost_per_token = get_model_cost(target_model) / 1_000_000

        # Estimate input tokens for energy calculation
        input_tokens_estimate = len(prompt.split()) * 1.3  # Rough estimate

        print(f"\n{'=' * 60}")
        print(f"BEST EXECUTION #{self.request_count} (UNIFIED ROUTER)")
        print(f"   Optimal Model: {optimal_model} (Tier: {optimal_tier})")
        print(f"   Predicted Quality: {predicted_quality:.2f}")
        print(f"   Draft: {draft_model} | Target: {target_model}")
        print(f"   Green Factor (lambda): {green_factor}")
        print(f"   Utility Score: {routing_decision.utility_score:.4f}")
        print(f"{'=' * 60}")

        # Step 1: Glimpse Signal
        if not skip_glimpse:
            should_try_draft, entropy = await self.glimpse_signal(prompt, draft_model)

            if not should_try_draft:
                # High entropy - skip draft, go straight to target
                print(
                    f"GLIMPSE EXIT: High entropy ({entropy:.2f}), executing target directly"
                )

                response, tokens, latency_ms = await self.execute_target(
                    prompt, messages, target_model, temperature
                )

                actual_cost = tokens * target_cost_per_token
                market_value = actual_cost * 1.2  # 20% markup

                # Calculate energy (target model used)
                output_tokens = len(response.split()) * 1.3
                energy_estimate = ENERGY_ORACLE.estimate_energy(
                    target_model, int(input_tokens_estimate), int(output_tokens)
                )
                energy_joules = energy_estimate.total_joules
                co2_grams = energy_estimate.co2_grams

                # Calculate what energy would have been with draft
                draft_energy = ENERGY_ORACLE.estimate_joules(
                    "draft_8b", int(input_tokens_estimate), int(output_tokens)
                )
                energy_saved = 0  # No savings - we used target

                self.blotter.log_trade(
                    request_id=request_id,
                    task_type=task_type,
                    execution_path="glimpse_exit",
                    draft_model=draft_model,
                    target_model=target_model,
                    actual_cost=actual_cost,
                    market_value=market_value,
                    latency_ms=latency_ms,
                    tokens=tokens,
                    energy_joules=energy_joules,
                    energy_saved_joules=energy_saved,
                    co2_grams=co2_grams,
                )

                return TradeResult(
                    response=response,
                    model_used=target_model,
                    actual_cost=actual_cost,
                    market_value=market_value,
                    spread_captured=market_value - actual_cost,
                    execution_path="glimpse_exit",
                    latency_ms=latency_ms,
                    tokens_used=tokens,
                    input_tokens=int(input_tokens_estimate),
                    output_tokens=int(output_tokens),
                    energy_joules=energy_joules,
                    energy_saved_joules=energy_saved,
                    co2_grams=co2_grams,
                )

        # Step 2: Speculative Step - Try draft model
        print("SPECULATIVE STEP: Trying draft model...")

        try:
            draft_response, draft_tokens, draft_latency = await self.execute_draft(
                prompt, messages, draft_model, temperature
            )

            # Step 3: Arbitrage Verification
            print("VERIFICATION: Checking draft quality...")

            is_verified, confidence = await self.oracle.verify(
                prompt, draft_response, task_type
            )

            if is_verified:
                # Draft accepted - capture the spread!
                draft_cost_per_token = get_model_cost(draft_model) / 1_000_000
                actual_cost = draft_tokens * draft_cost_per_token
                market_value = (
                    draft_tokens * target_cost_per_token
                )  # Charge target rate

                # Calculate energy savings (draft vs target)
                output_tokens = len(draft_response.split()) * 1.3
                energy_estimate = ENERGY_ORACLE.estimate_energy(
                    draft_model, int(input_tokens_estimate), int(output_tokens)
                )
                energy_joules = energy_estimate.total_joules
                co2_grams = energy_estimate.co2_grams

                # Calculate energy that would have been used with target
                target_energy = ENERGY_ORACLE.estimate_joules(
                    ENERGY_ORACLE.get_model_tier(target_model),
                    int(input_tokens_estimate),
                    int(output_tokens),
                )
                energy_saved = target_energy - energy_joules

                print(
                    f"DRAFT ACCEPTED: Confidence {confidence:.2f} | Energy: {energy_joules:.2f}J | Saved: {energy_saved:.2f}J"
                )

                self.blotter.log_trade(
                    request_id=request_id,
                    task_type=task_type,
                    execution_path="draft_accepted",
                    draft_model=draft_model,
                    target_model=target_model,
                    actual_cost=actual_cost,
                    market_value=market_value,
                    latency_ms=draft_latency,
                    tokens=draft_tokens,
                    energy_joules=energy_joules,
                    energy_saved_joules=energy_saved,
                    co2_grams=co2_grams,
                )

                return TradeResult(
                    response=draft_response,
                    model_used=draft_model,
                    actual_cost=actual_cost,
                    market_value=market_value,
                    spread_captured=market_value - actual_cost,
                    execution_path="draft_accepted",
                    latency_ms=draft_latency,
                    tokens_used=draft_tokens,
                    input_tokens=int(input_tokens_estimate),
                    output_tokens=int(output_tokens),
                    verification_score=confidence,
                    energy_joules=energy_joules,
                    energy_saved_joules=energy_saved,
                    co2_grams=co2_grams,
                )

            else:
                print(
                    f"DRAFT REJECTED: Confidence {confidence:.2f}, escalating to target"
                )

        except Exception as e:
            print(f"Warning: Draft execution failed: {e}, escalating to target")

        # Step 4: Target Execution (fallback)
        print(f"TARGET EXECUTION: Using {target_model}")

        response, tokens, latency_ms = await self.execute_target(
            prompt, messages, target_model, temperature
        )

        actual_cost = tokens * target_cost_per_token
        market_value = actual_cost * 1.2  # 20% markup

        total_latency = (time.time() - start_time) * 1000

        # Calculate energy (target model used, no savings)
        output_tokens = len(response.split()) * 1.3
        energy_estimate = ENERGY_ORACLE.estimate_energy(
            target_model, int(input_tokens_estimate), int(output_tokens)
        )
        energy_joules = energy_estimate.total_joules
        co2_grams = energy_estimate.co2_grams
        energy_saved = 0  # No savings - we used target

        self.blotter.log_trade(
            request_id=request_id,
            task_type=task_type,
            execution_path="target_executed",
            draft_model=draft_model,
            target_model=target_model,
            actual_cost=actual_cost,
            market_value=market_value,
            latency_ms=total_latency,
            tokens=tokens,
            energy_joules=energy_joules,
            energy_saved_joules=energy_saved,
            co2_grams=co2_grams,
        )

        return TradeResult(
            response=response,
            model_used=target_model,
            actual_cost=actual_cost,
            market_value=market_value,
            spread_captured=market_value - actual_cost,
            execution_path="target_executed",
            latency_ms=total_latency,
            tokens_used=tokens,
            input_tokens=int(input_tokens_estimate),
            output_tokens=int(output_tokens),
            energy_joules=energy_joules,
            energy_saved_joules=energy_saved,
            co2_grams=co2_grams,
        )

    def get_blotter_stats(self) -> Dict:
        """Get execution statistics from the blotter."""
        return self.blotter.get_stats()


# Global instance
BEST_EXECUTION_ENGINE = BestExecutionEngine()


async def execute_best(
    prompt: str,
    messages: List[Dict],
    user_id: str,
    request_id: str,
    temperature: float = 0.7,
    green_factor: float = DEFAULT_GREEN_FACTOR,
) -> TradeResult:
    """
    Execute a request using Best Execution Engine.

    This is the main entry point for the trading-style routing.

    Args:
        green_factor: Energy penalty weight (lambda). Higher = more aggressive energy savings.
                     0 = ignore energy, 0.01 = default, 0.1 = aggressive green mode
    """
    return await BEST_EXECUTION_ENGINE.execute(
        prompt=prompt,
        messages=messages,
        user_id=user_id,
        request_id=request_id,
        temperature=temperature,
        green_factor=green_factor,
    )
