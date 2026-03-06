# energy_oracle.py
# Astrai Energy Oracle - Energy-Aware AI Inference Routing
#
# Based on research from:
# - "From Prompts to Power" paper
# - "Measuring Energy Consumption" paper
#
# Energy Formula:
# E_total = (L_input × C_prefill) + (L_output × C_decode)
#
# Where:
# - L = Length (number of tokens)
# - C_prefill = Energy constant for processing input (Joules/token)
# - C_decode = Energy constant for generating output (Joules/token)
# - Decode phase is typically 5-10x more energy-intensive than prefill

from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EnergyEstimate:
    """Energy estimation result."""
    total_joules: float
    prefill_joules: float
    decode_joules: float
    watt_hours: float
    co2_grams: float  # Assuming average grid carbon intensity
    model_tier: str


@dataclass
class EfficiencyGain:
    """Efficiency comparison between draft and target models."""
    joules_saved: float
    watt_hours_saved: float
    co2_grams_saved: float
    efficiency_gain_pct: float
    draft_energy: EnergyEstimate
    target_energy: EnergyEstimate


class EnergyOracle:
    """
    Energy Oracle for estimating AI inference energy consumption.
    
    Uses research-based coefficients for different model tiers:
    - Draft (8B): Fast, cheap, low energy
    - Target (70B): High quality, moderate energy
    - Ultra (405B+): Frontier quality, high energy
    
    Coefficients are normalized for H100/A100 hardware with 4-bit quantization.
    """
    
    # Energy coefficients: Joules per token
    # Based on normalized 4-bit quantized runs on H100 hardware
    COEFFICIENTS = {
        # Draft models (8B parameters) - Groq Llama 3.1 8B, etc.
        "draft_8b": {
            "prefill": 0.002,   # J/token for input processing
            "decode": 0.015,    # J/token for output generation
            "idle_watts": 50,   # Baseline idle power
        },
        # Target models (70B parameters) - Llama 70B, GPT-4o, Claude 3.5
        "target_70b": {
            "prefill": 0.018,
            "decode": 0.120,
            "idle_watts": 150,
        },
        # Ultra models (405B+ parameters) - GPT-5.2, Claude Opus 4.5
        "ultra_400b": {
            "prefill": 0.085,
            "decode": 0.650,
            "idle_watts": 400,
        },
    }
    
    # Model name to tier mapping
    MODEL_TIERS = {
        # Draft tier (8B class) — fast, cheap, low energy
        "groq/llama-3.1-8b-instant": "draft_8b",
        "groq/groq/llama-3.1-8b-instant": "draft_8b",
        "together/llama-3.1-8b": "draft_8b",
        "deepseek/deepseek-chat": "draft_8b",
        "openai/gpt-4o-mini": "draft_8b",
        "cerebras/llama-3.1-8b": "draft_8b",
        "cerebras/llama-4-scout-17b-16e-instruct": "draft_8b",
        "google/gemini-2.5-flash": "draft_8b",
        "google/gemini-3-flash": "draft_8b",
        "anthropic/claude-haiku-4-5-20251001": "draft_8b",
        "mistral/mistral-small": "draft_8b",
        "ollama/llama3.2:3b": "draft_8b",
        "ollama/llama3.1:8b": "draft_8b",
        "ollama/qwen2.5:1.5b": "draft_8b",
        "ollama/qwen2.5:7b": "draft_8b",
        "ollama/mistral:7b": "draft_8b",

        # Target tier (70B class) — strong general-purpose
        "openai/gpt-4o": "target_70b",
        "openai/gpt-4.1": "target_70b",
        "openai/gpt-5.1": "target_70b",
        "openai/o1-mini": "target_70b",
        "openai/o3-mini": "target_70b",
        "anthropic/claude-sonnet-4-20250514": "target_70b",
        "anthropic/claude-sonnet-4-5-20250929": "target_70b",
        "anthropic/claude-3-5-sonnet-latest": "target_70b",
        "groq/llama-3.3-70b-versatile": "target_70b",
        "groq/groq/llama-3.3-70b-versatile": "target_70b",
        "google/gemini-2.5-pro": "target_70b",
        "google/gemini/gemini-2.5-pro": "target_70b",
        "mistral/mistral-large": "target_70b",
        "deepinfra/meta-llama/Llama-4-Maverick-17B-128E-Instruct": "target_70b",
        "deepseek/deepseek-v3.2": "target_70b",
        "ollama/qwen2.5:14b": "draft_8b",
        "ollama/qwen2.5:32b": "target_70b",
        "ollama/llama3.3:70b": "target_70b",

        # Ultra tier (400B+ class) — frontier, high energy
        "openai/gpt-5.2": "ultra_400b",
        "openai/gpt-5.2-pro": "ultra_400b",
        "anthropic/claude-opus-4-5-20251101": "ultra_400b",
        "anthropic/claude-opus-4-6": "ultra_400b",
        "google/gemini-3-pro": "ultra_400b",
        "google/gemini/gemini-3-pro": "ultra_400b",
        "together/llama-3.1-405b": "ultra_400b",
        "deepseek/deepseek-reasoner": "ultra_400b",
    }
    
    # Average grid carbon intensity (gCO2/kWh)
    # US average ~400, EU average ~250, varies by region
    CARBON_INTENSITY = 400  # gCO2 per kWh
    
    def __init__(self, carbon_intensity: float = None):
        """
        Initialize EnergyOracle.
        
        Args:
            carbon_intensity: Grid carbon intensity in gCO2/kWh (default: 400)
        """
        self.carbon_intensity = carbon_intensity or self.CARBON_INTENSITY
    
    def get_model_tier(self, model_name: str) -> str:
        """Get the energy tier for a model."""
        return self.MODEL_TIERS.get(model_name, "target_70b")
    
    def estimate_joules(
        self,
        model_tier: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Estimate energy consumption in Joules.
        
        Formula: E = (In × Prefill_Rate) + (Out × Decode_Rate)
        
        Args:
            model_tier: One of "draft_8b", "target_70b", "ultra_400b"
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Energy in Joules
        """
        coeffs = self.COEFFICIENTS.get(model_tier, self.COEFFICIENTS["target_70b"])
        
        # Energy = (Input × Prefill_Rate) + (Output × Decode_Rate)
        energy_joules = (input_tokens * coeffs["prefill"]) + (output_tokens * coeffs["decode"])
        
        return round(energy_joules, 4)
    
    def estimate_energy(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int
    ) -> EnergyEstimate:
        """
        Full energy estimation for a model request.
        
        Args:
            model_name: Full model name (e.g., "openai/gpt-5.2")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            EnergyEstimate with Joules, Watt-hours, and CO2
        """
        model_tier = self.get_model_tier(model_name)
        coeffs = self.COEFFICIENTS.get(model_tier, self.COEFFICIENTS["target_70b"])
        
        prefill_joules = input_tokens * coeffs["prefill"]
        decode_joules = output_tokens * coeffs["decode"]
        total_joules = prefill_joules + decode_joules
        
        # Convert to Watt-hours (1 Wh = 3600 J)
        watt_hours = total_joules / 3600
        
        # Calculate CO2 emissions
        # CO2 (g) = Wh × (gCO2/kWh) / 1000
        co2_grams = watt_hours * self.carbon_intensity / 1000
        
        return EnergyEstimate(
            total_joules=round(total_joules, 4),
            prefill_joules=round(prefill_joules, 4),
            decode_joules=round(decode_joules, 4),
            watt_hours=round(watt_hours, 6),
            co2_grams=round(co2_grams, 6),
            model_tier=model_tier
        )
    
    # Normalized decode cost per token for each tier (0-1 scale).
    # Used at routing time when output token count is unknown.
    _DECODE_COSTS = {
        "draft_8b": 0.015,
        "target_70b": 0.120,
        "ultra_400b": 0.650,
    }
    _MAX_DECODE = 0.650  # ultra decode cost, for normalization

    def get_energy_score(self, model_name: str) -> float:
        """
        Return a normalized energy cost score for a model (0.0 = greenest, 1.0 = most power-hungry).

        Uses the decode coefficient as proxy since decode dominates real energy
        spend and output token count is unknown at routing time.
        """
        tier = self.get_model_tier(model_name)
        decode_cost = self._DECODE_COSTS.get(tier, self._DECODE_COSTS["target_70b"])
        return decode_cost / self._MAX_DECODE

    def get_energy_score_by_family(self, model_family: str) -> float:
        """
        Like get_energy_score but accepts a normalized model family name
        (e.g. 'gpt-4o', 'llama-3.1-8b') instead of provider/model.

        Infers tier from model family keywords.
        """
        family = (model_family or "").lower()
        # Ultra tier
        if any(t in family for t in ("gpt-5.2", "gpt-5.2-pro", "opus-4.5", "opus-4.6",
                                      "opus-4-5", "opus-4-6", "405b", "o3", "o1",
                                      "gemini-3-pro", "deepseek-r1")):
            return 1.0
        # Draft tier
        if any(t in family for t in ("mini", "flash", "8b", "3b", "1.5b", "scout",
                                      "haiku", "mixtral", "7b", "14b")):
            return self._DECODE_COSTS["draft_8b"] / self._MAX_DECODE  # ~0.023
        # Target tier (default)
        return self._DECODE_COSTS["target_70b"] / self._MAX_DECODE  # ~0.185

    def get_efficiency_alpha(
        self,
        input_tokens: int,
        output_tokens: int,
        draft_tier: str = "draft_8b",
        target_tier: str = "ultra_400b"
    ) -> EfficiencyGain:
        """
        Calculate energy saved by using Draft vs Target model.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            draft_tier: Draft model tier (default: draft_8b)
            target_tier: Target model tier (default: ultra_400b)
            
        Returns:
            EfficiencyGain with savings breakdown
        """
        e_draft = self.estimate_joules(draft_tier, input_tokens, output_tokens)
        e_target = self.estimate_joules(target_tier, input_tokens, output_tokens)
        
        joules_saved = e_target - e_draft
        wh_saved = joules_saved / 3600
        co2_saved = wh_saved * self.carbon_intensity / 1000
        
        efficiency_pct = (1 - (e_draft / e_target)) * 100 if e_target > 0 else 0
        
        # Full estimates for both
        draft_estimate = EnergyEstimate(
            total_joules=e_draft,
            prefill_joules=input_tokens * self.COEFFICIENTS[draft_tier]["prefill"],
            decode_joules=output_tokens * self.COEFFICIENTS[draft_tier]["decode"],
            watt_hours=e_draft / 3600,
            co2_grams=(e_draft / 3600) * self.carbon_intensity / 1000,
            model_tier=draft_tier
        )
        
        target_estimate = EnergyEstimate(
            total_joules=e_target,
            prefill_joules=input_tokens * self.COEFFICIENTS[target_tier]["prefill"],
            decode_joules=output_tokens * self.COEFFICIENTS[target_tier]["decode"],
            watt_hours=e_target / 3600,
            co2_grams=(e_target / 3600) * self.carbon_intensity / 1000,
            model_tier=target_tier
        )
        
        return EfficiencyGain(
            joules_saved=round(joules_saved, 4),
            watt_hours_saved=round(wh_saved, 6),
            co2_grams_saved=round(co2_saved, 6),
            efficiency_gain_pct=round(efficiency_pct, 2),
            draft_energy=draft_estimate,
            target_energy=target_estimate
        )
    
    def calculate_utility(
        self,
        predicted_accuracy: float,
        estimated_joules: float,
        green_factor: float = 0.01
    ) -> float:
        """
        Calculate routing utility with energy penalty.
        
        Formula: Utility = Predicted_Accuracy - (λ × Estimated_Joules)
        
        Args:
            predicted_accuracy: Expected accuracy/quality score (0-1)
            estimated_joules: Energy cost in Joules
            green_factor: Lambda (λ) - energy penalty weight (default: 0.01)
                         Higher = more aggressive energy savings
                         
        Returns:
            Utility score (higher = better choice)
        """
        utility = predicted_accuracy - (green_factor * estimated_joules)
        return round(utility, 4)
    
    def should_escalate(
        self,
        draft_accuracy: float,
        target_accuracy: float,
        input_tokens: int,
        output_tokens: int,
        green_factor: float = 0.01
    ) -> Tuple[bool, Dict]:
        """
        Decide whether to escalate from draft to target model.
        
        Uses green-aware routing formula to prevent escalation
        if marginal accuracy gain isn't worth the energy spike.
        
        Args:
            draft_accuracy: Predicted accuracy of draft model (0-1)
            target_accuracy: Predicted accuracy of target model (0-1)
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            green_factor: Energy penalty weight (0 = ignore energy, 1 = max penalty)
            
        Returns:
            (should_escalate, details_dict)
        """
        # Calculate energy for both models
        e_draft = self.estimate_joules("draft_8b", input_tokens, output_tokens)
        e_target = self.estimate_joules("ultra_400b", input_tokens, output_tokens)
        
        # Calculate utility scores
        utility_draft = self.calculate_utility(draft_accuracy, e_draft, green_factor)
        utility_target = self.calculate_utility(target_accuracy, e_target, green_factor)
        
        # Escalate only if target utility is higher
        should_escalate = utility_target > utility_draft
        
        # Calculate marginal gains
        accuracy_gain = target_accuracy - draft_accuracy
        energy_cost = e_target - e_draft
        
        return should_escalate, {
            "draft_utility": utility_draft,
            "target_utility": utility_target,
            "utility_delta": utility_target - utility_draft,
            "accuracy_gain": accuracy_gain,
            "energy_cost_joules": energy_cost,
            "green_factor": green_factor,
            "recommendation": "escalate" if should_escalate else "use_draft",
            "reason": f"Target utility ({utility_target:.3f}) {'>' if should_escalate else '<='} Draft utility ({utility_draft:.3f})"
        }


# Global instance
ENERGY_ORACLE = EnergyOracle()


def estimate_request_energy(
    model_name: str,
    input_tokens: int,
    output_tokens: int
) -> Dict:
    """
    Estimate energy for a request.
    
    Returns dict with Joules, Watt-hours, and CO2.
    """
    estimate = ENERGY_ORACLE.estimate_energy(model_name, input_tokens, output_tokens)
    return {
        "model": model_name,
        "model_tier": estimate.model_tier,
        "energy_joules": estimate.total_joules,
        "energy_wh": estimate.watt_hours,
        "co2_grams": estimate.co2_grams,
        "breakdown": {
            "prefill_joules": estimate.prefill_joules,
            "decode_joules": estimate.decode_joules,
        }
    }


def get_energy_savings(input_tokens: int, output_tokens: int) -> Dict:
    """
    Calculate energy savings from using draft model.
    
    Returns dict with savings breakdown.
    """
    gain = ENERGY_ORACLE.get_efficiency_alpha(input_tokens, output_tokens)
    return {
        "joules_saved": gain.joules_saved,
        "watt_hours_saved": gain.watt_hours_saved,
        "co2_grams_saved": gain.co2_grams_saved,
        "efficiency_gain_pct": gain.efficiency_gain_pct,
    }
