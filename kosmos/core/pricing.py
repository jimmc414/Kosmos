"""
Canonical model pricing for Kosmos cost tracking.

Single source of truth for LLM pricing across the codebase.
All modules should import from here instead of hardcoding prices.

Pricing is per 1 million tokens in USD.

Prices come from the ``llm-prices`` package (dated, source-cited rows for
Anthropic, OpenAI, Google, DeepSeek and others). MODEL_PRICING below is the
fallback for models it does not list (local Ollama models, retired ids).
"""

from typing import Dict, Optional, Tuple

try:
    import llm_prices
except ImportError:  # pragma: no cover - llm-prices is a core dependency
    llm_prices = None

# Fallback model pricing per 1M tokens (input, output) in USD
# Updated pricing as of February 2026
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # Anthropic Claude 4.5 (current)
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-opus-4-5": (5.0, 25.0),
    # Anthropic Claude 3.5 (legacy)
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-3-5-haiku-20241022": (0.8, 4.0),
    # Anthropic Claude 3 (legacy)
    "claude-3-opus-20240229": (15.0, 75.0),
    "claude-3-sonnet-20240229": (3.0, 15.0),
    "claude-3-haiku-20240307": (0.25, 1.25),
    # OpenAI
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4-turbo-preview": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "gpt-4-32k": (60.0, 120.0),
    "gpt-3.5-turbo": (0.5, 1.5),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    # DeepSeek
    "deepseek/deepseek-chat": (0.14, 0.28),
    "deepseek/deepseek-coder": (0.14, 0.28),
    # Ollama (free, local)
    "ollama/llama3.1": (0.0, 0.0),
    "ollama/llama3.1:8b": (0.0, 0.0),
    "ollama/llama3.1:70b": (0.0, 0.0),
    "ollama/mistral": (0.0, 0.0),
    "ollama/codellama": (0.0, 0.0),
    "ollama/phi3": (0.0, 0.0),
}

# Convenience aliases mapping model family keywords to pricing
_FAMILY_PRICING: Dict[str, Tuple[float, float]] = {
    "haiku": (1.0, 5.0),
    "sonnet": (3.0, 15.0),
    "opus": (15.0, 75.0),
}


def _llm_prices_cost(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """Cost from the llm-prices table, or None when it has no price for the model."""
    if llm_prices is None or model.lower().startswith("ollama/"):
        return None
    price = llm_prices.lookup(model)
    if price is None:
        return None
    return price.cost(max(int(input_tokens), 0), max(int(output_tokens), 0))


def get_model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate cost for a given model and token counts.

    Looks up pricing from llm-prices first (it resolves dated, prefixed and
    Bedrock/Vertex model ids), then the fallback MODEL_PRICING dict, then
    family-based matching (haiku/sonnet/opus keywords) if no exact match.

    Args:
        model: Model identifier (e.g. "claude-sonnet-4-5", "gpt-4o")
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated

    Returns:
        Estimated cost in USD
    """
    cost = _llm_prices_cost(model, input_tokens, output_tokens)
    if cost is not None:
        return cost

    # Try exact model match
    if model in MODEL_PRICING:
        input_price, output_price = MODEL_PRICING[model]
    # Try base model name (without tags like :8b)
    elif model.split(":")[0] in MODEL_PRICING:
        input_price, output_price = MODEL_PRICING[model.split(":")[0]]
    else:
        # Fall back to family-based matching
        model_lower = model.lower()
        input_price, output_price = (0.0, 0.0)
        for family, pricing in _FAMILY_PRICING.items():
            if family in model_lower:
                input_price, output_price = pricing
                break

    cost = (input_tokens / 1_000_000) * input_price + \
           (output_tokens / 1_000_000) * output_price
    return cost
