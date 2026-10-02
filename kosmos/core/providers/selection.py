"""
Per-run provider and model selection for `kosmos run --provider/--model`.

The .env file sets the default provider. These helpers apply a command-line
override to the loaded KosmosConfig, resolve short model aliases, and reject
provider/model combinations that cannot work, with a message naming the fix.
"""

import logging
import os
from typing import Dict, Optional, Tuple

from kosmos.config import ANTHROPIC_KEY_ENV_VARS, _anthropic_key_from_env

logger = logging.getLogger(__name__)

# Values accepted by --provider
PROVIDER_CHOICES: Tuple[str, ...] = ("deepseek", "litellm", "anthropic", "claude-code")

# --provider value for each KosmosConfig.llm_provider
_CONFIG_TO_CHOICE: Dict[str, str] = {
    "litellm": "litellm",
    "anthropic": "anthropic",
    "claude_code": "claude-code",
    "openai": "openai",
}

# Short names accepted by --model; any other string passes through unchanged
MODEL_ALIASES: Dict[str, str] = {
    "opus": "claude-opus-5-5",
    "sonnet": "claude-sonnet-5-5",
    "haiku": "claude-haiku-4-5",
    "fable": "claude-fable-5-1",
    "deepseek": "deepseek/deepseek-chat",
    "deepseek-chat": "deepseek/deepseek-chat",
    "deepseek-reasoner": "deepseek/deepseek-reasoner",
}

DEFAULT_DEEPSEEK_MODEL = "deepseek/deepseek-chat"


class ProviderSelectionError(ValueError):
    """A --provider/--model combination that cannot work; the message names the fix."""


def resolve_model(model: Optional[str]) -> Optional[str]:
    """Map a short alias to its model id; return other strings unchanged."""
    if not model:
        return None
    return MODEL_ALIASES.get(model.strip().lower(), model.strip())


def _is_claude_model(model: str) -> bool:
    return model.startswith("claude-")


def _is_deepseek_model(model: str) -> bool:
    return "deepseek" in model.lower()


def apply_provider_selection(
    config,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Apply --provider/--model to a loaded KosmosConfig in place.

    Args:
        config: The KosmosConfig from get_config()
        provider: One of PROVIDER_CHOICES, or None to keep the .env provider
        model: A model id or alias from MODEL_ALIASES, or None for the provider default

    Returns:
        (llm_provider, model_id) now active in config

    Raises:
        ProviderSelectionError: When the combination cannot work

    Example:
        ```python
        config = get_config()
        apply_provider_selection(config, "claude-code", "opus")
        get_client(reset=True)
        ```
    """
    resolved = resolve_model(model)
    effective = provider or _CONFIG_TO_CHOICE.get(config.llm_provider, config.llm_provider)

    if provider is not None and provider not in PROVIDER_CHOICES:
        raise ProviderSelectionError(
            f"Unknown provider '{provider}'. Choose one of: {', '.join(PROVIDER_CHOICES)}"
        )

    if resolved and _is_claude_model(resolved) and effective in ("deepseek", "litellm", "openai"):
        raise ProviderSelectionError(
            f"Model '{resolved}' is an Anthropic model, but the provider is '{effective}'. "
            f"Use --provider claude-code (your Claude Code login, no API key) or "
            f"--provider anthropic (API key in KOSMOS_ANTHROPIC_API_KEY)."
        )
    if resolved and _is_deepseek_model(resolved) and effective in ("anthropic", "claude-code"):
        raise ProviderSelectionError(
            f"Model '{resolved}' is a DeepSeek model, but the provider is '{effective}'. "
            f"Use --provider deepseek."
        )
    if effective == "anthropic" and not _anthropic_key_from_env():
        raise ProviderSelectionError(
            "--provider anthropic needs an Anthropic API key: set KOSMOS_ANTHROPIC_API_KEY "
            "(preferred; ANTHROPIC_API_KEY also works). To use your Claude Code login with "
            "no API key, use --provider claude-code."
        )

    if effective == "deepseek":
        config.llm_provider = "litellm"
        config.litellm.model = resolved or DEFAULT_DEEPSEEK_MODEL
        # LiteLLM reads DEEPSEEK_API_KEY itself; a LITELLM_* key or base set for another
        # backend (an OpenAI key, a local Ollama URL) would misroute the request
        config.litellm.api_key = os.environ.get("DEEPSEEK_API_KEY") or config.litellm.api_key
        config.litellm.api_base = None
    elif effective == "litellm":
        config.llm_provider = "litellm"
        if resolved:
            config.litellm.model = resolved
    elif effective == "anthropic":
        from kosmos.config import ClaudeConfig

        config.llm_provider = "anthropic"
        if config.claude is None:
            config.claude = ClaudeConfig()
        if resolved:
            config.claude.model = resolved
    elif effective == "claude-code":
        config.llm_provider = "claude_code"
        if resolved:
            config.claude_code.model = resolved
    elif resolved:
        raise ProviderSelectionError(
            f"--model is not supported for provider '{effective}'; set the model in .env instead."
        )

    active_model = config.get_active_model()
    logger.info(f"LLM provider selected: {config.llm_provider} ({active_model})")
    return config.llm_provider, active_model


def describe_credentials() -> Dict[str, bool]:
    """Report which credential variables are set, never their values."""
    return {
        "DEEPSEEK_API_KEY": bool(os.environ.get("DEEPSEEK_API_KEY")),
        **{name: bool(os.environ.get(name)) for name in ANTHROPIC_KEY_ENV_VARS},
    }
