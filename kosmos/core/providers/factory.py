"""
Provider Factory for Multi-LLM Support.

Provides factory functions to instantiate the correct LLM provider
based on configuration.
"""

import logging
from typing import Dict, Any, Optional

from kosmos.core.providers.base import LLMProvider, ProviderAPIError

logger = logging.getLogger(__name__)

# Provider registry
_PROVIDER_REGISTRY: Dict[str, type] = {}


def register_provider(name: str, provider_class: type):
    """
    Register a provider class.

    Args:
        name: Provider name (e.g., "anthropic", "openai")
        provider_class: Provider class inheriting from LLMProvider
    """
    if not issubclass(provider_class, LLMProvider):
        raise ValueError(f"{provider_class} must inherit from LLMProvider")

    _PROVIDER_REGISTRY[name.lower()] = provider_class
    logger.debug(f"Registered provider: {name}")


def get_provider(provider_name: str, config: Dict[str, Any]) -> LLMProvider:
    """
    Get a provider instance by name.

    Args:
        provider_name: Provider name ("anthropic", "openai", etc.)
        config: Provider-specific configuration dictionary

    Returns:
        LLMProvider: Instantiated provider

    Raises:
        ProviderAPIError: If provider is unknown or initialization fails

    Example:
        ```python
        config = {
            'api_key': 'sk-ant-...',
            'model': 'claude-3-5-sonnet-20241022',
            'max_tokens': 4096
        }
        provider = get_provider("anthropic", config)
        response = provider.generate("Hello world")
        ```
    """
    provider_name_lower = provider_name.lower()

    if provider_name_lower not in _PROVIDER_REGISTRY:
        available = ", ".join(_PROVIDER_REGISTRY.keys())
        raise ProviderAPIError(
            provider_name,
            f"Unknown provider '{provider_name}'. Available providers: {available}"
        )

    provider_class = _PROVIDER_REGISTRY[provider_name_lower]

    try:
        provider = provider_class(config)
        logger.info(f"Instantiated {provider_name} provider")
        return provider
    except Exception as e:
        logger.error(f"Failed to instantiate {provider_name} provider: {e}")
        raise ProviderAPIError(
            provider_name,
            f"Failed to initialize provider: {e}",
            raw_error=e
        )


def get_provider_from_config(kosmos_config) -> LLMProvider:
    """
    Get provider from Kosmos configuration object.

    Args:
        kosmos_config: KosmosConfig object with provider settings

    Returns:
        LLMProvider: Instantiated provider based on config

    Example:
        ```python
        from kosmos.config import get_config
        from kosmos.core.providers import get_provider_from_config

        config = get_config()
        provider = get_provider_from_config(config)
        ```
    """
    # Determine provider from config
    # For now, default to Anthropic (backward compatibility)
    provider_name = getattr(kosmos_config, 'llm_provider', 'anthropic')

    # Get provider-specific config
    if provider_name.lower() == 'anthropic':
        if hasattr(kosmos_config, 'claude') and kosmos_config.claude:
            # Backward compatibility: old config uses 'claude' field
            claude_config = kosmos_config.claude
            provider_config = {
                'api_key': claude_config.api_key,
                'model': claude_config.model,
                'max_tokens': claude_config.max_tokens,
                'temperature': claude_config.temperature,
                'enable_cache': claude_config.enable_cache,
                'enable_auto_model_selection': getattr(claude_config, 'enable_auto_model_selection', False)
            }
        elif hasattr(kosmos_config, 'anthropic'):
            # New config uses 'anthropic' field
            anthropic_config = kosmos_config.anthropic
            provider_config = {
                'api_key': anthropic_config.api_key,
                'model': anthropic_config.model,
                'max_tokens': anthropic_config.max_tokens,
                'temperature': anthropic_config.temperature,
                'enable_cache': anthropic_config.enable_cache,
                'enable_auto_model_selection': getattr(anthropic_config, 'enable_auto_model_selection', False)
            }
        else:
            raise ValueError("No Anthropic/Claude configuration found")

    elif provider_name.lower() == 'openai':
        if not hasattr(kosmos_config, 'openai'):
            raise ValueError("No OpenAI configuration found")

        openai_config = kosmos_config.openai
        provider_config = {
            'api_key': openai_config.api_key,
            'model': openai_config.model,
            'max_tokens': getattr(openai_config, 'max_tokens', 4096),
            'temperature': getattr(openai_config, 'temperature', 0.7),
            'base_url': getattr(openai_config, 'base_url', None),
            'organization': getattr(openai_config, 'organization', None),
        }

    else:
        raise ValueError(f"Unknown provider in config: {provider_name}")

    return get_provider(provider_name, provider_config)


def list_providers() -> list:
    """
    List all registered providers.

    Returns:
        list: Names of registered providers
    """
    return list(_PROVIDER_REGISTRY.keys())


# Auto-register built-in providers
def _register_builtin_providers():
    """Register built-in provider implementations."""
    try:
        from kosmos.core.providers.anthropic import AnthropicProvider
        register_provider("anthropic", AnthropicProvider)
        register_provider("claude", AnthropicProvider)  # Alias
    except ImportError:
        logger.warning("Anthropic provider not available (anthropic package not installed)")

    try:
        from kosmos.core.providers.openai import OpenAIProvider
        register_provider("openai", OpenAIProvider)
    except ImportError:
        logger.debug("OpenAI provider not available (openai package not installed or not implemented yet)")


# Register on module import
_register_builtin_providers()
