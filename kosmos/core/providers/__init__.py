"""
Multi-Provider LLM Support for Kosmos.

This module provides a unified interface for working with different LLM providers
(Anthropic, OpenAI, LiteLLM for 100+ providers, and other OpenAI-compatible APIs).

Basic usage:
    ```python
    from kosmos.core.providers import get_provider
    from kosmos.config import get_config

    config = get_config()
    provider = get_provider(config)

    response = provider.generate("What is machine learning?")
    print(response.content)
    ```

LiteLLM provider usage:
    ```python
    from kosmos.core.providers import get_provider

    # Ollama (local)
    provider = get_provider("litellm", {
        'model': 'ollama/llama3.1:8b',
        'api_base': 'http://localhost:11434'
    })

    # DeepSeek
    provider = get_provider("litellm", {
        'model': 'deepseek/deepseek-chat',
        'api_key': 'sk-...'
    })

    response = provider.generate("Hello!")
    ```
"""

from kosmos.core.providers.base import (
    LLMProvider,
    Message,
    UsageStats,
    LLMResponse,
    ProviderAPIError
)
from kosmos.core.providers.factory import (
    get_provider,
    get_provider_from_config,
    list_providers,
    register_provider
)

# Optional: Import LiteLLMProvider if available
try:
    from kosmos.core.providers.litellm_provider import LiteLLMProvider
    _LITELLM_AVAILABLE = True
except ImportError:
    LiteLLMProvider = None
    _LITELLM_AVAILABLE = False

# Optional: ClaudeCodeProvider (needs claude-agent-sdk at construction time)
try:
    from kosmos.core.providers.claude_code import ClaudeCodeProvider
except ImportError:
    ClaudeCodeProvider = None

__all__ = [
    "LLMProvider",
    "Message",
    "UsageStats",
    "LLMResponse",
    "ProviderAPIError",
    "get_provider",
    "get_provider_from_config",
    "list_providers",
    "register_provider",
    "LiteLLMProvider",
    "ClaudeCodeProvider",
]
