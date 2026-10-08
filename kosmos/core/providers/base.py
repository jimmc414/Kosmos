"""
Base Provider Interface for Multi-LLM Support.

Defines the abstract interface that all LLM providers must implement,
enabling Kosmos to work with Anthropic, OpenAI, and other providers.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, AsyncIterator, Iterator
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """
    Unified message format for all providers.

    Attributes:
        role: Message role (system, user, assistant)
        content: Message content text
        name: Optional name for the message sender
        metadata: Optional provider-specific metadata
    """
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class UsageStats:
    """
    Token usage statistics across providers.

    Attributes:
        input_tokens: Number of tokens in the prompt
        output_tokens: Number of tokens in the response
        total_tokens: Total tokens used
        cost_usd: Estimated cost in USD (if available)
        model: Model name used
        provider: Provider name
        timestamp: When the request was made
    """
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: Optional[float] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class LLMResponse:
    """
    Unified response format from all providers.

    Supports string-like operations (strip, lower, etc.) by delegating to
    self.content, so callers that expect str from generate() work seamlessly.

    Attributes:
        content: The generated text content
        usage: Token usage statistics
        model: Model used for generation
        finish_reason: Reason the generation stopped
        raw_response: Original provider-specific response object
        metadata: Additional provider-specific metadata
    """
    content: str
    usage: UsageStats
    model: str
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None

    # -- String compatibility methods --
    # These allow LLMResponse to be used where str is expected (e.g. response.strip())

    def __str__(self) -> str:
        return self.content

    def __contains__(self, item: str) -> bool:
        return item in self.content

    def __len__(self) -> int:
        return len(self.content)

    def __add__(self, other: str) -> str:
        return self.content + other

    def __radd__(self, other: str) -> str:
        return other + self.content

    def __bool__(self) -> bool:
        return bool(self.content)

    def __iter__(self):
        return iter(self.content)

    def __getitem__(self, key):
        return self.content[key]

    def strip(self, chars: Optional[str] = None) -> str:
        return self.content.strip(chars) if chars else self.content.strip()

    def lstrip(self, chars: Optional[str] = None) -> str:
        return self.content.lstrip(chars) if chars else self.content.lstrip()

    def rstrip(self, chars: Optional[str] = None) -> str:
        return self.content.rstrip(chars) if chars else self.content.rstrip()

    def lower(self) -> str:
        return self.content.lower()

    def upper(self) -> str:
        return self.content.upper()

    def startswith(self, prefix, *args) -> bool:
        return self.content.startswith(prefix, *args)

    def endswith(self, suffix, *args) -> bool:
        return self.content.endswith(suffix, *args)

    def split(self, *args, **kwargs):
        return self.content.split(*args, **kwargs)

    def rsplit(self, *args, **kwargs):
        return self.content.rsplit(*args, **kwargs)

    def replace(self, old: str, new: str, count: int = -1) -> str:
        return self.content.replace(old, new, count)

    def find(self, sub: str, *args) -> int:
        return self.content.find(sub, *args)

    def rfind(self, sub: str, *args) -> int:
        return self.content.rfind(sub, *args)

    def count(self, sub: str, *args) -> int:
        return self.content.count(sub, *args)

    def encode(self, *args, **kwargs) -> bytes:
        return self.content.encode(*args, **kwargs)

    def format(self, *args, **kwargs) -> str:
        return self.content.format(*args, **kwargs)

    def join(self, iterable) -> str:
        return self.content.join(iterable)


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    Provides a unified interface for:
    - Text generation (sync and async)
    - Structured JSON output
    - Multi-turn conversations
    - Streaming responses
    - Usage tracking and cost estimation

    All provider implementations (Anthropic, OpenAI, etc.) must inherit
    from this class and implement all abstract methods.

    Example:
        ```python
        class MyProvider(LLMProvider):
            def generate(self, prompt: str, **kwargs) -> LLMResponse:
                # Implementation here
                pass
        ```
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize provider with configuration.

        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()

        # Usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.request_count = 0

        logger.info(f"Initialized {self.provider_name} provider")

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from a prompt (synchronous).

        Args:
            prompt: The user prompt/query
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            stop_sequences: Optional list of stop sequences
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse: Unified response object

        Raises:
            ProviderAPIError: If the API call fails
        """
        pass

    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from a prompt (asynchronous).

        Args:
            prompt: The user prompt/query
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            stop_sequences: Optional list of stop sequences
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse: Unified response object

        Raises:
            ProviderAPIError: If the API call fails
        """
        pass

    @abstractmethod
    def generate_with_messages(
        self,
        messages: List[Message],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from a conversation history.

        Args:
            messages: List of Message objects (conversation history)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse: Unified response object

        Raises:
            ProviderAPIError: If the API call fails
        """
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output matching a schema.

        Args:
            prompt: The user prompt/query
            schema: JSON schema or example structure
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Provider-specific parameters

        Returns:
            Dict[str, Any]: Parsed JSON object matching schema

        Raises:
            ProviderAPIError: If the API call fails
            JSONDecodeError: If response is not valid JSON
        """
        pass

    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate text with streaming (optional, not all providers support).

        Args:
            prompt: The user prompt/query
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Provider-specific parameters

        Yields:
            str: Chunks of generated text

        Raises:
            NotImplementedError: If provider doesn't support streaming
        """
        raise NotImplementedError(f"{self.provider_name} does not support streaming")

    async def generate_stream_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate text with async streaming (optional).

        Args:
            prompt: The user prompt/query
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Provider-specific parameters

        Yields:
            str: Chunks of generated text

        Raises:
            NotImplementedError: If provider doesn't support async streaming
        """
        raise NotImplementedError(f"{self.provider_name} does not support async streaming")
        # Make this a proper async generator
        if False:
            yield

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dict with keys: name, max_tokens, cost_per_input_token, cost_per_output_token
        """
        pass

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get cumulative usage statistics for this provider instance.

        Returns:
            Dict with usage metrics and cost estimates
        """
        return {
            "provider": self.provider_name,
            "total_requests": self.request_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
        }

    def _update_usage_stats(self, usage: UsageStats):
        """
        Update cumulative usage statistics.

        Args:
            usage: UsageStats object from a request
        """
        self.request_count += 1
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        if usage.cost_usd is not None:
            self.total_cost_usd += usage.cost_usd

        # Feed the process-wide metrics collector so budget enforcement sees every call
        try:
            from kosmos.core.metrics import get_metrics
            get_metrics().record_api_call(
                model=usage.model or getattr(self, "model", "unknown"),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                duration_seconds=0.0,
                success=True,
                cost_usd=usage.cost_usd,
            )
        except Exception as e:
            logger.debug(f"Metrics recording skipped: {e}")

    def reset_usage_stats(self):
        """Reset usage statistics to zero."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.request_count = 0
        logger.info(f"Reset usage stats for {self.provider_name}")

    def __repr__(self) -> str:
        """String representation of the provider."""
        return f"{self.__class__.__name__}(provider={self.provider_name})"


class ProviderAPIError(Exception):
    """
    Exception raised when a provider API call fails.

    Attributes:
        provider: Name of the provider
        message: Error message
        status_code: HTTP status code (if applicable)
        raw_error: Original error object
        recoverable: Whether this error can be retried
    """

    def __init__(
        self,
        provider: str,
        message: str,
        status_code: Optional[int] = None,
        raw_error: Optional[Exception] = None,
        recoverable: bool = True
    ):
        self.provider = provider
        self.message = message
        self.status_code = status_code
        self.raw_error = raw_error
        self.recoverable = recoverable

        super().__init__(f"[{provider}] {message}")

    def is_recoverable(self) -> bool:
        """
        Check if this error is recoverable through retry.

        Returns:
            True if the error might succeed on retry (network issues, rate limits),
            False if retry is pointless (auth failures, JSON parse errors).
        """
        # If explicitly marked as non-recoverable, respect that
        if not self.recoverable:
            return False

        # Check status code for known non-recoverable errors
        if self.status_code is not None:
            # 4xx client errors (except 429 rate limit) are not recoverable
            if 400 <= self.status_code < 500 and self.status_code != 429:
                return False

        # Check message patterns for recoverable errors
        message_lower = self.message.lower()
        recoverable_patterns = (
            'timeout', 'connection', 'network', 'rate_limit', 'rate limit',
            'overloaded', 'service_unavailable', 'service unavailable',
            'temporarily', 'retry', '429', '503', '502', '504'
        )
        non_recoverable_patterns = (
            'json', 'parse', 'invalid', 'authentication', 'unauthorized',
            'forbidden', 'not found', 'bad request', '401', '403', '404', '400'
        )

        # If explicitly mentions recoverable error type
        if any(pattern in message_lower for pattern in recoverable_patterns):
            return True

        # If explicitly mentions non-recoverable error type
        if any(pattern in message_lower for pattern in non_recoverable_patterns):
            return False

        # Default to recoverable for unknown errors
        return True
