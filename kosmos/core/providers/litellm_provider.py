"""
LiteLLM Provider for Multi-LLM Support.

Supports 100+ LLM providers through the LiteLLM library:
- Anthropic (claude-3-5-sonnet, claude-3-5-haiku, etc.)
- OpenAI (gpt-4-turbo, gpt-4, gpt-3.5-turbo, etc.)
- Ollama (ollama/llama3.1, ollama/mistral, etc.)
- DeepSeek (deepseek/deepseek-chat, deepseek/deepseek-coder)
- Azure OpenAI (azure/deployment-name)
- And many more...

Model Format Examples:
    Anthropic: "claude-3-5-sonnet-20241022"
    OpenAI: "gpt-4-turbo"
    Ollama: "ollama/llama3.1:8b"
    DeepSeek: "deepseek/deepseek-chat"
    LM Studio: Set api_base to "http://localhost:1234/v1"
"""

import json
import logging
from typing import List, Dict, Any, Optional, Iterator, AsyncIterator
from datetime import datetime

from kosmos.core.providers.base import (
    LLMProvider,
    Message,
    UsageStats,
    LLMResponse,
    ProviderAPIError
)

logger = logging.getLogger(__name__)


from kosmos.config import _DEFAULT_CLAUDE_SONNET_MODEL, _DEFAULT_CLAUDE_HAIKU_MODEL

# Model pricing per 1M tokens (input, output) in USD
# Updated pricing as of November 2025
MODEL_PRICING: Dict[str, tuple] = {
    # Anthropic Claude 4.5 (current)
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-opus-4-5": (15.0, 75.0),
    # Anthropic Claude 3.5 (legacy)
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-3-5-haiku-20241022": (1.0, 5.0),
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
    "gpt-4o": (5.0, 15.0),
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


class LiteLLMProvider(LLMProvider):
    """
    LiteLLM-based provider supporting 100+ LLM backends.

    This provider uses the LiteLLM library to provide a unified interface
    to multiple LLM providers including Anthropic, OpenAI, Ollama, DeepSeek,
    Azure, and many more.

    Example usage:
        ```python
        # Ollama (local)
        provider = LiteLLMProvider({
            'model': 'ollama/llama3.1:8b',
            'api_base': 'http://localhost:11434'
        })

        # DeepSeek
        provider = LiteLLMProvider({
            'model': 'deepseek/deepseek-chat',
            'api_key': 'sk-...'
        })

        # Anthropic
        provider = LiteLLMProvider({
            'model': 'claude-3-5-sonnet-20241022',
            'api_key': 'sk-ant-...'
        })

        response = provider.generate("Hello!")
        print(response.content)
        ```
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LiteLLM provider.

        Args:
            config: Configuration dictionary with:
                - model: Model identifier (e.g., "ollama/llama3.1", "gpt-4-turbo")
                - api_key: API key (optional for local models like Ollama)
                - api_base: Custom API base URL (for Ollama, LM Studio, etc.)
                - max_tokens: Default max tokens
                - temperature: Default temperature
                - timeout: Request timeout in seconds
        """
        super().__init__(config)

        # Import litellm here to make it an optional dependency
        try:
            import litellm
            self.litellm = litellm
        except ImportError:
            raise ImportError(
                "LiteLLM is required for LiteLLMProvider. "
                "Install it with: pip install litellm"
            )

        # Configuration
        self.model = config.get('model', 'gpt-3.5-turbo')
        self.api_key = config.get('api_key')
        self.api_base = config.get('api_base')
        self.max_tokens_default = config.get('max_tokens', 4096)
        self.temperature_default = config.get('temperature', 0.7)
        self.timeout = config.get('timeout', 120)

        # Configure LiteLLM
        if self.api_key:
            # Set API key based on model provider
            if 'claude' in self.model.lower() or 'anthropic' in self.model.lower():
                litellm.anthropic_key = self.api_key
            elif 'deepseek' in self.model.lower():
                litellm.deepseek_key = self.api_key
            elif 'openai' in self.model.lower() or 'gpt' in self.model.lower():
                litellm.openai_key = self.api_key

        # Set up LiteLLM settings
        litellm.set_verbose = False  # Reduce logging

        # Determine provider type from model name
        self._detect_provider_type()

        logger.info(
            f"LiteLLM provider initialized: model={self.model}, "
            f"provider_type={self.provider_type}, api_base={self.api_base}"
        )

    def _detect_provider_type(self):
        """Detect provider type from model name for cost tracking."""
        model_lower = self.model.lower()

        if model_lower.startswith('ollama/'):
            self.provider_type = 'ollama'
        elif model_lower.startswith('deepseek/'):
            self.provider_type = 'deepseek'
        elif model_lower.startswith('azure/'):
            self.provider_type = 'azure'
        elif 'claude' in model_lower:
            self.provider_type = 'anthropic'
        elif 'gpt' in model_lower or model_lower.startswith('openai/'):
            self.provider_type = 'openai'
        else:
            self.provider_type = 'unknown'

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on model pricing."""
        # Try exact model match
        if self.model in MODEL_PRICING:
            input_price, output_price = MODEL_PRICING[self.model]
        # Try base model name (without tags like :8b)
        elif self.model.split(':')[0] in MODEL_PRICING:
            input_price, output_price = MODEL_PRICING[self.model.split(':')[0]]
        # Default: free (local models or unknown)
        else:
            input_price, output_price = (0.0, 0.0)

        cost = (input_tokens / 1_000_000) * input_price + \
               (output_tokens / 1_000_000) * output_price
        return cost

    def _build_messages(
        self,
        prompt: str,
        system: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Build messages list for LiteLLM."""
        messages = []

        # Handle Qwen models - they need explicit no-think directive to avoid empty responses
        # when max_tokens is set. Qwen3 models have thinking mode enabled by default.
        is_qwen = "qwen" in self.model.lower()
        no_think_directive = "Do not use thinking mode. Respond directly without <think> tags."

        if system:
            # Add no-think directive for Qwen models
            if is_qwen and no_think_directive not in system:
                system = f"{system}\n\n{no_think_directive}"
            messages.append({"role": "system", "content": system})
        elif is_qwen:
            # Add minimal system message for Qwen models to disable thinking
            messages.append({"role": "system", "content": no_think_directive})

        messages.append({"role": "user", "content": prompt})
        return messages

    def _parse_response(self, response) -> LLMResponse:
        """Parse LiteLLM response into unified format."""
        content = response.choices[0].message.content or ""

        # Extract usage stats
        usage_data = response.usage if hasattr(response, 'usage') else None
        input_tokens = getattr(usage_data, 'prompt_tokens', 0) if usage_data else 0
        output_tokens = getattr(usage_data, 'completion_tokens', 0) if usage_data else 0
        total_tokens = getattr(usage_data, 'total_tokens', input_tokens + output_tokens) if usage_data else 0

        cost = self._estimate_cost(input_tokens, output_tokens)

        usage = UsageStats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            model=self.model,
            provider=f"litellm/{self.provider_type}",
            timestamp=datetime.now()
        )

        self._update_usage_stats(usage)

        return LLMResponse(
            content=content,
            usage=usage,
            model=response.model if hasattr(response, 'model') else self.model,
            finish_reason=response.choices[0].finish_reason if response.choices else None,
            raw_response=response
        )

    def _get_effective_max_tokens(self, max_tokens: Optional[int]) -> int:
        """
        Get effective max_tokens, with special handling for Qwen models.

        Qwen3 models use thinking mode by default, which consumes tokens even when
        disabled via system prompt. We need a higher minimum to ensure the actual
        response isn't cut off.
        """
        requested = max_tokens or self.max_tokens_default

        # Qwen models need at least 8192 tokens to handle potential thinking overhead
        is_qwen = "qwen" in self.model.lower()
        if is_qwen:
            return max(requested, 8192)

        return requested

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from a prompt (synchronous).

        Args:
            prompt: The user prompt/query
            system: Optional system prompt
            max_tokens: Maximum tokens to generate (default from config)
            temperature: Sampling temperature (default from config)
            stop_sequences: Optional list of stop sequences
            **kwargs: Additional LiteLLM parameters

        Returns:
            LLMResponse: Unified response object
        """
        import time as time_module

        # Check if LLM call logging is enabled
        log_llm = False
        try:
            from kosmos.config import get_config
            config = get_config()
            log_llm = config.logging.log_llm_calls
        except Exception:
            pass

        messages = self._build_messages(prompt, system)
        effective_max_tokens = self._get_effective_max_tokens(max_tokens)
        effective_temperature = temperature if temperature is not None else self.temperature_default

        # Pre-call logging
        if log_llm:
            logger.debug(
                "[LLM] Request: model=%s, prompt_len=%d, system_len=%d, "
                "max_tokens=%d, temp=%.2f",
                self.model,
                len(prompt),
                len(system or ""),
                effective_max_tokens,
                effective_temperature
            )

        start_time = time_module.time()

        try:
            response = self.litellm.completion(
                model=self.model,
                messages=messages,
                max_tokens=effective_max_tokens,
                temperature=effective_temperature,
                stop=stop_sequences,
                api_key=self.api_key,
                api_base=self.api_base,
                timeout=self.timeout,
                **kwargs
            )

            # Post-call logging
            latency_ms = int((time_module.time() - start_time) * 1000)
            if log_llm:
                usage_data = response.usage if hasattr(response, 'usage') else None
                input_tokens = getattr(usage_data, 'prompt_tokens', 0) if usage_data else 0
                output_tokens = getattr(usage_data, 'completion_tokens', 0) if usage_data else 0
                logger.debug(
                    "[LLM] Response: model=%s, in_tokens=%d, out_tokens=%d, "
                    "latency=%dms, finish=%s",
                    self.model,
                    input_tokens,
                    output_tokens,
                    latency_ms,
                    response.choices[0].finish_reason if response.choices else "unknown"
                )

            return self._parse_response(response)

        except Exception as e:
            logger.error(f"LiteLLM generation failed: {e}")
            raise ProviderAPIError(
                "litellm",
                f"Generation failed: {e}",
                raw_error=e
            )

    async def generate_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from a prompt (asynchronous).

        Args:
            prompt: The user prompt/query
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Optional list of stop sequences
            **kwargs: Additional LiteLLM parameters

        Returns:
            LLMResponse: Unified response object
        """
        messages = self._build_messages(prompt, system)
        effective_max_tokens = self._get_effective_max_tokens(max_tokens)

        try:
            response = await self.litellm.acompletion(
                model=self.model,
                messages=messages,
                max_tokens=effective_max_tokens,
                temperature=temperature if temperature is not None else self.temperature_default,
                stop=stop_sequences,
                api_key=self.api_key,
                api_base=self.api_base,
                timeout=self.timeout,
                **kwargs
            )
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"LiteLLM async generation failed: {e}")
            raise ProviderAPIError(
                "litellm",
                f"Async generation failed: {e}",
                raw_error=e
            )

    def generate_with_messages(
        self,
        messages: List[Message],
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from a conversation history.

        Args:
            messages: List of Message objects (conversation history)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional LiteLLM parameters

        Returns:
            LLMResponse: Unified response object
        """
        # Convert Message objects to LiteLLM format
        litellm_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        try:
            response = self.litellm.completion(
                model=self.model,
                messages=litellm_messages,
                max_tokens=max_tokens or self.max_tokens_default,
                temperature=temperature if temperature is not None else self.temperature_default,
                api_key=self.api_key,
                api_base=self.api_base,
                timeout=self.timeout,
                **kwargs
            )
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"LiteLLM message generation failed: {e}")
            raise ProviderAPIError(
                "litellm",
                f"Message generation failed: {e}",
                raw_error=e
            )

    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output matching a schema.

        Args:
            prompt: The user prompt/query
            schema: JSON schema or example structure
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional LiteLLM parameters

        Returns:
            Dict[str, Any]: Parsed JSON object

        Raises:
            ProviderAPIError: If generation or parsing fails
        """
        # Build system prompt for JSON output
        json_system = (system or "") + """

IMPORTANT: Respond ONLY with valid JSON. No explanations, no markdown code blocks, just pure JSON.
The response must match this schema: """ + json.dumps(schema, indent=2)

        response = self.generate(
            prompt=prompt,
            system=json_system.strip(),
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else 0.3,  # Lower temp for structured output
            **kwargs
        )

        # Clean and parse JSON
        content = response.content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {content[:200]}...")
            raise ProviderAPIError(
                "litellm",
                f"Invalid JSON response: {e}",
                raw_error=e,
                recoverable=False
            )

    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate text with streaming.

        Args:
            prompt: The user prompt/query
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional LiteLLM parameters

        Yields:
            str: Chunks of generated text
        """
        messages = self._build_messages(prompt, system)

        try:
            response = self.litellm.completion(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens or self.max_tokens_default,
                temperature=temperature if temperature is not None else self.temperature_default,
                api_key=self.api_key,
                api_base=self.api_base,
                timeout=self.timeout,
                stream=True,
                **kwargs
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"LiteLLM streaming failed: {e}")
            raise ProviderAPIError(
                "litellm",
                f"Streaming failed: {e}",
                raw_error=e
            )

    async def generate_stream_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate text with async streaming.

        Args:
            prompt: The user prompt/query
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional LiteLLM parameters

        Yields:
            str: Chunks of generated text
        """
        messages = self._build_messages(prompt, system)

        try:
            response = await self.litellm.acompletion(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens or self.max_tokens_default,
                temperature=temperature if temperature is not None else self.temperature_default,
                api_key=self.api_key,
                api_base=self.api_base,
                timeout=self.timeout,
                stream=True,
                **kwargs
            )

            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"LiteLLM async streaming failed: {e}")
            raise ProviderAPIError(
                "litellm",
                f"Async streaming failed: {e}",
                raw_error=e
            )

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dict with model information
        """
        # Get pricing if available
        if self.model in MODEL_PRICING:
            input_price, output_price = MODEL_PRICING[self.model]
        else:
            input_price, output_price = (0.0, 0.0)

        return {
            "name": self.model,
            "provider": f"litellm/{self.provider_type}",
            "max_tokens": self.max_tokens_default,
            "cost_per_input_token": input_price / 1_000_000,
            "cost_per_output_token": output_price / 1_000_000,
            "api_base": self.api_base,
            "supports_streaming": True,
            "supports_async": True,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"LiteLLMProvider(model={self.model}, provider_type={self.provider_type})"
