"""
OpenAI provider implementation.

Supports OpenAI API and OpenAI-compatible endpoints (Ollama, OpenRouter, etc.).
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from kosmos.core.providers.base import (
    LLMProvider,
    Message,
    UsageStats,
    LLMResponse,
    ProviderAPIError
)

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider implementation.

    Supports:
    - OpenAI official API (GPT-4, GPT-3.5, etc.)
    - OpenAI-compatible APIs (OpenRouter, Together AI, etc.)
    - Local models (Ollama, LM Studio, LocalAI, etc.)

    Features:
    - Unified interface matching Anthropic provider
    - Response caching (via generic LLM cache)
    - Usage tracking and cost estimation
    - Custom base URLs for compatible providers

    Example:
        ```python
        # OpenAI official
        config = {
            'api_key': 'sk-...',
            'model': 'gpt-4-turbo',
            'max_tokens': 4096,
            'temperature': 0.7,
        }

        # Ollama local
        config = {
            'api_key': 'ollama',  # Dummy key
            'base_url': 'http://localhost:11434/v1',
            'model': 'llama3.1:70b',
        }

        # OpenRouter
        config = {
            'api_key': 'sk-or-...',
            'base_url': 'https://openrouter.ai/api/v1',
            'model': 'anthropic/claude-3.5-sonnet',
        }

        provider = OpenAIProvider(config)
        response = provider.generate("Explain quantum computing")
        print(response.content)
        ```
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI provider.

        Args:
            config: Configuration dict with keys:
                - api_key: OpenAI API key (or dummy for local)
                - model: Model name (e.g., gpt-4-turbo, llama3.1:70b)
                - max_tokens: Max tokens (default: 4096)
                - temperature: Sampling temperature (default: 0.7)
                - base_url: Custom base URL for OpenAI-compatible APIs
                - organization: OpenAI organization ID (optional)
        """
        super().__init__(config)

        if not HAS_OPENAI:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )

        # Extract configuration (handle both dict and Pydantic model)
        def get_config_value(key, default=None):
            """Get value from config (dict or Pydantic model)."""
            if isinstance(config, dict):
                return config.get(key, default)
            else:
                return getattr(config, key, default)

        self.api_key = get_config_value('api_key') or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not provided in config or environment."
            )

        self.model = get_config_value('model') or 'gpt-4-turbo'
        self.max_tokens = get_config_value('max_tokens') or 4096
        temperature = get_config_value('temperature')
        self.temperature = temperature if temperature is not None else 0.7
        self.base_url = get_config_value('base_url') or os.environ.get('OPENAI_BASE_URL')
        self.organization = get_config_value('organization') or os.environ.get('OPENAI_ORGANIZATION')

        # Detect provider type from base_url
        if self.base_url:
            if 'ollama' in self.base_url or 'localhost' in self.base_url or '127.0.0.1' in self.base_url:
                self.provider_type = 'local'
            elif 'openrouter' in self.base_url:
                self.provider_type = 'openrouter'
            elif 'together' in self.base_url:
                self.provider_type = 'together'
            else:
                self.provider_type = 'compatible'
        else:
            self.provider_type = 'openai'

        # Initialize OpenAI client
        try:
            client_args = {
                'api_key': self.api_key,
            }
            if self.base_url:
                client_args['base_url'] = self.base_url
            if self.organization:
                client_args['organization'] = self.organization

            self.client = OpenAI(**client_args)
            logger.info(f"OpenAI provider initialized (type: {self.provider_type}, model: {self.model})")

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise ProviderAPIError("openai", f"Failed to initialize: {e}", raw_error=e)

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
        Generate text from OpenAI.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            stop_sequences: Optional list of stop sequences
            **kwargs: Additional args

        Returns:
            LLMResponse: Unified response object

        Raises:
            ProviderAPIError: If the API call fails
        """
        try:
            # Build messages (OpenAI format: system is first message)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            # Prepare API call arguments
            api_args = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            if stop_sequences:
                api_args["stop"] = stop_sequences

            # Call OpenAI API
            response = self.client.chat.completions.create(**api_args)

            # Extract text and usage
            text = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            # Handle usage stats (may not be present for local models)
            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
            else:
                # Estimate for local models without usage stats
                input_tokens = self._estimate_tokens(prompt + (system or ""))
                output_tokens = self._estimate_tokens(text)
                total_tokens = input_tokens + output_tokens

            # Calculate cost (only for OpenAI official)
            cost = self._calculate_cost(input_tokens, output_tokens) if self.provider_type == 'openai' else None

            usage_stats = UsageStats(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                model=self.model,
                provider="openai",
                timestamp=datetime.now()
            )

            # Update stats
            self._update_usage_stats(usage_stats)

            logger.debug(f"Generated {len(text)} characters from OpenAI")

            return LLMResponse(
                content=text,
                usage=usage_stats,
                model=self.model,
                finish_reason=finish_reason,
                raw_response=response,
                metadata={'provider_type': self.provider_type}
            )

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise ProviderAPIError("openai", f"Generation failed: {e}", raw_error=e)

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
        Generate text asynchronously.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Optional stop sequences
            **kwargs: Additional arguments

        Returns:
            LLMResponse: Unified response object

        Note:
            Currently delegates to sync version.
            TODO: Implement true async with AsyncOpenAI
        """
        # For now, delegate to sync version
        # TODO: Implement true async with AsyncOpenAI
        return self.generate(prompt, system, max_tokens, temperature, stop_sequences, **kwargs)

    def generate_with_messages(
        self,
        messages: List[Message],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from conversation history.

        Args:
            messages: List of Message objects
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments

        Returns:
            LLMResponse: Unified response object
        """
        try:
            # Convert Message objects to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]

            # Call API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Extract and convert
            text = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            # Handle usage stats
            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
            else:
                # Estimate for local models
                all_text = " ".join([msg.content for msg in messages])
                input_tokens = self._estimate_tokens(all_text)
                output_tokens = self._estimate_tokens(text)
                total_tokens = input_tokens + output_tokens

            cost = self._calculate_cost(input_tokens, output_tokens) if self.provider_type == 'openai' else None

            usage_stats = UsageStats(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                model=self.model,
                provider="openai",
                timestamp=datetime.now()
            )

            self._update_usage_stats(usage_stats)

            return LLMResponse(
                content=text,
                usage=usage_stats,
                model=self.model,
                finish_reason=finish_reason,
                raw_response=response
            )

        except Exception as e:
            logger.error(f"OpenAI multi-turn generation failed: {e}")
            raise ProviderAPIError("openai", f"Multi-turn generation failed: {e}", raw_error=e)

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
        Generate structured JSON output.

        Args:
            prompt: The user prompt
            schema: JSON schema or example structure
            system: Optional system prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Parsed JSON object

        Raises:
            ProviderAPIError: If generation or parsing fails
        """
        try:
            # Add JSON instruction to system prompt
            json_system = (system or "") + "\n\nYou must respond with valid JSON matching this schema:\n" + json.dumps(schema, indent=2)
            json_system += "\n\nIMPORTANT: Return ONLY valid JSON, no additional text or explanations."

            # Generate response
            response = self.generate(
                prompt=prompt,
                system=json_system,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            response_text = response.content

            # Parse JSON (handle markdown code blocks)
            try:
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()

                return json.loads(response_text)

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                logger.error(f"Response text: {response_text[:500]}")
                raise ProviderAPIError("openai", f"Invalid JSON response: {e}", raw_error=e)

        except Exception as e:
            if isinstance(e, ProviderAPIError):
                raise
            logger.error(f"Structured generation failed: {e}")
            raise ProviderAPIError("openai", f"Structured generation failed: {e}", raw_error=e)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dict with model details
        """
        model_info = {
            "name": self.model,
            "provider": "openai",
            "provider_type": self.provider_type,
            "base_url": self.base_url or "https://api.openai.com/v1",
        }

        # Add pricing and context for known OpenAI models
        if self.provider_type == 'openai':
            if "gpt-4-turbo" in self.model.lower() or "gpt-4-1106" in self.model.lower():
                model_info["max_tokens"] = 128000
                model_info["cost_per_million_input_tokens"] = 10.00
                model_info["cost_per_million_output_tokens"] = 30.00
            elif "gpt-4" in self.model.lower():
                model_info["max_tokens"] = 8192
                model_info["cost_per_million_input_tokens"] = 30.00
                model_info["cost_per_million_output_tokens"] = 60.00
            elif "gpt-3.5-turbo" in self.model.lower():
                model_info["max_tokens"] = 16385
                model_info["cost_per_million_input_tokens"] = 0.50
                model_info["cost_per_million_output_tokens"] = 1.50
            elif "o1-preview" in self.model.lower():
                model_info["max_tokens"] = 128000
                model_info["cost_per_million_input_tokens"] = 15.00
                model_info["cost_per_million_output_tokens"] = 60.00
            elif "o1-mini" in self.model.lower():
                model_info["max_tokens"] = 128000
                model_info["cost_per_million_input_tokens"] = 3.00
                model_info["cost_per_million_output_tokens"] = 12.00

        return model_info

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for OpenAI official API.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            float: Cost in USD
        """
        if self.provider_type != 'openai':
            return 0.0  # No cost tracking for non-OpenAI providers

        # Pricing per million tokens (as of Nov 2024)
        if "gpt-4-turbo" in self.model.lower() or "gpt-4-1106" in self.model.lower():
            input_cost_per_m = 10.00
            output_cost_per_m = 30.00
        elif "gpt-4" in self.model.lower():
            input_cost_per_m = 30.00
            output_cost_per_m = 60.00
        elif "gpt-3.5-turbo" in self.model.lower():
            input_cost_per_m = 0.50
            output_cost_per_m = 1.50
        elif "o1-preview" in self.model.lower():
            input_cost_per_m = 15.00
            output_cost_per_m = 60.00
        elif "o1-mini" in self.model.lower():
            input_cost_per_m = 3.00
            output_cost_per_m = 12.00
        else:
            # Default to GPT-4 pricing
            input_cost_per_m = 30.00
            output_cost_per_m = 60.00

        input_cost = (input_tokens / 1_000_000) * input_cost_per_m
        output_cost = (output_tokens / 1_000_000) * output_cost_per_m

        return input_cost + output_cost

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token count estimate for local models without usage stats.

        Args:
            text: Text to estimate

        Returns:
            int: Estimated token count
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get detailed usage statistics.

        Returns:
            Dict with usage metrics
        """
        stats = super().get_usage_stats()

        # Add OpenAI-specific stats
        stats.update({
            "provider_type": self.provider_type,
            "base_url": self.base_url or "https://api.openai.com/v1",
        })

        return stats
