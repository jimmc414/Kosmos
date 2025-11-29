"""
Anthropic (Claude) provider implementation.

Supports both Anthropic API and Claude Code CLI routing.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from kosmos.core.providers.base import (
    LLMProvider,
    Message,
    UsageStats,
    LLMResponse,
    ProviderAPIError
)
from kosmos.core.utils.json_parser import parse_json_response, JSONParseError
from kosmos.core.claude_cache import get_claude_cache, ClaudeCache
from datetime import datetime

logger = logging.getLogger(__name__)

from kosmos.config import _DEFAULT_CLAUDE_SONNET_MODEL, _DEFAULT_CLAUDE_HAIKU_MODEL

class AnthropicProvider(LLMProvider):
    """
    Anthropic (Claude) provider implementation.

    Supports both:
    - Anthropic API (real API keys starting with sk-ant-)
    - Claude Code CLI (API key of all 9s for routing)

    Features:
    - Response caching
    - Auto model selection (Haiku/Sonnet)
    - Usage tracking and cost estimation
    - Multi-turn conversations

    Example:
        ```python
        config = {
            'api_key': 'sk-ant-...',
            'model': 'claude-3-5-sonnet-20241022',
            'max_tokens': 4096,
            'temperature': 0.7,
            'enable_cache': True,
        }
        provider = AnthropicProvider(config)
        response = provider.generate("Explain quantum computing")
        print(response.content)
        ```
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Anthropic provider.

        Args:
            config: Configuration dict with keys:
                - api_key: Anthropic API key or '999...' for CLI mode
                - model: Model name (default: claude-3-5-sonnet-20241022)
                - max_tokens: Max tokens (default: 4096)
                - temperature: Sampling temperature (default: 0.7)
                - enable_cache: Enable caching (default: True)
                - enable_auto_model_selection: Auto-select Haiku/Sonnet (default: False)
                - base_url: (Optional) Alternative endpoint for Anthropic API
                - timeout: Request timeout in seconds (default: 120)
        """
        super().__init__(config)

        if not HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package is required. Install with: pip install anthropic"
            )

        # Extract configuration
        self.api_key = config.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not provided in config or environment. "
                "Set to your API key or '999999999999999999999999999999999999999999999999' for CLI mode."
            )

        self.model = config.get('model', _DEFAULT_CLAUDE_SONNET_MODEL)
        self.default_model = self.model
        self.max_tokens = config.get('max_tokens', 4096)
        self.temperature = config.get('temperature', 0.7)
        self.enable_cache = config.get('enable_cache', True)
        self.enable_auto_model_selection = config.get('enable_auto_model_selection', False)
        self.timeout = config.get('timeout', 120)

        # Model variants for auto-selection
        self.haiku_model = _DEFAULT_CLAUDE_HAIKU_MODEL
        self.sonnet_model = _DEFAULT_CLAUDE_SONNET_MODEL

        self.base_url = config.get('base_url') or os.environ.get('CLAUDE_BASE_URL') # NEW: get user-supplied endpoint if any

        # Detect mode (CLI or API)
        self.is_cli_mode = self.api_key.replace('9', '') == ''

        # Initialize Anthropic client
        try:
            if self.base_url:
                self.client = Anthropic(api_key=self.api_key, base_url=self.base_url)  # NEW: pass base_url if present
                logger.info(f"Anthropic provider initialized with custom base_url={self.base_url}")
            else:
                self.client = Anthropic(api_key=self.api_key)
                logger.info(f"Anthropic provider initialized in {'CLI' if self.is_cli_mode else 'API'} mode")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            raise ProviderAPIError("anthropic", f"Failed to initialize: {e}", raw_error=e)

        # Initialize cache
        self.cache: Optional[ClaudeCache] = None
        if self.enable_cache:
            self.cache = get_claude_cache()
            logger.info("Claude response caching enabled")

        # Model selection statistics
        self.haiku_requests = 0
        self.sonnet_requests = 0
        self.model_overrides = 0
        self.cache_hits = 0
        self.cache_misses = 0

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
        Generate text from Claude.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            stop_sequences: Optional list of stop sequences
            **kwargs: Additional args (bypass_cache, model_override)

        Returns:
            LLMResponse: Unified response object

        Raises:
            ProviderAPIError: If the API call fails
        """
        import time as time_module
        try:
            bypass_cache = kwargs.get('bypass_cache', False)
            model_override = kwargs.get('model_override', None)

            # Check if LLM call logging is enabled
            log_llm = False
            try:
                from kosmos.config import get_config
                config = get_config()
                log_llm = config.logging.log_llm_calls
            except Exception:
                pass

            # Model selection logic
            selected_model = self.model

            if model_override:
                selected_model = model_override
                self.model_overrides += 1
                logger.debug(f"Model override: {selected_model}")
            elif self.enable_auto_model_selection and not self.is_cli_mode:
                # Auto-select based on complexity
                from kosmos.core.llm import ModelComplexity
                complexity_analysis = ModelComplexity.estimate_complexity(prompt, system)

                if complexity_analysis['recommendation'] == 'haiku':
                    selected_model = self.haiku_model
                    self.haiku_requests += 1
                else:
                    selected_model = self.sonnet_model
                    self.sonnet_requests += 1

                logger.info(
                    f"Auto-selected {selected_model} "
                    f"(complexity: {complexity_analysis['complexity_score']}, "
                    f"reason: {complexity_analysis['reason']})"
                )
            else:
                # Track model usage
                if 'haiku' in selected_model.lower():
                    self.haiku_requests += 1
                elif 'sonnet' in selected_model.lower():
                    self.sonnet_requests += 1

            # Check cache first
            if self.cache and not bypass_cache:
                cache_key_params = {
                    'system': system or "",
                    'max_tokens': max_tokens,
                    'temperature': temperature,
                    'stop_sequences': stop_sequences or [],
                }

                cached_response = self.cache.get(
                    prompt=prompt,
                    model=selected_model,
                    bypass=False,
                    **cache_key_params
                )

                if cached_response is not None:
                    # Cache hit!
                    self.cache_hits += 1
                    response_text = cached_response['response']
                    logger.info(f"Cache hit: saved API call")

                    # Return as LLMResponse
                    usage = UsageStats(
                        input_tokens=cached_response.get('metadata', {}).get('input_tokens', 0),
                        output_tokens=cached_response.get('metadata', {}).get('output_tokens', 0),
                        total_tokens=cached_response.get('metadata', {}).get('input_tokens', 0) +
                                     cached_response.get('metadata', {}).get('output_tokens', 0),
                        model=selected_model,
                        provider="anthropic",
                        timestamp=datetime.now()
                    )
                    return LLMResponse(
                        content=response_text,
                        usage=usage,
                        model=selected_model,
                        finish_reason="stop",
                        metadata={'cache_hit': True}
                    )
                else:
                    self.cache_misses += 1

            # Build message
            messages = [{"role": "user", "content": prompt}]

            # Pre-call logging
            if log_llm:
                logger.debug(
                    "[LLM] Request: model=%s, prompt_len=%d, system_len=%d, "
                    "max_tokens=%d, temp=%.2f",
                    selected_model,
                    len(prompt),
                    len(system or ""),
                    max_tokens,
                    temperature
                )

            start_time = time_module.time()

            # Call Anthropic API
            response = self.client.messages.create(
                model=selected_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "",
                messages=messages,
                stop_sequences=stop_sequences or [],
                timeout=self.timeout,
            )

            # Extract text and usage
            text = response.content[0].text

            # Post-call logging
            latency_ms = int((time_module.time() - start_time) * 1000)
            if log_llm:
                logger.debug(
                    "[LLM] Response: model=%s, in_tokens=%d, out_tokens=%d, "
                    "latency=%dms, finish=%s",
                    selected_model,
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                    latency_ms,
                    response.stop_reason if hasattr(response, 'stop_reason') else "unknown"
                )
            usage_stats = UsageStats(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                cost_usd=self._calculate_cost(response.usage.input_tokens, response.usage.output_tokens, selected_model) if not self.is_cli_mode else None,
                model=selected_model,
                provider="anthropic",
                timestamp=datetime.now()
            )

            # Update stats
            self._update_usage_stats(usage_stats)

            # Cache the response
            if self.cache and not bypass_cache:
                metadata = {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens,
                }
                cache_key_params = {
                    'system': system or "",
                    'max_tokens': max_tokens,
                    'temperature': temperature,
                    'stop_sequences': stop_sequences or [],
                }
                self.cache.set(
                    prompt=prompt,
                    model=selected_model,
                    response=text,
                    metadata=metadata,
                    **cache_key_params
                )

            logger.debug(f"Generated {len(text)} characters from Claude")

            return LLMResponse(
                content=text,
                usage=usage_stats,
                model=selected_model,
                finish_reason=response.stop_reason if hasattr(response, 'stop_reason') else "stop",
                raw_response=response,
                metadata={'cache_hit': False}
            )

        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise ProviderAPIError("anthropic", f"Generation failed: {e}", raw_error=e)

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
        Generate text asynchronously (delegated to sync for now).

        Args:
            prompt: The user prompt
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Optional stop sequences
            **kwargs: Additional arguments

        Returns:
            LLMResponse: Unified response object
        """
        # For now, delegate to sync version
        # TODO: Implement true async with AsyncAnthropic
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
            # Convert Message objects to Anthropic format
            anthropic_messages = []
            system_prompt = None

            for msg in messages:
                if msg.role == "system":
                    # Anthropic handles system as separate parameter
                    system_prompt = msg.content
                else:
                    anthropic_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })

            # Call API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "",
                messages=anthropic_messages,
                timeout=self.timeout,
            )

            # Extract and convert
            text = response.content[0].text
            usage_stats = UsageStats(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                cost_usd=self._calculate_cost(response.usage.input_tokens, response.usage.output_tokens, self.model) if not self.is_cli_mode else None,
                model=self.model,
                provider="anthropic",
                timestamp=datetime.now()
            )

            self._update_usage_stats(usage_stats)

            return LLMResponse(
                content=text,
                usage=usage_stats,
                model=self.model,
                finish_reason=response.stop_reason if hasattr(response, 'stop_reason') else "stop",
                raw_response=response
            )

        except Exception as e:
            logger.error(f"Anthropic multi-turn generation failed: {e}")
            raise ProviderAPIError("anthropic", f"Multi-turn generation failed: {e}", raw_error=e)

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

            # Generate response
            response = self.generate(
                prompt=prompt,
                system=json_system,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            response_text = response.content

            # Parse JSON with robust fallback strategies
            try:
                return parse_json_response(response_text, schema=schema)

            except JSONParseError as e:
                logger.error(f"Failed to parse JSON after {e.attempts} attempts")
                logger.error(f"Response text: {response_text[:500]}")
                # JSON parse errors are NOT recoverable - retrying won't help
                raise ProviderAPIError(
                    "anthropic",
                    f"Invalid JSON response: {e.message}",
                    raw_error=e,
                    recoverable=False
                )

        except Exception as e:
            if isinstance(e, ProviderAPIError):
                raise
            logger.error(f"Structured generation failed: {e}")
            raise ProviderAPIError("anthropic", f"Structured generation failed: {e}", raw_error=e)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dict with model details
        """
        # Pricing as of Nov 2024
        model_info = {
            "name": self.model,
            "max_tokens": 200000,  # Claude 3.5 context window
            "provider": "anthropic",
            "mode": "cli" if self.is_cli_mode else "api",
        }

        # Add pricing for API mode
        if not self.is_cli_mode:
            if "haiku" in self.model.lower():
                model_info["cost_per_million_input_tokens"] = 0.80
                model_info["cost_per_million_output_tokens"] = 4.00
            elif "sonnet" in self.model.lower():
                model_info["cost_per_million_input_tokens"] = 3.00
                model_info["cost_per_million_output_tokens"] = 15.00
            elif "opus" in self.model.lower():
                model_info["cost_per_million_input_tokens"] = 15.00
                model_info["cost_per_million_output_tokens"] = 75.00

        return model_info

    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """
        Calculate cost for a request.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name used

        Returns:
            float: Cost in USD
        """
        if self.is_cli_mode:
            return 0.0

        # Pricing per million tokens
        if "haiku" in model.lower():
            input_cost_per_m = 0.80
            output_cost_per_m = 4.00
        elif "sonnet" in model.lower():
            input_cost_per_m = 3.00
            output_cost_per_m = 15.00
        elif "opus" in model.lower():
            input_cost_per_m = 15.00
            output_cost_per_m = 75.00
        else:
            # Default to Sonnet pricing
            input_cost_per_m = 3.00
            output_cost_per_m = 15.00

        input_cost = (input_tokens / 1_000_000) * input_cost_per_m
        output_cost = (output_tokens / 1_000_000) * output_cost_per_m

        return input_cost + output_cost

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get detailed usage statistics.

        Returns:
            Dict with usage metrics including cache stats
        """
        stats = super().get_usage_stats()

        # Add Anthropic-specific stats
        total_requests_with_cache = self.request_count + self.cache_hits
        cache_hit_rate = (
            (self.cache_hits / total_requests_with_cache * 100)
            if total_requests_with_cache > 0
            else 0.0
        )

        stats.update({
            "cache_enabled": self.enable_cache,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "mode": "cli" if self.is_cli_mode else "api",
        })

        # Model selection stats
        if self.enable_auto_model_selection:
            total_model_requests = self.haiku_requests + self.sonnet_requests
            stats["model_selection"] = {
                "auto_selection_enabled": True,
                "haiku_requests": self.haiku_requests,
                "sonnet_requests": self.sonnet_requests,
                "total_model_requests": total_model_requests,
                "haiku_percent": round(
                    (self.haiku_requests / total_model_requests * 100)
                    if total_model_requests > 0 else 0, 2
                ),
                "model_overrides": self.model_overrides,
            }

        # Cache stats if available
        if self.cache:
            stats["cache_stats"] = self.cache.get_stats()

        return stats


# Backward compatibility alias
ClaudeClient = AnthropicProvider
