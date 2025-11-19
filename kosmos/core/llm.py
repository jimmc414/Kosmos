"""
Multi-provider LLM integration for Kosmos.

Supports both Anthropic (Claude) and OpenAI providers through a unified interface.
Maintains backward compatibility with existing ClaudeClient usage.

This module provides:
1. Anthropic API (direct API or Claude Code CLI routing)
2. OpenAI API (official OpenAI or OpenAI-compatible providers like Ollama, OpenRouter)
3. Backward-compatible ClaudeClient interface
4. Provider-agnostic get_client() function
"""

import os
from typing import Any, Dict, List, Optional, Union
import json
import logging

try:
    from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    Anthropic = None  # Define as None so later checks don't fail
    HUMAN_PROMPT = None
    AI_PROMPT = None
    print("Warning: anthropic package not installed. Install with: pip install anthropic")

from kosmos.core.claude_cache import get_claude_cache, ClaudeCache
from kosmos.core.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class ModelComplexity:
    """Estimate prompt complexity for model selection."""

    # Complexity keywords that suggest Sonnet should be used
    COMPLEX_KEYWORDS = [
        'analyze', 'synthesis', 'complex', 'design', 'architecture',
        'research', 'hypothesis', 'experiment', 'optimize', 'algorithm',
        'proof', 'theorem', 'mathematical', 'scientific', 'reasoning',
        'creative', 'novel', 'innovative', 'strategy', 'plan'
    ]

    @staticmethod
    def estimate_complexity(
        prompt: str,
        system: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Estimate prompt complexity.

        Args:
            prompt: The user prompt
            system: Optional system prompt

        Returns:
            dict: Complexity analysis with score and recommendation
        """
        # Count tokens (rough estimate: ~4 chars per token)
        prompt_tokens = len(prompt) / 4
        system_tokens = len(system or "") / 4
        total_tokens = prompt_tokens + system_tokens

        # Check for complex keywords
        prompt_lower = prompt.lower()
        keyword_matches = sum(
            1 for keyword in ModelComplexity.COMPLEX_KEYWORDS
            if keyword in prompt_lower
        )

        # Scoring (0-100)
        # - Token count contributes up to 50 points
        # - Keyword matches contribute up to 50 points
        token_score = min(total_tokens / 20, 50)  # Max at 1000 tokens
        keyword_score = min(keyword_matches * 10, 50)  # Max at 5 keywords

        complexity_score = token_score + keyword_score

        # Recommendation
        if complexity_score < 30:
            recommendation = "haiku"  # Simple task
        elif complexity_score < 60:
            recommendation = "sonnet"  # Moderate complexity
        else:
            recommendation = "sonnet"  # High complexity

        return {
            'complexity_score': round(complexity_score, 2),
            'total_tokens_estimate': int(total_tokens),
            'keyword_matches': keyword_matches,
            'recommendation': recommendation,
            'reason': (
                'simple query' if complexity_score < 30
                else 'moderate complexity' if complexity_score < 60
                else 'high complexity task'
            )
        }


class ClaudeClient:
    """
    Unified Claude client supporting both API and CLI modes.

    Automatically detects mode based on API key:
    - API mode: API key starts with 'sk-ant-'
    - CLI mode: API key is all 9s (routes to Claude Code CLI)

    Features:
    - Automatic model selection (Haiku vs Sonnet) based on complexity
    - Response caching for cost savings
    - Support for both API and CLI modes

    Example:
        ```python
        # API mode with auto model selection
        os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-...'
        client = ClaudeClient(enable_auto_model_selection=True)

        # CLI mode (uses Claude Code Max)
        os.environ['ANTHROPIC_API_KEY'] = '999999999...'
        client = ClaudeClient()
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        enable_cache: bool = True,
        enable_auto_model_selection: bool = False,
    ):
        """
        Initialize Claude client.

        Args:
            api_key: Anthropic API key or '999...' for CLI mode.
                     If None, reads from ANTHROPIC_API_KEY env var.
            model: Claude model to use (default model, can be auto-selected)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-1.0)
            enable_cache: Enable response caching (default: True)
            enable_auto_model_selection: Auto-select Haiku/Sonnet based on complexity
        """
        if not HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package is required. Install with: pip install anthropic\n"
                "For CLI routing support: pip install git+https://github.com/jimmc414/claude_n_codex_api_proxy.git"
            )

        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set to your API key or '999999999999999999999999999999999999999999999999' for CLI mode."
            )

        self.model = model
        self.default_model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_cache = enable_cache
        self.enable_auto_model_selection = enable_auto_model_selection

        # Model variants for auto-selection
        self.haiku_model = "claude-3-5-haiku-20241022"
        self.sonnet_model = "claude-3-5-sonnet-20241022"

        # Detect mode
        self.is_cli_mode = self.api_key.replace('9', '') == ''

        # Initialize Anthropic client (will auto-route based on API key)
        try:
            self.client = Anthropic(api_key=self.api_key)
            logger.info(f"Claude client initialized in {'CLI' if self.is_cli_mode else 'API'} mode")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            raise

        # Initialize cache
        self.cache: Optional[ClaudeCache] = None
        if self.enable_cache:
            self.cache = get_claude_cache()
            logger.info("Claude response caching enabled")

        # Usage statistics
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Model selection statistics
        self.haiku_requests = 0
        self.sonnet_requests = 0
        self.model_overrides = 0

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        bypass_cache: bool = False,
        model_override: Optional[str] = None,
    ) -> str:
        """
        Generate text from Claude.

        Args:
            prompt: The user prompt/question
            system: Optional system prompt for instructions
            max_tokens: Override default max_tokens
            temperature: Override default temperature
            stop_sequences: Optional list of stop sequences
            bypass_cache: Force bypass cache for this request
            model_override: Override auto model selection with specific model

        Returns:
            str: Generated text from Claude

        Example:
            ```python
            client = ClaudeClient()
            response = client.generate(
                prompt="Explain quantum entanglement",
                system="You are a physics professor"
            )
            print(response)
            ```
        """
        try:
            # Model selection logic
            selected_model = self.model

            if model_override:
                # User override takes precedence
                selected_model = model_override
                self.model_overrides += 1
                logger.debug(f"Model override: {selected_model}")
            elif self.enable_auto_model_selection and not self.is_cli_mode:
                # Auto-select based on complexity
                complexity_analysis = ModelComplexity.estimate_complexity(
                    prompt, system
                )

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

            # Check cache first (if enabled and not bypassed)
            if self.cache and not bypass_cache:
                cache_key_params = {
                    'system': system or "",
                    'max_tokens': max_tokens or self.max_tokens,
                    'temperature': temperature or self.temperature,
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
                    logger.info(
                        f"Cache hit ({cached_response.get('cache_hit_type', 'exact')}): "
                        f"saved API call"
                    )
                    return response_text
                else:
                    # Cache miss
                    self.cache_misses += 1

            # Build message
            messages = [{"role": "user", "content": prompt}]

            # Call Claude API (auto-routes to CLI if API key is all 9s)
            response = self.client.messages.create(
                model=selected_model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
                system=system or "",
                messages=messages,
                stop_sequences=stop_sequences or [],
            )

            # Update statistics
            self.total_requests += 1
            if hasattr(response, 'usage') and response.usage:
                if hasattr(response.usage, 'input_tokens'):
                    self.total_input_tokens += response.usage.input_tokens
                if hasattr(response.usage, 'output_tokens'):
                    self.total_output_tokens += response.usage.output_tokens

            # Validate response has content
            if not response.content or len(response.content) == 0:
                raise ValueError("Claude API returned empty response content")

            # Extract text
            text = response.content[0].text

            # Cache the response (if caching enabled)
            if self.cache and not bypass_cache:
                metadata = {}
                if hasattr(response, 'usage') and response.usage:
                    if hasattr(response.usage, 'input_tokens') and hasattr(response.usage, 'output_tokens'):
                        metadata = {
                            'input_tokens': response.usage.input_tokens,
                            'output_tokens': response.usage.output_tokens,
                        }

                cache_key_params = {
                    'system': system or "",
                    'max_tokens': max_tokens or self.max_tokens,
                    'temperature': temperature or self.temperature,
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
            return text

        except Exception as e:
            logger.error(f"Claude generation failed: {e}")
            raise

    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate text using multi-turn conversation format.

        Args:
            messages: List of message dicts with 'role' and 'content'
                     Example: [{"role": "user", "content": "Hello"}]
            system: Optional system prompt
            max_tokens: Override default max_tokens
            temperature: Override default temperature

        Returns:
            str: Generated text from Claude
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
                system=system or "",
                messages=messages,
            )

            # Update statistics
            self.total_requests += 1
            if hasattr(response, 'usage') and response.usage:
                if hasattr(response.usage, 'input_tokens'):
                    self.total_input_tokens += response.usage.input_tokens
                if hasattr(response.usage, 'output_tokens'):
                    self.total_output_tokens += response.usage.output_tokens

            # Validate response has content
            if not response.content or len(response.content) == 0:
                raise ValueError("Claude API returned empty response content")

            return response.content[0].text

        except Exception as e:
            logger.error(f"Claude multi-turn generation failed: {e}")
            raise

    def generate_structured(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured output (JSON) from Claude.

        Args:
            prompt: The user prompt
            output_schema: JSON schema describing expected output structure
            system: Optional system prompt

        Returns:
            dict: Parsed JSON response

        Example:
            ```python
            schema = {
                "type": "object",
                "properties": {
                    "hypothesis": {"type": "string"},
                    "confidence": {"type": "number"}
                }
            }
            result = client.generate_structured(
                prompt="Generate a hypothesis about dark matter",
                output_schema=schema
            )
            ```
        """
        # Add JSON instruction to system prompt
        json_system = (system or "") + "\n\nYou must respond with valid JSON matching this schema:\n" + json.dumps(output_schema, indent=2)

        response_text = self.generate(
            prompt=prompt,
            system=json_system,
        )

        # Parse JSON
        try:
            # Try to extract JSON from markdown code blocks if present
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
            logger.error(f"Failed to parse JSON from Claude response: {e}")
            logger.error(f"Response text: {response_text[:500]}")
            raise ValueError(f"Claude did not return valid JSON: {e}")

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics including cache metrics.

        Returns:
            dict: Statistics including requests, tokens, cost, and cache metrics
        """
        total_requests_with_cache = self.total_requests + self.cache_hits
        cache_hit_rate = (
            (self.cache_hits / total_requests_with_cache * 100)
            if total_requests_with_cache > 0
            else 0.0
        )

        # Calculate cost savings from cache
        if self.cache and self.cache_hits > 0:
            # Estimate average tokens per request
            avg_input_tokens = (
                self.total_input_tokens / self.total_requests
                if self.total_requests > 0
                else 1000
            )
            avg_output_tokens = (
                self.total_output_tokens / self.total_requests
                if self.total_requests > 0
                else 500
            )

            # Estimate cost savings (API mode only)
            if not self.is_cli_mode:
                input_saved = (avg_input_tokens * self.cache_hits / 1_000_000) * 3.0
                output_saved = (avg_output_tokens * self.cache_hits / 1_000_000) * 15.0
                cost_saved = input_saved + output_saved
            else:
                cost_saved = 0.0  # CLI mode has no per-token cost
        else:
            cost_saved = 0.0

        stats = {
            "total_api_requests": self.total_requests,
            "total_cache_hits": self.cache_hits,
            "total_cache_misses": self.cache_misses,
            "total_requests": total_requests_with_cache,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": self._estimate_cost(),
            "estimated_cost_saved_usd": round(cost_saved, 2),
            "mode": "cli" if self.is_cli_mode else "api",
            "cache_enabled": self.enable_cache,
        }

        # Add model selection stats
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

            # Estimate cost savings from using Haiku
            # Haiku is ~5x cheaper: Sonnet $3/$15, Haiku ~$0.60/$3 per M tokens
            # Simplified: assume each Haiku request saved ~80% of cost
            if self.haiku_requests > 0 and not self.is_cli_mode:
                avg_tokens_per_request = (
                    (self.total_input_tokens + self.total_output_tokens) /
                    self.total_requests if self.total_requests > 0 else 1500
                )
                # Estimate savings: 80% of what Sonnet would have cost
                estimated_haiku_savings = (
                    (avg_tokens_per_request / 1_000_000) *
                    self.haiku_requests * 12 * 0.8  # ~80% savings
                )
                stats["model_selection"]["estimated_cost_saved_usd"] = round(
                    estimated_haiku_savings, 2
                )

        # Add detailed cache stats if available
        if self.cache:
            stats["cache_stats"] = self.cache.get_stats()

        return stats

    def _estimate_cost(self) -> float:
        """
        Estimate API cost (only relevant for API mode, CLI is unlimited).

        Returns:
            float: Estimated cost in USD
        """
        if self.is_cli_mode:
            return 0.0  # CLI mode has no per-token cost

        # Pricing for Claude 3.5 Sonnet (as of Nov 2025)
        # Input: $3 per million tokens
        # Output: $15 per million tokens
        input_cost = (self.total_input_tokens / 1_000_000) * 3.0
        output_cost = (self.total_output_tokens / 1_000_000) * 15.0

        return input_cost + output_cost

    def reset_stats(self):
        """Reset usage statistics."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.haiku_requests = 0
        self.sonnet_requests = 0
        self.model_overrides = 0


# Singleton instance for convenience
_default_client: Optional[Union[ClaudeClient, LLMProvider]] = None


def get_client(reset: bool = False, use_provider_system: bool = True) -> Union[ClaudeClient, LLMProvider]:
    """
    Get or create default LLM client singleton.

    This function provides backward compatibility while enabling the new multi-provider system.

    Args:
        reset: If True, create a new client instance
        use_provider_system: If True, use new provider system based on config (default).
                           If False, use legacy ClaudeClient (backward compatibility).

    Returns:
        Union[ClaudeClient, LLMProvider]: Client instance

    Examples:
        ```python
        # Default: Uses config to select provider
        client = get_client()

        # Legacy mode: Always use ClaudeClient
        client = get_client(use_provider_system=False)

        # Reset and recreate
        client = get_client(reset=True)
        ```
    """
    global _default_client

    if _default_client is None or reset:
        if use_provider_system:
            # Use new provider system
            try:
                from kosmos.config import get_config
                from kosmos.core.providers import get_provider_from_config

                config = get_config()
                _default_client = get_provider_from_config(config)
                logger.info(f"Initialized {config.llm_provider} provider via config")

            except Exception as e:
                logger.warning(f"Failed to initialize provider from config: {e}. Falling back to AnthropicProvider")
                # Use AnthropicProvider (which implements LLMProvider) instead of ClaudeClient
                from kosmos.core.providers.anthropic import AnthropicProvider
                _default_client = AnthropicProvider()
        else:
            # Legacy mode: use ClaudeClient directly
            _default_client = ClaudeClient()
            logger.info("Initialized legacy ClaudeClient")

    return _default_client


def get_provider() -> LLMProvider:
    """
    Get the current LLM provider instance.

    This is the recommended way to access the LLM provider in new code.

    Returns:
        LLMProvider: Current provider instance

    Example:
        ```python
        from kosmos.core.llm import get_provider

        provider = get_provider()
        response = provider.generate("What is machine learning?")
        print(response.content)
        ```
    """
    client = get_client(use_provider_system=True)

    # Ensure we return a provider instance
    if not isinstance(client, LLMProvider):
        raise TypeError(f"Expected LLMProvider, got {type(client)}")

    return client
