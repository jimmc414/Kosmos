"""
Claude Code provider: Anthropic models through the user's Claude Code login.

Runs each request as a single-turn, tool-free Claude Agent SDK query. The SDK
launches the `claude` CLI, which authenticates with the Claude Code login
(for example a Max subscription), so no Anthropic API key is involved. The
provider never reads ~/.claude/.credentials.json itself.

ANTHROPIC_API_KEY overrides the Claude Code login, so it is removed from the
process environment when this provider is constructed. Cost figures are the
SDK's API-equivalent estimate; the subscription is not billed per token.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )
    HAS_CLAUDE_AGENT_SDK = True
except ImportError:
    HAS_CLAUDE_AGENT_SDK = False
    AssistantMessage = ClaudeAgentOptions = ResultMessage = TextBlock = query = None

from kosmos.core.providers.base import (
    LLMProvider,
    LLMResponse,
    Message,
    ProviderAPIError,
    UsageStats,
)
from kosmos.core.utils.json_parser import parse_json_response

logger = logging.getLogger(__name__)

DEFAULT_CLAUDE_CODE_MODEL = "claude-opus-5-5"
DEFAULT_CLAUDE_CODE_FALLBACK_MODEL = "claude-sonnet-5-5"
COST_LABEL = "api_equivalent_usd"

# Structured output may take an extra turn for the CLI's schema tool
_PLAIN_MAX_TURNS = 1
_STRUCTURED_MAX_TURNS = 3


class ClaudeCodeProvider(LLMProvider):
    """
    Anthropic models through the Claude Code login, via the Claude Agent SDK.

    Requires `pip install claude-agent-sdk`, the `claude` CLI on PATH (or
    `cli_path`), and a completed `claude login`.

    Example:
        ```python
        provider = ClaudeCodeProvider({"model": "claude-opus-5-5"})
        response = provider.generate("Summarize the hypothesis in one line")
        print(response.content)
        ```
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the provider.

        Args:
            config: Configuration dict with keys:
                - model: Model id (default claude-opus-5-5)
                - fallback_model: Model the CLI falls back to (default claude-sonnet-5-5)
                - timeout: Seconds per request (default 300)
                - cli_path: Path to the claude binary (default: found on PATH)
                - max_thinking_tokens: Optional thinking budget passed to the CLI
                - oauth_token: Optional CLAUDE_CODE_OAUTH_TOKEN for the CLI

        Raises:
            ProviderAPIError: If claude-agent-sdk is not installed
        """
        super().__init__(config)

        if not HAS_CLAUDE_AGENT_SDK:
            raise ProviderAPIError(
                "claude_code",
                "claude-agent-sdk is not installed. Install it with: pip install claude-agent-sdk",
                recoverable=False,
            )

        if os.environ.pop("ANTHROPIC_API_KEY", None):
            logger.warning(
                "ANTHROPIC_API_KEY removed from the process environment so the Claude Code login is used"
            )

        self.model = config.get("model") or DEFAULT_CLAUDE_CODE_MODEL
        fallback = config.get("fallback_model", DEFAULT_CLAUDE_CODE_FALLBACK_MODEL)
        # The CLI rejects a fallback identical to the main model
        self.fallback_model = fallback if fallback and fallback != self.model else None
        self.timeout = int(config.get("timeout") or 300)
        self.cli_path = config.get("cli_path") or None
        self.max_thinking_tokens = config.get("max_thinking_tokens")
        self.oauth_token = config.get("oauth_token") or None

        logger.info(f"Claude Code provider initialized with model {self.model}")

    # ------------------------------------------------------------------
    # Core SDK call
    # ------------------------------------------------------------------

    def _options(self, system: Optional[str], schema: Optional[Dict[str, Any]]):
        return ClaudeAgentOptions(
            model=self.model,
            fallback_model=self.fallback_model,
            system_prompt=system or "",
            tools=[],
            allowed_tools=[],
            max_turns=_STRUCTURED_MAX_TURNS if schema else _PLAIN_MAX_TURNS,
            setting_sources=[],  # never load the user's CLAUDE.md or settings into a Kosmos call
            cli_path=self.cli_path,
            max_thinking_tokens=self.max_thinking_tokens,
            env={"CLAUDE_CODE_OAUTH_TOKEN": self.oauth_token} if self.oauth_token else {},
            output_format={"type": "json_schema", "schema": schema} if schema else None,
        )

    async def _run(
        self,
        prompt: str,
        system: Optional[str],
        schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Any, Any]:
        """Run one query; return (text, structured_output, ResultMessage)."""
        texts: List[str] = []
        result = None
        async for message in query(prompt=prompt, options=self._options(system, schema)):
            if isinstance(message, AssistantMessage):
                texts.extend(b.text for b in message.content if isinstance(b, TextBlock))
            elif isinstance(message, ResultMessage):
                result = message

        if result is None or result.is_error:
            detail = getattr(result, "result", None) or getattr(result, "subtype", None) or "no result message"
            raise ProviderAPIError("claude_code", f"Claude Code run failed: {detail}", recoverable=True)

        text = "".join(texts) or (result.result or "")
        self._record_usage(result)
        return text, result.structured_output, result

    async def _run_with_timeout(self, prompt: str, system: Optional[str], schema: Optional[Dict[str, Any]] = None):
        try:
            return await asyncio.wait_for(self._run(prompt, system, schema), timeout=self.timeout)
        except asyncio.TimeoutError as e:
            raise ProviderAPIError(
                "claude_code", f"Claude Code run timed out after {self.timeout}s", raw_error=e, recoverable=True
            )

    def _run_sync(self, prompt: str, system: Optional[str], schema: Optional[Dict[str, Any]] = None):
        """Run a query from synchronous code, including from inside a running event loop."""
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False

        try:
            if not in_loop:
                return asyncio.run(self._run_with_timeout(prompt, system, schema))
            # The director's handlers run inside the CLI's event loop; run the query on a
            # worker thread with its own loop instead of blocking on this one.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._run_with_timeout(prompt, system, schema))
                return future.result(timeout=self.timeout + 30)
        except ProviderAPIError:
            raise
        except Exception as e:
            raise ProviderAPIError("claude_code", f"Claude Code call failed: {e}", raw_error=e, recoverable=True)

    @staticmethod
    def _token_counts(result) -> Tuple[int, int]:
        """Return (input, output) tokens; input includes cache creation and cache reads.

        The CLI reports cached prompt tokens separately. Counting them keeps the metrics
        collector, and so --budget, close to the SDK's own cost estimate.
        """
        usage = result.usage or {}
        input_tokens = sum(
            int(usage.get(key, 0) or 0)
            for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        return input_tokens, int(usage.get("output_tokens", 0) or 0)

    def _record_usage(self, result) -> UsageStats:
        input_tokens, output_tokens = self._token_counts(result)
        stats = UsageStats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=result.total_cost_usd or 0.0,
            model=self.model,
            provider="claudecode/subscription",
            timestamp=datetime.now(timezone.utc),
        )
        self._update_usage_stats(stats)
        return stats

    def _response(self, text: str, result) -> LLMResponse:
        input_tokens, output_tokens = self._token_counts(result)
        return LLMResponse(
            content=text,
            usage=UsageStats(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=result.total_cost_usd or 0.0,
                model=self.model,
                provider="claudecode/subscription",
                timestamp=datetime.now(timezone.utc),
            ),
            model=self.model,
            finish_reason="stop",
            raw_response=result,
            metadata={"cost_label": COST_LABEL},
        )

    @staticmethod
    def _ignored_params(max_tokens, temperature, stop_sequences) -> None:
        logger.debug(
            "Claude Code provider ignores max_tokens=%s, temperature=%s, stop_sequences=%s "
            "(the Agent SDK has no such parameters)",
            max_tokens, temperature, stop_sequences,
        )

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

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
        Generate text through Claude Code.

        Args:
            prompt: The user prompt
            system: Optional system prompt (replaces Claude Code's default prompt)
            max_tokens: Accepted for interface compatibility; ignored
            temperature: Accepted for interface compatibility; ignored
            stop_sequences: Accepted for interface compatibility; ignored
            **kwargs: Ignored

        Returns:
            LLMResponse: Unified response object

        Raises:
            ProviderAPIError: If the CLI run fails or times out
        """
        self._ignored_params(max_tokens, temperature, stop_sequences)
        text, _, result = self._run_sync(prompt, system)
        return self._response(text, result)

    async def generate_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text asynchronously through Claude Code."""
        self._ignored_params(max_tokens, temperature, stop_sequences)
        text, _, result = await self._run_with_timeout(prompt, system)
        return self._response(text, result)

    def generate_with_messages(
        self,
        messages: List[Message],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """
        Generate from a conversation history as one single-turn query.

        System messages become the system prompt; other turns are joined with role prefixes.
        """
        system = "\n\n".join(m.content for m in messages if m.role == "system") or None
        turns = [f"{m.role.upper()}: {m.content}" for m in messages if m.role != "system"]
        return self.generate("\n\n".join(turns), system=system, max_tokens=max_tokens, temperature=temperature)

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
        Generate a JSON object, using the CLI's JSON-schema output when available.

        Falls back to parsing the reply text when the CLI returns no structured output.

        Raises:
            ProviderAPIError: If the run fails or no valid JSON comes back
        """
        self._ignored_params(max_tokens, temperature, None)
        json_system = (system or "") + (
            "\n\nYou must respond with valid JSON matching this schema:\n" + json.dumps(schema, indent=2)
        )
        text, structured, _ = self._run_sync(prompt, json_system, schema=schema)
        if isinstance(structured, dict):
            return structured
        try:
            return parse_json_response(text, schema=schema)
        except Exception as e:
            raise ProviderAPIError(
                "claude_code", f"Invalid JSON response: {e}", raw_error=e, recoverable=False
            )

    def get_model_info(self) -> Dict[str, Any]:
        """Return the model, provider, auth method and CLI version."""
        return {
            "name": self.model,
            "fallback_model": self.fallback_model,
            "provider": "claude_code",
            "auth": "claude-code-login",
            "cost_label": COST_LABEL,
            "cli_version": self.cli_version(self.cli_path),
        }

    def get_usage_stats(self) -> Dict[str, Any]:
        """Usage statistics; cost is the SDK's API-equivalent estimate."""
        stats = super().get_usage_stats()
        stats.update({"auth": "claude-code-login", "cost_label": COST_LABEL})
        return stats

    @staticmethod
    def cli_version(cli_path: Optional[str] = None) -> Optional[str]:
        """Return `claude --version` output, or None when the CLI is missing or fails."""
        binary = cli_path or shutil.which("claude")
        if not binary:
            return None
        try:
            out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
