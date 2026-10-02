"""
Tests for the Anthropic API options (viability plan A-1): KOSMOS_ANTHROPIC_API_KEY,
current-model request shapes, pricing, and selecting the claude_code provider.
"""

from unittest.mock import MagicMock, patch

import pytest

from kosmos.config import KosmosConfig
from kosmos.core.pricing import get_model_cost


@pytest.fixture
def no_key_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("KOSMOS_ANTHROPIC_API_KEY", raising=False)


def _response():
    r = MagicMock()
    r.content = [MagicMock(text="ok")]
    r.usage.input_tokens = 1
    r.usage.output_tokens = 1
    r.stop_reason = "end_turn"
    return r


def test_kosmos_key_selects_anthropic_without_anthropic_api_key(no_key_env, monkeypatch):
    from kosmos.core.providers.anthropic import AnthropicProvider

    monkeypatch.setenv("KOSMOS_ANTHROPIC_API_KEY", "sk-ant-api03-kosmos")
    config = KosmosConfig(LLM_PROVIDER="anthropic")
    assert config.claude.api_key == "sk-ant-api03-kosmos"

    with patch("kosmos.core.providers.anthropic.Anthropic") as mock_anthropic:
        AnthropicProvider({"enable_cache": False})
    assert mock_anthropic.call_args.kwargs["api_key"] == "sk-ant-api03-kosmos"


def test_anthropic_without_any_key_names_kosmos_variable(no_key_env):
    with pytest.raises(ValueError, match="KOSMOS_ANTHROPIC_API_KEY"):
        KosmosConfig(LLM_PROVIDER="anthropic")


@pytest.mark.parametrize("model, sends_temperature", [
    ("claude-opus-5-5", False),
    ("claude-sonnet-5-5", False),
    ("claude-fable-5-1", False),
    ("claude-haiku-4-5", True),
])
def test_temperature_only_sent_to_models_that_accept_it(no_key_env, model, sends_temperature):
    from kosmos.core.providers.anthropic import AnthropicProvider

    with patch("kosmos.core.providers.anthropic.Anthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create.return_value = _response()
        provider = AnthropicProvider({"api_key": "sk-ant-api03-x", "model": model, "enable_cache": False})
        provider.generate("hello", temperature=0.3)

    kwargs = mock_anthropic.return_value.messages.create.call_args.kwargs
    assert ("temperature" in kwargs) is sends_temperature


def test_default_model_is_current_opus(no_key_env):
    from kosmos.core.providers.anthropic import AnthropicProvider

    with patch("kosmos.core.providers.anthropic.Anthropic"):
        provider = AnthropicProvider({"api_key": "sk-ant-api03-x", "enable_cache": False})
    assert provider.model == "claude-opus-5-5"


def test_current_model_pricing():
    assert get_model_cost("claude-opus-5-5", 1_000_000, 1_000_000) == pytest.approx(24.0)
    assert get_model_cost("claude-sonnet-5-5", 1_000_000, 1_000_000) == pytest.approx(12.0)
    assert get_model_cost("claude-fable-5-1", 1_000_000, 1_000_000) == pytest.approx(60.0)


def test_claude_code_provider_needs_no_key(no_key_env):
    from kosmos.core.providers import get_provider_from_config
    from kosmos.core.providers.claude_code import ClaudeCodeProvider

    config = KosmosConfig(LLM_PROVIDER="claude_code")
    provider = get_provider_from_config(config)

    assert isinstance(provider, ClaudeCodeProvider)
    assert provider.model == config.claude_code.model
    assert config.get_active_model() == "claude-opus-5-5"
