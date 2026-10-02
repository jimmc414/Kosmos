"""
Tests for ClaudeCodeProvider (viability plan A-1).

The Agent SDK's query() is patched, so no test launches the claude binary.
"""

import os
from unittest.mock import patch

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from kosmos.core.metrics import get_metrics
from kosmos.core.providers.base import ProviderAPIError
from kosmos.core.providers.claude_code import ClaudeCodeProvider


def _result(is_error=False, structured=None, text="hi"):
    return ResultMessage(
        subtype="error_during_execution" if is_error else "success",
        duration_ms=1, duration_api_ms=1, is_error=is_error, num_turns=1, session_id="s",
        total_cost_usd=0.0012, usage={"input_tokens": 10, "output_tokens": 5},
        result=text, structured_output=structured,
    )


def _fake_query(captured, text="hi", structured=None, is_error=False):
    async def fake(prompt, options):
        captured["prompt"] = prompt
        captured["options"] = options
        yield AssistantMessage(content=[TextBlock(text=text)], model="claude-opus-5-5", parent_tool_use_id=None, error=None)
        yield _result(is_error=is_error, structured=structured, text=text)
    return fake


@pytest.fixture(autouse=True)
def fresh_metrics():
    get_metrics(reset=True)
    yield
    get_metrics(reset=True)


def test_generate_runs_a_tool_free_single_turn_query():
    captured = {}
    with patch("kosmos.core.providers.claude_code.query", _fake_query(captured)):
        provider = ClaudeCodeProvider({"model": "claude-opus-5-5"})
        response = provider.generate("x", system="be brief")

    assert response.content == "hi"
    assert provider.request_count == 1
    assert provider.total_cost_usd == pytest.approx(0.0012)
    options = captured["options"]
    assert options.tools == []
    assert options.allowed_tools == []
    assert options.max_turns == 1
    assert options.setting_sources == []
    assert options.model == "claude-opus-5-5"
    assert options.system_prompt == "be brief"
    assert options.env == {}


def test_construction_removes_anthropic_api_key(monkeypatch, caplog):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    ClaudeCodeProvider({})
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert "ANTHROPIC_API_KEY removed" in caplog.text


@pytest.mark.parametrize("structured, text, expected", [
    ({"a": 1}, "ignored", {"a": 1}),
    (None, 'Here: {"a": 2}', {"a": 2}),
])
def test_generate_structured(structured, text, expected):
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    captured = {}
    with patch("kosmos.core.providers.claude_code.query", _fake_query(captured, text=text, structured=structured)):
        result = ClaudeCodeProvider({}).generate_structured("x", schema)

    assert result == expected
    assert captured["options"].output_format == {"type": "json_schema", "schema": schema}


async def test_sync_generate_works_inside_a_running_loop():
    captured = {}
    with patch("kosmos.core.providers.claude_code.query", _fake_query(captured)):
        response = ClaudeCodeProvider({}).generate("x")
    assert response.content == "hi"


def test_error_result_raises():
    with patch("kosmos.core.providers.claude_code.query", _fake_query({}, is_error=True)):
        with pytest.raises(ProviderAPIError):
            ClaudeCodeProvider({}).generate("x")


def test_calls_reach_the_metrics_collector():
    with patch("kosmos.core.providers.claude_code.query", _fake_query({})):
        ClaudeCodeProvider({}).generate("x")
    assert get_metrics().api_calls == 1


def test_identical_fallback_model_is_dropped():
    provider = ClaudeCodeProvider({"model": "claude-sonnet-5-5", "fallback_model": "claude-sonnet-5-5"})
    assert provider.fallback_model is None


def test_oauth_token_is_passed_to_the_cli_only():
    captured = {}
    with patch("kosmos.core.providers.claude_code.query", _fake_query(captured)):
        ClaudeCodeProvider({"oauth_token": "tok"}).generate("x")
    assert captured["options"].env == {"CLAUDE_CODE_OAUTH_TOKEN": "tok"}
    assert os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") != "tok"


def test_cached_prompt_tokens_count_as_input():
    async def fake(prompt, options):
        yield ResultMessage(
            subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=1,
            session_id="s", total_cost_usd=0.05, result="hi", structured_output=None,
            usage={"input_tokens": 4, "cache_creation_input_tokens": 9000,
                   "cache_read_input_tokens": 1000, "output_tokens": 50},
        )

    with patch("kosmos.core.providers.claude_code.query", fake):
        provider = ClaudeCodeProvider({})
        response = provider.generate("x")

    assert provider.total_input_tokens == 10004
    assert response.usage.input_tokens == 10004
    assert response.content == "hi"
