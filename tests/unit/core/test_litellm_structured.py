"""
Unit tests for LiteLLMProvider.generate_structured: JSON mode, tolerant
parsing and the single repair call (plan P2-6). litellm.completion is
patched, so no network calls are made.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("litellm")

from kosmos.core.providers.base import ProviderAPIError
from kosmos.core.providers.litellm_provider import LiteLLMProvider
from kosmos.core.utils.json_parser import (
    JSONParseError,
    parse_json_array_response,
    parse_json_response,
)


def _completion(content: str) -> SimpleNamespace:
    """A litellm.completion return value carrying `content`."""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="deepseek/deepseek-chat",
    )


@pytest.fixture
def provider():
    with patch("kosmos.core.metrics.get_metrics", return_value=MagicMock()):
        yield LiteLLMProvider({"model": "deepseek/deepseek-chat"})


SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}}


class TestJSONMode:

    def test_fenced_preamble_trailing_comma_parsed_with_json_mode(self, provider):
        content = 'Sure. ```json\n{"name": "x", "steps": [],}\n```'
        with patch("litellm.completion", return_value=_completion(content)) as completion:
            result = provider.generate_structured("make a plan", schema=SCHEMA)

        assert result == {"name": "x", "steps": []}
        assert completion.call_count == 1
        assert completion.call_args_list[0].kwargs["response_format"] == {"type": "json_object"}

    def test_rejected_json_mode_falls_back_once_per_instance(self, provider):
        clean = _completion('{"name": "y"}')
        with patch(
            "litellm.completion",
            side_effect=[
                Exception("litellm.BadRequestError: response_format not supported"),
                clean,
                clean,
            ],
        ) as completion:
            first = provider.generate_structured("q1", schema=SCHEMA)
            second = provider.generate_structured("q2", schema=SCHEMA)

        assert first == {"name": "y"}
        assert second == {"name": "y"}
        assert completion.call_count == 3
        calls = completion.call_args_list
        assert calls[0].kwargs["response_format"] == {"type": "json_object"}
        assert "response_format" not in calls[1].kwargs
        assert "response_format" not in calls[2].kwargs
        assert provider._supports_json_mode is False

    def test_unrelated_error_is_not_retried(self, provider):
        with patch(
            "litellm.completion", side_effect=Exception("connection reset by peer")
        ) as completion:
            with pytest.raises(ProviderAPIError):
                provider.generate_structured("q", schema=SCHEMA)

        assert completion.call_count == 1
        assert provider._supports_json_mode is True


class TestRepair:

    def test_invalid_json_gets_one_repair_call(self, provider):
        with patch(
            "litellm.completion",
            side_effect=[_completion("not json"), _completion('{"a":1}')],
        ) as completion:
            result = provider.generate_structured("q", schema=SCHEMA)

        assert result == {"a": 1}
        assert completion.call_count == 2
        repair_kwargs = completion.call_args_list[1].kwargs
        assert repair_kwargs["temperature"] == 0.0
        assert "not valid JSON" in repair_kwargs["messages"][-1]["content"]
        assert "not json" in repair_kwargs["messages"][-1]["content"]

    def test_failed_repair_raises_recoverable_error(self, provider):
        with patch(
            "litellm.completion",
            side_effect=[_completion("not json"), _completion("still not json")],
        ):
            with pytest.raises(ProviderAPIError) as exc_info:
                provider.generate_structured("q", schema=SCHEMA)

        assert exc_info.value.recoverable is True
        assert "still not json" in exc_info.value.message

    def test_missing_required_keys_logged(self, provider, caplog, monkeypatch):
        # An earlier test that migrates a fresh database runs alembic/env.py's
        # fileConfig(), which disables every logger that already exists; re-enable
        # this module's logger so the assertion does not depend on test order.
        monkeypatch.setattr(
            logging.getLogger("kosmos.core.providers.litellm_provider"), "disabled", False
        )
        schema = {"type": "object", "required": ["name", "steps"]}
        with patch("litellm.completion", return_value=_completion('{"name": "x"}')):
            with caplog.at_level(logging.WARNING, logger="kosmos.core.providers.litellm_provider"):
                result = provider.generate_structured("q", schema=schema)

        assert result == {"name": "x"}
        assert "missing required keys: steps" in caplog.text


class TestArrayParser:

    def test_array_with_preamble(self):
        result = parse_json_array_response('Here: [{"statement":"s"}]')
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["statement"] == "s"

    def test_array_in_fence_with_trailing_comma(self):
        text = '```json\n[{"statement": "a"}, {"statement": "b"},]\n```'
        assert parse_json_array_response(text) == [{"statement": "a"}, {"statement": "b"}]

    def test_array_parser_rejects_object(self):
        with pytest.raises(JSONParseError):
            parse_json_array_response('{"statement": "s"}')

    def test_object_parser_rejects_top_level_array(self):
        with pytest.raises(JSONParseError):
            parse_json_response('[1, 2, 3]')
