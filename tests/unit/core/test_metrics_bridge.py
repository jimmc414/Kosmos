"""
Tests for the provider-to-metrics cost bridge (viability plan P2-4 (4)).

A provider call must reach the metrics collector with the cost the provider
computed, and get_api_statistics and the budget period cost must sum the
stored cost, pricing by model only when a call carries none.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("litellm")

from kosmos.core.metrics import get_metrics
from kosmos.core.pricing import get_model_cost
from kosmos.core.providers.litellm_provider import LiteLLMProvider


@pytest.fixture(autouse=True)
def fresh_metrics():
    get_metrics(reset=True)
    yield
    get_metrics(reset=True)


def _completion(prompt_tokens: int, completion_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        model="deepseek/deepseek-chat",
    )


def test_litellm_call_cost_reaches_metrics():
    provider = LiteLLMProvider({"model": "deepseek/deepseek-chat"})
    with patch("litellm.completion", return_value=_completion(1000, 500)):
        provider.generate("x")

    stats = get_metrics().get_api_statistics()

    assert stats["estimated_cost_usd"] == pytest.approx(
        get_model_cost("deepseek/deepseek-chat", 1000, 500)
    )
    assert stats["total_calls"] == 1
    assert get_metrics().api_calls == 1
    assert get_metrics().api_call_history[-1]["cost_usd"] == pytest.approx(provider.total_cost_usd)


def test_stored_cost_wins_over_model_pricing():
    m = get_metrics()
    m.record_api_call("claude-sonnet-4-5", 1000, 500, 0.0, cost_usd=0.5)
    m.record_api_call("deepseek/deepseek-chat", 1000, 500, 0.0)

    expected = 0.5 + get_model_cost("deepseek/deepseek-chat", 1000, 500)
    assert m.get_api_statistics()["estimated_cost_usd"] == pytest.approx(expected)
    assert m._calculate_period_cost() == pytest.approx(expected)


def test_zero_cost_is_priced_by_model():
    """A reported 0.0 means the backend gave no cost; the budget prices it by model."""
    m = get_metrics()
    m.record_api_call("claude-opus-4-5", 1000, 500, 0.0, cost_usd=0.0)
    m.record_api_call("ollama/llama3.1", 1000, 500, 0.0, cost_usd=0.0)

    expected = get_model_cost("claude-opus-4-5", 1000, 500)
    assert expected > 0
    assert m.get_api_statistics()["estimated_cost_usd"] == pytest.approx(expected)
    assert m._calculate_period_cost() == pytest.approx(expected)


def test_total_survives_history_truncation():
    m = get_metrics()
    for _ in range(1005):
        m.record_api_call("x", 0, 0, 0.0, cost_usd=0.001)

    assert len(m.api_call_history) == 1000
    assert m.get_api_statistics()["estimated_cost_usd"] == pytest.approx(1.005)
