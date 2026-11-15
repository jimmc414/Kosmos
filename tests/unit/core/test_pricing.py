"""
Unit tests for model pricing.

Tests that get_model_cost uses current per-token prices from llm-prices and
falls back to the local table for models it does not list.
"""

import pytest

from kosmos.core.pricing import get_model_cost


class TestGetModelCost:
    """Tests for get_model_cost."""

    def test_gpt_4o_current_price(self):
        # $2.50 input / $10 output per 1M tokens (was $5 / $15)
        assert get_model_cost("gpt-4o", 1_000_000, 1_000_000) == pytest.approx(12.5)

    def test_claude_opus_4_5_current_price(self):
        # $5 input / $25 output per 1M tokens (was $15 / $75)
        assert get_model_cost("claude-opus-4-5", 1_000_000, 1_000_000) == pytest.approx(30.0)

    def test_claude_3_5_haiku_current_price(self):
        assert get_model_cost("claude-3-5-haiku-20241022", 1_000_000, 1_000_000) == pytest.approx(
            4.8
        )

    def test_claude_sonnet_unchanged(self):
        assert get_model_cost("claude-sonnet-4-5", 1_000_000, 1_000_000) == pytest.approx(18.0)

    @pytest.mark.parametrize("model", ["gpt-5", "gpt-5.1", "gemini-2.5-pro"])
    def test_models_missing_from_table_now_have_a_cost(self, model):
        assert get_model_cost(model, 1_000_000, 1_000_000) > 0

    def test_dated_snapshot_id_resolves(self):
        assert get_model_cost("gpt-4o-2024-08-06", 1_000_000, 0) == pytest.approx(2.5)

    def test_ollama_models_stay_free(self):
        assert get_model_cost("ollama/llama3.1:8b", 1_000_000, 1_000_000) == 0.0

    def test_fallback_table_for_unlisted_model(self):
        # gpt-4 is not in llm-prices; the local table still prices it
        assert get_model_cost("gpt-4", 1_000_000, 1_000_000) == pytest.approx(90.0)

    def test_family_fallback(self):
        assert get_model_cost("my-custom-sonnet", 1_000_000, 1_000_000) == pytest.approx(18.0)

    def test_unknown_model_is_free(self):
        assert get_model_cost("totally-unknown-model", 1_000, 1_000) == 0.0
