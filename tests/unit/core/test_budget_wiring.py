"""
Tests for budget wiring (viability plan P1-5).

--budget must arm enforcement, every provider call must reach the metrics
collector, and period cost must be priced by the model that served each call.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from kosmos.core.metrics import get_metrics
from kosmos.core.pricing import get_model_cost
from kosmos.core.providers.base import LLMProvider, UsageStats
from kosmos.core.workflow import NextAction, WorkflowState


@pytest.fixture(autouse=True)
def fresh_metrics():
    get_metrics(reset=True)
    yield
    get_metrics(reset=True)


def _director(config):
    from kosmos.agents.research_director import ResearchDirectorAgent

    with patch('kosmos.agents.research_director.get_client', return_value=MagicMock()), \
         patch('kosmos.agents.research_director.get_world_model', return_value=MagicMock()), \
         patch('kosmos.agents.research_director.SkillLoader') as mock_skills, \
         patch('kosmos.db.init_from_config'):
        mock_skills.return_value.load_skills_for_task.return_value = ""
        return ResearchDirectorAgent(research_question="q", domain="climate", config=config)


class _P(LLMProvider):
    def generate(self, prompt, **kwargs):
        pass

    async def generate_async(self, prompt, **kwargs):
        pass

    def generate_with_messages(self, messages, **kwargs):
        pass

    def generate_structured(self, prompt, schema, **kwargs):
        pass

    def get_model_info(self):
        pass


def test_budget_flag_arms_enforcement():
    _director({"max_iterations": 1, "budget_usd": 1.0})
    assert get_metrics().budget_enabled is True
    assert get_metrics().budget_limit_usd == 1.0


def test_no_budget_leaves_enforcement_off():
    _director({"max_iterations": 1})
    assert get_metrics().budget_enabled is False


def test_provider_calls_are_recorded_and_priced_by_model():
    _P({})._update_usage_stats(UsageStats(
        input_tokens=1000, output_tokens=500, total_tokens=1500, cost_usd=0.0,
        model="deepseek/deepseek-chat", provider="litellm/deepseek",
        timestamp=datetime.now(timezone.utc),
    ))

    metrics = get_metrics()
    assert metrics.api_calls == 1
    assert metrics._calculate_period_cost() == pytest.approx(
        get_model_cost("deepseek/deepseek-chat", 1000, 500)
    )


def test_exceeded_budget_converges():
    director = _director({"max_iterations": 10})
    director.workflow = MagicMock()
    director.workflow.current_state = WorkflowState.GENERATING_HYPOTHESES
    get_metrics().configure_budget(limit_usd=0.0001)
    get_metrics().record_api_call("claude-sonnet-4-5", 10000, 10000, 1.0)

    assert director.decide_next_action() == NextAction.CONVERGE
    targets = [c.args[0] for c in director.workflow.transition_to.call_args_list]
    assert WorkflowState.CONVERGED in targets
