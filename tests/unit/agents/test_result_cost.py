"""
Per-result LLM cost attribution in ResearchDirectorAgent (viability plan P2-4 (3)).

The cost spent while designing, executing and analysing an experiment is the
growth of llm_client.total_cost_usd across those handlers; it is stored on the
result row at creation and completed through update_result_validation.
"""

from unittest.mock import MagicMock, Mock

import pytest

from kosmos.agents.data_analyst import ResultInterpretation
from kosmos.db import get_session
from kosmos.db.models import Result
from kosmos.execution.executor import ExecutionResult
from tests.unit.agents.conftest import EXP_ID, H_ID


class _CostingClient(MagicMock):
    """An LLM client whose cumulative cost is a plain float."""


def _client(cost=0.0):
    client = _CostingClient()
    client.total_cost_usd = cost
    return client


def _interpretation():
    return ResultInterpretation(
        experiment_id=EXP_ID, hypothesis_supported=True, confidence=0.9,
        summary="S", key_findings=["k"], significance_interpretation="",
        biological_significance=None, comparison_to_prior_work=None,
        potential_confounds=[], follow_up_experiments=[], anomalies_detected=[],
        patterns_detected=[], overall_assessment="",
    )


def _cost_of_only_result():
    with get_session() as session:
        r = session.query(Result).one()
        return r.id, r.cost_usd


async def test_design_execute_and_analyze_costs_land_on_the_result(db_director):
    d = db_director
    d.llm_client = _client(1.0)
    d._protocol_cost[EXP_ID] = 0.002  # spent by the design handler for this experiment

    def generate(*args, **kwargs):
        d.llm_client.total_cost_usd += 0.003  # code generation calls the LLM
        return "results = {}"

    d._code_generator = Mock(generate=Mock(side_effect=generate))
    d._code_executor = Mock(use_sandbox=True)
    d._code_executor.execute_with_data.return_value = ExecutionResult(
        success=True, return_value={"data_source": "synthetic"}, execution_time=0.1,
    )

    await d._handle_execute_experiment_action(EXP_ID)

    result_id, cost = _cost_of_only_result()
    assert cost == pytest.approx(0.005)
    assert EXP_ID not in d._protocol_cost

    def interpret(**kwargs):
        d.llm_client.total_cost_usd += 0.004  # the analyst calls the LLM
        return _interpretation()

    d._data_analyst = Mock(interpret_results=Mock(side_effect=interpret))

    await d._handle_analyze_result_action(result_id)

    _, cost = _cost_of_only_result()
    assert cost == pytest.approx(0.009)
    with get_session() as session:
        assert session.query(Result).one().validation_status == "unvalidated"


async def test_design_cost_is_kept_per_experiment(db_director):
    d = db_director
    d.llm_client = _client(0.5)
    protocol = Mock(id="exp-new-1")

    def design(**kwargs):
        d.llm_client.total_cost_usd += 0.001
        return Mock(protocol=protocol)

    d._experiment_designer = Mock(design_experiment=Mock(side_effect=design))
    d._persist_protocol_to_graph = Mock()

    await d._handle_design_experiment_action(H_ID)

    assert d._protocol_cost["exp-new-1"] == pytest.approx(0.001)


async def test_client_without_numeric_cost_leaves_cost_null(db_director):
    d = db_director
    d.llm_client = MagicMock()  # total_cost_usd is a MagicMock, not a number
    d._code_executor = Mock(use_sandbox=True)
    d._code_executor.execute_with_data.return_value = ExecutionResult(success=True, return_value={})

    await d._handle_execute_experiment_action(EXP_ID)

    _, cost = _cost_of_only_result()
    assert cost is None
