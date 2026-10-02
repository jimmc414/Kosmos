"""
Unit tests for ResearchDirectorAgent._handle_execute_experiment_action (viability plan P0-4).

The handler must read ExecutionResult.success, store an honest result row and
experiment status for both outcomes, keep the executed code, pin the sandbox
image from config, and halt the run when the sandbox is unavailable.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from kosmos.core.workflow import WorkflowState
from kosmos.db import get_session
from kosmos.db.models import Experiment, ExperimentStatus, Result
from kosmos.execution.executor import ExecutionResult
from tests.unit.agents.conftest import CODE, EXP_ID


@pytest.fixture
def director(db_director):
    return db_director


def _rows():
    with get_session() as session:
        results = session.query(Result).all()
        exp = session.query(Experiment).filter_by(id=EXP_ID).one()
        return (
            [(r.id, dict(r.data), r.p_value, r.effect_size) for r in results],
            (exp.status, exp.code_generated, exp.error_message),
        )


def _transitions(d):
    return [c.args[0] for c in d.workflow.transition_to.call_args_list]


async def test_success_stores_honest_row(director):
    director._code_executor = Mock(use_sandbox=True)
    director._code_executor.execute_with_data.return_value = ExecutionResult(
        success=True,
        return_value={"p_value": np.float64(0.01), "effect_size": 0.9, "data_source": "file", "n_samples": 64},
        execution_time=1.2,
    )

    await director._handle_execute_experiment_action(EXP_ID)

    rows, (status, code, _) = _rows()
    assert len(rows) == 1
    row_id, data, p_value, _ = rows[0]
    assert data["execution_success"] is True
    assert data["data_source"] == "file"
    assert data["executor_mode"] == "sandbox"
    assert p_value == 0.01
    assert status == ExperimentStatus.COMPLETED
    assert code == CODE
    assert director.research_plan.completed_experiments == [EXP_ID]
    assert director.research_plan.results == [row_id]
    assert WorkflowState.ANALYZING in _transitions(director)


async def test_failure_stores_failed_row(director):
    director._code_executor = Mock(use_sandbox=True)
    director._code_executor.execute_with_data.return_value = ExecutionResult(
        success=False,
        error="Container exited with code 1",
        error_type="ExecutionError",
        stderr="Traceback ... KeyError: 'group'",
    )

    await director._handle_execute_experiment_action(EXP_ID)

    rows, (status, _, error_message) = _rows()
    assert len(rows) == 1
    _, data, p_value, _ = rows[0]
    assert data["execution_success"] is False
    assert p_value is None
    assert "KeyError" in data["stderr_tail"]
    assert status == ExperimentStatus.FAILED
    assert error_message.startswith("ExecutionError:")
    assert director.research_plan.completed_experiments == []
    assert EXP_ID not in director.research_plan.experiment_queue
    assert WorkflowState.ANALYZING in _transitions(director)


async def test_sandbox_unavailable_halts_run(director):
    director._code_executor = None
    with patch("kosmos.execution.executor.CodeExecutor", side_effect=RuntimeError("Docker not available")):
        await director._handle_execute_experiment_action(EXP_ID)

    rows, (status, _, _) = _rows()
    assert len(rows) == 1
    assert rows[0][1]["error_type"] == "SandboxUnavailable"
    assert rows[0][1]["executor_mode"] == "none"
    assert status == ExperimentStatus.FAILED
    assert director.research_plan.has_converged is True
    assert director.research_plan.convergence_reason.startswith("halted: sandbox unavailable")
    assert _transitions(director) == [WorkflowState.ERROR]
    assert EXP_ID not in director.research_plan.experiment_queue


async def test_db_failure_leaves_no_phantom_result(director):
    director._code_executor = Mock(use_sandbox=True)
    director._code_executor.execute_with_data.return_value = ExecutionResult(
        success=True, return_value={"p_value": 0.01}
    )
    director._handle_error_with_recovery = Mock()

    with patch("kosmos.db.operations.create_result", side_effect=RuntimeError("db down")):
        await director._handle_execute_experiment_action(EXP_ID)

    director._handle_error_with_recovery.assert_called_once()
    assert director.research_plan.results == []


async def test_nan_p_value_is_stored_as_null(director):
    director._code_executor = Mock(use_sandbox=True)
    director._code_executor.execute_with_data.return_value = ExecutionResult(
        success=True, return_value={"p_value": float("nan")}
    )

    await director._handle_execute_experiment_action(EXP_ID)

    rows, _ = _rows()
    _, data, p_value, _ = rows[0]
    assert p_value is None
    assert data["p_value"] is None


@pytest.mark.parametrize("configured, expected", [(None, "kosmos-sandbox:latest"), ("kosmos-sandbox:pinned", "kosmos-sandbox:pinned")])
async def test_sandbox_image_comes_from_config(director, configured, expected):
    director._code_executor = None
    if configured:
        director.config["sandbox_image"] = configured
    with patch("kosmos.execution.executor.CodeExecutor") as mock_cls:
        mock_cls.return_value.execute_with_data.return_value = ExecutionResult(success=True, return_value={})
        mock_cls.return_value.use_sandbox = True
        await director._handle_execute_experiment_action(EXP_ID)

    assert mock_cls.call_args.kwargs["sandbox_config"] == {"image": expected}


def test_convergence_check_receives_hypotheses_and_results(director):
    """P1-4: the detector gets the real hypotheses and results, not empty lists."""
    from kosmos.db import operations
    from kosmos.models.result import ResultStatus
    from tests.unit.agents.conftest import H_ID

    with get_session() as session:
        operations.create_hypothesis(
            session, id="hyp-exec-2", research_question="Does CO2 predict temperature?",
            statement="Solar irradiance explains the temperature anomaly",
            rationale="Solar forcing changes the energy balance of the climate system",
            domain="climate",
        )
        operations.create_result(
            session, id="res-conv-1", experiment_id=EXP_ID,
            data={"execution_success": True}, supports_hypothesis=True,
        )
    director.research_plan.hypothesis_pool = [H_ID, "hyp-exec-2"]
    director.research_plan.completed_experiments = [EXP_ID]
    director.convergence_detector = Mock(check_convergence=Mock(return_value=MagicMock(
        should_stop=False, reason=MagicMock(value="x"), details="",
    )))

    director._check_convergence_direct()

    kwargs = director.convergence_detector.check_convergence.call_args.kwargs
    assert len(kwargs["hypotheses"]) == 2
    assert len(kwargs["results"]) == 1
    assert kwargs["results"][0].supports_hypothesis is True
    assert kwargs["results"][0].status == ResultStatus.SUCCESS
