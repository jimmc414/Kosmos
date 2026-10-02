"""
Unit tests for ResearchDirectorAgent._handle_execute_experiment_action (viability plan P0-4).

The handler must read ExecutionResult.success, store an honest result row and
experiment status for both outcomes, keep the executed code, pin the sandbox
image from config, and halt the run when the sandbox is unavailable.
"""

from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest

import kosmos.db as kosmos_db
from kosmos.agents.research_director import ResearchDirectorAgent
from kosmos.core.workflow import ResearchPlan, WorkflowState
from kosmos.db import get_session, init_database
from kosmos.db import operations
from kosmos.db.models import Experiment, ExperimentStatus, Result
from kosmos.execution.executor import ExecutionResult
from kosmos.models.experiment import (
    ExperimentProtocol,
    ExperimentType,
    ProtocolStep,
    ResourceRequirements,
    StatisticalTest,
    StatisticalTestSpec,
    Variable,
    VariableType,
)

H_ID = "hyp-exec-1"
EXP_ID = "exp-exec-1"
CODE = "results = {}"


def _ttest_protocol() -> ExperimentProtocol:
    """Same shape as the ttest_protocol fixture in tests/unit/execution/test_code_generator.py."""
    return ExperimentProtocol(
        id=EXP_ID,
        name="T-Test Experiment Protocol",
        hypothesis_id=H_ID,
        domain="statistics",
        description="T-test comparison experiment for statistical analysis of treatment vs control groups",
        objective="Compare means between two groups using T-test",
        experiment_type=ExperimentType.DATA_ANALYSIS,
        statistical_tests=[
            StatisticalTestSpec(
                test_type=StatisticalTest.T_TEST,
                description="Two-sample T-test for group comparison",
                null_hypothesis="No difference between group means",
                variables=["group", "measurement"],
            )
        ],
        steps=[
            ProtocolStep(
                step_number=1,
                title="Execute T-test",
                description="Load data and run T-test analysis",
                action="run_ttest",
                expected_duration_minutes=5,
            )
        ],
        variables={
            "group": Variable(name="group", type=VariableType.INDEPENDENT, description="Group variable"),
            "measurement": Variable(name="measurement", type=VariableType.DEPENDENT, description="Measurement"),
        },
        resource_requirements=ResourceRequirements(
            estimated_runtime_seconds=300, cpu_cores=1, memory_gb=1, storage_gb=0.1
        ),
        data_requirements={"format": "csv", "columns": ["group", "measurement"]},
        expected_duration_minutes=10,
    )


@pytest.fixture
def in_memory_db():
    """Point kosmos.db at a fresh in-memory SQLite database; restore the old engine afterwards."""
    saved = (kosmos_db._engine, kosmos_db._SessionLocal)
    init_database("sqlite:///:memory:")
    yield
    kosmos_db.reset_database()
    kosmos_db._engine, kosmos_db._SessionLocal = saved


@pytest.fixture
def director(in_memory_db):
    """A director with a real ResearchPlan, a mocked workflow, and one seeded experiment."""
    with patch('kosmos.agents.research_director.get_client') as mock_client, \
         patch('kosmos.agents.research_director.get_world_model') as mock_wm, \
         patch('kosmos.agents.research_director.SkillLoader') as mock_skills, \
         patch('kosmos.db.init_from_config'):
        mock_client.return_value = MagicMock()
        mock_wm.return_value = MagicMock()
        mock_skills.return_value = MagicMock()
        mock_skills.return_value.load_skills_for_task.return_value = ""

        d = ResearchDirectorAgent(
            research_question="Does CO2 predict temperature?",
            domain="climate",
            config={"max_iterations": 10},
        )

        proto = _ttest_protocol()
        protocol_json = proto.to_dict()
        ExperimentProtocol.model_validate(protocol_json)  # round-trip must hold for the handler
        with get_session() as session:
            operations.create_hypothesis(
                session, id=H_ID, research_question="Does CO2 predict temperature?",
                statement="CO2 concentration predicts the temperature anomaly",
                rationale="Radiative forcing increases with CO2 concentration",
                domain="climate",
            )
            operations.create_experiment(
                session, id=EXP_ID, hypothesis_id=H_ID, experiment_type="computational",
                description="d", protocol=protocol_json, domain="climate",
            )

        d.research_plan = ResearchPlan(research_question="q", max_iterations=10)
        d.research_plan.add_hypothesis(H_ID)
        d.research_plan.add_experiment(EXP_ID)
        d.workflow = MagicMock()
        d._code_generator = Mock(generate=Mock(return_value=CODE))
        d.data_path = "/x/data.csv"
        yield d


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
