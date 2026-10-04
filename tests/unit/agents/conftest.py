"""
Shared fixtures for director tests that need a real database and ResearchPlan.

db_director seeds one hypothesis (H_ID) and one experiment (EXP_ID) in an
in-memory SQLite database and gives the director a real ResearchPlan and a
mocked workflow. Tests import the constants from this module.
"""

from unittest.mock import Mock, MagicMock, patch

import pytest

import kosmos.db as kosmos_db
from kosmos.agents.research_director import ResearchDirectorAgent
from kosmos.core.workflow import ResearchPlan
from kosmos.db import get_session, init_database
from kosmos.db import operations
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
def db_director(in_memory_db, tmp_path):
    """A director with a real ResearchPlan, a mocked workflow, and one seeded experiment.

    Run seed 7; artifacts go under tmp_path/artifacts.
    """
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
            config={"max_iterations": 10, "random_seed": 7,
                    "artifacts_dir": str(tmp_path / "artifacts")},
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
