"""
Tests for dataset schema and variable-to-column binding (P2-1).

The designer binds every independent and dependent variable to a real column
of the supplied dataset; the code generator analyses those columns instead of
positional numeric columns or LLM-invented names.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from kosmos.agents.experiment_designer import ExperimentDesignerAgent, UnboundVariableError
from kosmos.core.workflow import ResearchPlan
from kosmos.execution.code_generator import ExperimentCodeGenerator
from kosmos.execution.data_schema import describe_dataset, resolve_column
from kosmos.execution.executor import CodeExecutor
from kosmos.models.experiment import (
    ExperimentProtocol,
    ExperimentType,
    ProtocolStep,
    ResourceRequirements,
    Variable,
    VariableType,
)
from kosmos.models.hypothesis import Hypothesis

CLIMATE_CSV = Path(__file__).resolve().parents[3] / "evaluation" / "data" / "climate_co2_temperature_test.csv"
CLIMATE_COLUMNS = [
    "year", "co2_ppm", "temp_anomaly_c", "solar_irradiance_wm2",
    "volcanic_aerosol_index", "enso_index", "co2_growth_rate", "decade",
]


@pytest.fixture(scope="module")
def schema():
    return describe_dataset(str(CLIMATE_CSV))


@pytest.fixture
def hypothesis():
    return Hypothesis(
        id="hyp-bind-1",
        research_question="Does CO2 concentration predict temperature anomaly?",
        statement="Higher atmospheric CO2 concentration predicts a higher temperature anomaly",
        rationale="CO2 is a greenhouse gas that traps outgoing infrared radiation",
        domain="climate_science",
    )


def _llm_protocol(variables):
    """A structured protocol as the LLM returns it."""
    return {
        "name": "CO2 and temperature anomaly",
        "description": "Test whether CO2 concentration tracks the temperature anomaly over time",
        "objective": "Measure the association between CO2 and temperature",
        "steps": [
            {"step_number": 1, "title": "Load", "description": "Load the dataset", "action": "load"},
            {"step_number": 2, "title": "Test", "description": "Run the test", "action": "test"},
            {"step_number": 3, "title": "Report", "description": "Report results", "action": "report"},
        ],
        "variables": variables,
        "statistical_tests": [{"test_type": "correlation", "description": "Pearson correlation"}],
    }


def _designer(response):
    with patch("kosmos.agents.experiment_designer.get_client") as mock_get_client:
        llm = MagicMock()
        llm.generate_structured.return_value = response
        mock_get_client.return_value = llm
        agent = ExperimentDesignerAgent(config={"use_llm_enhancement": False})
    return agent, llm


def _bound_protocol(x_col, y_col):
    return ExperimentProtocol(
        id="exp-bind-1",
        name="Bound analysis",
        hypothesis_id="hyp-bind-1",
        domain="climate_science",
        description="Analysis of two bound columns of the climate dataset",
        objective="Test the association between the bound columns",
        experiment_type=ExperimentType.COMPUTATIONAL,
        steps=[ProtocolStep(step_number=1, title="Analyse", description="Run the analysis", action="analyse")],
        variables={
            "predictor": Variable(name="predictor", type=VariableType.INDEPENDENT,
                                  description="Independent variable from the dataset", column=x_col),
            "outcome": Variable(name="outcome", type=VariableType.DEPENDENT,
                                description="Dependent variable from the dataset", column=y_col),
        },
        resource_requirements=ResourceRequirements(),
    )


def _run(code):
    result = CodeExecutor(use_sandbox=False).execute_with_data(code, str(CLIMATE_CSV))
    assert result.success, result.error
    return result.return_value


class TestDescribeDataset:
    def test_climate_schema(self, schema):  # (a)
        assert schema.columns == CLIMATE_COLUMNS
        assert len(schema.numeric_columns) == 7
        assert "decade" not in schema.numeric_columns
        assert len(schema.categorical_columns["decade"]) == 7
        assert schema.n_rows == 64
        assert len(schema.sha256) == 64
        assert schema.dtypes["decade"] == "object"

    def test_prompt_block_lists_every_column(self, schema):
        block = schema.to_prompt_block()
        for col in CLIMATE_COLUMNS:
            assert f"- {col} (" in block
        assert "1960s" in block

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            describe_dataset(str(tmp_path / "absent.csv"))

    @pytest.mark.parametrize("proposed,expected", [
        ("co2_ppm", "co2_ppm"),
        ("CO2_ppm", "co2_ppm"),
        ("CO2 ppm", "co2_ppm"),
        ("temp_anomaly", "temp_anomaly_c"),
        ("ocean_heat", None),
        (None, None),
    ])
    def test_resolve_column(self, proposed, expected):
        assert resolve_column(proposed, CLIMATE_COLUMNS) == expected


class TestDesignerBinding:
    def test_columns_resolved(self, schema, hypothesis):  # (b)
        agent, llm = _designer(_llm_protocol({
            "co2": {"type": "independent", "description": "CO2 concentration in ppm", "column": "CO2_ppm"},
            "temp": {"type": "dependent", "description": "Temperature anomaly in C", "column": "temp_anomaly_c"},
        }))

        response = agent.design_experiment(hypothesis, store_in_db=False, dataset_schema=schema)

        variables = response.protocol.variables
        assert variables["co2"].column == "co2_ppm"
        assert variables["temp"].column == "temp_anomaly_c"
        assert response.protocol.to_dict()["variables"]["co2"]["column"] == "co2_ppm"
        prompt = llm.generate_structured.call_args.kwargs["prompt"]
        assert "temp_anomaly_c" in prompt
        assert "MUST include" in prompt

    def test_invented_column_raises(self, schema, hypothesis):  # (c)
        agent, _ = _designer(_llm_protocol({
            "heat": {"type": "independent", "description": "Ocean heat content", "column": "ocean_heat"},
            "temp": {"type": "dependent", "description": "Temperature anomaly in C", "column": "temp_anomaly_c"},
        }))

        with pytest.raises(UnboundVariableError) as exc_info:
            agent.design_experiment(hypothesis, store_in_db=False, dataset_schema=schema)

        assert exc_info.value.unbound_names == ["heat"]
        assert exc_info.value.available_columns == CLIMATE_COLUMNS
        assert exc_info.value.hypothesis_id == "hyp-bind-1"

    def test_same_column_for_x_and_y_raises(self, schema, hypothesis):
        agent, _ = _designer(_llm_protocol({
            "co2": {"type": "independent", "description": "CO2 concentration in ppm", "column": "co2_ppm"},
            "co2b": {"type": "dependent", "description": "CO2 concentration again", "column": "co2_ppm"},
        }))

        with pytest.raises(UnboundVariableError, match="both bound"):
            agent.design_experiment(hypothesis, store_in_db=False, dataset_schema=schema)

    def test_schema_skips_templates(self, schema, hypothesis):
        agent, _ = _designer(_llm_protocol({
            "co2": {"type": "independent", "description": "CO2 concentration in ppm", "column": "co2_ppm"},
            "temp": {"type": "dependent", "description": "Temperature anomaly in C", "column": "temp_anomaly_c"},
        }))
        agent._generate_from_template = Mock()

        agent.design_experiment(hypothesis, store_in_db=False, dataset_schema=schema)

        agent._generate_from_template.assert_not_called()

    def test_without_schema_prompt_has_no_dataset_block(self, hypothesis):
        agent, llm = _designer(_llm_protocol({
            "co2": {"type": "independent", "description": "CO2 concentration in ppm"},
            "temp": {"type": "dependent", "description": "Temperature anomaly in C"},
        }))
        agent.use_templates = False

        response = agent.design_experiment(hypothesis, store_in_db=False)

        assert "MUST include" not in llm.generate_structured.call_args.kwargs["prompt"]
        assert all(v.column is None for v in response.protocol.variables.values())


class TestCodeGeneratorBinding:
    def test_correlation_on_bound_columns(self):  # (d)
        code = ExperimentCodeGenerator(use_llm=False).generate(_bound_protocol("co2_ppm", "temp_anomaly_c"))

        results = _run(code)

        assert results["columns"] == {"x": "co2_ppm", "y": "temp_anomaly_c"}
        assert results["test_type"] == "pearson_correlation"
        assert results["statistic"] > 0.85
        assert results["p_value"] < 1e-6
        assert results["n"] == 64
        assert results["data_source"] == "file"

    def test_anova_on_categorical_column(self):  # (e)
        code = ExperimentCodeGenerator(use_llm=False).generate(_bound_protocol("decade", "temp_anomaly_c"))

        results = _run(code)

        assert results["test_type"] == "one_way_anova"
        assert results["columns"] == {"x": "decade", "y": "temp_anomaly_c"}
        assert results["statistic"] > 0
        assert 0 <= results["effect_size"] <= 1
        assert len(results["groups"]) == 7

    def test_welch_on_two_level_column(self, tmp_path):
        csv = tmp_path / "two.csv"
        rows = ["arm,score"] + [f"a,{1.0 + 0.1 * i}" for i in range(10)] + [f"b,{2.0 + 0.2 * i}" for i in range(10)]
        csv.write_text("\n".join(rows) + "\n")
        code = ExperimentCodeGenerator(use_llm=False).generate(_bound_protocol("arm", "score"))

        result = CodeExecutor(use_sandbox=False).execute_with_data(code, str(csv))

        assert result.success, result.error
        assert result.return_value["test_type"] == "welch_t_test"
        assert result.return_value["effect_size"] < 0  # a minus b

    def test_missing_column_fails_loudly(self, tmp_path):
        csv = tmp_path / "other.csv"
        csv.write_text("a,b\n1,2\n3,4\n5,7\n")
        code = ExperimentCodeGenerator(use_llm=False).generate(_bound_protocol("co2_ppm", "temp_anomaly_c"))

        result = CodeExecutor(use_sandbox=False).execute_with_data(code, str(csv))

        assert not result.success
        assert "Dataset is missing required columns" in (result.error or "")

    def test_unbound_variable_with_schema_raises(self, schema):
        protocol = _bound_protocol("co2_ppm", None)

        with pytest.raises(ValueError, match="protocol has unbound variables"):
            ExperimentCodeGenerator(use_llm=False).generate(protocol, dataset_schema=schema)

    def test_llm_prompt_lists_dataset_columns(self, schema):
        generator = ExperimentCodeGenerator(use_llm=False)

        prompt = generator._create_code_generation_prompt(_bound_protocol("co2_ppm", "temp_anomaly_c"), schema)

        assert "dataset column 'co2_ppm'" in prompt
        assert "- decade (" in prompt


class TestDirectorUntestable:
    @pytest.fixture
    def director(self):
        from kosmos.agents.research_director import ResearchDirectorAgent

        with patch("kosmos.agents.research_director.get_client", return_value=MagicMock()), \
             patch("kosmos.agents.research_director.get_world_model", return_value=MagicMock()), \
             patch("kosmos.agents.research_director.SkillLoader") as mock_skills, \
             patch("kosmos.db.init_from_config"):
            mock_skills.return_value.load_skills_for_task.return_value = ""
            director = ResearchDirectorAgent(
                research_question="Does CO2 concentration predict temperature anomaly?",
                domain="climate_science",
                config={"max_iterations": 3, "data_path": str(CLIMATE_CSV)},
            )
        director.research_plan = ResearchPlan(research_question=director.research_question)
        director.workflow = MagicMock()
        return director

    def test_director_describes_dataset(self, director):
        assert director.dataset_schema is not None
        assert director.dataset_schema.columns == CLIMATE_COLUMNS

    async def test_unbound_hypothesis_is_excluded(self, director):  # (f)
        director.research_plan.add_hypothesis("h-unbound")
        director.research_plan.add_hypothesis("h-other")
        director._experiment_designer = Mock(design_experiment=Mock(
            side_effect=UnboundVariableError("h-unbound", ["heat"], CLIMATE_COLUMNS)
        ))
        director._handle_error_with_recovery = Mock()

        await director._handle_design_experiment_action("h-unbound")

        assert director.research_plan.hypothesis_pool == ["h-unbound", "h-other"]
        assert director.research_plan.untestable_hypotheses == ["h-unbound"]
        assert director.research_plan.get_untested_hypotheses() == ["h-other"]
        assert "h-unbound" not in director.research_plan.tested_hypotheses
        director._handle_error_with_recovery.assert_not_called()
        director.workflow.transition_to.assert_not_called()
        kwargs = director._experiment_designer.design_experiment.call_args.kwargs
        assert kwargs["dataset_schema"] is director.dataset_schema

    async def test_hypothesis_generation_gets_dataset_context(self, director):
        director._hypothesis_agent = Mock()
        director._hypothesis_agent.generate_hypotheses.return_value = Mock(hypotheses=[])

        await director._handle_generate_hypothesis_action()

        kwargs = director._hypothesis_agent.generate_hypotheses.call_args.kwargs
        assert kwargs["research_question"] == director.research_question
        assert "temp_anomaly_c" in kwargs["dataset_context"]
