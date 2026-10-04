"""
Tests for the run seed and result provenance (viability plan P2-3).

One seed travels from `kosmos run --seed` through the director, the designed
protocol, the generated code and the executor into the stored result, and every
result row carries a provenance record that is enough to re-run it.
"""

import hashlib
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import typer
from typer.testing import CliRunner

from kosmos.cli.commands.run import run_research
from kosmos.config import reset_config
from kosmos.db import get_session
from kosmos.db.models import Experiment, Result
from kosmos.execution.code_generator import ExperimentCodeGenerator
from kosmos.execution.executor import CodeExecutor, ExecutionResult
from kosmos.execution.provenance import build_run_provenance
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
from tests.unit.agents.conftest import CODE, EXP_ID, db_director, in_memory_db  # noqa: F401
from tests.unit.execution.test_column_binding import (
    CLIMATE_CSV,
    _designer,
    _llm_protocol,
    hypothesis,  # noqa: F401
)

SYNTHETIC_FALLBACK_CODE = """
import numpy as np
import pandas as pd
try:
    df = pd.read_csv(data_path)
except Exception:
    df = pd.DataFrame({'x': np.random.normal(size=50)})
results = {'mean': float(df['x'].mean()), 'first': float(df['x'].iloc[0])}
"""


def _run_seeded(seed):
    result = CodeExecutor(use_sandbox=False).execute_with_data(
        SYNTHETIC_FALLBACK_CODE, "/nonexistent.csv", seed=seed
    )
    assert result.success, result.error
    return result


class TestExecutorSeed:
    def test_same_seed_same_result(self):  # (a)
        first, second = _run_seeded(7), _run_seeded(7)
        assert first.return_value == second.return_value
        assert first.random_seed == 7
        assert first.sandbox_used is False
        assert _run_seeded(8).return_value != first.return_value

    def test_seed_defines_random_seed_variable(self):
        result = CodeExecutor(use_sandbox=False).execute_with_data(
            "results = {'seed': random_seed}", "/nonexistent.csv", seed=11
        )
        assert result.success, result.error
        assert result.return_value == {"seed": 11}

    def test_no_seed_leaves_code_unchanged(self):
        executor = CodeExecutor(use_sandbox=False)
        with patch.object(executor, "execute", return_value=ExecutionResult(success=True)) as execute:
            result = executor.execute_with_data("results = {}", "/data.csv")
        assert "random_seed" not in execute.call_args.args[0]
        assert result.random_seed is None

    def test_sandbox_path_gets_the_seed_prelude(self):
        executor = CodeExecutor(use_sandbox=False)
        executor.use_sandbox = True
        with patch.object(executor, "execute", return_value=ExecutionResult(success=True)) as execute:
            result = executor.execute_with_data("results = {}", "/host/data.csv", seed=5)
        code, local_vars = execute.call_args.args[0], execute.call_args.args[1]
        assert code.startswith("random_seed = 5\n")
        assert "/host/data.csv" not in code  # the sandbox assigns the container path itself
        assert local_vars == {"data_path": "/host/data.csv"}
        assert result.random_seed == 5

    def test_seed_is_keyword_only(self):
        with pytest.raises(TypeError):
            CodeExecutor(use_sandbox=False).execute_with_data("results = {}", "/x.csv", False, 7)


class TestBuildRunProvenance:
    def test_climate_csv(self):  # (b)
        llm = Mock(model="deepseek/deepseek-chat", provider_name="litellm", temperature_default=0.7)
        protocol = Mock(template_name=None)
        prov = build_run_provenance(CODE, str(CLIMATE_CSV), llm, 7, protocol, sandbox_used=True)

        assert prov["data_sha256"] == hashlib.sha256(open(CLIMATE_CSV, "rb").read()).hexdigest()
        assert re.fullmatch(r"[0-9a-f]{40}", prov["git_sha"])
        assert prov["model"] == "deepseek/deepseek-chat"
        assert prov["provider"] == "litellm"
        assert prov["temperature"] == 0.7
        assert prov["data_rows"] == 64
        assert prov["code_sha256"] == hashlib.sha256(CODE.encode()).hexdigest()
        assert prov["seed"] == 7
        assert prov["sandbox_used"] is True
        assert prov["template"] == "llm"
        assert prov["kosmos_version"]
        assert prov["python_version"].count(".") == 2

    def test_no_data_and_non_string_client_fields(self):
        prov = build_run_provenance(CODE, None, MagicMock(), 3, Mock(template_name="t"),
                                    sandbox_used=False, template="generic_computational")
        assert prov["data_sha256"] is None and prov["data_rows"] is None
        assert prov["model"] is None and prov["provider"] is None and prov["temperature"] is None
        assert prov["template"] == "generic_computational"

    def test_missing_data_file(self, tmp_path):
        prov = build_run_provenance(CODE, str(tmp_path / "gone.csv"), None, 1, None, sandbox_used=False)
        assert prov["data_sha256"] is None


class TestDirectorRecordsSeedAndProvenance:
    async def test_result_row(self, db_director):  # (c)
        director = db_director
        director._code_generator.last_template_name = "ttest_comparison"
        director.llm_client = Mock(model="deepseek/deepseek-chat", provider_name="litellm",
                                   temperature_default=0.7)
        director._code_executor = Mock(use_sandbox=True)
        director._code_executor.execute_with_data.return_value = ExecutionResult(
            success=True, return_value={"p_value": 0.01, "data_source": "file"},
            random_seed=7, sandbox_used=True,
        )

        await director._handle_execute_experiment_action(EXP_ID)

        assert director._code_executor.execute_with_data.call_args.kwargs["seed"] == 7
        with get_session() as session:
            row = session.query(Result).one()
            exp = session.query(Experiment).filter_by(id=EXP_ID).one()
            assert row.random_seed == 7
            assert row.run_id == director.run_id
            assert row.provenance["code_sha256"] == hashlib.sha256(CODE.encode()).hexdigest()
            assert row.provenance["seed"] == 7
            assert row.provenance["sandbox_used"] is True
            assert row.provenance["template"] == "ttest_comparison"
            assert row.provenance["model"] == "deepseek/deepseek-chat"
            assert exp.code_generated == CODE
            code_path = director.artifacts_dir / director.run_id / "code" / f"{row.id}.py"
            assert code_path.read_text() == CODE
            assert row.provenance["code_path"] == str(code_path)

            metadata = director._db_result_to_experiment_result(row).metadata
        assert metadata.random_seed == 7
        assert metadata.data_source == "file"
        assert metadata.sandbox_used is True

    async def test_run_without_data_seeds_the_code(self, db_director):
        director = db_director
        director.data_path = None
        director._code_executor = Mock(use_sandbox=False)
        director._code_executor.execute.return_value = ExecutionResult(success=True, return_value={})

        await director._handle_execute_experiment_action(EXP_ID)

        assert director._code_executor.execute.call_args.args[0].startswith("random_seed = 7\n")
        with get_session() as session:
            row = session.query(Result).one()
            assert row.random_seed == 7
            assert row.provenance["data_sha256"] is None

    def test_init_records_research_session(self, db_director):
        from kosmos.db.operations import get_research_session
        assert re.fullmatch(r"run_[0-9a-f]{12}", db_director.run_id)
        assert db_director.random_seed == 7
        with get_session() as session:
            rs = get_research_session(session, db_director.run_id)
            assert rs is not None
            assert rs.research_question == "Does CO2 predict temperature?"

    async def test_design_passes_the_run_seed(self, db_director):
        db_director._experiment_designer = Mock()
        db_director._experiment_designer.design_experiment.side_effect = RuntimeError("stop")
        await db_director._handle_design_experiment_action("hyp-exec-1")
        assert db_director._experiment_designer.design_experiment.call_args.kwargs["random_seed"] == 7


class TestDesignerSeed:
    def test_run_seed_overrides_llm_seed(self, hypothesis):
        response = _llm_protocol({
            "co2": {"type": "independent", "description": "CO2 concentration in ppm"},
            "temp": {"type": "dependent", "description": "Temperature anomaly in C"},
        })
        response["random_seed"] = 123
        agent, _ = _designer(response)
        agent.use_templates = False

        assert agent.design_experiment(hypothesis, store_in_db=False).protocol.random_seed == 123
        assert agent.design_experiment(hypothesis, store_in_db=False, random_seed=7).protocol.random_seed == 7


def _protocol(name, description, seed, experiment_type=ExperimentType.COMPUTATIONAL, tests=()):
    return ExperimentProtocol(
        id="exp-seed-1",
        name=name,
        hypothesis_id="hyp-seed-1",
        domain="climate_science",
        description=description,
        objective="Test the association between the bound columns",
        experiment_type=experiment_type,
        statistical_tests=[
            StatisticalTestSpec(test_type=t, description="The planned test", null_hypothesis="No effect",
                                variables=["predictor", "outcome"])
            for t in tests
        ],
        steps=[ProtocolStep(step_number=1, title="Analyse", description="Run the analysis", action="analyse")],
        variables={
            "predictor": Variable(name="predictor", type=VariableType.INDEPENDENT,
                                  description="Independent variable from the dataset", column="co2_ppm"),
            "outcome": Variable(name="outcome", type=VariableType.DEPENDENT,
                                description="Dependent variable from the dataset", column="temp_anomaly_c"),
        },
        resource_requirements=ResourceRequirements(),
        random_seed=seed,
    )


class TestTemplatesEmitSeed:
    @pytest.mark.parametrize("name,description,experiment_type,tests,template", [
        ("Bound analysis", "Analysis of two bound columns of the climate dataset",
         ExperimentType.COMPUTATIONAL, (), "generic_computational"),
        ("Correlation analysis", "Correlation analysis of two bound columns of the climate dataset",
         ExperimentType.DATA_ANALYSIS, (), "correlation_analysis"),
        ("Group comparison", "Comparison of two groups of the climate dataset",
         ExperimentType.DATA_ANALYSIS, (StatisticalTest.T_TEST,), "ttest_comparison"),
        ("Scaling analysis", "Power law scaling between two bound columns of the climate dataset",
         ExperimentType.DATA_ANALYSIS, (), "log_log_scaling"),
        ("ML classification", "Machine learning classification with cross-validation on the dataset",
         ExperimentType.COMPUTATIONAL, (), "ml_experiment"),
    ])
    def test_results_carry_the_seed(self, name, description, experiment_type, tests, template):
        generator = ExperimentCodeGenerator(use_llm=False)
        code = generator.generate(_protocol(name, description, 7, experiment_type, tests))
        assert generator.last_template_name == template
        assert "random_seed'] = 7" in code
        assert "42" not in code

    def test_seed_zero_is_kept(self):
        code = ExperimentCodeGenerator(use_llm=False).generate(
            _protocol("Bound analysis", "Analysis of two bound columns of the climate dataset", 0)
        )
        assert "random_seed'] = 0" in code

    def test_generic_on_climate_csv(self):
        code = ExperimentCodeGenerator(use_llm=False).generate(
            _protocol("Bound analysis", "Analysis of two bound columns of the climate dataset", 7)
        )
        result = CodeExecutor(use_sandbox=False).execute_with_data(code, str(CLIMATE_CSV), seed=7)
        assert result.success, result.error
        assert result.return_value["random_seed"] == 7


app = typer.Typer()


@app.callback()
def _root():
    """Test root."""


app.command("run")(run_research)

RESULTS = {"id": "r", "question": "Q", "domain": None, "state": "converged",
           "current_iteration": 1, "max_iterations": 1, "convergence_reason": None,
           "hypotheses": [], "experiments": [], "metrics": {}}


class TestCliSeed:
    @pytest.fixture
    def director_cls(self, monkeypatch):
        monkeypatch.delenv("DEFAULT_RANDOM_SEED", raising=False)
        monkeypatch.delenv("KOSMOS_ARTIFACTS_DIR", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        reset_config()
        with patch("kosmos.agents.research_director.ResearchDirectorAgent") as director, \
             patch("kosmos.cli.commands.run.run_with_progress_async", new=AsyncMock(return_value=RESULTS)), \
             patch("kosmos.agents.registry.get_registry", return_value=MagicMock()):
            yield director
        reset_config()

    def _config(self, director_cls, *args):
        result = CliRunner().invoke(app, ["run", "Q", "--max-iterations", "1", *args])
        assert result.exit_code == 0, result.output
        return director_cls.call_args.kwargs["config"]

    def test_seed_flag(self, director_cls):
        config = self._config(director_cls, "--seed", "7")
        assert config["random_seed"] == 7
        assert config["artifacts_dir"] is None

    def test_default_seed(self, director_cls):
        assert self._config(director_cls)["random_seed"] == 42
