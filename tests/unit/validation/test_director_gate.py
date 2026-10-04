"""
Tests for the result validation gate in ResearchDirectorAgent._handle_analyze_result_action
(viability plan P2-2).

A result is 'validated' only when its statistic matches an independent recomputation
on the dataset and survives a permutation null on that same data; ScholarEval is
advisory. A hypothesis is marked supported only on a validated result.
"""

import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from kosmos.agents.data_analyst import ResultInterpretation
from kosmos.db import get_session
from kosmos.db import operations
from kosmos.db.models import Hypothesis as DBHypothesis, HypothesisStatus, Result
from kosmos.execution.code_generator import (
    CorrelationAnalysisCodeTemplate,
    GenericComputationalCodeTemplate,
    TTestComparisonCodeTemplate,
)
from kosmos.execution.data_schema import describe_dataset
from kosmos.execution.executor import CodeExecutor
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
from kosmos.validation.analysis_fn import build_analysis_fn, shuffle_target
from kosmos.validation.null_model import NullModelValidator
from kosmos.validation.scholar_eval import ScholarEvalValidator
from tests.unit.agents.conftest import EXP_ID, H_ID, db_director, in_memory_db  # noqa: F401

DATA_DIR = Path(__file__).resolve().parents[3] / "evaluation" / "data"
CLIMATE_CSV = DATA_DIR / "climate_co2_temperature_test.csv"
RESULT_ID = "res-gate-1"

SCHOLAR_JSON = json.dumps({
    "novelty": 0.85, "rigor": 0.85, "clarity": 0.85, "reproducibility": 0.85,
    "impact": 0.85, "coherence": 0.85, "limitations": 0.85, "ethics": 0.85,
    "reasoning": "Sound correlation on the real dataset.",
})
ANALYST_JSON = json.dumps({
    "hypothesis_supported": True,
    "confidence": 0.9,
    "summary": "CO2 concentration is strongly correlated with the temperature anomaly.",
    "key_findings": ["Pearson r is high"],
    "significance_interpretation": "p is far below 0.05",
    "potential_confounds": [],
    "follow_up_experiments": [],
    "overall_assessment": "Supported",
})


def _pearson_stats(df, statistic=None):
    fn = build_analysis_fn("pearson_correlation", "co2_ppm", "temp_anomaly_c")
    out = fn(df)
    return {
        "test_type": "pearson_correlation",
        "statistic": out["statistic"] if statistic is None else statistic,
        "p_value": out["p_value"],
        "effect_size": out["effect_size"],
        "n": out["n"],
        "columns": {"x": "co2_ppm", "y": "temp_anomaly_c"},
    }


def _seed(stats, execution_success=True, data_source="file", data=None):
    with get_session() as session:
        operations.create_result(
            session, id=RESULT_ID, experiment_id=EXP_ID,
            data={"execution_success": execution_success} if data is None else data,
            statistical_tests=stats,
            p_value=(stats or {}).get("p_value"),
            effect_size=(stats or {}).get("effect_size"),
            execution_success=execution_success,
            data_source=data_source,
        )


def _row():
    with get_session() as session:
        r = session.query(Result).filter_by(id=RESULT_ID).one()
        h = session.query(DBHypothesis).filter_by(id=H_ID).one()
        return r.validation_status, r.validation_detail, r.supports_hypothesis, h.status


def _use_dataset(director, path):
    director.data_path = str(path)
    director.dataset_schema = describe_dataset(str(path))


@pytest.fixture
def gate_director(db_director):
    """The DB-backed director with a dataset and LLM clients that return valid JSON."""
    _use_dataset(db_director, CLIMATE_CSV)
    db_director.llm_client = Mock(generate=Mock(return_value=Mock(content=SCHOLAR_JSON)))
    db_director.config["null_permutations"] = 200
    db_director.config["random_seed"] = 0
    analyst_llm = Mock(generate=Mock(return_value=Mock(content=ANALYST_JSON)))
    with patch("kosmos.agents.data_analyst.get_client", return_value=analyst_llm):
        yield db_director


async def test_real_correlation_is_validated(gate_director):  # (a)
    _seed(_pearson_stats(pd.read_csv(CLIMATE_CSV)))

    await gate_director._handle_analyze_result_action(RESULT_ID)

    status, detail, supports, h_status = _row()
    assert status == "validated"
    assert detail["recomputed_match"] is True
    assert detail["null_model"]["passes_null_test"] is True
    assert detail["null_model"]["persists_in_noise"] is False
    assert detail["null_model"]["shuffle_method"] == "target"
    assert detail["scholar_eval"]["passes_threshold"] is True
    assert supports is True
    assert h_status == HypothesisStatus.SUPPORTED
    assert H_ID in gate_director.research_plan.supported_hypotheses


async def test_shuffled_data_is_rejected(gate_director, tmp_path):  # (b)
    df = pd.read_csv(CLIMATE_CSV)
    df["temp_anomaly_c"] = np.random.default_rng(0).permutation(df["temp_anomaly_c"].values)
    shuffled_csv = tmp_path / "climate_shuffled.csv"
    df.to_csv(shuffled_csv, index=False)
    _use_dataset(gate_director, shuffled_csv)
    _seed(_pearson_stats(df))

    await gate_director._handle_analyze_result_action(RESULT_ID)

    status, detail, supports, h_status = _row()
    assert status == "rejected"
    assert detail["recomputed_match"] is True
    assert detail["null_model"]["passes_null_test"] is False
    assert supports is None
    assert h_status == HypothesisStatus.INCONCLUSIVE
    plan = gate_director.research_plan
    assert H_ID in plan.tested_hypotheses
    assert H_ID not in plan.supported_hypotheses
    assert H_ID not in plan.rejected_hypotheses


async def test_unsupported_verdict_on_real_data_rejects_the_hypothesis(gate_director, tmp_path):
    df = pd.read_csv(CLIMATE_CSV)
    df["temp_anomaly_c"] = np.random.default_rng(0).permutation(df["temp_anomaly_c"].values)
    shuffled_csv = tmp_path / "climate_shuffled.csv"
    df.to_csv(shuffled_csv, index=False)
    _use_dataset(gate_director, shuffled_csv)
    _seed(_pearson_stats(df))
    gate_director._data_analyst = Mock(interpret_results=Mock(return_value=ResultInterpretation(
        experiment_id=EXP_ID, hypothesis_supported=False, confidence=0.9, summary="No association",
        key_findings=[], significance_interpretation="p well above 0.05", biological_significance=None,
        comparison_to_prior_work=None, potential_confounds=[], follow_up_experiments=[],
        anomalies_detected=[], patterns_detected=[], overall_assessment="",
    )))

    await gate_director._handle_analyze_result_action(RESULT_ID)

    status, _, supports, h_status = _row()
    assert status == "rejected"
    assert supports is False
    assert h_status == HypothesisStatus.REJECTED
    assert H_ID in gate_director.research_plan.rejected_hypotheses


async def test_tampered_statistic_is_rejected(gate_director):  # (c)
    _seed(_pearson_stats(pd.read_csv(CLIMATE_CSV), statistic=0.99))

    await gate_director._handle_analyze_result_action(RESULT_ID)

    status, detail, supports, _ = _row()
    assert detail["recomputed_match"] is False
    assert status == "rejected"
    assert supports is None
    assert H_ID not in gate_director.research_plan.supported_hypotheses


@pytest.mark.parametrize("case, reason", [
    ("failed", "execution_failed"),
    ("empty_data", "execution_failed"),
    ("synthetic", "synthetic_data"),
    ("no_statistic", "no_statistic"),
    ("no_dataset", "no_dataset"),
])
async def test_unvalidated_results_never_support(gate_director, case, reason):  # (d)
    stats = _pearson_stats(pd.read_csv(CLIMATE_CSV))
    if case == "failed":
        _seed(stats, execution_success=False)
    elif case == "empty_data":
        _seed(None, execution_success=None, data_source=None, data={})
    elif case == "synthetic":
        _seed(stats, data_source="synthetic")
    elif case == "no_statistic":
        _seed({"n": 64})
    else:
        gate_director.data_path, gate_director.dataset_schema = None, None
        _seed(stats)

    await gate_director._handle_analyze_result_action(RESULT_ID)

    status, detail, supports, h_status = _row()
    assert status == "unvalidated"
    assert detail["reason"] == reason
    assert supports is None
    assert h_status == HypothesisStatus.INCONCLUSIVE
    assert H_ID not in gate_director.research_plan.supported_hypotheses
    assert H_ID in gate_director.research_plan.tested_hypotheses


def test_statistic_without_test_type_is_not_recomputable():
    """A log2 t-test (or LogLog/ML result) carries a statistic but no recomputable test."""
    from types import SimpleNamespace
    from kosmos.agents.research_director import ResearchDirectorAgent

    director = Mock(data_path=str(CLIMATE_CSV), dataset_schema=object(), config={})
    row = SimpleNamespace(
        id="r", execution_success=True, data_source="file", random_seed=None,
        statistical_tests={"t_statistic": 3.0, "p_value": 0.01},
    )
    status, detail = ResearchDirectorAgent._validate_result(director, row, Mock(), "p")
    assert status == "unvalidated"
    assert detail["reason"] == "no_statistic"
    assert "not recomputable" in detail["note"]


def test_scholar_eval_failure_scores_zero():  # (e), validator half
    validator = ScholarEvalValidator(llm_client=Mock(generate=Mock(side_effect=RuntimeError("down"))))

    score = validator.evaluate_finding({"summary": "s", "statistics": {"p_value": 0.01}})

    assert score.passes_threshold is False
    assert score.overall_score == 0.0
    assert score.feedback.startswith("evaluation_error")


def test_scholar_eval_without_client_is_an_error_unless_mock_allowed():
    assert ScholarEvalValidator().evaluate_finding({"summary": "s"}).feedback.startswith("evaluation_error")
    assert ScholarEvalValidator(allow_mock=True).evaluate_finding({"summary": "s"}).passes_threshold is True


def test_scholar_eval_garbled_reply_is_an_error():
    validator = ScholarEvalValidator(llm_client=Mock(generate=Mock(return_value=Mock(content="no json here"))))

    score = validator.evaluate_finding({"summary": "s"})

    assert score.passes_threshold is False
    assert score.feedback.startswith("evaluation_error")


def test_scholar_eval_llm_client_skips_null_model_when_asked():
    llm = Mock(generate=Mock(return_value=Mock(content=SCHOLAR_JSON)))
    validator = ScholarEvalValidator(llm_client=llm, run_null_model=False)

    score = validator.evaluate_finding({"summary": "s", "statistics": {"statistic": 0.9, "p_value": 0.01}})

    assert score.passes_threshold is True
    assert score.null_model_result is None
    assert llm.generate.call_args.kwargs["max_tokens"] == 1500


async def test_scholar_eval_failure_is_advisory(gate_director):  # (e), director half
    gate_director.llm_client = Mock(generate=Mock(side_effect=RuntimeError("provider down")))
    _seed(_pearson_stats(pd.read_csv(CLIMATE_CSV)))

    await gate_director._handle_analyze_result_action(RESULT_ID)

    status, detail, supports, _ = _row()
    assert status == "validated"
    assert detail["scholar_eval"]["passes_threshold"] is False
    assert detail["scholar_eval"]["feedback"].startswith("evaluation_error")
    assert supports is True


async def test_scholar_eval_gate_flag_makes_it_binding(gate_director):
    gate_director.llm_client = Mock(generate=Mock(side_effect=RuntimeError("provider down")))
    gate_director.config["scholar_eval_gate"] = True
    _seed(_pearson_stats(pd.read_csv(CLIMATE_CSV)))

    await gate_director._handle_analyze_result_action(RESULT_ID)

    status, _, supports, _ = _row()
    assert status == "rejected"
    assert supports is None


# --- (f) template code and build_analysis_fn agree ----------------------------------

def _protocol(x_col, y_col, experiment_type=ExperimentType.COMPUTATIONAL, test=None):
    tests = []
    if test is not None:
        tests = [StatisticalTestSpec(
            test_type=test, description="Primary test of the bound columns",
            null_hypothesis="No association between the bound columns", variables=[x_col, y_col],
        )]
    return ExperimentProtocol(
        id="exp-consistency",
        name="Bound analysis",
        hypothesis_id="hyp-consistency",
        domain="general",
        description="Analysis of two bound columns of an evaluation dataset",
        objective="Test the association between the bound columns",
        experiment_type=experiment_type,
        statistical_tests=tests,
        steps=[ProtocolStep(step_number=1, title="Analyse", description="Run the analysis", action="analyse")],
        variables={
            "predictor": Variable(name="predictor", type=VariableType.INDEPENDENT,
                                  description="Independent variable from the dataset", column=x_col),
            "outcome": Variable(name="outcome", type=VariableType.DEPENDENT,
                                description="Dependent variable from the dataset", column=y_col),
        },
        resource_requirements=ResourceRequirements(),
    )


CONSISTENCY_CASES = [
    # (template, csv, x, y, test spec, expected test_type)
    ("generic", "climate_co2_temperature_test.csv", "co2_ppm", "temp_anomaly_c", None, "pearson_correlation"),
    ("generic", "climate_co2_temperature_test.csv", "decade", "temp_anomaly_c", None, "one_way_anova"),
    ("generic", "enzyme_kinetics_test.csv", "temperature", "enzyme_activity", None, "pearson_correlation"),
    ("generic", "gene_expression_test.csv", "condition", "expression_level", None, "welch_t_test"),
    ("generic", "perovskite_solar_cell_test.csv", "annealing_temp_c", "power_conversion_efficiency", None,
     "pearson_correlation"),
    ("correlation", "perovskite_solar_cell_test.csv", "halide_ratio", "power_conversion_efficiency",
     StatisticalTest.CORRELATION, "pearson_correlation"),
    ("correlation", "enzyme_kinetics_test.csv", "pH", "enzyme_activity", "spearman_correlation",
     "spearman_correlation"),
    ("ttest", "gene_expression_test.csv", "condition", "expression_level", StatisticalTest.T_TEST, "welch_t_test"),
]
TEMPLATES = {
    "generic": GenericComputationalCodeTemplate,
    "correlation": CorrelationAnalysisCodeTemplate,
    "ttest": TTestComparisonCodeTemplate,
}


@pytest.mark.parametrize("template, csv, x_col, y_col, test, expected", CONSISTENCY_CASES)
def test_template_matches_analysis_fn(template, csv, x_col, y_col, test, expected):  # (f)
    path = DATA_DIR / csv
    experiment_type = ExperimentType.COMPUTATIONAL if test is None else ExperimentType.DATA_ANALYSIS
    if isinstance(test, str):  # spearman: the enum has no member, the template reads the string
        protocol = _protocol(x_col, y_col, experiment_type, StatisticalTest.CORRELATION)
        protocol.statistical_tests[0].test_type = test
    else:
        protocol = _protocol(x_col, y_col, experiment_type, test)
    code = TEMPLATES[template]().generate(protocol, dataset_schema=describe_dataset(str(path)))

    result = CodeExecutor(use_sandbox=False).execute_with_data(code, str(path))

    assert result.success, result.error
    stats = result.return_value
    assert stats["test_type"] == expected
    assert stats["data_source"] == "file"
    columns = stats["columns"]
    recomputed = build_analysis_fn(
        stats["test_type"], columns["x"], columns["y"], groups=columns.get("groups")
    )(pd.read_csv(path))
    for key in ("statistic", "p_value", "effect_size"):
        assert recomputed[key] == pytest.approx(stats[key], rel=1e-9, abs=1e-12), key
    assert recomputed["n"] == stats["n"]


# --- analysis_fn and the null model ---------------------------------------------------

@pytest.mark.parametrize("test_type, x_col", [
    ("pearson_correlation", "co2_ppm"),
    ("spearman_correlation", "co2_ppm"),
    ("linear_regression", "co2_ppm"),
    ("one_way_anova", "decade"),
])
def test_analysis_fn_returns_the_standard_keys(test_type, x_col):
    out = build_analysis_fn(test_type, x_col, "temp_anomaly_c")(pd.read_csv(CLIMATE_CSV))

    assert set(out) >= {"statistic", "p_value", "effect_size", "test_type", "n"}
    assert out["test_type"] == test_type
    assert out["n"] == 64


def test_two_group_tests():
    df = pd.DataFrame({"arm": ["a"] * 10 + ["b"] * 10, "score": list(range(10)) + list(range(5, 15))})

    welch = build_analysis_fn("welch_t_test", "arm", "score")(df)
    reversed_welch = build_analysis_fn("welch_t_test", "arm", "score", groups=["b", "a"])(df)
    mw = build_analysis_fn("mann_whitney", "arm", "score")(df)

    assert welch["statistic"] < 0
    assert reversed_welch["statistic"] == pytest.approx(-welch["statistic"])
    assert -1.0 <= mw["statistic"] < 0
    assert mw["u_statistic"] >= 0


def test_unsupported_test_type_raises():
    with pytest.raises(ValueError, match="Unsupported test_type"):
        build_analysis_fn("chi_square", "a", "b")


@pytest.mark.parametrize("test_type, shuffled_col, kept_col", [
    ("pearson_correlation", "temp_anomaly_c", "co2_ppm"),
    ("one_way_anova", "decade", "temp_anomaly_c"),
])
def test_shuffle_target_permutes_only_the_tested_column(test_type, shuffled_col, kept_col):
    df = pd.read_csv(CLIMATE_CSV)
    x_col = "decade" if test_type == "one_way_anova" else "co2_ppm"

    out = shuffle_target(df, test_type, x_col, "temp_anomaly_c", np.random.default_rng(0))

    assert sorted(out[shuffled_col]) == sorted(df[shuffled_col])
    assert not out[shuffled_col].equals(df[shuffled_col])
    assert out[kept_col].equals(df[kept_col])
    assert df.equals(pd.read_csv(CLIMATE_CSV))  # the input is not modified


def test_500_permutations_on_64_rows_under_a_second():
    df = pd.read_csv(CLIMATE_CSV)
    fn = build_analysis_fn("pearson_correlation", "co2_ppm", "temp_anomaly_c")

    start = time.perf_counter()
    null = NullModelValidator(n_permutations=500, random_seed=0).validate_finding(
        {"statistics": fn(df)}, data=df, analysis_func=fn,
        shuffle_func=lambda d, rng: shuffle_target(d, "pearson_correlation", "co2_ppm", "temp_anomaly_c", rng),
    )
    elapsed = time.perf_counter() - start

    assert null.n_permutations == 500
    assert null.passes_null_test is True
    assert elapsed < 1.0
