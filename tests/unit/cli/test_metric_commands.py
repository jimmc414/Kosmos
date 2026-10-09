"""
Tests for `kosmos validate-null` and `kosmos rerun` (viability plan M-1).

The database holds one validated pearson_correlation result on the climate CSV:
its statistics come from build_analysis_fn, its provenance carries the file's
sha256, seed 42 and sandbox_used False (so rerun takes the host executor path,
no Docker), and its experiment's code_generated is the P2-1 template code for
the bound protocol. No LLM is called.
"""

import hashlib
import shutil
from unittest.mock import patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from kosmos.cli.main import app
from kosmos.db import get_session
from kosmos.db import operations
from kosmos.db.models import Result
from kosmos.execution.code_generator import ExperimentCodeGenerator
from kosmos.validation.analysis_fn import build_analysis_fn
from tests.unit.agents.conftest import EXP_ID, H_ID, in_memory_db  # noqa: F401
from tests.unit.execution.test_column_binding import CLIMATE_CSV, _bound_protocol

RUN_ID = "r1"
RESULT_ID = "res-metric-1"
X_COL, Y_COL = "co2_ppm", "temp_anomaly_c"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _add_result(session, result_id, data_path, sha=None):
    stats = build_analysis_fn("pearson_correlation", X_COL, Y_COL)(pd.read_csv(data_path))
    stats["columns"] = {"x": X_COL, "y": Y_COL}
    operations.create_result(
        session, id=result_id, experiment_id=EXP_ID, data={"execution_success": True},
        statistical_tests=stats, p_value=stats["p_value"], effect_size=stats["effect_size"],
        supports_hypothesis=True, run_id=RUN_ID, execution_success=True, data_source="file",
        random_seed=42, validation_status="validated",
        validation_detail={"recomputed_match": True, "null_model": {"passes_null_test": True}},
        provenance={
            "data_path": str(data_path), "data_sha256": sha or _sha256(data_path),
            "seed": 42, "sandbox_used": False, "git_sha": "0" * 40,
        },
    )


@pytest.fixture
def metric_db(in_memory_db):  # noqa: F811
    """Run r1 with one validated result whose experiment holds the P2-1 generated code."""
    code = ExperimentCodeGenerator(use_llm=False).generate(_bound_protocol(X_COL, Y_COL))
    with get_session() as session:
        operations.create_research_session(
            session, id=RUN_ID, research_question="Does CO2 predict temperature?", domain="climate",
        )
        operations.create_hypothesis(
            session, id=H_ID, research_question="Does CO2 predict temperature?",
            statement="CO2 concentration predicts the temperature anomaly",
            rationale="Radiative forcing increases with CO2 concentration", domain="climate",
        )
        operations.create_experiment(
            session, id=EXP_ID, hypothesis_id=H_ID, experiment_type="data_analysis",
            description="d", protocol={"name": "p"}, domain="climate",
        )
        _add_result(session, RESULT_ID, CLIMATE_CSV)
        session.query(Result).filter_by(id=RESULT_ID).one().experiment.code_generated = code


def _invoke(*args):
    # The CLI callback re-initializes the database from config; keep the test's database
    with patch("kosmos.db.init_from_config"):
        return CliRunner().invoke(app, list(args))


def _row(result_id=RESULT_ID):
    with get_session() as session:
        r = session.query(Result).filter_by(id=result_id).one()
        return {
            "status": r.validation_status, "detail": dict(r.validation_detail or {}),
            "supports": r.supports_hypothesis, "provenance": dict(r.provenance or {}),
        }


def _tamper_statistic(value):
    with get_session() as session:
        r = session.query(Result).filter_by(id=RESULT_ID).one()
        r.statistical_tests = {**r.statistical_tests, "statistic": value}


# --- kosmos validate-null ------------------------------------------------------------------

def test_validate_null_stores_a_low_shuffled_pass_rate(metric_db):
    with patch("kosmos.core.llm.get_client") as get_client:
        result = _invoke("validate-null", "--run-id", RUN_ID, "--k", "20", "--seed", "0")

    assert result.exit_code == 0, result.output
    get_client.assert_not_called()
    row = _row()
    assert row["detail"]["shuffled_pass_rate"] <= 0.15
    assert row["detail"]["shuffled_null"] == {"k": 20, "seed": 0, "n_permutations": 500}
    # The gate's verdict and its earlier detail are kept
    assert row["status"] == "validated" and row["supports"] is True
    assert row["detail"]["recomputed_match"] is True
    assert "mean shuffled_pass_rate" in result.output


def test_validate_null_skips_a_changed_data_file(metric_db, tmp_path):
    changed = tmp_path / "climate.csv"
    shutil.copy(CLIMATE_CSV, changed)
    with get_session() as session:
        _add_result(session, "res-metric-changed", changed, sha="e" * 64)

    result = _invoke("validate-null", "--run-id", RUN_ID, "--k", "2", "--seed", "0")

    assert result.exit_code == 0, result.output
    assert "res-metric-changed  skipped: data file changed" in result.output
    assert "shuffled_pass_rate" not in _row("res-metric-changed")["detail"]
    assert "shuffled_pass_rate" in _row()["detail"]


def test_validate_null_without_an_eligible_result_exits_1(metric_db):
    result = _invoke("validate-null", "--run-id", "run_unknown", "--k", "2")

    assert result.exit_code == 1


def test_validate_null_exits_1_when_the_mean_exceeds_alpha(metric_db):
    with patch("kosmos.cli.commands.metrics.shuffled_pass_rate", return_value=0.3):
        result = _invoke("validate-null", "--run-id", RUN_ID, "--k", "2", "--alpha", "0.05")

    assert result.exit_code == 1
    assert _row()["detail"]["shuffled_pass_rate"] == 0.3



def test_validate_null_mean_exactly_at_alpha_passes(metric_db):
    """Three rows at 1/20 average to 0.05000000000000001 in floats; that is not above alpha
    (LIVE-25 hit this: every row 0.050, exit 1)."""
    with get_session() as session:
        _add_result(session, "res-metric-2", CLIMATE_CSV)
        _add_result(session, "res-metric-3", CLIMATE_CSV)
    with patch("kosmos.cli.commands.metrics.shuffled_pass_rate", return_value=0.05):
        result = _invoke("validate-null", "--run-id", RUN_ID, "--k", "20", "--alpha", "0.05")

    assert result.exit_code == 0, result.output
    assert "over 3 result(s)" in result.output

# --- kosmos rerun ----------------------------------------------------------------------

def test_rerun_reproduces_the_statistic_then_fails_after_tampering(metric_db):
    with patch("kosmos.core.llm.get_client") as get_client:
        result = _invoke("rerun", "--result-id", RESULT_ID, "--seeds", "1,2")

    assert result.exit_code == 0, result.output
    get_client.assert_not_called()
    assert "exact_match = True" in result.output
    repro = _row()["provenance"]["reproducibility"]
    assert set(repro) == {"exact_match", "seeds", "conclusion_stable"}
    assert repro["exact_match"] is True
    assert set(repro["seeds"]) == {"1", "2"}
    assert all(p < 1e-6 for p in repro["seeds"].values())
    assert repro["conclusion_stable"] is True
    assert _row()["provenance"]["data_sha256"] == _sha256(CLIMATE_CSV)  # the rest is kept

    _tamper_statistic(0.5)
    result = _invoke("rerun", "--result-id", RESULT_ID, "--seeds", "1,2")

    assert result.exit_code == 1
    assert _row()["provenance"]["reproducibility"]["exact_match"] is False


def test_rerun_exits_1_on_a_sha256_mismatch(metric_db):
    with get_session() as session:
        r = session.query(Result).filter_by(id=RESULT_ID).one()
        r.provenance = {**r.provenance, "data_sha256": "e" * 64}

    with patch("kosmos.execution.executor.CodeExecutor.execute_with_data") as execute:
        result = _invoke("rerun", "--result-id", RESULT_ID, "--seeds", "1")

    assert result.exit_code == 1
    execute.assert_not_called()
    assert "reproducibility" not in _row()["provenance"]


def test_rerun_unknown_result_exits_1(metric_db):
    result = _invoke("rerun", "--result-id", "nope")

    assert result.exit_code == 1
