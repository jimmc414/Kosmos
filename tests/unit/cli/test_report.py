"""
Tests for the findings artifacts and `kosmos report` (viability plan C-1).

The analyze handler writes a Finding JSON for every validated or rejected result to
<artifacts_dir>/<run_id>/findings/<result_id>.json, beside the saved code, and a
failure to write it never changes the verdict. `kosmos report --run-id` renders the
run's result rows as Markdown without calling an LLM.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kosmos.cli.main import app
from kosmos.db import get_session
from kosmos.db import operations
from kosmos.db.models import Result
from kosmos.world_model.artifacts import ArtifactStateManager
from tests.unit.agents.conftest import EXP_ID, H_ID, db_director, in_memory_db  # noqa: F401
from tests.unit.validation.test_director_gate import (  # noqa: F401
    CLIMATE_CSV,
    _pearson_stats,
    _use_dataset,
    gate_director,
)

RUN_ID = "r1"
QUESTION = "Does CO2 predict temperature?"
VALID_ID = "res-report-validated"
REJECT_ID = "res-report-rejected"
FAIL_ID = "res-report-failed"
GIT_SHA = "0123456789abcdef0123456789abcdef01234567"
DATA_SHA = "d" * 64
ERROR = "KeyError: 'temp_anomaly' [column missing]"
GATE_RESULT_ID = "res-findings-1"


def _provenance(code_path, seed=42):
    return {"git_sha": GIT_SHA, "data_sha256": DATA_SHA, "seed": seed, "code_path": code_path}


@pytest.fixture
def report_db(in_memory_db):  # noqa: F811
    """Run r1: one validated, one rejected and one failed result; plus another run's result."""
    with get_session() as session:
        operations.create_research_session(
            session, id=RUN_ID, research_question=QUESTION, domain="climate",
        )
        operations.create_hypothesis(
            session, id=H_ID, research_question=QUESTION,
            statement="CO2 concentration predicts the temperature anomaly",
            rationale="Radiative forcing increases with CO2 concentration", domain="climate",
        )
        operations.create_experiment(
            session, id=EXP_ID, hypothesis_id=H_ID, experiment_type="data_analysis",
            description="d", protocol={"name": "p"}, domain="climate",
        )
        operations.create_result(
            session, id=VALID_ID, experiment_id=EXP_ID, data={},
            statistical_tests={"test_type": "pearson_correlation", "statistic": 0.9317, "n": 64},
            p_value=5.8e-29, effect_size=0.9317, supports_hypothesis=True,
            interpretation="CO2 is strongly correlated with the temperature anomaly.",
            run_id=RUN_ID, execution_success=True, data_source="file", random_seed=42,
            validation_status="validated",
            validation_detail={
                "recomputed_match": True,
                "null_model": {"permutation_p_value": 0.004, "passes_null_test": True,
                               "persists_in_noise": False},
                "scholar_eval": {"overall_score": 0.85, "passes_threshold": True},
            },
            provenance=_provenance(f"/runs/{RUN_ID}/code/{VALID_ID}.py"), cost_usd=0.004,
        )
        operations.create_result(
            session, id=REJECT_ID, experiment_id=EXP_ID, data={},
            statistical_tests={"test_type": "pearson_correlation", "statistic": 0.05, "n": 64},
            p_value=0.69, effect_size=0.05,
            run_id=RUN_ID, execution_success=True, data_source="file", random_seed=42,
            validation_status="rejected",
            validation_detail={
                "recomputed_match": True,
                "null_model": {"permutation_p_value": 0.71, "passes_null_test": False,
                               "persists_in_noise": False},
            },
            provenance=_provenance(f"/runs/{RUN_ID}/code/{REJECT_ID}.py"), cost_usd=0.002,
        )
        operations.create_result(
            session, id=FAIL_ID, experiment_id=EXP_ID, data={},
            run_id=RUN_ID, execution_success=False, data_source=None, random_seed=42,
            validation_status="unvalidated", validation_detail={"reason": "execution_failed"},
            error_message=ERROR, provenance=_provenance(f"/runs/{RUN_ID}/code/{FAIL_ID}.py"),
        )
        operations.create_result(
            session, id="res-other-run", experiment_id=EXP_ID, data={},
            run_id="run_other", execution_success=True, data_source="file",
            validation_status="validated",
        )


def _invoke(args):
    # The CLI callback re-initializes the database from config; keep the test's database
    with patch("kosmos.db.init_from_config"):
        return CliRunner().invoke(app, ["report", *args])


# --- kosmos report ---------------------------------------------------------------------

def test_report_renders_findings_metrics_and_failures(report_db, tmp_path):
    out = tmp_path / "report.md"

    result = _invoke(["--run-id", RUN_ID, "--output", str(out)])

    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert VALID_ID in text and REJECT_ID in text
    assert "validated" in text
    assert GIT_SHA in text
    assert DATA_SHA in text
    assert ERROR in text
    assert QUESTION in text
    assert f"/runs/{RUN_ID}/code/{VALID_ID}.py" in text
    assert "res-other-run" not in text
    validated_part = text.split("## Validated Findings")[1].split("## Rejected Findings")[0]
    rejected_part = text.split("## Rejected Findings")[1].split("## Failed Experiments")[0]
    failed_part = text.split("## Failed Experiments")[1]
    assert VALID_ID in validated_part and REJECT_ID not in validated_part
    assert REJECT_ID in rejected_part and "fails the null test" in rejected_part
    assert FAIL_ID in failed_part and ERROR in failed_part
    assert "| Experiments attempted | 3 |" in text
    assert "| Experiments failed | 1 |" in text
    assert "| Findings validated | 1 |" in text
    assert "| Findings rejected | 1 |" in text
    assert "| LLM cost attributed to results | $0.0060 |" in text
    assert "| Cost per validated finding | $0.0060 |" in text


def test_report_default_path_is_under_the_run_artifacts(report_db, tmp_path):
    config = Mock()
    config.research.artifacts_dir = str(tmp_path / "runs")
    with patch("kosmos.config.get_config", return_value=config):
        result = _invoke(["--run-id", RUN_ID])

    assert result.exit_code == 0, result.output
    assert VALID_ID in (tmp_path / "runs" / RUN_ID / "report.md").read_text()


def test_report_unknown_run_exits_1(report_db, tmp_path):
    out = tmp_path / "report.md"

    result = _invoke(["--run-id", "nope", "--output", str(out)])

    assert result.exit_code == 1
    assert not out.exists()


def test_report_calls_no_llm(report_db, tmp_path):
    with patch("kosmos.core.llm.get_client") as get_client:
        result = _invoke(["--run-id", RUN_ID, "--output", str(tmp_path / "r.md")])

    assert result.exit_code == 0, result.output
    get_client.assert_not_called()


# --- findings artifacts from the analyze handler ------------------------------------------

def _seed_gate_result(director, stats, data_source="file"):
    """A result of the director's run, with its code saved the way P2-3 saves it."""
    code_path = director._save_run_code(GATE_RESULT_ID, "results = {}")
    with get_session() as session:
        operations.create_result(
            session, id=GATE_RESULT_ID, experiment_id=EXP_ID, data={"execution_success": True},
            statistical_tests=stats, p_value=stats.get("p_value"), effect_size=stats.get("effect_size"),
            execution_success=True, data_source=data_source, run_id=director.run_id,
            random_seed=0, provenance=_provenance(str(code_path), seed=0),
        )
    return code_path


def _findings_path(director):
    return director.artifacts_dir / director.run_id / "findings" / f"{GATE_RESULT_ID}.json"


def _status():
    with get_session() as session:
        r = session.query(Result).filter_by(id=GATE_RESULT_ID).one()
        return r.validation_status, r.supports_hypothesis


async def test_validated_result_writes_the_findings_json(gate_director, tmp_path):  # noqa: F811
    code_path = _seed_gate_result(gate_director, _pearson_stats(pd.read_csv(CLIMATE_CSV)))

    await gate_director._handle_analyze_result_action(GATE_RESULT_ID)

    assert _status() == ("validated", True)
    path = _findings_path(gate_director)
    assert path.parent.parent == code_path.parent.parent  # beside <run_id>/code/
    finding = json.loads(path.read_text())
    assert finding["finding_id"] == GATE_RESULT_ID
    assert finding["hypothesis_id"] == H_ID
    assert finding["scholar_eval"]["passes_threshold"] is True
    assert "overall_score" in finding["scholar_eval"]
    assert finding["null_model_result"]["passes_null_test"] is True
    assert finding["null_model_result"]["shuffle_method"] == "target"
    assert finding["code_provenance"] == {"notebook_path": str(code_path), "cell_index": 0}
    assert code_path.suffix == ".py" and code_path.exists()
    assert finding["statistics"]["test_type"] == "pearson_correlation"
    assert finding["summary"].startswith("CO2 concentration is strongly correlated")
    assert finding["metadata"]["validation_status"] == "validated"
    assert finding["metadata"]["recomputed_match"] is True
    assert finding["metadata"]["provenance"]["git_sha"] == GIT_SHA

    # The report of the director's run shows the finding
    out = tmp_path / "run_report.md"
    result = _invoke(["--run-id", gate_director.run_id, "--output", str(out)])
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert GATE_RESULT_ID in text.split("## Validated Findings")[1].split("## Rejected Findings")[0]
    assert "Does CO2 predict temperature?" in text


async def test_rejected_result_writes_the_findings_json(gate_director, tmp_path):  # noqa: F811
    df = pd.read_csv(CLIMATE_CSV)
    df["temp_anomaly_c"] = np.random.default_rng(0).permutation(df["temp_anomaly_c"].values)
    shuffled_csv = tmp_path / "climate_shuffled.csv"
    df.to_csv(shuffled_csv, index=False)
    _use_dataset(gate_director, shuffled_csv)
    _seed_gate_result(gate_director, _pearson_stats(df))

    await gate_director._handle_analyze_result_action(GATE_RESULT_ID)

    assert _status()[0] == "rejected"
    finding = json.loads(_findings_path(gate_director).read_text())
    assert finding["metadata"]["validation_status"] == "rejected"
    assert finding["null_model_result"]["passes_null_test"] is False
    assert "scholar_eval" in finding


async def test_unvalidated_result_writes_no_findings_json(gate_director):  # noqa: F811
    _seed_gate_result(gate_director, _pearson_stats(pd.read_csv(CLIMATE_CSV)), data_source="synthetic")

    await gate_director._handle_analyze_result_action(GATE_RESULT_ID)

    assert _status()[0] == "unvalidated"
    assert not _findings_path(gate_director).exists()
    assert not _findings_path(gate_director).parent.exists()


async def test_artifact_write_failure_keeps_the_verdict(gate_director):  # noqa: F811
    _seed_gate_result(gate_director, _pearson_stats(pd.read_csv(CLIMATE_CSV)))

    with patch.object(ArtifactStateManager, "save_finding_artifact",
                      new=AsyncMock(side_effect=OSError("disk full"))) as save, \
         patch.object(gate_director, "_handle_error_with_recovery") as recovery:
        await gate_director._handle_analyze_result_action(GATE_RESULT_ID)

    save.assert_awaited_once()
    recovery.assert_not_called()
    assert _status() == ("validated", True)
    assert H_ID in gate_director.research_plan.supported_hypotheses
    assert not _findings_path(gate_director).exists()
