"""
Tests for the honest end-of-run report (viability plan P2-4).

build_run_results must read hypotheses, experiments and results from the
database columns, take LLM usage from get_usage_stats, and count execution
success, data source and validation per result. The director attributes LLM
cost per result. The results table and the exports show all of it.
"""

import io
import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from rich.console import Console

import kosmos.db as kosmos_db
from kosmos.cli.commands.run_results import build_run_results
from kosmos.cli.views.results_viewer import ResultsViewer
from kosmos.core.workflow import ResearchPlan
from kosmos.db import get_session, init_database
from kosmos.db import operations

RUN_ID = "run_abc123def456"
H_ID = "hyp-results-1"
EXP_ID = "exp-results-1"
OK_ID = "res-ok-0001"
FAIL_ID = "res-fail-0002"
STATEMENT = "CO2 concentration predicts the temperature anomaly"


@pytest.fixture
def in_memory_db():
    saved = (kosmos_db._engine, kosmos_db._SessionLocal)
    init_database("sqlite:///:memory:")
    yield
    kosmos_db.reset_database()
    kosmos_db._engine, kosmos_db._SessionLocal = saved


@pytest.fixture
def loops(in_memory_db):
    """1 hypothesis, 1 experiment, 2 results: one validated from the file, one failed."""
    with get_session() as session:
        operations.create_hypothesis(
            session, id=H_ID, research_question="Does CO2 predict temperature?",
            statement=STATEMENT,
            rationale="Radiative forcing increases with CO2 concentration",
            domain="climate", novelty_score=0.7, testability_score=0.9,
        )
        operations.create_experiment(
            session, id=EXP_ID, hypothesis_id=H_ID, experiment_type="data_analysis",
            description="d", protocol={"name": "p"}, domain="climate",
        )
        operations.create_result(
            session, id=OK_ID, experiment_id=EXP_ID, data={"x": 1},
            statistical_tests={"test_type": "pearson_correlation", "statistic": 0.9317, "n": 64},
            p_value=5.8e-29, effect_size=0.9317, supports_hypothesis=True,
            run_id=RUN_ID, execution_success=True, data_source="file", random_seed=42,
            validation_status="validated", cost_usd=0.004,
        )
        operations.create_result(
            session, id=FAIL_ID, experiment_id=EXP_ID, data={"x": 2},
            run_id=RUN_ID, execution_success=False, data_source=None, random_seed=42,
            validation_status="unvalidated", validation_detail={"reason": "execution_failed"},
            error_message="KeyError: 'temp_anomaly' [column missing]",
        )
        # A result of another run must not appear
        operations.create_result(
            session, id="res-other-run", experiment_id=EXP_ID, data={},
            run_id="run_other", execution_success=True, data_source="file",
            validation_status="validated",
        )

    plan = ResearchPlan(research_question="q", max_iterations=3)
    plan.hypothesis_pool = [H_ID]
    plan.tested_hypotheses = [H_ID]
    plan.supported_hypotheses = [H_ID]
    plan.untestable_hypotheses = ["hyp-untestable-1"]
    plan.completed_experiments = [EXP_ID]

    director = Mock()
    director.run_id = RUN_ID
    director.research_plan = plan
    director.llm_client = Mock(
        get_usage_stats=lambda: {
            "total_requests": 9, "total_cost_usd": 0.0123,
            "total_input_tokens": 4000, "total_output_tokens": 1500,
        },
        total_cost_usd=0.0123,
    )
    director.get_research_status.return_value = {
        "domain": "climate", "workflow_state": "converged", "iteration": 1,
        "has_converged": True, "convergence_reason": "done",
        "hypothesis_pool_size": 1, "hypotheses_tested": 1,
        "hypotheses_supported": 1, "hypotheses_rejected": 0,
    }
    return director


def test_run_results_read_columns_and_real_cost(loops):
    results = build_run_results(loops, "Does CO2 predict temperature?", 3)

    hyp = results["hypotheses"][0]
    assert hyp["claim"] == STATEMENT
    assert hyp["status"] == "generated"
    assert hyp["tested"] is True and hyp["supported"] is True and hyp["untestable"] is False
    assert hyp["priority_score"] is None

    assert results["experiments"] == [{
        "id": EXP_ID, "type": "data_analysis", "status": "created",
        "created_at": results["experiments"][0]["created_at"], "hypothesis_id": H_ID,
    }]

    by_id = {r["result_id"]: r for r in results["results"]}
    assert set(by_id) == {OK_ID, FAIL_ID}
    ok, fail = by_id[OK_ID], by_id[FAIL_ID]
    assert ok["hypothesis_id"] == H_ID
    assert (ok["execution_success"], ok["data_source"], ok["validation_status"]) == (True, "file", "validated")
    assert (ok["test_type"], ok["statistic"], ok["n"]) == ("pearson_correlation", 0.9317, 64)
    assert ok["cost_usd"] == pytest.approx(0.004)
    assert (fail["execution_success"], fail["data_source"]) == (False, None)
    assert fail["validation_reason"] == "execution_failed"
    assert fail["error_message"].startswith("KeyError")

    m = results["metrics"]
    assert m["api_calls"] == 9
    assert m["experiments_attempted"] == 2
    assert m["experiments_succeeded"] == 1
    assert m["experiments_failed"] == 1
    assert m["results_from_file"] == 1
    assert m["results_synthetic"] == 0
    assert m["findings_validated"] == 1
    assert m["findings_rejected"] == 0
    assert m["total_cost_usd"] == 0.0123
    assert m["cost_per_validated_finding"] == 0.0123
    assert (m["input_tokens"], m["output_tokens"]) == (4000, 1500)
    assert m["hypotheses_untestable"] == 1
    assert results["run_id"] == RUN_ID and results["id"] == RUN_ID

    json.dumps(results, default=str)  # the --output JSON export must serialize


def test_no_validated_finding_gives_no_cost_per_finding(loops):
    with get_session() as session:
        operations.update_result_validation(session, OK_ID, "rejected", detail={"reason": "null_model"})

    m = build_run_results(loops, "q", 3)["metrics"]

    assert m["findings_validated"] == 0
    assert m["findings_rejected"] == 1
    assert m["cost_per_validated_finding"] is None


def test_client_without_usage_stats_reports_unknown_cost(loops):
    loops.llm_client = object()

    m = build_run_results(loops, "q", 3)["metrics"]

    assert m["api_calls"] == 0
    assert m["total_cost_usd"] is None
    assert m["cost_per_validated_finding"] is None


def test_failed_experiment_is_listed(loops):
    """A failed experiment never enters completed_experiments; its result still names it."""
    loops.research_plan.completed_experiments = []

    results = build_run_results(loops, "q", 3)

    assert [e["id"] for e in results["experiments"]] == [EXP_ID]


def test_results_table_marks_failed_rows(loops):
    out = io.StringIO()
    viewer = ResultsViewer(console_instance=Console(file=out, width=200))

    results = build_run_results(loops, "q", 3)
    viewer.display_results_table(results["results"])
    viewer.display_metrics_summary(results["metrics"])
    text = out.getvalue()

    assert "pearson_correlation" in text
    assert "validated" in text
    assert "FAIL" in text and "OK" in text
    assert "KeyError: 'temp_anomaly' [column missing]" in text  # markup escaped, not swallowed
    assert "$0.0123" in text
    assert "Findings Validated" in text
    assert "Cache Hits" not in text  # no provider reports cache counts; show none rather than 0


def test_results_table_empty():
    out = io.StringIO()
    ResultsViewer(console_instance=Console(file=out, width=120)).display_results_table([])
    assert "No results yet" in out.getvalue()


def test_exports_include_results(loops, tmp_path):
    viewer = ResultsViewer(console_instance=Console(file=io.StringIO()))
    results = build_run_results(loops, "q", 3)

    viewer.export_to_json(results, tmp_path / "r.json")
    viewer.export_to_markdown(results, tmp_path / "r.md")

    exported = json.loads((tmp_path / "r.json").read_text())
    assert {r["result_id"] for r in exported["results"]} == {OK_ID, FAIL_ID}
    md = (tmp_path / "r.md").read_text()
    assert "## Results" in md
    assert "pearson_correlation" in md
    assert "FAIL" in md
    assert "**Total cost:** $0.0123" in md
    assert "- **Priority:** -" in md


# --- per-result cost attribution in the director ---------------------------------------

def _director(cost_box):
    """Callers use in_memory_db: the director records a ResearchSession row in the current DB."""
    from kosmos.agents.research_director import ResearchDirectorAgent

    with patch('kosmos.agents.research_director.get_client', return_value=MagicMock()), \
         patch('kosmos.agents.research_director.get_world_model', return_value=MagicMock()), \
         patch('kosmos.agents.research_director.SkillLoader') as mock_skills, \
         patch('kosmos.db.init_from_config'):
        mock_skills.return_value.load_skills_for_task.return_value = ""
        d = ResearchDirectorAgent(research_question="q", domain="climate", config={"max_iterations": 3})

    client = MagicMock()
    type(client).total_cost_usd = property(lambda self: cost_box[0])
    d.llm_client = client
    return d


def test_cost_since_snapshot(in_memory_db):
    cost = [0.010]
    d = _director(cost)

    start = d._llm_cost_total()
    cost[0] = 0.0125

    assert start == 0.010
    assert d._llm_cost_since(start) == pytest.approx(0.0025)


def test_cost_unknown_for_mock_client(in_memory_db):
    """A client whose total_cost_usd is not a number attributes no cost."""
    d = _director([0.0])
    d.llm_client = MagicMock()

    assert d._llm_cost_total() is None
    assert d._llm_cost_since(d._llm_cost_total()) is None
