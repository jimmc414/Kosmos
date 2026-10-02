"""
Unit tests for ResearchDirectorAgent._handle_analyze_result_action (viability plan P1-1).

The verdict must be persisted on the result row and the hypothesis status,
failed executions must skip the LLM, and the analyst fallback must accept a
missing p-value.
"""

from unittest.mock import Mock, patch

import pytest

from kosmos.agents.data_analyst import DataAnalystAgent, ResultInterpretation
from kosmos.core.workflow import WorkflowState
from kosmos.db import get_session
from kosmos.db import operations
from kosmos.db.models import Hypothesis as DBHypothesis, HypothesisStatus, Result
from kosmos.models.result import ExperimentResult, ResultStatus
from tests.unit.agents.conftest import EXP_ID, H_ID

RESULT_ID = "res-analyze-1"


def _seed_result(data, p_value=None, effect_size=None):
    with get_session() as session:
        operations.create_result(
            session, id=RESULT_ID, experiment_id=EXP_ID, data=data,
            p_value=p_value, effect_size=effect_size,
        )


def _interpretation(supported):
    return ResultInterpretation(
        experiment_id=EXP_ID, hypothesis_supported=supported, confidence=0.9,
        summary="S", key_findings=["k"], significance_interpretation="",
        biological_significance=None, comparison_to_prior_work=None,
        potential_confounds=[], follow_up_experiments=[], anomalies_detected=[],
        patterns_detected=[], overall_assessment="",
    )


def _stored():
    with get_session() as session:
        r = session.query(Result).filter_by(id=RESULT_ID).one()
        h = session.query(DBHypothesis).filter_by(id=H_ID).one()
        return r.supports_hypothesis, r.interpretation, r.key_findings, h.status


async def test_supported_verdict_is_persisted(db_director):
    _seed_result({"execution_success": True, "p_value": 0.01}, p_value=0.01, effect_size=0.9)
    db_director._data_analyst = Mock(interpret_results=Mock(return_value=_interpretation(True)))

    await db_director._handle_analyze_result_action(RESULT_ID)

    supports, interpretation, key_findings, status = _stored()
    assert supports is True
    assert interpretation == "S"
    assert key_findings == ["k"]
    assert status == HypothesisStatus.SUPPORTED
    assert H_ID in db_director.research_plan.supported_hypotheses
    db_director.workflow.transition_to.assert_called()
    assert db_director.workflow.transition_to.call_args.args[0] == WorkflowState.REFINING


async def test_failed_execution_skips_the_llm(db_director):
    _seed_result({"execution_success": False, "error_type": "ExecutionError", "error": "exit 1"})
    db_director._data_analyst = Mock()

    await db_director._handle_analyze_result_action(RESULT_ID)

    db_director._data_analyst.interpret_results.assert_not_called()
    supports, interpretation, _, status = _stored()
    assert supports is None
    assert interpretation.startswith("Execution failed")
    assert status == HypothesisStatus.INCONCLUSIVE
    assert H_ID in db_director.research_plan.tested_hypotheses


def test_fallback_interpretation_accepts_missing_p_value():
    from datetime import datetime, timezone
    from kosmos.models.result import ExecutionMetadata

    now = datetime.now(timezone.utc)
    result = ExperimentResult(
        id="r", experiment_id="e", protocol_id="e", status=ResultStatus.FAILED,
        raw_data={}, primary_p_value=None,
        metadata=ExecutionMetadata(
            start_time=now, end_time=now, duration_seconds=0.0, python_version="3",
            platform="p", experiment_id="e", protocol_id="e",
        ),
    )
    with patch("kosmos.agents.data_analyst.get_client"):
        interpretation = DataAnalystAgent()._create_fallback_interpretation(result)

    assert "undetermined" in interpretation.significance_interpretation
