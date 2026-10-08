"""
The director's handling of code it may not run (viability plan P3-1 (4)).

A rejected_unsafe result is analysed without an LLM call and keeps its status;
under an emergency stop the generated code is neither validated into a result
nor executed.
"""

from unittest.mock import Mock

from kosmos.db import get_session
from kosmos.db.models import Result
from kosmos.safety.guardrails import SafetyGuardrails
from tests.unit.agents.conftest import EXP_ID, H_ID

UNSAFE_CODE = "import subprocess\nsubprocess.run(['ls'])\nresults = {}"


async def test_rejected_unsafe_result_is_analysed_without_llm_and_keeps_its_status(db_director):
    d = db_director
    d._code_generator = Mock(generate=Mock(return_value=UNSAFE_CODE))
    d._code_executor = Mock(use_sandbox=True)
    d._data_analyst = Mock()

    await d._handle_execute_experiment_action(EXP_ID)
    with get_session() as session:
        result_id = session.query(Result).one().id

    await d._handle_analyze_result_action(result_id)

    d._data_analyst.interpret_results.assert_not_called()
    with get_session() as session:
        row = session.query(Result).one()
        assert row.validation_status == "rejected_unsafe"
        assert row.validation_detail == {"reason": "execution_failed", "rejected_unsafe": True}
        assert row.supports_hypothesis is None
        assert row.interpretation.startswith("Code rejected as unsafe, not executed: ")
    assert H_ID in d.research_plan.tested_hypotheses
    assert H_ID not in d.research_plan.supported_hypotheses


async def test_emergency_stop_blocks_execution(db_director, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the stop flag file and the incident log live in the cwd
    SafetyGuardrails.STOP_FLAG_FILE.write_text("{}")
    d = db_director
    d._code_executor = Mock(use_sandbox=True)

    await d._handle_execute_experiment_action(EXP_ID)

    d._code_executor.execute_with_data.assert_not_called()
    assert d._guardrails.is_emergency_stop_active()
    assert d._consecutive_errors == 1
    with get_session() as session:
        assert session.query(Result).count() == 0
