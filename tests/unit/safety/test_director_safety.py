"""
CodeValidator and the emergency stop on the director path (viability plan P3-1).

Generated code is checked before it runs: unsafe code becomes a rejected_unsafe
result and never reaches the executor. The guardrails the director and
execute_protocol_code build leave process signals alone; when signal handlers
are asked for, they chain to the handler installed before them.
"""

import inspect
import json
import os
import signal
from unittest.mock import Mock, patch

from kosmos.db import get_session
from kosmos.db.models import Experiment, Result
from kosmos.execution.executor import ExecutionResult
from kosmos.execution.sandbox import DockerSandbox
from kosmos.safety.guardrails import SafetyGuardrails
from tests.unit.agents.conftest import EXP_ID

UNSAFE_CODE = "import os\nos.system('echo hi')\nresults={}"


def test_emergency_stop_logs_an_incident_without_a_violation(tmp_path):
    log = tmp_path / "x.jsonl"
    guardrails = SafetyGuardrails(enable_signal_handlers=False, incident_log_path=str(log))

    guardrails.trigger_emergency_stop("test", "reason")

    lines = log.read_text().splitlines()
    assert len(lines) == 1
    assert '"violation": null' in lines[0]
    assert json.loads(lines[0])["context"]["reason"] == "reason"
    assert guardrails.is_emergency_stop_active()


async def test_unsafe_code_is_stored_as_rejected_unsafe_and_never_executed(db_director):
    d = db_director
    d._code_generator = Mock(generate=Mock(return_value=UNSAFE_CODE))
    d._code_executor = Mock(use_sandbox=True)

    await d._handle_execute_experiment_action(EXP_ID)

    d._code_executor.execute_with_data.assert_not_called()
    d._code_executor.execute.assert_not_called()
    with get_session() as session:
        row = session.query(Result).one()
        assert row.validation_status == "rejected_unsafe"
        assert row.execution_success is False
        assert row.data_source is None
        assert "Dangerous import detected: os" in row.error_message
        assert row.run_id == d.run_id
        result_id = row.id
        experiment = session.get(Experiment, EXP_ID)
        assert experiment.code_generated == UNSAFE_CODE
        assert experiment.status.value == "failed"
    assert d.research_plan.results == [result_id]
    assert EXP_ID not in d.research_plan.experiment_queue
    assert EXP_ID not in d.research_plan.completed_experiments
    assert d.workflow.transition_to.call_args.args[0].value == "analyzing"


def test_signal_handler_chains_to_the_previous_handler(tmp_path):
    calls = []

    def custom(signum, frame):
        calls.append(signum)

    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        signal.signal(signal.SIGINT, custom)
        guardrails = SafetyGuardrails(
            enable_signal_handlers=True, incident_log_path=str(tmp_path / "x.jsonl")
        )

        os.kill(os.getpid(), signal.SIGINT)

        assert calls == [signal.SIGINT]
        assert guardrails.is_emergency_stop_active()
        assert guardrails.emergency_stop.triggered_by == "signal"
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


async def test_director_guardrails_leave_signals_alone_and_limit_the_sandbox(db_director):
    d = db_director
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    executor = Mock(use_sandbox=True)
    executor.execute_with_data.return_value = ExecutionResult(success=True, return_value={})

    with patch("kosmos.execution.executor.CodeExecutor", return_value=executor) as executor_cls:
        await d._handle_execute_experiment_action(EXP_ID)
    SafetyGuardrails()  # the default registers no handlers either

    assert {sig: signal.getsignal(sig) for sig in before} == before
    executor.execute_with_data.assert_called_once()
    sandbox_config = executor_cls.call_args.kwargs["sandbox_config"]
    limits = d._guardrails.enforce_resource_limits()
    assert sandbox_config["memory_limit"] == f"{limits.max_memory_mb}m"
    assert sandbox_config["timeout"] == limits.max_execution_time_seconds
    assert set(sandbox_config) <= set(inspect.signature(DockerSandbox.__init__).parameters)
