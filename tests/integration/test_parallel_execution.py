"""
Integration tests for parallel experiment execution.

Tests ParallelExperimentExecutor, ExperimentTask, ParallelExecutionResult,
_execute_single_experiment and ResourceAwareScheduler against the API in
kosmos/execution/parallel.py.

Worker stubs are module-level functions: ProcessPoolExecutor pickles the worker
by reference, so a stub patched over _execute_single_experiment must be a
top-level function, never a Mock.
"""

import os
import sys
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

from kosmos.execution.parallel import (
    ExperimentTask,
    ParallelExecutionResult,
    ParallelExperimentExecutor,
    ResourceAwareScheduler,
    _execute_single_experiment,
)

WORKER = "kosmos.execution.parallel._execute_single_experiment"


# ============================================================================
# Module-level worker stubs (picklable for the process pool)
# ============================================================================

def _stub_success(task, use_sandbox=False, timeout=None):
    """Succeed after a short delay, recording when and where the task ran."""
    started = time.time()
    time.sleep(task.config.get("sleep", 0.0) if task.config else 0.0)
    return ParallelExecutionResult(
        experiment_id=task.experiment_id,
        success=True,
        result={"pid": os.getpid(), "start": started, "end": time.time(),
                "use_sandbox": use_sandbox, "timeout": timeout},
        execution_time=time.time() - started,
    )


def _stub_fail_some(task, use_sandbox=False, timeout=None):
    """Raise for tasks whose id contains 'failure', succeed otherwise."""
    if "failure" in task.experiment_id:
        raise RuntimeError(f"Experiment {task.experiment_id} failed")
    return _stub_success(task, use_sandbox, timeout)


# ============================================================================
# Data classes
# ============================================================================

class TestExperimentTask:
    """Test the ExperimentTask data class."""

    def test_defaults(self):
        task = ExperimentTask(experiment_id="exp_1", code="x = 1")
        assert task.data_path is None
        assert task.config is None
        assert task.priority == 0


class TestParallelExecutionResult:
    """Test ParallelExecutionResult data class."""

    def test_success_result(self):
        """Test creating success result."""
        result = ParallelExecutionResult(
            experiment_id="exp_1",
            success=True,
            result={"metric": 0.95},
            execution_time=5.5,
        )

        assert result.success is True
        assert result.error is None
        assert result.result["metric"] == 0.95
        assert result.execution_time == 5.5
        assert result.started_at is None and result.completed_at is None

    def test_failure_result(self):
        """Test creating failure result."""
        result = ParallelExecutionResult(
            experiment_id="exp_1",
            success=False,
            result=None,
            execution_time=0.0,
            error="Experiment execution failed",
        )

        assert result.success is False
        assert result.result is None
        assert "failed" in result.error.lower()


# ============================================================================
# Executor configuration
# ============================================================================

class TestParallelExperimentExecutor:
    """Test ParallelExperimentExecutor configuration and batch execution."""

    def test_initialization(self):
        """Test executor initialization."""
        executor = ParallelExperimentExecutor(max_workers=4)

        assert executor.max_workers == 4
        assert executor.max_workers_io == 8  # 2x CPU workers by default
        assert executor.enable_progress_logging is True
        assert executor.chunk_size == 1

    def test_max_workers_configuration(self):
        """Test configuring max workers."""
        assert ParallelExperimentExecutor(max_workers=2).max_workers == 2
        executor = ParallelExperimentExecutor(max_workers=8, max_workers_io=3)
        assert executor.max_workers == 8
        assert executor.max_workers_io == 3

    def test_default_workers_leave_one_core(self):
        """Default worker count is CPU count - 1, never below 1."""
        with patch("kosmos.execution.parallel.multiprocessing.cpu_count", return_value=8):
            executor = ParallelExperimentExecutor()
        assert executor.max_workers == 7
        assert executor.max_workers_io == 14

        with patch("kosmos.execution.parallel.multiprocessing.cpu_count", return_value=1):
            assert ParallelExperimentExecutor().max_workers == 1

    def test_empty_batch(self):
        """An empty batch returns no results and starts no pool."""
        executor = ParallelExperimentExecutor(max_workers=2)
        with patch("kosmos.execution.parallel.ProcessPoolExecutor") as pool:
            assert executor.execute_batch([]) == []
        pool.assert_not_called()

    def test_execute_batch(self):
        """Test executing batch of experiments: one result per task."""
        executor = ParallelExperimentExecutor(max_workers=4)
        tasks = [ExperimentTask(f"exp_{i}", "x = 1") for i in range(10)]

        with patch(WORKER, _stub_success):
            results = executor.execute_batch(tasks, use_sandbox=True, timeout_per_task=30.0)

        assert len(results) == 10
        assert all(r.success for r in results)
        assert {r.experiment_id for r in results} == {t.experiment_id for t in tasks}
        # Batch options reach every worker call
        assert all(r.result["use_sandbox"] is True for r in results)
        assert all(r.result["timeout"] == 30.0 for r in results)

    def test_tasks_run_concurrently(self):
        """Tasks run in more than one process at overlapping times."""
        executor = ParallelExperimentExecutor(max_workers=4)
        tasks = [ExperimentTask(f"exp_{i}", "x = 1", config={"sleep": 0.5}) for i in range(4)]

        with patch(WORKER, _stub_success):
            results = executor.execute_batch(tasks)

        assert len(results) == 4
        pids = {r.result["pid"] for r in results}
        assert os.getpid() not in pids  # executed in worker processes
        assert len(pids) >= 2
        spans = sorted((r.result["start"], r.result["end"]) for r in results)
        overlaps = sum(1 for a, b in zip(spans, spans[1:]) if b[0] < a[1])
        assert overlaps >= 1, f"no two tasks overlapped in time: {spans}"

    def test_priority_ordering(self):
        """Higher-priority tasks are submitted, and so start, first."""
        executor = ParallelExperimentExecutor(max_workers=1)
        tasks = [
            ExperimentTask("low", "x = 1", priority=1, config={"sleep": 0.05}),
            ExperimentTask("high", "x = 1", priority=10, config={"sleep": 0.05}),
            ExperimentTask("mid", "x = 1", priority=5, config={"sleep": 0.05}),
        ]

        with patch(WORKER, _stub_success):
            results = executor.execute_batch(tasks)

        by_start = sorted(results, key=lambda r: r.result["start"])
        assert [r.experiment_id for r in by_start] == ["high", "mid", "low"]

    def test_error_handling(self):
        """A worker that raises yields a failed result; the rest still succeed."""
        executor = ParallelExperimentExecutor(max_workers=2)
        tasks = [ExperimentTask(i, "x = 1") for i in ["success_1", "failure_1", "success_2"]]

        with patch(WORKER, _stub_fail_some):
            results = executor.execute_batch(tasks)

        assert len(results) == 3
        by_id = {r.experiment_id: r for r in results}
        assert by_id["success_1"].success is True
        assert by_id["success_2"].success is True
        failed = by_id["failure_1"]
        assert failed.success is False
        assert failed.result is None
        assert failed.execution_time == 0.0
        assert "failed" in failed.error.lower()

    def test_queue_management(self):
        """More tasks than workers: all complete eventually."""
        executor = ParallelExperimentExecutor(max_workers=2)
        tasks = [ExperimentTask(f"exp_{i}", "x = 1", config={"sleep": 0.05}) for i in range(10)]

        with patch(WORKER, _stub_success):
            results = executor.execute_batch(tasks)

        assert len(results) == 10
        assert all(r.success for r in results)

    def test_execute_batch_async_invokes_callback(self):
        """execute_batch_async returns a Future of the results and calls back per result."""
        executor = ParallelExperimentExecutor(max_workers=2)
        tasks = [ExperimentTask(f"exp_{i}", "x = 1") for i in range(3)]
        seen = []

        with patch(WORKER, _stub_success):
            future = executor.execute_batch_async(tasks, callback=seen.append)
            results = future.result(timeout=60)

        assert len(results) == 3
        assert sorted(r.experiment_id for r in seen) == ["exp_0", "exp_1", "exp_2"]

    def test_callback_errors_do_not_lose_results(self):
        """A raising callback is logged; every result is still returned."""
        executor = ParallelExperimentExecutor(max_workers=2)
        tasks = [ExperimentTask(f"exp_{i}", "x = 1") for i in range(2)]
        callback = MagicMock(side_effect=ValueError("callback boom"))

        with patch(WORKER, _stub_success):
            results = executor._execute_with_callbacks(tasks, callback)

        assert len(results) == 2
        assert callback.call_count == 2


# ============================================================================
# Single-task worker
# ============================================================================

class TestExecuteSingleExperiment:
    """Test the per-task worker in-process (execute_protocol_code mocked)."""

    def test_maps_execution_result(self):
        """The executor's dict maps onto ParallelExecutionResult."""
        task = ExperimentTask("exp_1", "x = 1", data_path="/data/x.csv", config={"timeout_seconds": 5})
        payload = {"success": True, "error": None, "return_value": {"p_value": 0.01}}

        with patch("kosmos.execution.executor.execute_protocol_code", return_value=payload) as run:
            result = _execute_single_experiment(task, use_sandbox=True)

        run.assert_called_once_with(
            code="x = 1",
            data_path="/data/x.csv",
            use_sandbox=True,
            sandbox_config={"timeout_seconds": 5},
        )
        assert result.experiment_id == "exp_1"
        assert result.success is True
        assert result.error is None
        assert result.result is payload
        assert isinstance(result.started_at, datetime)
        assert result.completed_at >= result.started_at
        assert result.execution_time >= 0.0

    def test_reports_failed_execution(self):
        """A failed execution keeps the executor's error message."""
        payload = {"success": False, "error": "Code validation failed"}
        with patch("kosmos.execution.executor.execute_protocol_code", return_value=payload):
            result = _execute_single_experiment(ExperimentTask("exp_1", "x = 1"))

        assert result.success is False
        assert result.error == "Code validation failed"

    def test_exception_becomes_failed_result(self):
        """An exception from the executor becomes a failed result, never a raise."""
        with patch("kosmos.execution.executor.execute_protocol_code",
                   side_effect=RuntimeError("sandbox unavailable")):
            result = _execute_single_experiment(ExperimentTask("exp_1", "x = 1"))

        assert result.success is False
        assert result.result is None
        assert result.error == "sandbox unavailable"
        assert result.completed_at is not None


class TestParallelExecutionWithRealExperiments:
    """Run real (tiny, validated) code through the pool without the Docker sandbox."""

    def test_parallel_experiment_workflow(self):
        """Valid code succeeds with its return value; unsafe code is refused by validation."""
        executor = ParallelExperimentExecutor(max_workers=2)
        tasks = [
            ExperimentTask(f"exp_{i}", f"x = {i} + 1\nresults = {{'value': x}}")
            for i in range(3)
        ]
        tasks.append(ExperimentTask("exp_unsafe", "import os\nos.system('echo unsafe')\nresults = {}"))

        results = executor.execute_batch(tasks, use_sandbox=False)

        assert len(results) == 4
        by_id = {r.experiment_id: r for r in results}
        for i in range(3):
            r = by_id[f"exp_{i}"]
            assert r.success is True, r.error
            assert r.result["return_value"] == {"value": i + 1}
            assert r.result["sandbox_used"] is False
        unsafe = by_id["exp_unsafe"]
        assert unsafe.success is False
        assert unsafe.error == "Code validation failed"
        assert unsafe.result["validation_errors"]


# ============================================================================
# Resource-aware scheduling
# ============================================================================

class TestResourceAwareScheduler:
    """Test ResourceAwareScheduler worker recommendations."""

    @staticmethod
    def _psutil(cpu, mem):
        fake = MagicMock()
        fake.cpu_percent.return_value = cpu
        fake.virtual_memory.return_value = MagicMock(percent=mem)
        return fake

    def test_idle_system_uses_all_but_one_core(self):
        scheduler = ResourceAwareScheduler()
        with patch("kosmos.execution.parallel.multiprocessing.cpu_count", return_value=8), \
                patch.dict(sys.modules, {"psutil": self._psutil(10.0, 20.0)}):
            assert scheduler.get_optimal_workers() == 7

    def test_busy_system_reduces_workers(self):
        scheduler = ResourceAwareScheduler(max_cpu_percent=50.0, max_memory_percent=50.0)
        with patch("kosmos.execution.parallel.multiprocessing.cpu_count", return_value=8), \
                patch.dict(sys.modules, {"psutil": self._psutil(90.0, 80.0)}):
            # 7 - 4 (CPU over by 40) - 3 (memory over by 30) = 0, floored at min_workers
            assert scheduler.get_optimal_workers() == 1

        scheduler = ResourceAwareScheduler(max_cpu_percent=85.0, max_memory_percent=85.0)
        with patch("kosmos.execution.parallel.multiprocessing.cpu_count", return_value=8), \
                patch.dict(sys.modules, {"psutil": self._psutil(100.0, 95.0)}):
            # CPU over by 15 -> -1; memory over by 10 -> -1
            assert scheduler.get_optimal_workers() == 5

    def test_min_workers_floor(self):
        scheduler = ResourceAwareScheduler(min_workers=3)
        with patch("kosmos.execution.parallel.multiprocessing.cpu_count", return_value=2), \
                patch.dict(sys.modules, {"psutil": self._psutil(10.0, 10.0)}):
            assert scheduler.get_optimal_workers() == 3

    def test_without_psutil_uses_static_count(self):
        scheduler = ResourceAwareScheduler()
        with patch("kosmos.execution.parallel.multiprocessing.cpu_count", return_value=8), \
                patch.dict(sys.modules, {"psutil": None}):
            assert scheduler.get_optimal_workers() == 7
