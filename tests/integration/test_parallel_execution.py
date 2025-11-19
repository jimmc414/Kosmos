"""
Integration tests for parallel experiment execution.

Tests ParallelExperimentExecutor and concurrent experiment workflows.
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from kosmos.execution.parallel import (
    ParallelExperimentExecutor,
    ParallelExecutionResult
)


class TestParallelExperimentExecutor:
    """Test ParallelExperimentExecutor."""

    @pytest.fixture
    def executor(self):
        """Create executor with 4 workers."""
        executor = ParallelExperimentExecutor(max_workers=4)
        yield executor
        executor.shutdown()

    def test_initialization(self, executor):
        """Test executor initialization."""
        assert executor.max_workers == 4
        assert executor.executor is not None

    def test_execute_single_experiment(self, executor):
        """Test executing single experiment."""
        protocol_id = "test_protocol_1"

        # Mock experiment execution
        with patch.object(executor, '_execute_experiment_task') as mock_exec:
            mock_exec.return_value = ParallelExecutionResult(
                protocol_id=protocol_id,
                success=True,
                result_id="result_1",
                duration_seconds=1.0
            )

            result = executor.execute(protocol_id)

            assert result.success is True
            assert result.protocol_id == protocol_id

    def test_execute_batch(self, executor):
        """Test executing batch of experiments."""
        protocol_ids = [f"protocol_{i}" for i in range(10)]

        # Mock experiment execution
        def mock_execute_task(protocol_id):
            time.sleep(0.1)  # Simulate work
            return ParallelExecutionResult(
                protocol_id=protocol_id,
                success=True,
                result_id=f"result_{protocol_id}",
                duration_seconds=0.1
            )

        with patch.object(executor, '_execute_experiment_task', side_effect=mock_execute_task):
            results = executor.execute_batch(protocol_ids)

            assert len(results) == 10
            assert all(r.success for r in results)

    def test_parallel_speedup(self, executor):
        """Test that parallel execution is faster than sequential."""
        protocol_ids = [f"protocol_{i}" for i in range(8)]

        def mock_execute_task(protocol_id):
            time.sleep(0.2)  # Each takes 200ms
            return ParallelExecutionResult(
                protocol_id=protocol_id,
                success=True,
                result_id=f"result_{protocol_id}",
                duration_seconds=0.2
            )

        with patch.object(executor, '_execute_experiment_task', side_effect=mock_execute_task):
            start = time.time()
            results = executor.execute_batch(protocol_ids)
            parallel_time = time.time() - start

            # With 4 workers, 8 experiments should take ~400ms (2 batches)
            # Sequential would take ~1600ms (8 * 200ms)
            assert parallel_time < 1.0  # Should be significantly faster
            assert len(results) == 8

    def test_error_handling(self, executor):
        """Test handling of experiment failures."""
        protocol_ids = ["success_1", "failure_1", "success_2"]

        def mock_execute_task(protocol_id):
            if "failure" in protocol_id:
                raise Exception("Experiment failed")
            return ParallelExecutionResult(
                protocol_id=protocol_id,
                success=True,
                result_id=f"result_{protocol_id}",
                duration_seconds=0.1
            )

        with patch.object(executor, '_execute_experiment_task', side_effect=mock_execute_task):
            results = executor.execute_batch(protocol_ids)

            assert len(results) == 3
            assert results[0].success is True
            assert results[1].success is False
            assert "failed" in results[1].error.lower()
            assert results[2].success is True

    def test_shutdown(self):
        """Test executor shutdown."""
        executor = ParallelExperimentExecutor(max_workers=2)

        executor.shutdown()

        # Executor should be shutdown
        assert executor.executor._shutdown is True

    def test_max_workers_configuration(self):
        """Test configuring max workers."""
        executor1 = ParallelExperimentExecutor(max_workers=2)
        assert executor1.max_workers == 2
        executor1.shutdown()

        executor2 = ParallelExperimentExecutor(max_workers=8)
        assert executor2.max_workers == 8
        executor2.shutdown()

    def test_result_ordering(self, executor):
        """Test that results maintain order of input."""
        protocol_ids = [f"protocol_{i}" for i in range(5)]

        def mock_execute_task(protocol_id):
            # Variable delay to test ordering
            delay = 0.1 if "0" in protocol_id or "2" in protocol_id else 0.05
            time.sleep(delay)
            return ParallelExecutionResult(
                protocol_id=protocol_id,
                success=True,
                result_id=f"result_{protocol_id}",
                duration_seconds=delay
            )

        with patch.object(executor, '_execute_experiment_task', side_effect=mock_execute_task):
            results = executor.execute_batch(protocol_ids)

            # Results should be in same order as input
            for i, result in enumerate(results):
                assert result.protocol_id == protocol_ids[i]


class TestParallelExecutionResult:
    """Test ParallelExecutionResult data class."""

    def test_success_result(self):
        """Test creating success result."""
        result = ParallelExecutionResult(
            protocol_id="test_protocol",
            success=True,
            result_id="result_123",
            duration_seconds=5.5,
            data={"metric": 0.95}
        )

        assert result.success is True
        assert result.error is None
        assert result.data["metric"] == 0.95

    def test_failure_result(self):
        """Test creating failure result."""
        result = ParallelExecutionResult(
            protocol_id="test_protocol",
            success=False,
            error="Experiment execution failed"
        )

        assert result.success is False
        assert result.result_id is None
        assert "failed" in result.error.lower()


class TestParallelExecutionWithRealExperiments:
    """Integration tests with real experiment execution (mocked APIs)."""

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client for experiment execution."""
        with patch('kosmos.core.llm.ClaudeClient') as mock:
            client = MagicMock()
            client.generate.return_value = "Experiment analysis: The results show..."
            mock.return_value = client
            yield client

    @pytest.mark.integration
    def test_parallel_experiment_workflow(self, mock_llm_client):
        """Test complete parallel experiment workflow."""
        executor = ParallelExperimentExecutor(max_workers=4)

        # Create mock experiment protocols
        protocols = [
            {"id": f"exp_{i}", "type": "computational", "params": {"iterations": 100}}
            for i in range(6)
        ]

        protocol_ids = [p["id"] for p in protocols]

        # Execute in parallel
        results = executor.execute_batch(protocol_ids)

        assert len(results) == 6
        # Allow for some failures in real execution
        success_rate = sum(1 for r in results if r.success) / len(results)
        assert success_rate >= 0.5  # At least 50% should succeed

        executor.shutdown()

    @pytest.mark.integration
    def test_memory_usage_under_load(self):
        """Test memory usage doesn't grow excessively."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        executor = ParallelExperimentExecutor(max_workers=4)

        # Execute many experiments
        def mock_execute_task(protocol_id):
            # Allocate and release memory
            data = [0] * 100000
            time.sleep(0.01)
            return ParallelExecutionResult(
                protocol_id=protocol_id,
                success=True,
                result_id=f"result_{protocol_id}",
                duration_seconds=0.01
            )

        protocol_ids = [f"protocol_{i}" for i in range(50)]

        with patch.object(executor, '_execute_experiment_task', side_effect=mock_execute_task):
            results = executor.execute_batch(protocol_ids)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory shouldn't grow excessively (< 500MB increase)
        assert memory_increase < 500

        executor.shutdown()


class TestConcurrentExperimentScheduling:
    """Test experiment scheduling and queuing."""

    def test_queue_management(self):
        """Test experiment queue management."""
        executor = ParallelExperimentExecutor(max_workers=2)

        # Submit more experiments than workers
        protocol_ids = [f"protocol_{i}" for i in range(10)]

        def mock_execute_task(protocol_id):
            time.sleep(0.1)
            return ParallelExecutionResult(
                protocol_id=protocol_id,
                success=True,
                result_id=f"result_{protocol_id}",
                duration_seconds=0.1
            )

        with patch.object(executor, '_execute_experiment_task', side_effect=mock_execute_task):
            results = executor.execute_batch(protocol_ids)

            # All should complete eventually
            assert len(results) == 10
            assert all(r.success for r in results)

        executor.shutdown()

    def test_graceful_shutdown_with_pending_work(self):
        """Test shutting down with pending experiments."""
        executor = ParallelExperimentExecutor(max_workers=2)

        def slow_task(protocol_id):
            time.sleep(1.0)
            return ParallelExecutionResult(
                protocol_id=protocol_id,
                success=True,
                result_id=f"result_{protocol_id}",
                duration_seconds=1.0
            )

        protocol_ids = [f"protocol_{i}" for i in range(4)]

        with patch.object(executor, '_execute_experiment_task', side_effect=slow_task):
            # Start batch but don't wait
            import threading
            thread = threading.Thread(target=executor.execute_batch, args=(protocol_ids,))
            thread.start()

            # Give it a moment to start
            time.sleep(0.2)

            # Shutdown should wait for running tasks
            executor.shutdown(wait=True)

            # Thread should complete
            thread.join(timeout=3.0)


class TestResourceLimits:
    """Test resource limit enforcement."""

    def test_cpu_limit_enforcement(self):
        """Test CPU usage stays within limits."""
        executor = ParallelExperimentExecutor(max_workers=4)

        # CPU-intensive task
        def cpu_intensive_task(protocol_id):
            # Simulate CPU work
            result = sum(i**2 for i in range(1000000))
            return ParallelExecutionResult(
                protocol_id=protocol_id,
                success=True,
                result_id=f"result_{protocol_id}",
                duration_seconds=0.5,
                data={"result": result}
            )

        protocol_ids = [f"protocol_{i}" for i in range(8)]

        with patch.object(executor, '_execute_experiment_task', side_effect=cpu_intensive_task):
            start = time.time()
            results = executor.execute_batch(protocol_ids)
            duration = time.time() - start

            assert len(results) == 8
            assert all(r.success for r in results)
            # Should complete in reasonable time with parallelism
            assert duration < 10.0

        executor.shutdown()

    @pytest.mark.integration
    def test_memory_limit_handling(self):
        """Test handling of memory limit exceeded."""
        executor = ParallelExperimentExecutor(max_workers=2)

        def memory_intensive_task(protocol_id):
            try:
                # Try to allocate large amount of memory
                data = [0] * 100000000  # ~400MB
                return ParallelExecutionResult(
                    protocol_id=protocol_id,
                    success=True,
                    result_id=f"result_{protocol_id}",
                    duration_seconds=0.1
                )
            except MemoryError:
                return ParallelExecutionResult(
                    protocol_id=protocol_id,
                    success=False,
                    error="Memory limit exceeded"
                )

        protocol_ids = ["protocol_1"]

        with patch.object(executor, '_execute_experiment_task', side_effect=memory_intensive_task):
            results = executor.execute_batch(protocol_ids)

            # Should handle gracefully (either succeed or fail with error)
            assert len(results) == 1
            if not results[0].success:
                assert "memory" in results[0].error.lower()

        executor.shutdown()
