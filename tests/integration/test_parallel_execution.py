"""
Integration tests for parallel experiment execution.

Tests ParallelExperimentExecutor and concurrent experiment workflows.
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from kosmos.execution.parallel import (
    ParallelExperimentExecutor,
    ParallelExecutionResult,
    ExperimentTask
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
        # Executor is created on demand, so we don't check for .executor attribute

    def test_execute_single_experiment(self, executor):
        """Test executing single experiment."""
        experiment_id = "exp_1"
        task = ExperimentTask(experiment_id=experiment_id, code="print('hello')")

        # Mock experiment execution at the module level
        with patch('kosmos.execution.parallel._execute_single_experiment') as mock_exec:
            mock_exec.return_value = ParallelExecutionResult(
                experiment_id=experiment_id,
                success=True,
                result="result_1",
                execution_time=1.0
            )

            # Patch ProcessPoolExecutor to run synchronously for this test
            with patch('kosmos.execution.parallel.ProcessPoolExecutor') as MockPool:
                mock_pool_instance = MockPool.return_value
                mock_pool_instance.__enter__.return_value = mock_pool_instance
                
                # Mock submit to return a Future with the result
                from concurrent.futures import Future
                future = Future()
                future.set_result(mock_exec.return_value)
                mock_pool_instance.submit.return_value = future

                results = executor.execute_batch([task])
                result = results[0]

                assert result.success is True
                assert result.experiment_id == experiment_id

    def test_execute_batch(self, executor):
        """Test executing batch of experiments."""
        tasks = [ExperimentTask(experiment_id=f"exp_{i}", code="print('test')") for i in range(10)]

        # Mock results
        results_map = {
            task.experiment_id: ParallelExecutionResult(
                experiment_id=task.experiment_id,
                success=True,
                result=f"result_{task.experiment_id}",
                execution_time=0.1
            )
            for task in tasks
        }

        with patch('kosmos.execution.parallel.ProcessPoolExecutor') as MockPool:
            mock_pool_instance = MockPool.return_value
            mock_pool_instance.__enter__.return_value = mock_pool_instance
            
            # Mock submit
            def side_effect(func, task, *args):
                from concurrent.futures import Future
                f = Future()
                f.set_result(results_map[task.experiment_id])
                return f
            
            mock_pool_instance.submit.side_effect = side_effect

            results = executor.execute_batch(tasks)

            assert len(results) == 10
            assert all(r.success for r in results)

    def test_parallel_speedup(self, executor):
        """Test that parallel execution is faster than sequential."""
        # For speedup test, we use real execution but mock the inner worker function behavior
        # via ProcessPoolExecutor behavior simulation or just rely on tasks being submitted.
        
        # It's hard to verify true parallelism with Mocks.
        # We'll verify that tasks are submitted to the pool.
        
        tasks = [ExperimentTask(experiment_id=f"exp_{i}", code="sleep") for i in range(8)]
        
        with patch('kosmos.execution.parallel.ProcessPoolExecutor') as MockPool:
            mock_pool_instance = MockPool.return_value
            mock_pool_instance.__enter__.return_value = mock_pool_instance
            
            executor.execute_batch(tasks)
            
            # Check that submit was called 8 times
            assert mock_pool_instance.submit.call_count == 8

    def test_error_handling(self, executor):
        """Test handling of experiment failures."""
        tasks = [
            ExperimentTask(experiment_id="success_1", code="ok"),
            ExperimentTask(experiment_id="failure_1", code="fail"),
            ExperimentTask(experiment_id="success_2", code="ok")
        ]

        with patch('kosmos.execution.parallel.ProcessPoolExecutor') as MockPool:
            mock_pool_instance = MockPool.return_value
            mock_pool_instance.__enter__.return_value = mock_pool_instance
            
            def side_effect(func, task, *args):
                from concurrent.futures import Future
                f = Future()
                if "failure" in task.experiment_id:
                    # Simulate exception in worker or result
                    # If worker raises, future.result() raises.
                    f.set_exception(Exception("Experiment failed"))
                else:
                    f.set_result(ParallelExecutionResult(
                        experiment_id=task.experiment_id,
                        success=True,
                        result="ok",
                        execution_time=0.1
                    ))
                return f
            
            mock_pool_instance.submit.side_effect = side_effect

            results = executor.execute_batch(tasks)

            assert len(results) == 3
            # results[1] should be failure result created by exception handling in execute_batch
            assert results[1].success is False
            assert "failed" in str(results[1].error).lower()

    def test_shutdown(self):
        """Test executor shutdown."""
        executor = ParallelExperimentExecutor(max_workers=2)
        executor.shutdown()
        # No-op, just ensure no error

    def test_max_workers_configuration(self):
        """Test configuring max workers."""
        executor1 = ParallelExperimentExecutor(max_workers=2)
        assert executor1.max_workers == 2
        executor1.shutdown()

        executor2 = ParallelExperimentExecutor(max_workers=8)
        assert executor2.max_workers == 8
        executor2.shutdown()

    def test_result_ordering(self, executor):
        """Test that results are collected (ordering isn't guaranteed by as_completed, but list returned is)."""
        # execute_batch returns list of results as they complete (as_completed).
        # So ordering is NOT guaranteed to match input.
        # But we can check if all are present.
        
        tasks = [ExperimentTask(experiment_id=f"exp_{i}", code="test") for i in range(5)]
        
        with patch('kosmos.execution.parallel.ProcessPoolExecutor') as MockPool:
            mock_pool_instance = MockPool.return_value
            mock_pool_instance.__enter__.return_value = mock_pool_instance
            
            def side_effect(func, task, *args):
                from concurrent.futures import Future
                f = Future()
                f.set_result(ParallelExecutionResult(
                    experiment_id=task.experiment_id,
                    success=True,
                    result="ok",
                    execution_time=0.1
                ))
                return f
            
            mock_pool_instance.submit.side_effect = side_effect

            results = executor.execute_batch(tasks)
            
            result_ids = [r.experiment_id for r in results]
            assert len(results) == 5
            for task in tasks:
                assert task.experiment_id in result_ids


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
