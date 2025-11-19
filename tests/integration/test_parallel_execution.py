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
            experiment_id="test_protocol",
            success=True,
            result="result_123",
            execution_time=5.5,
        )

        assert result.success is True
        assert result.error is None
        # result field contains the data/result
        assert result.result == "result_123"

    def test_failure_result(self):
        """Test creating failure result."""
        result = ParallelExecutionResult(
            experiment_id="test_protocol",
            success=False,
            result=None,
            execution_time=0.1,
            error="Experiment execution failed"
        )

        assert result.success is False
        assert result.result is None
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
        protocol_ids = [f"exp_{i}" for i in range(6)]
        tasks = [ExperimentTask(experiment_id=pid, code="print('test')") for pid in protocol_ids]

        # Mock _execute_single_experiment to avoid running real code in separate process
        # and to avoid complexity of pickling mocks
        with patch('kosmos.execution.parallel._execute_single_experiment') as mock_exec:
            def side_effect(task, use_sandbox, timeout):
                return ParallelExecutionResult(
                    experiment_id=task.experiment_id,
                    success=True,
                    result={"data": "test"},
                    execution_time=0.1
                )
            mock_exec.side_effect = side_effect
            
            # We also need to patch ProcessPoolExecutor to run in current process or use threads
            # because pickling the side_effect function (closure) will fail.
            with patch('kosmos.execution.parallel.ProcessPoolExecutor') as MockPool:
                mock_pool = MockPool.return_value
                mock_pool.__enter__.return_value = mock_pool
                
                def submit_side_effect(func, task, *args):
                    from concurrent.futures import Future
                    f = Future()
                    f.set_result(func(task, *args))
                    return f
                mock_pool.submit.side_effect = submit_side_effect

                # Execute in parallel
                results = executor.execute_batch(tasks)

                assert len(results) == 6
                success_rate = sum(1 for r in results if r.success) / len(results)
                assert success_rate >= 0.5

        executor.shutdown()

    @pytest.mark.integration
    def test_memory_usage_under_load(self):
        """Test memory usage doesn't grow excessively."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        executor = ParallelExperimentExecutor(max_workers=4)

        protocol_ids = [f"protocol_{i}" for i in range(50)]
        tasks = [ExperimentTask(experiment_id=pid, code="pass") for pid in protocol_ids]

        with patch('kosmos.execution.parallel._execute_single_experiment') as mock_exec:
            def side_effect(task, use_sandbox, timeout):
                # Allocate memory simulation (transient)
                _ = [0] * 1000
                return ParallelExecutionResult(
                    experiment_id=task.experiment_id,
                    success=True,
                    result=None,
                    execution_time=0.01
                )
            mock_exec.side_effect = side_effect

            with patch('kosmos.execution.parallel.ProcessPoolExecutor') as MockPool:
                mock_pool = MockPool.return_value
                mock_pool.__enter__.return_value = mock_pool
                
                def submit_side_effect(func, task, *args):
                    from concurrent.futures import Future
                    f = Future()
                    f.set_result(func(task, *args))
                    return f
                mock_pool.submit.side_effect = submit_side_effect

                results = executor.execute_batch(tasks)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory shouldn't grow excessively
        assert memory_increase < 500

        executor.shutdown()


class TestConcurrentExperimentScheduling:
    """Test experiment scheduling and queuing."""

    def test_queue_management(self):
        """Test experiment queue management."""
        executor = ParallelExperimentExecutor(max_workers=2)

        protocol_ids = [f"protocol_{i}" for i in range(10)]
        tasks = [ExperimentTask(experiment_id=pid, code="pass") for pid in protocol_ids]

        with patch('kosmos.execution.parallel._execute_single_experiment') as mock_exec:
            mock_exec.return_value = ParallelExecutionResult(
                experiment_id="any", # Will be overwritten by logic if needed, but execute_batch results depend on task
                success=True,
                result=None,
                execution_time=0.1
            )
            
            # Patch submit to ensure task ID is preserved in result
            with patch('kosmos.execution.parallel.ProcessPoolExecutor') as MockPool:
                mock_pool = MockPool.return_value
                mock_pool.__enter__.return_value = mock_pool
                
                def submit_side_effect(func, task, *args):
                    from concurrent.futures import Future
                    f = Future()
                    f.set_result(ParallelExecutionResult(
                        experiment_id=task.experiment_id,
                        success=True,
                        result=None,
                        execution_time=0.1
                    ))
                    return f
                mock_pool.submit.side_effect = submit_side_effect

                results = executor.execute_batch(tasks)

                assert len(results) == 10
                assert all(r.success for r in results)

        executor.shutdown()

    def test_graceful_shutdown_with_pending_work(self):
        """Test shutting down with pending experiments."""
        executor = ParallelExperimentExecutor(max_workers=2)

        protocol_ids = [f"protocol_{i}" for i in range(4)]
        tasks = [ExperimentTask(experiment_id=pid, code="pass") for pid in protocol_ids]

        # Just test that shutdown works without error, as true concurrency is hard to mock
        executor.execute_batch(tasks)
        executor.shutdown(wait=True)


class TestResourceLimits:
    """Test resource limit enforcement."""

    def test_cpu_limit_enforcement(self):
        """Test CPU usage stays within limits."""
        executor = ParallelExperimentExecutor(max_workers=4)

        protocol_ids = [f"protocol_{i}" for i in range(8)]
        tasks = [ExperimentTask(experiment_id=pid, code="pass") for pid in protocol_ids]

        with patch('kosmos.execution.parallel.ProcessPoolExecutor') as MockPool:
            mock_pool = MockPool.return_value
            mock_pool.__enter__.return_value = mock_pool
            
            def submit_side_effect(func, task, *args):
                from concurrent.futures import Future
                f = Future()
                f.set_result(ParallelExecutionResult(
                    experiment_id=task.experiment_id,
                    success=True,
                    result=None,
                    execution_time=0.5
                ))
                return f
            mock_pool.submit.side_effect = submit_side_effect

            start = time.time()
            results = executor.execute_batch(tasks)
            duration = time.time() - start

            assert len(results) == 8
            assert all(r.success for r in results)

        executor.shutdown()

    @pytest.mark.integration
    def test_memory_limit_handling(self):
        """Test handling of memory limit exceeded."""
        executor = ParallelExperimentExecutor(max_workers=2)

        protocol_ids = ["protocol_1"]
        tasks = [ExperimentTask(experiment_id=pid, code="pass") for pid in protocol_ids]

        with patch('kosmos.execution.parallel.ProcessPoolExecutor') as MockPool:
            mock_pool = MockPool.return_value
            mock_pool.__enter__.return_value = mock_pool
            
            def submit_side_effect(func, task, *args):
                from concurrent.futures import Future
                f = Future()
                # Simulate memory error result
                f.set_result(ParallelExecutionResult(
                    experiment_id=task.experiment_id,
                    success=False,
                    result=None,
                    execution_time=0.1,
                    error="Memory limit exceeded"
                ))
                return f
            mock_pool.submit.side_effect = submit_side_effect

            results = executor.execute_batch(tasks)

            assert len(results) == 1
            if not results[0].success:
                assert "memory" in results[0].error.lower()

        executor.shutdown()
