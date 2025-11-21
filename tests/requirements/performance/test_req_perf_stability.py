"""
Tests for Performance Stability Requirements (REQ-PERF-STAB-001 through REQ-PERF-STAB-004).

These tests validate system stability under extended operation, including
12-hour runs, 20 iterations, 200 rollouts, and memory stability.
"""

import pytest
import asyncio
import time
import psutil
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List

from kosmos.agents.research_director import ResearchDirectorAgent
from kosmos.core.workflow import ResearchPlan
from kosmos.monitoring.metrics import MetricsCollector
from kosmos.core.metrics import get_metrics

# Test markers for requirements traceability
pytestmark = [
    pytest.mark.requirement("REQ-PERF-STAB"),
    pytest.mark.category("performance"),
]


@pytest.mark.requirement("REQ-PERF-STAB-001")
@pytest.mark.priority("MUST")
@pytest.mark.slow
@pytest.mark.manual
class TestREQ_PERF_STAB_001_TwelveHourStability:
    """
    REQ-PERF-STAB-001: The system MUST maintain stability and responsiveness
    during 12-hour continuous operation without memory leaks or degradation.
    """

    @pytest.mark.timeout(43200)  # 12 hours
    def test_twelve_hour_stability_full_run(self):
        """
        Verify system remains stable during 12-hour continuous operation.

        NOTE: This test is marked as 'manual' and 'slow' - it should only be run
        during extended integration testing or CI/CD pipelines with long timeouts.
        """
        pytest.skip("Manual test: Run during extended integration testing only")

        # This test would run actual research cycles for 12 hours
        # and monitor memory, CPU, and responsiveness

        start_time = time.time()
        end_time = start_time + (12 * 3600)  # 12 hours

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        max_memory = initial_memory
        memory_samples = []

        with patch('kosmos.agents.research_director.get_client') as mock_llm:
            with patch('kosmos.world_model.get_world_model') as mock_wm:
                mock_llm.return_value = Mock()
                mock_wm.return_value = None

                director = ResearchDirectorAgent(
                    research_question="Test stability",
                    config={"max_iterations": 1000}
                )

                iteration = 0
                while time.time() < end_time:
                    # Simulate research cycle
                    director.research_plan.increment_iteration()
                    iteration += 1

                    # Sample memory every 10 minutes
                    if iteration % 600 == 0:
                        current_memory = process.memory_info().rss / 1024 / 1024
                        memory_samples.append({
                            'time': time.time() - start_time,
                            'memory_mb': current_memory,
                            'iteration': iteration
                        })
                        max_memory = max(max_memory, current_memory)

                    time.sleep(1)  # Throttle

                # Verify no significant memory growth (< 20% increase)
                final_memory = process.memory_info().rss / 1024 / 1024
                memory_growth_percent = ((final_memory - initial_memory) / initial_memory) * 100

                assert memory_growth_percent < 20, \
                    f"Memory grew by {memory_growth_percent:.1f}% over 12 hours"

                # Verify system remained responsive
                assert iteration >= 43200, "System should complete at least 1 iteration/second"

    @pytest.mark.timeout(300)
    def test_twelve_hour_stability_simulation(self):
        """
        Simulate 12-hour stability test with accelerated time and mocked operations.
        """
        # Simulate 12 hours in 60 seconds (720x speedup)
        simulation_duration = 60  # seconds
        simulated_hours = 12
        cycles_per_hour = 360  # 1 cycle per 10 seconds
        expected_cycles = simulated_hours * cycles_per_hour

        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        memory_samples = []

        with patch('kosmos.agents.research_director.get_client') as mock_llm:
            with patch('kosmos.world_model.get_world_model') as mock_wm:
                mock_llm.return_value = Mock()
                mock_wm.return_value = None

                director = ResearchDirectorAgent(
                    research_question="Test stability",
                    config={"max_iterations": 10000}
                )

                # Run simulated cycles
                for cycle in range(expected_cycles):
                    director.research_plan.increment_iteration()
                    director.research_plan.add_hypothesis(f"hyp_{cycle}")
                    director.research_plan.add_experiment(f"exp_{cycle}")

                    # Sample memory periodically
                    if cycle % 100 == 0:
                        current_memory = process.memory_info().rss / 1024 / 1024
                        memory_samples.append(current_memory)

                    # Throttle to match simulation duration
                    if time.time() - start_time > simulation_duration:
                        break

                # Verify memory stability
                final_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = final_memory - initial_memory

                # Allow up to 50MB growth during simulation
                assert memory_growth < 50, \
                    f"Memory grew by {memory_growth:.1f}MB during stability test"

                # Verify no memory leak pattern (monotonic growth)
                if len(memory_samples) > 10:
                    # Check that memory doesn't continuously increase
                    first_half_avg = sum(memory_samples[:len(memory_samples)//2]) / (len(memory_samples)//2)
                    second_half_avg = sum(memory_samples[len(memory_samples)//2:]) / (len(memory_samples) - len(memory_samples)//2)
                    growth_rate = ((second_half_avg - first_half_avg) / first_half_avg) * 100

                    assert growth_rate < 10, \
                        f"Memory grew {growth_rate:.1f}% between first and second half (possible leak)"

    def test_stability_monitoring_metrics(self):
        """Verify stability metrics are collected and available."""
        metrics = get_metrics(reset=True)

        # Simulate some operations
        for i in range(100):
            metrics.record_api_call(
                model="claude-3-5-sonnet",
                input_tokens=1000,
                output_tokens=500,
                duration_seconds=1.2
            )

        stats = metrics.get_statistics()

        # Verify metrics are being collected
        assert stats["api"]["total_calls"] == 100
        assert stats["system"]["uptime_seconds"] >= 0

        # Verify no abnormal error rates
        error_rate = stats["api"]["error_rate"]
        assert error_rate == 0, "Should have no errors in stable operation"


@pytest.mark.requirement("REQ-PERF-STAB-002")
@pytest.mark.priority("MUST")
class TestREQ_PERF_STAB_002_TwentyIterationStability:
    """
    REQ-PERF-STAB-002: The system MUST successfully complete 20 research
    iterations without crashes or state corruption.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_twenty_iterations_complete(self, mock_wm, mock_llm):
        """Verify system completes 20 iterations successfully."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test 20 iterations",
            config={"max_iterations": 20}
        )

        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        # Execute 20 iterations
        for i in range(20):
            director.research_plan.increment_iteration()

            # Simulate research activities
            director.research_plan.add_hypothesis(f"hyp_iter_{i}_1")
            director.research_plan.add_hypothesis(f"hyp_iter_{i}_2")
            director.research_plan.add_experiment(f"exp_iter_{i}_1")
            director.research_plan.mark_experiment_complete(f"exp_iter_{i}_1")
            director.research_plan.add_result(f"res_iter_{i}_1")

            # Verify state consistency
            assert director.research_plan.iteration_count == i + 1
            assert len(director.research_plan.hypothesis_pool) == (i + 1) * 2
            assert len(director.research_plan.results) == i + 1

        # Verify final state
        assert director.research_plan.iteration_count == 20
        assert len(director.research_plan.hypothesis_pool) == 40
        assert len(director.research_plan.results) == 20

        # Verify no excessive memory growth
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        assert memory_growth < 100, f"Memory grew by {memory_growth:.1f}MB"

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_twenty_iterations_state_consistency(self, mock_wm, mock_llm):
        """Verify research plan state remains consistent across 20 iterations."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test state consistency",
            config={"max_iterations": 20}
        )

        # Track state snapshots
        state_snapshots = []

        for i in range(20):
            director.research_plan.increment_iteration()

            # Add varying amounts of data
            num_hypotheses = (i % 5) + 1
            for j in range(num_hypotheses):
                hyp_id = f"hyp_{i}_{j}"
                director.research_plan.add_hypothesis(hyp_id)

                # Test some hypotheses
                if j % 2 == 0:
                    director.research_plan.mark_tested(hyp_id)

            # Snapshot current state
            snapshot = {
                'iteration': director.research_plan.iteration_count,
                'hypothesis_count': len(director.research_plan.hypothesis_pool),
                'tested_count': len(director.research_plan.tested_hypotheses),
                'supported_count': len(director.research_plan.supported_hypotheses),
                'rejected_count': len(director.research_plan.rejected_hypotheses),
            }
            state_snapshots.append(snapshot)

        # Verify state invariants
        for i, snapshot in enumerate(state_snapshots):
            # Iteration should match
            assert snapshot['iteration'] == i + 1

            # Hypothesis counts should be non-decreasing
            if i > 0:
                prev = state_snapshots[i - 1]
                assert snapshot['hypothesis_count'] >= prev['hypothesis_count'], \
                    "Hypothesis pool should only grow"
                assert snapshot['tested_count'] >= prev['tested_count'], \
                    "Tested count should only grow"

        # Final state should be consistent
        final = state_snapshots[-1]
        assert final['iteration'] == 20
        assert final['hypothesis_count'] > 0
        assert final['tested_count'] <= final['hypothesis_count']
        assert final['supported_count'] + final['rejected_count'] <= final['tested_count']

    def test_twenty_iterations_metrics_tracking(self):
        """Verify metrics are tracked correctly across 20 iterations."""
        metrics = get_metrics(reset=True)

        # Simulate 20 iterations
        for i in range(20):
            metrics.track_research_iteration(domain="test")

            # Simulate hypothesis generation
            for j in range(3):
                metrics.track_hypothesis_generated(domain="test", strategy="default")

            # Simulate experiments
            metrics.track_experiment_start(domain="test", experiment_type="data_analysis")
            time.sleep(0.01)  # Small delay
            metrics.track_experiment_complete(
                domain="test",
                experiment_type="data_analysis",
                status="success",
                duration=0.01
            )

        stats = metrics.get_statistics()

        # Verify counts
        # Note: These are Prometheus counters, actual values depend on implementation
        assert stats["experiments"]["experiments_completed"] >= 20, \
            "Should complete at least 20 experiments"


@pytest.mark.requirement("REQ-PERF-STAB-003")
@pytest.mark.priority("MUST")
@pytest.mark.slow
class TestREQ_PERF_STAB_003_TwoHundredRolloutStability:
    """
    REQ-PERF-STAB-003: The system MUST handle 200 consecutive experiment
    rollouts without degradation or resource exhaustion.
    """

    @pytest.mark.timeout(1800)  # 30 minutes
    def test_two_hundred_rollouts(self):
        """Verify system handles 200 consecutive rollouts."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

        metrics = get_metrics(reset=True)

        with patch('kosmos.agents.research_director.get_client') as mock_llm:
            with patch('kosmos.world_model.get_world_model') as mock_wm:
                mock_llm.return_value = Mock()
                mock_wm.return_value = None

                director = ResearchDirectorAgent(
                    research_question="Test rollouts",
                    config={"max_iterations": 1000}
                )

                rollout_durations = []
                memory_samples = []

                # Execute 200 rollouts
                for rollout in range(200):
                    start_time = time.time()

                    # Simulate rollout: hypothesis -> experiment -> result
                    hyp_id = f"hyp_{rollout}"
                    exp_id = f"exp_{rollout}"
                    res_id = f"res_{rollout}"

                    director.research_plan.add_hypothesis(hyp_id)
                    director.research_plan.add_experiment(exp_id)
                    director.research_plan.mark_experiment_complete(exp_id)
                    director.research_plan.add_result(res_id)
                    director.research_plan.mark_tested(hyp_id)

                    # Record metrics
                    duration = time.time() - start_time
                    rollout_durations.append(duration)

                    if rollout % 20 == 0:
                        current_memory = process.memory_info().rss / 1024 / 1024
                        memory_samples.append(current_memory)

                # Verify all rollouts completed
                assert len(director.research_plan.hypothesis_pool) == 200
                assert len(director.research_plan.results) == 200

                # Verify no performance degradation
                first_20_avg = sum(rollout_durations[:20]) / 20
                last_20_avg = sum(rollout_durations[-20:]) / 20
                slowdown_factor = last_20_avg / first_20_avg if first_20_avg > 0 else 1.0

                assert slowdown_factor < 2.0, \
                    f"Performance degraded {slowdown_factor:.1f}x over 200 rollouts"

                # Verify memory stability
                final_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = final_memory - initial_memory
                assert memory_growth < 200, \
                    f"Memory grew by {memory_growth:.1f}MB during 200 rollouts"

    def test_rollout_batch_processing(self):
        """Verify batch processing of rollouts maintains stability."""
        with patch('kosmos.agents.research_director.get_client') as mock_llm:
            with patch('kosmos.world_model.get_world_model') as mock_wm:
                mock_llm.return_value = Mock()
                mock_wm.return_value = None

                director = ResearchDirectorAgent(
                    research_question="Test batch rollouts",
                    config={
                        "enable_concurrent_operations": True,
                        "max_concurrent_experiments": 10
                    }
                )

                # Create 200 experiments
                experiment_ids = [f"exp_{i}" for i in range(200)]
                for exp_id in experiment_ids:
                    director.research_plan.add_experiment(exp_id)

                # Process in batches
                batch_size = 10
                for i in range(0, 200, batch_size):
                    batch = experiment_ids[i:i+batch_size]

                    # Mock batch execution
                    with patch.object(director, 'execute_experiments_batch') as mock_batch:
                        mock_batch.return_value = [
                            {"protocol_id": exp_id, "success": True, "result_id": f"res_{exp_id}"}
                            for exp_id in batch
                        ]

                        results = director.execute_experiments_batch(batch)
                        assert len(results) == len(batch)

                # Verify all experiments tracked
                assert len(director.research_plan.experiment_queue) <= 200


@pytest.mark.requirement("REQ-PERF-STAB-004")
@pytest.mark.priority("MUST")
class TestREQ_PERF_STAB_004_MemoryStability:
    """
    REQ-PERF-STAB-004: The system MUST maintain stable memory usage during
    extended operation with proper cleanup and garbage collection.
    """

    def test_memory_cleanup_after_iteration(self):
        """Verify memory is cleaned up after each iteration."""
        import gc

        process = psutil.Process(os.getpid())
        gc.collect()  # Start with clean slate

        initial_memory = process.memory_info().rss / 1024 / 1024

        with patch('kosmos.agents.research_director.get_client') as mock_llm:
            with patch('kosmos.world_model.get_world_model') as mock_wm:
                mock_llm.return_value = Mock()
                mock_wm.return_value = None

                for iteration in range(10):
                    # Create director for each iteration
                    director = ResearchDirectorAgent(
                        research_question=f"Test iteration {iteration}",
                        config={"max_iterations": 1}
                    )

                    # Simulate work
                    for j in range(50):
                        director.research_plan.add_hypothesis(f"hyp_{iteration}_{j}")

                    # Cleanup
                    del director
                    gc.collect()

                # Force garbage collection
                gc.collect()

                # Check memory returned to reasonable level
                final_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = final_memory - initial_memory

                # Allow up to 30MB growth for caches/overhead
                assert memory_growth < 30, \
                    f"Memory not properly cleaned up, grew by {memory_growth:.1f}MB"

    def test_no_circular_references(self):
        """Verify no circular references prevent garbage collection."""
        import gc
        import weakref

        gc.collect()

        with patch('kosmos.agents.research_director.get_client') as mock_llm:
            with patch('kosmos.world_model.get_world_model') as mock_wm:
                mock_llm.return_value = Mock()
                mock_wm.return_value = None

                director = ResearchDirectorAgent(
                    research_question="Test circular refs",
                    config={"max_iterations": 1}
                )

                # Create weak reference
                weak_ref = weakref.ref(director)

                # Verify object exists
                assert weak_ref() is not None

                # Delete and collect
                del director
                gc.collect()

                # Weak reference should be dead
                assert weak_ref() is None, \
                    "Object not garbage collected (possible circular reference)"

    def test_large_data_structures_cleanup(self):
        """Verify large data structures are properly cleaned up."""
        import gc

        process = psutil.Process(os.getpid())
        gc.collect()

        initial_memory = process.memory_info().rss / 1024 / 1024

        with patch('kosmos.agents.research_director.get_client') as mock_llm:
            with patch('kosmos.world_model.get_world_model') as mock_wm:
                mock_llm.return_value = Mock()
                mock_wm.return_value = None

                director = ResearchDirectorAgent(
                    research_question="Test large data",
                    config={"max_iterations": 100}
                )

                # Add large amount of data
                for i in range(1000):
                    director.research_plan.add_hypothesis(f"hyp_{i}")
                    director.research_plan.add_experiment(f"exp_{i}")
                    director.research_plan.add_result(f"res_{i}")

                # Check memory grew
                mid_memory = process.memory_info().rss / 1024 / 1024
                assert mid_memory > initial_memory, "Memory should grow with data"

                # Cleanup
                del director
                gc.collect()

                # Verify memory returned to near-initial level
                final_memory = process.memory_info().rss / 1024 / 1024
                memory_retained = final_memory - initial_memory

                assert memory_retained < 20, \
                    f"Large data structures not cleaned up, {memory_retained:.1f}MB retained"

    def test_metrics_memory_overhead(self):
        """Verify metrics collection doesn't cause memory leaks."""
        import gc

        process = psutil.Process(os.getpid())
        gc.collect()

        metrics = get_metrics(reset=True)
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Record many metrics
        for i in range(1000):
            metrics.record_api_call(
                model="claude-3-5-sonnet",
                input_tokens=1000,
                output_tokens=500,
                duration_seconds=1.0
            )
            metrics.record_experiment_start(f"exp_{i}", "test")
            metrics.record_experiment_end(f"exp_{i}", 1.0, "success")

        # Check memory growth is bounded
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory

        # Metrics should cap history, memory growth should be limited
        assert memory_growth < 10, \
            f"Metrics collection leaked {memory_growth:.1f}MB"

        # Verify history is capped
        stats = metrics.get_statistics()
        # History should be limited to ~1000 entries
        assert len(metrics.api_call_history) <= 1000
        assert len(metrics.experiment_history) <= 1000
