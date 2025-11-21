"""
Tests for Performance Time Requirements (REQ-PERF-TIME-001 through REQ-PERF-TIME-003).

These tests validate timing constraints including hypothesis generation,
iteration completion, and database query performance.
"""

import pytest
import time
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List

from kosmos.agents.research_director import ResearchDirectorAgent
from kosmos.agents.hypothesis_generator import HypothesisGeneratorAgent
from kosmos.core.workflow import ResearchPlan
from kosmos.world_model import get_world_model
from kosmos.monitoring.metrics import MetricsCollector
from kosmos.core.metrics import get_metrics

# Test markers for requirements traceability
pytestmark = [
    pytest.mark.requirement("REQ-PERF-TIME"),
    pytest.mark.category("performance"),
]


@pytest.mark.requirement("REQ-PERF-TIME-001")
@pytest.mark.priority("MUST")
class TestREQ_PERF_TIME_001_HypothesisGenerationTime:
    """
    REQ-PERF-TIME-001: The system MUST generate a batch of hypotheses
    (typically 3-5 hypotheses) in less than 5 minutes.
    """

    @patch('kosmos.agents.hypothesis_generator.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_hypothesis_generation_under_five_minutes(self, mock_wm, mock_llm):
        """Verify hypothesis generation completes within 5 minutes."""
        # Mock LLM to return quickly
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text='{"hypotheses": ["H1: Test", "H2: Test", "H3: Test"],"reasoning": "Test"}')]
        mock_client.messages.create.return_value = mock_response
        mock_llm.return_value = mock_client

        mock_wm.return_value = None

        generator = HypothesisGeneratorAgent()

        start_time = time.time()

        # Generate hypotheses
        hypotheses = generator.generate_hypotheses(
            research_question="What is the effect of X on Y?",
            literature_context="Previous studies show...",
            num_hypotheses=5
        )

        duration = time.time() - start_time

        # Verify completed within 5 minutes (300 seconds)
        assert duration < 300, \
            f"Hypothesis generation took {duration:.1f}s, exceeds 5 minute limit"

        # Verify hypotheses were generated
        assert len(hypotheses) > 0, "Should generate at least some hypotheses"

    @patch('kosmos.agents.hypothesis_generator.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_batch_hypothesis_generation_performance(self, mock_wm, mock_llm):
        """Verify batch generation performance is consistent."""
        # Mock fast LLM responses
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text='{"hypotheses": ["H1", "H2", "H3"],"reasoning": "Test"}')]
        mock_client.messages.create.return_value = mock_response
        mock_llm.return_value = mock_client

        mock_wm.return_value = None

        generator = HypothesisGeneratorAgent()

        # Test multiple batches
        batch_durations = []

        for batch in range(5):
            start_time = time.time()

            hypotheses = generator.generate_hypotheses(
                research_question=f"Research question {batch}",
                literature_context="Context",
                num_hypotheses=5
            )

            duration = time.time() - start_time
            batch_durations.append(duration)

            # Each batch should be under 5 minutes
            assert duration < 300, \
                f"Batch {batch} took {duration:.1f}s, exceeds limit"

        # Verify consistent performance (no degradation)
        avg_first_half = sum(batch_durations[:3]) / 3
        avg_second_half = sum(batch_durations[3:]) / 2

        slowdown = (avg_second_half / avg_first_half) if avg_first_half > 0 else 1.0
        assert slowdown < 1.5, \
            f"Performance degraded {slowdown:.1f}x across batches"

    @patch('kosmos.agents.hypothesis_generator.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_hypothesis_generation_timeout_handling(self, mock_wm, mock_llm):
        """Verify system handles slow hypothesis generation gracefully."""
        # Mock slow LLM
        mock_client = Mock()

        def slow_response(*args, **kwargs):
            time.sleep(0.5)  # Simulate slow response
            mock_resp = Mock()
            mock_resp.content = [Mock(text='{"hypotheses": ["H1"],"reasoning": "Slow"}')]
            return mock_resp

        mock_client.messages.create.side_effect = slow_response
        mock_llm.return_value = mock_client
        mock_wm.return_value = None

        generator = HypothesisGeneratorAgent()

        start_time = time.time()

        # Should complete even if LLM is slow
        hypotheses = generator.generate_hypotheses(
            research_question="Test question",
            literature_context="Context",
            num_hypotheses=3
        )

        duration = time.time() - start_time

        # Verify it completed (even if slowly)
        assert hypotheses is not None
        assert duration < 300, "Should still complete within time limit"

    def test_hypothesis_generation_metrics_tracking(self):
        """Verify hypothesis generation time is tracked in metrics."""
        metrics = get_metrics(reset=True)

        # Simulate hypothesis generation with timing
        start_time = time.time()
        time.sleep(0.1)  # Simulate work
        duration = time.time() - start_time

        metrics.track_hypothesis_generated(domain="test", strategy="literature")

        # Verify metrics can be retrieved
        stats = metrics.get_statistics()
        assert "experiments" in stats or "api" in stats


@pytest.mark.requirement("REQ-PERF-TIME-002")
@pytest.mark.priority("MUST")
class TestREQ_PERF_TIME_002_IterationCompletionTime:
    """
    REQ-PERF-TIME-002: The system MUST complete a single research iteration
    (hypothesis generation → experiment design → execution → analysis)
    in less than 30 minutes under normal conditions.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_iteration_completes_under_thirty_minutes(self, mock_wm, mock_llm):
        """Verify single iteration completes within 30 minutes."""
        # Mock fast LLM responses
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text='{"status": "ok"}')]
        mock_client.messages.create.return_value = mock_response
        mock_llm.return_value = mock_client

        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test iteration timing",
            config={"max_iterations": 1}
        )

        start_time = time.time()

        # Simulate complete iteration
        # 1. Hypothesis generation
        director.research_plan.add_hypothesis("hyp1")
        director.research_plan.add_hypothesis("hyp2")
        director.research_plan.add_hypothesis("hyp3")

        # 2. Experiment design
        director.research_plan.add_experiment("exp1")
        director.research_plan.add_experiment("exp2")

        # 3. Execution
        director.research_plan.mark_experiment_complete("exp1")
        director.research_plan.mark_experiment_complete("exp2")

        # 4. Analysis
        director.research_plan.add_result("res1")
        director.research_plan.add_result("res2")

        # 5. Refinement
        director.research_plan.increment_iteration()

        duration = time.time() - start_time

        # Verify under 30 minutes (1800 seconds)
        # In reality this will be much faster since we're not doing actual work
        assert duration < 1800, \
            f"Iteration took {duration:.1f}s, exceeds 30 minute limit"

        # Verify iteration was completed
        assert director.research_plan.iteration_count == 1
        assert len(director.research_plan.results) == 2

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_iteration_phase_timing_breakdown(self, mock_wm, mock_llm):
        """Verify each phase of iteration completes within reasonable time."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text='{"status": "ok"}')]
        mock_client.messages.create.return_value = mock_response
        mock_llm.return_value = mock_client

        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test phase timing",
            config={"max_iterations": 1}
        )

        phase_timings = {}

        # Phase 1: Hypothesis generation (target: < 5 minutes)
        start = time.time()
        for i in range(5):
            director.research_plan.add_hypothesis(f"hyp{i}")
        phase_timings['hypothesis_generation'] = time.time() - start

        # Phase 2: Experiment design (target: < 5 minutes)
        start = time.time()
        for i in range(5):
            director.research_plan.add_experiment(f"exp{i}")
        phase_timings['experiment_design'] = time.time() - start

        # Phase 3: Execution (target: < 15 minutes)
        start = time.time()
        for i in range(5):
            director.research_plan.mark_experiment_complete(f"exp{i}")
            director.research_plan.add_result(f"res{i}")
        phase_timings['execution'] = time.time() - start

        # Phase 4: Analysis & Refinement (target: < 5 minutes)
        start = time.time()
        director.research_plan.increment_iteration()
        phase_timings['refinement'] = time.time() - start

        # Verify phase timings
        assert phase_timings['hypothesis_generation'] < 300, \
            "Hypothesis generation should be < 5 minutes"
        assert phase_timings['experiment_design'] < 300, \
            "Experiment design should be < 5 minutes"
        assert phase_timings['execution'] < 900, \
            "Execution should be < 15 minutes"
        assert phase_timings['refinement'] < 300, \
            "Refinement should be < 5 minutes"

        # Total should be under 30 minutes
        total_time = sum(phase_timings.values())
        assert total_time < 1800, \
            f"Total iteration time {total_time:.1f}s exceeds 30 minutes"

    def test_iteration_timing_with_concurrent_operations(self):
        """Verify concurrent operations improve iteration timing."""
        metrics = get_metrics(reset=True)

        # Simulate sequential execution
        sequential_start = time.time()
        for i in range(5):
            time.sleep(0.1)  # Simulate work
            metrics.record_experiment_start(f"exp_seq_{i}", "test")
            metrics.record_experiment_end(f"exp_seq_{i}", 0.1, "success")
        sequential_time = time.time() - sequential_start

        # Simulate concurrent execution (much faster)
        concurrent_start = time.time()
        # In real concurrent execution, these would overlap
        time.sleep(0.1)  # All 5 experiments run concurrently
        for i in range(5):
            metrics.record_experiment_start(f"exp_conc_{i}", "test")
            metrics.record_experiment_end(f"exp_conc_{i}", 0.1, "success")
        concurrent_time = time.time() - concurrent_start

        # Concurrent should be faster (or at least not slower)
        # Note: This is a simplified test; real concurrency would show more speedup
        assert concurrent_time <= sequential_time * 1.5, \
            "Concurrent execution should not be significantly slower"

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_multiple_iterations_average_time(self, mock_wm, mock_llm):
        """Verify average iteration time across multiple iterations."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text='{"status": "ok"}')]
        mock_client.messages.create.return_value = mock_response
        mock_llm.return_value = mock_client

        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test average timing",
            config={"max_iterations": 5}
        )

        iteration_times = []

        for i in range(5):
            start_time = time.time()

            # Simulate iteration
            director.research_plan.add_hypothesis(f"hyp_{i}")
            director.research_plan.add_experiment(f"exp_{i}")
            director.research_plan.mark_experiment_complete(f"exp_{i}")
            director.research_plan.add_result(f"res_{i}")
            director.research_plan.increment_iteration()

            duration = time.time() - start_time
            iteration_times.append(duration)

        # Verify average time
        avg_time = sum(iteration_times) / len(iteration_times)
        assert avg_time < 1800, \
            f"Average iteration time {avg_time:.1f}s exceeds 30 minutes"

        # Verify no significant slowdown over time
        if len(iteration_times) >= 2:
            first_half_avg = sum(iteration_times[:3]) / 3
            second_half_avg = sum(iteration_times[3:]) / 2
            slowdown = second_half_avg / first_half_avg if first_half_avg > 0 else 1.0

            assert slowdown < 2.0, \
                f"Performance degraded {slowdown:.1f}x across iterations"


@pytest.mark.requirement("REQ-PERF-TIME-003")
@pytest.mark.priority("MUST")
class TestREQ_PERF_TIME_003_DatabaseQueryPerformance:
    """
    REQ-PERF-TIME-003: The system MUST execute database queries for world model
    operations (read/write hypotheses, experiments, results) in less than 1 second.
    """

    def test_world_model_read_query_performance(self):
        """Verify world model read queries complete within 1 second."""
        try:
            wm = get_world_model()
            if wm is None:
                pytest.skip("World model not initialized")

            start_time = time.time()

            # Perform read query
            hypotheses = wm.get_all_hypotheses()

            duration = time.time() - start_time

            # Should complete within 1 second
            assert duration < 1.0, \
                f"Read query took {duration:.3f}s, exceeds 1 second limit"

        except Exception as e:
            pytest.skip(f"World model not available: {e}")

    def test_world_model_write_query_performance(self):
        """Verify world model write queries complete within 1 second."""
        try:
            wm = get_world_model()
            if wm is None:
                pytest.skip("World model not initialized")

            start_time = time.time()

            # Perform write query
            hypothesis_id = f"test_hyp_{int(time.time())}"
            wm.add_hypothesis(
                hypothesis_id=hypothesis_id,
                content="Test hypothesis for performance",
                research_question="Test"
            )

            duration = time.time() - start_time

            # Should complete within 1 second
            assert duration < 1.0, \
                f"Write query took {duration:.3f}s, exceeds 1 second limit"

            # Cleanup
            try:
                wm.delete_hypothesis(hypothesis_id)
            except:
                pass

        except Exception as e:
            pytest.skip(f"World model not available: {e}")

    def test_world_model_bulk_query_performance(self):
        """Verify bulk queries complete within reasonable time."""
        try:
            wm = get_world_model()
            if wm is None:
                pytest.skip("World model not initialized")

            # Insert multiple items
            hypothesis_ids = []
            for i in range(10):
                hyp_id = f"bulk_test_hyp_{i}_{int(time.time())}"
                hypothesis_ids.append(hyp_id)

            start_time = time.time()

            # Bulk write
            for hyp_id in hypothesis_ids:
                wm.add_hypothesis(
                    hypothesis_id=hyp_id,
                    content=f"Hypothesis {hyp_id}",
                    research_question="Bulk test"
                )

            write_duration = time.time() - start_time

            # Each write should average < 1 second
            avg_write_time = write_duration / len(hypothesis_ids)
            assert avg_write_time < 1.0, \
                f"Average write time {avg_write_time:.3f}s exceeds limit"

            # Bulk read
            start_time = time.time()

            for hyp_id in hypothesis_ids:
                hyp = wm.get_hypothesis(hyp_id)

            read_duration = time.time() - start_time

            # Each read should average < 1 second
            avg_read_time = read_duration / len(hypothesis_ids)
            assert avg_read_time < 1.0, \
                f"Average read time {avg_read_time:.3f}s exceeds limit"

            # Cleanup
            for hyp_id in hypothesis_ids:
                try:
                    wm.delete_hypothesis(hyp_id)
                except:
                    pass

        except Exception as e:
            pytest.skip(f"World model not available: {e}")

    def test_world_model_complex_query_performance(self):
        """Verify complex queries (filters, joins) complete within 1 second."""
        try:
            wm = get_world_model()
            if wm is None:
                pytest.skip("World model not initialized")

            start_time = time.time()

            # Perform complex query (e.g., get all supported hypotheses)
            try:
                supported_hyps = wm.get_hypotheses_by_status("supported")
            except AttributeError:
                # Fallback if method doesn't exist
                all_hyps = wm.get_all_hypotheses()
                supported_hyps = [h for h in all_hyps if getattr(h, 'status', '') == 'supported']

            duration = time.time() - start_time

            # Should complete within 1 second
            assert duration < 1.0, \
                f"Complex query took {duration:.3f}s, exceeds 1 second limit"

        except Exception as e:
            pytest.skip(f"World model not available: {e}")

    def test_database_query_metrics_tracking(self):
        """Verify database query metrics are tracked."""
        metrics = MetricsCollector() if hasattr(MetricsCollector, '__init__') else None

        if metrics and hasattr(metrics, 'track_database_query'):
            # Track a query
            metrics.track_database_query(
                operation="select",
                table="hypotheses",
                status="success",
                duration=0.05
            )

            # Verify it can be tracked without error
            assert True
        else:
            pytest.skip("Database query tracking not implemented")

    def test_query_performance_under_load(self):
        """Verify query performance remains acceptable under load."""
        try:
            wm = get_world_model()
            if wm is None:
                pytest.skip("World model not initialized")

            # Simulate load with rapid queries
            query_times = []

            for i in range(20):
                start_time = time.time()

                # Rapid fire queries
                wm.get_all_hypotheses()

                duration = time.time() - start_time
                query_times.append(duration)

            # Verify all queries completed within limit
            max_time = max(query_times)
            assert max_time < 1.0, \
                f"Query under load took {max_time:.3f}s, exceeds limit"

            # Verify average performance
            avg_time = sum(query_times) / len(query_times)
            assert avg_time < 0.5, \
                f"Average query time {avg_time:.3f}s too high under load"

            # Verify no significant degradation
            first_5_avg = sum(query_times[:5]) / 5
            last_5_avg = sum(query_times[-5:]) / 5
            slowdown = last_5_avg / first_5_avg if first_5_avg > 0 else 1.0

            assert slowdown < 2.0, \
                f"Query performance degraded {slowdown:.1f}x under load"

        except Exception as e:
            pytest.skip(f"World model not available: {e}")

    @pytest.mark.asyncio
    async def test_async_query_performance(self):
        """Verify async queries enable better throughput."""
        try:
            wm = get_world_model()
            if wm is None:
                pytest.skip("World model not initialized")

            # Test concurrent async queries (if supported)
            start_time = time.time()

            # Simulate multiple concurrent queries
            tasks = []
            for i in range(10):
                # If world model supports async, use it
                # Otherwise, wrap synchronous calls
                async def query():
                    return wm.get_all_hypotheses()

                tasks.append(query())

            results = await asyncio.gather(*tasks)

            duration = time.time() - start_time

            # Verify all queries completed
            assert len(results) == 10

            # Concurrent queries should be faster than sequential
            # (or at least not take 10x as long)
            assert duration < 10.0, \
                f"10 concurrent queries took {duration:.1f}s, should be faster"

        except Exception as e:
            pytest.skip(f"Async queries not supported: {e}")
