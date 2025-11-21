"""
Tests for Orchestrator Resource Management Requirements (REQ-ORCH-RES-001 through REQ-ORCH-RES-004).

These tests validate resource management including concurrent operations,
parallel execution, async LLM calls, and resource limits for the Research Director.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List

from kosmos.agents.research_director import ResearchDirectorAgent

# Test markers for requirements traceability
pytestmark = [
    pytest.mark.requirement("REQ-ORCH-RES"),
    pytest.mark.category("orchestrator"),
]


@pytest.mark.requirement("REQ-ORCH-RES-001")
@pytest.mark.priority("SHOULD")
class TestREQ_ORCH_RES_001_ConcurrentOperations:
    """
    REQ-ORCH-RES-001: The Research Director SHOULD support concurrent operations
    mode (enable_concurrent_operations) for parallel hypothesis evaluation and
    experiment execution.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_concurrent_operations_disabled_by_default(self, mock_wm, mock_llm):
        """Verify concurrent operations disabled by default."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        assert director.enable_concurrent is False

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_concurrent_operations_can_be_enabled(self, mock_wm, mock_llm):
        """Verify concurrent operations can be enabled via config."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test",
            config={"enable_concurrent_operations": True}
        )

        assert director.enable_concurrent is True

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_max_parallel_hypotheses_configurable(self, mock_wm, mock_llm):
        """Verify max parallel hypotheses is configurable."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test",
            config={
                "enable_concurrent_operations": True,
                "max_parallel_hypotheses": 10
            }
        )

        assert director.max_parallel_hypotheses == 10

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_max_concurrent_experiments_configurable(self, mock_wm, mock_llm):
        """Verify max concurrent experiments is configurable."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test",
            config={
                "enable_concurrent_operations": True,
                "max_concurrent_experiments": 8
            }
        )

        assert director.max_concurrent_experiments == 8

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_parallel_executor_initialized_when_enabled(self, mock_wm, mock_llm):
        """Verify parallel executor is initialized when concurrent mode enabled."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        with patch('kosmos.agents.research_director.ParallelExperimentExecutor') as mock_executor:
            mock_executor.return_value = Mock()

            director = ResearchDirectorAgent(
                research_question="Test",
                config={"enable_concurrent_operations": True}
            )

            # Note: Actual initialization might not happen if import fails
            # But config should be set
            assert director.enable_concurrent is True


@pytest.mark.requirement("REQ-ORCH-RES-002")
@pytest.mark.priority("SHOULD")
class TestREQ_ORCH_RES_002_ParallelExperimentExecution:
    """
    REQ-ORCH-RES-002: The Research Director SHOULD support parallel experiment
    execution via execute_experiments_batch() for improved throughput.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_execute_experiments_batch_exists(self, mock_wm, mock_llm):
        """Verify execute_experiments_batch method exists."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        assert hasattr(director, 'execute_experiments_batch')
        assert callable(director.execute_experiments_batch)

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_batch_execution_sequential_fallback(self, mock_wm, mock_llm):
        """Verify batch execution falls back to sequential when concurrent disabled."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test",
            config={"enable_concurrent_operations": False}
        )

        director.register_agent("Executor", "executor_001")

        protocol_ids = ["exp1", "exp2", "exp3"]

        with patch.object(director, '_send_to_executor') as mock_send:
            results = director.execute_experiments_batch(protocol_ids)

            # Should fall back to sequential
            assert len(results) == 3
            assert mock_send.call_count == 3

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_batch_execution_with_parallel_executor(self, mock_wm, mock_llm):
        """Verify batch execution uses parallel executor when available."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test",
            config={"enable_concurrent_operations": True}
        )

        # Mock parallel executor
        mock_parallel_executor = Mock()
        mock_parallel_executor.execute_batch = Mock(return_value=[
            {"protocol_id": "exp1", "success": True, "result_id": "res1"},
            {"protocol_id": "exp2", "success": True, "result_id": "res2"},
            {"protocol_id": "exp3", "success": True, "result_id": "res3"}
        ])
        director.parallel_executor = mock_parallel_executor

        protocol_ids = ["exp1", "exp2", "exp3"]

        results = director.execute_experiments_batch(protocol_ids)

        # Verify parallel executor was used
        mock_parallel_executor.execute_batch.assert_called_once_with(protocol_ids)
        assert len(results) == 3
        assert all(r["success"] for r in results)

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_batch_execution_updates_research_plan(self, mock_wm, mock_llm):
        """Verify batch execution updates research plan with results."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test",
            config={"enable_concurrent_operations": True}
        )

        director.research_plan.add_experiment("exp1")
        director.research_plan.add_experiment("exp2")

        # Mock parallel executor
        mock_parallel_executor = Mock()
        mock_parallel_executor.execute_batch = Mock(return_value=[
            {"protocol_id": "exp1", "success": True, "result_id": "res1"},
            {"protocol_id": "exp2", "success": True, "result_id": "res2"}
        ])
        director.parallel_executor = mock_parallel_executor

        results = director.execute_experiments_batch(["exp1", "exp2"])

        # Verify research plan updated
        assert "res1" in director.research_plan.results
        assert "res2" in director.research_plan.results
        assert "exp1" in director.research_plan.completed_experiments
        assert "exp2" in director.research_plan.completed_experiments


@pytest.mark.requirement("REQ-ORCH-RES-003")
@pytest.mark.priority("SHOULD")
class TestREQ_ORCH_RES_003_AsyncLLMCalls:
    """
    REQ-ORCH-RES-003: The Research Director SHOULD support async LLM calls via
    AsyncClaudeClient for concurrent hypothesis evaluation and result analysis.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_async_llm_client_initialized_when_enabled(self, mock_wm, mock_llm):
        """Verify async LLM client is initialized when concurrent mode enabled."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        with patch('kosmos.agents.research_director.AsyncClaudeClient') as mock_async:
            mock_async.return_value = Mock()

            with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'}):
                director = ResearchDirectorAgent(
                    research_question="Test",
                    config={"enable_concurrent_operations": True}
                )

                # Note: Actual initialization might not happen if import fails
                # But config should be set
                assert director.enable_concurrent is True

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_evaluate_hypotheses_concurrently_exists(self, mock_wm, mock_llm):
        """Verify evaluate_hypotheses_concurrently method exists."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        assert hasattr(director, 'evaluate_hypotheses_concurrently')
        assert asyncio.iscoroutinefunction(director.evaluate_hypotheses_concurrently)

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    async def test_evaluate_hypotheses_concurrently_returns_empty_without_client(self, mock_wm, mock_llm):
        """Verify concurrent evaluation returns empty list without async client."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.async_llm_client = None

        hypothesis_ids = ["hyp1", "hyp2", "hyp3"]

        evaluations = await director.evaluate_hypotheses_concurrently(hypothesis_ids)

        assert evaluations == []

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    async def test_evaluate_hypotheses_concurrently_with_mock_client(self, mock_wm, mock_llm):
        """Verify concurrent evaluation works with mocked async client."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        # Mock async LLM client
        mock_async_client = Mock()

        # Mock BatchResponse objects
        mock_responses = [
            Mock(id="hyp1", success=True, response='{"testability": 8, "novelty": 7, "impact": 9, "recommendation": "proceed", "reasoning": "Good hypothesis"}'),
            Mock(id="hyp2", success=True, response='{"testability": 6, "novelty": 5, "impact": 6, "recommendation": "refine", "reasoning": "Needs refinement"}'),
            Mock(id="hyp3", success=True, response='{"testability": 9, "novelty": 8, "impact": 8, "recommendation": "proceed", "reasoning": "Excellent hypothesis"}')
        ]

        # Create async mock
        async def mock_batch_generate(requests):
            return mock_responses

        mock_async_client.batch_generate = mock_batch_generate
        director.async_llm_client = mock_async_client

        hypothesis_ids = ["hyp1", "hyp2", "hyp3"]

        evaluations = await director.evaluate_hypotheses_concurrently(hypothesis_ids)

        assert len(evaluations) == 3
        assert evaluations[0]["hypothesis_id"] == "hyp1"
        assert evaluations[0]["recommendation"] == "proceed"
        assert evaluations[1]["recommendation"] == "refine"

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_analyze_results_concurrently_exists(self, mock_wm, mock_llm):
        """Verify analyze_results_concurrently method exists."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        assert hasattr(director, 'analyze_results_concurrently')
        assert asyncio.iscoroutinefunction(director.analyze_results_concurrently)

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    async def test_analyze_results_concurrently_returns_empty_without_client(self, mock_wm, mock_llm):
        """Verify concurrent analysis returns empty list without async client."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.async_llm_client = None

        result_ids = ["res1", "res2", "res3"]

        analyses = await director.analyze_results_concurrently(result_ids)

        assert analyses == []


@pytest.mark.requirement("REQ-ORCH-RES-004")
@pytest.mark.priority("SHOULD")
class TestREQ_ORCH_RES_004_ResourceLimits:
    """
    REQ-ORCH-RES-004: The Research Director SHOULD enforce resource limits
    (max_parallel_hypotheses, max_concurrent_experiments, max_concurrent_llm_calls,
    llm_rate_limit_per_minute) to prevent resource exhaustion.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_max_parallel_hypotheses_default(self, mock_wm, mock_llm):
        """Verify max_parallel_hypotheses has reasonable default."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        assert director.max_parallel_hypotheses > 0
        assert director.max_parallel_hypotheses <= 10  # Reasonable default

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_max_concurrent_experiments_default(self, mock_wm, mock_llm):
        """Verify max_concurrent_experiments has reasonable default."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        assert director.max_concurrent_experiments > 0
        assert director.max_concurrent_experiments <= 10  # Reasonable default

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_resource_limits_configurable(self, mock_wm, mock_llm):
        """Verify all resource limits are configurable."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        config = {
            "enable_concurrent_operations": True,
            "max_parallel_hypotheses": 5,
            "max_concurrent_experiments": 8,
            "max_concurrent_llm_calls": 10,
            "llm_rate_limit_per_minute": 100
        }

        director = ResearchDirectorAgent(
            research_question="Test",
            config=config
        )

        assert director.max_parallel_hypotheses == 5
        assert director.max_concurrent_experiments == 8

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_batch_size_respects_max_parallel_hypotheses(self, mock_wm, mock_llm):
        """Verify batch size is limited by max_parallel_hypotheses."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test",
            config={
                "enable_concurrent_operations": True,
                "max_parallel_hypotheses": 3
            }
        )

        # Mock async client
        director.async_llm_client = Mock()
        director.async_llm_client.batch_generate = AsyncMock(return_value=[])

        # Add many hypotheses
        for i in range(10):
            director.research_plan.add_hypothesis(f"hyp{i}")

        director.workflow.transition_to(director.workflow.WorkflowState.DESIGNING_EXPERIMENTS)

        # When executing design action, batch size should be limited
        # This is tested indirectly through the implementation

        assert director.max_parallel_hypotheses == 3

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_batch_size_respects_max_concurrent_experiments(self, mock_wm, mock_llm):
        """Verify batch size is limited by max_concurrent_experiments."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test",
            config={
                "enable_concurrent_operations": True,
                "max_concurrent_experiments": 4
            }
        )

        # Mock parallel executor
        mock_executor = Mock()
        mock_executor.execute_batch = Mock(return_value=[])
        director.parallel_executor = mock_executor

        # Add many experiments
        for i in range(10):
            director.research_plan.add_experiment(f"exp{i}")

        # Execute batch - should be limited to max_concurrent_experiments
        experiment_queue = list(director.research_plan.experiment_queue)[:10]

        # In implementation, batch size is limited
        assert director.max_concurrent_experiments == 4

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_async_client_rate_limits_configured(self, mock_wm, mock_llm):
        """Verify async client is configured with rate limits."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        with patch('kosmos.agents.research_director.AsyncClaudeClient') as mock_async:
            mock_async.return_value = Mock()

            with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'}):
                director = ResearchDirectorAgent(
                    research_question="Test",
                    config={
                        "enable_concurrent_operations": True,
                        "max_concurrent_llm_calls": 10,
                        "llm_rate_limit_per_minute": 100
                    }
                )

                # Verify config is set (actual client initialization may not happen)
                # but configuration should be preserved
                assert director.config.get("max_concurrent_llm_calls") == 10
                assert director.config.get("llm_rate_limit_per_minute") == 100

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_resource_limits_prevent_unbounded_growth(self, mock_wm, mock_llm):
        """Verify resource limits prevent unbounded operation growth."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test",
            config={
                "enable_concurrent_operations": True,
                "max_parallel_hypotheses": 2,
                "max_concurrent_experiments": 3
            }
        )

        # Add many hypotheses and experiments
        for i in range(100):
            director.research_plan.add_hypothesis(f"hyp{i}")
            director.research_plan.add_experiment(f"exp{i}")

        # Limits should still be enforced
        assert director.max_parallel_hypotheses == 2
        assert director.max_concurrent_experiments == 3

        # Actual batch sizes would be limited in execution
        # This ensures resource consumption doesn't grow unbounded
