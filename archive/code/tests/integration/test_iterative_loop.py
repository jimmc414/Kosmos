"""
Integration tests for Phase 7 iterative research loop.

Tests the complete research cycle with agent coordination, state transitions,
message passing, and feedback integration.
"""

import json
from unittest.mock import Mock, patch, MagicMock
import pytest

from kosmos.agents.research_director import ResearchDirectorAgent, NextAction
from kosmos.core.workflow import WorkflowState, ResearchWorkflow, ResearchPlan
from kosmos.models.hypothesis import Hypothesis, HypothesisStatus
from kosmos.models.result import ExperimentResult, ResultStatus
from kosmos.models.experiment import ExperimentProtocol


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_agents():
    """Create mocked agents for testing."""
    return {
        "hypothesis_generator": Mock(),
        "experiment_designer": Mock(),
        "executor": Mock(),
        "data_analyst": Mock(),
        "hypothesis_refiner": Mock(),
        "convergence_detector": Mock(),
    }


@pytest.fixture
def director(mock_llm_client):
    """Create ResearchDirectorAgent with mocked LLM."""
    return ResearchDirectorAgent(
        research_question="Does caffeine improve cognitive performance?",
        domain="neuroscience",
        config={"max_iterations": 5}
    )


# ============================================================================
# Test Class 1: Single Iteration
# ============================================================================


# ============================================================================
# Test Class 2: Multiple Iterations
# ============================================================================


# ============================================================================
# Test Class 3: Message Passing
# ============================================================================


# ============================================================================
# Test Class 4: State Transitions
# ============================================================================


# ============================================================================
# Test Class 5: Feedback Integration
# ============================================================================

class TestFeedbackIntegration:
    """Test feedback loop integration during iterations."""

    def test_feedback_loop_processes_success(self, director):
        """Test feedback loop processes successful results."""
        # Create feedback loop if not exists
        from kosmos.core.feedback import FeedbackLoop

        if not hasattr(director, 'feedback_loop'):
            director.feedback_loop = FeedbackLoop()

        hypothesis = Hypothesis(
            id="hyp_001",
            research_question=director.research_question,
            statement="Caffeine improves memory",
            rationale="Stimulant effects",
            domain="neuroscience",
        )

        result = ExperimentResult(
            id="result_001",
            hypothesis_id=hypothesis.id,
            supports_hypothesis=True,
            primary_p_value=0.01,
            primary_effect_size=0.75,
            primary_test="t-test",
            status=ResultStatus.SUCCESS,
        )

        # Process feedback
        signals = director.feedback_loop.process_result_feedback(result, hypothesis)

        assert len(signals) > 0
        assert any(s.signal_type.value == "success_pattern" for s in signals)

    def test_feedback_loop_processes_failure(self, director):
        """Test feedback loop processes failed results."""
        from kosmos.core.feedback import FeedbackLoop

        if not hasattr(director, 'feedback_loop'):
            director.feedback_loop = FeedbackLoop()

        hypothesis = Hypothesis(
            id="hyp_002",
            research_question=director.research_question,
            statement="Caffeine reduces errors",
            rationale="Attention enhancement",
            domain="neuroscience",
        )

        result = ExperimentResult(
            id="result_002",
            hypothesis_id=hypothesis.id,
            supports_hypothesis=False,
            primary_p_value=0.65,
            primary_effect_size=0.12,
            primary_test="t-test",
            status=ResultStatus.SUCCESS,
        )

        signals = director.feedback_loop.process_result_feedback(result, hypothesis)

        assert len(signals) > 0
        assert any(s.signal_type.value == "failure_pattern" for s in signals)

    def test_memory_prevents_duplicate_experiments(self, director):
        """Test memory system prevents duplicate experiments."""
        from kosmos.core.memory import MemoryStore

        if not hasattr(director, 'memory'):
            director.memory = MemoryStore()

        hypothesis = Hypothesis(
            id="hyp_001",
            research_question=director.research_question,
            statement="Caffeine improves memory",
            rationale="Stimulant effects",
            domain="neuroscience",
        )

        protocol = ExperimentProtocol(
            id="protocol_001",
            hypothesis_id=hypothesis.id,
            experiment_type="computational",
            methodology="Statistical analysis",
            description="Test caffeine effects",
        )

        # Record first experiment
        director.memory.record_experiment(hypothesis, protocol)

        # Check for duplicate
        is_dup, reason = director.memory.is_duplicate_experiment(hypothesis, protocol)

        assert is_dup is True
        assert "duplicate" in reason.lower()
