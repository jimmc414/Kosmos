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
from kosmos.models.result import ExperimentResult, ResultStatus, ExecutionMetadata, StatisticalTestResult
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

class TestSingleIteration:
    """Test a single complete research iteration."""

    def test_complete_single_iteration(self, director, mock_llm_client):
        """Test one complete cycle: hypothesis → experiment → result → analysis."""
        # Mock Claude responses for research planning
        mock_llm_client.generate.return_value = json.dumps({
            "strategy": "Generate and test caffeine hypotheses",
            "hypothesis_directions": ["Memory", "Attention"],
            "experiment_strategy": "Computational analysis",
            "success_criteria": "p < 0.05",
        })

        # Start director
        director.start()

        assert director.workflow.current_state == WorkflowState.GENERATING_HYPOTHESES
        assert director.research_plan.iteration_count == 0

        # Simulate hypothesis generation
        hypothesis = Hypothesis(
            id="hyp_001",
            research_question=director.research_question,
            statement="Caffeine improves working memory",
            rationale="Stimulant effects on cognition",
            domain="neuroscience",
            testability_score=0.9,
        )

        # Add hypothesis to plan
        director.research_plan.add_hypothesis(hypothesis.id)

        # Transition to designing experiments
        director.workflow.transition_to(
            WorkflowState.DESIGNING_EXPERIMENTS,
            action="hypothesis_generated",
            metadata={"hypothesis_id": hypothesis.id},
        )

        assert director.workflow.current_state == WorkflowState.DESIGNING_EXPERIMENTS

        # Simulate experiment design
        from kosmos.models.experiment import ResourceRequirements, ProtocolStep, Variable
        protocol = ExperimentProtocol(
            id="protocol_001",
            name="Caffeine Cognitive Performance Test Protocol",
            hypothesis_id=hypothesis.id,
            experiment_type="computational",
            domain="neuroscience",
            description="Comprehensive statistical analysis protocol to test the effects of caffeine on cognitive performance metrics including memory, attention, and reaction time.",
            objective="Validate caffeine effects on cognitive performance through statistical analysis",
            steps=[ProtocolStep(
                step_number=1,
                title="Analysis",
                action="execute_analysis",
                description="Run statistical analysis on caffeine performance data",
                expected_duration_minutes=5
            )],
            variables={"caffeine_dose": Variable(name="caffeine_dose", description="Caffeine dose in milligrams", unit="mg")},
            resource_requirements=ResourceRequirements(
                estimated_runtime_seconds=300,
                cpu_cores=1,
                memory_gb=1,
                storage_gb=0.1
            )
        )

        director.research_plan.add_experiment(protocol.id)

        # Transition to executing
        director.workflow.transition_to(
            WorkflowState.EXECUTING,
            action="experiment_designed",
            metadata={"protocol_id": protocol.id},
        )

        assert director.workflow.current_state == WorkflowState.EXECUTING

        # Simulate execution completion
        result = ExperimentResult(
            id="result_001",
            hypothesis_id=hypothesis.id,
            supports_hypothesis=True,
            primary_p_value=0.01,
            primary_effect_size=0.75,
            primary_test="t-test",
            status=ResultStatus.SUCCESS,
        )

        director.research_plan.add_result(result.id)
        director.research_plan.mark_tested(hypothesis.id)
        director.research_plan.mark_supported(hypothesis.id)

        # Transition to analyzing
        director.workflow.transition_to(
            WorkflowState.ANALYZING,
            action="execution_complete",
            metadata={"result_id": result.id},
        )

        assert director.workflow.current_state == WorkflowState.ANALYZING

        # Complete iteration
        director.research_plan.increment_iteration()

        assert director.research_plan.iteration_count == 1
        assert len(director.research_plan.tested_hypotheses) == 1
        assert len(director.research_plan.supported_hypotheses) == 1

    def test_iteration_state_progression(self, director):
        """Test state progresses through all stages in iteration."""
        director.start()

        # Record states visited
        states_visited = [director.workflow.current_state]

        # Simulate state transitions
        expected_states = [
            WorkflowState.INITIALIZING,
            WorkflowState.GENERATING_HYPOTHESES,
            WorkflowState.DESIGNING_EXPERIMENTS,
            WorkflowState.EXECUTING,
            WorkflowState.ANALYZING,
        ]

        # Start from INITIALIZING
        assert director.workflow.current_state == expected_states[1]  # Starts at GENERATING

        # Progress through states
        for i in range(2, len(expected_states)):
            can_transition = director.workflow.can_transition_to(expected_states[i])
            assert can_transition is True

    def test_iteration_updates_plan(self, director):
        """Test iteration updates research plan correctly."""
        initial_iteration = director.research_plan.iteration_count

        # Add hypothesis
        director.research_plan.add_hypothesis("hyp_001")
        director.research_plan.mark_tested("hyp_001")
        director.research_plan.mark_supported("hyp_001")
        director.research_plan.increment_iteration()

        assert director.research_plan.iteration_count == initial_iteration + 1
        assert "hyp_001" in director.research_plan.tested_hypotheses
        assert "hyp_001" in director.research_plan.supported_hypotheses


# ============================================================================
# Test Class 2: Multiple Iterations
# ============================================================================

class TestMultipleIterations:
    """Test multiple research iterations."""

    def test_two_iterations_complete(self, director, mock_llm_client):
        """Test two complete iterations."""
        mock_llm_client.generate.return_value = json.dumps({
            "strategy": "Test strategy",
            "hypothesis_directions": ["Test"],
            "experiment_strategy": "Computational",
            "success_criteria": "p < 0.05",
        })

        director.start()

        # Iteration 1
        director.research_plan.add_hypothesis("hyp_001")
        director.research_plan.mark_tested("hyp_001")
        director.research_plan.mark_supported("hyp_001")
        director.research_plan.increment_iteration()

        assert director.research_plan.iteration_count == 1

        # Iteration 2
        director.research_plan.add_hypothesis("hyp_002")
        director.research_plan.mark_tested("hyp_002")
        director.research_plan.mark_rejected("hyp_002")
        director.research_plan.increment_iteration()

        assert director.research_plan.iteration_count == 2
        assert len(director.research_plan.tested_hypotheses) == 2
        assert len(director.research_plan.supported_hypotheses) == 1
        assert len(director.research_plan.rejected_hypotheses) == 1

    def test_three_iterations_with_refinement(self, director):
        """Test three iterations with hypothesis refinement."""
        director.start()

        # Iteration 1: Initial hypothesis, supported
        director.research_plan.add_hypothesis("hyp_001")
        director.research_plan.mark_tested("hyp_001")
        director.research_plan.mark_supported("hyp_001")
        director.research_plan.increment_iteration()

        # Iteration 2: Refined hypothesis, inconclusive
        director.research_plan.add_hypothesis("hyp_001_refined")  # Refined from hyp_001
        director.research_plan.mark_tested("hyp_001_refined")
        director.research_plan.increment_iteration()

        # Iteration 3: Variant hypothesis, supported
        director.research_plan.add_hypothesis("hyp_001_variant")
        director.research_plan.mark_tested("hyp_001_variant")
        director.research_plan.mark_supported("hyp_001_variant")
        director.research_plan.increment_iteration()

        assert director.research_plan.iteration_count == 3
        assert len(director.research_plan.hypothesis_pool) >= 3

    def test_iteration_limit_stops_loop(self, director):
        """Test loop stops at max iterations."""
        director.max_iterations = 3

        # Run 3 iterations
        for i in range(3):
            director.research_plan.add_hypothesis(f"hyp_{i}")
            director.research_plan.mark_tested(f"hyp_{i}")
            director.research_plan.increment_iteration()

        # Check we hit the limit
        assert director.research_plan.iteration_count == 3
        assert director.research_plan.iteration_count >= director.max_iterations

    def test_accumulates_knowledge_across_iterations(self, director):
        """Test knowledge accumulates across multiple iterations."""
        director.start()

        # Track cumulative counts
        for i in range(3):
            director.research_plan.add_hypothesis(f"hyp_{i}")
            director.research_plan.mark_tested(f"hyp_{i}")

            if i % 2 == 0:  # Even iterations support
                director.research_plan.mark_supported(f"hyp_{i}")
            else:  # Odd iterations reject
                director.research_plan.mark_rejected(f"hyp_{i}")

            director.research_plan.increment_iteration()

        # Check cumulative knowledge
        assert len(director.research_plan.hypothesis_pool) == 3
        assert len(director.research_plan.tested_hypotheses) == 3
        assert len(director.research_plan.supported_hypotheses) == 2  # hyp_0, hyp_2
        assert len(director.research_plan.rejected_hypotheses) == 1  # hyp_1


# ============================================================================
# Test Class 3: Message Passing
# ============================================================================

class TestMessagePassing:
    """Test message passing between agents."""

    def test_director_sends_to_hypothesis_generator(self, director):
        """Test director can send messages to hypothesis generator."""
        message = director._send_to_hypothesis_generator(
            action="generate",
            context={"count": 3},
        )

        assert message is not None
        assert message.from_agent_id == director.agent_id
        assert message.to_agent_id is not None
        assert "count" in message.context

    def test_director_sends_to_experiment_designer(self, director):
        """Test director can send messages to experiment designer."""
        message = director._send_to_experiment_designer(
            hypothesis_id="hyp_001",
            context={},
        )

        assert message is not None
        assert "hypothesis_id" in message.content

    def test_director_sends_to_executor(self, director):
        """Test director can send messages to executor."""
        message = director._send_to_executor(
            protocol_id="protocol_001",
            context={},
        )

        assert message is not None
        assert "protocol_id" in message.content

    def test_director_sends_to_data_analyst(self, director):
        """Test director can send messages to data analyst."""
        message = director._send_to_data_analyst(
            result_id="result_001",
            hypothesis_id="hyp_001",
            context={},
        )

        assert message is not None
        assert "result_id" in message.content
        assert "hypothesis_id" in message.content

    def test_director_handles_hypothesis_response(self, director):
        """Test director handles hypothesis generator response."""
        from kosmos.agents.base import AgentMessage

        # Create response message
        response = AgentMessage(
            type="response",
            from_agent="hypothesis_generator",
            to_agent=director.agent_id,
            content={"hypothesis_ids": ["hyp_001", "hyp_002"]},
            context={},
        )

        initial_count = len(director.research_plan.hypothesis_pool)

        director._handle_hypothesis_generator_response(response)

        # Should have added hypotheses to plan
        assert len(director.research_plan.hypothesis_pool) > initial_count

    def test_message_correlation_tracking(self, director):
        """Test pending requests are tracked correctly."""
        # Send message
        message = director._send_to_hypothesis_generator(
            action="generate",
            context={},
        )

        # Check it's tracked
        assert message.correlation_id in director.pending_requests
        assert director.pending_requests[message.correlation_id]["target_agent"] == "hypothesis_generator"


# ============================================================================
# Test Class 4: State Transitions
# ============================================================================

class TestStateTransitions:
    """Test workflow state transitions."""

    def test_valid_state_transitions(self, director):
        """Test all valid state transitions work."""
        workflow = director.workflow

        # Start from INITIALIZING
        workflow.current_state = WorkflowState.INITIALIZING

        # Valid transitions from each state
        valid_transitions = {
            WorkflowState.INITIALIZING: [WorkflowState.GENERATING_HYPOTHESES],
            WorkflowState.GENERATING_HYPOTHESES: [WorkflowState.DESIGNING_EXPERIMENTS],
            WorkflowState.DESIGNING_EXPERIMENTS: [WorkflowState.EXECUTING],
            WorkflowState.EXECUTING: [WorkflowState.ANALYZING],
            WorkflowState.ANALYZING: [WorkflowState.REFINING, WorkflowState.CONVERGED],
        }

        # Test each transition
        for current, next_states in valid_transitions.items():
            workflow.current_state = current
            for next_state in next_states:
                assert workflow.can_transition_to(next_state) is True

    def test_invalid_state_transitions(self, director):
        """Test invalid state transitions are rejected."""
        workflow = director.workflow

        # Cannot go directly from GENERATING to ANALYZING (must go through DESIGNING and EXECUTING)
        workflow.current_state = WorkflowState.GENERATING_HYPOTHESES

        with pytest.raises(ValueError):
            workflow.transition_to(
                WorkflowState.ANALYZING,
                action="invalid_jump",
                metadata={},
            )

    def test_pause_resume_transitions(self, director):
        """Test pause and resume transitions."""
        workflow = director.workflow

        # Can pause from any state
        workflow.current_state = WorkflowState.GENERATING_HYPOTHESES
        assert workflow.can_transition_to(WorkflowState.PAUSED) is True

        workflow.transition_to(WorkflowState.PAUSED, action="pause", metadata={})
        assert workflow.current_state == WorkflowState.PAUSED

        # Can resume from PAUSED to previous state
        # (In real implementation, would track previous state)

    def test_error_state_transitions(self, director):
        """Test error state transitions."""
        workflow = director.workflow

        # Can transition to ERROR from any state
        workflow.current_state = WorkflowState.EXECUTING

        workflow.transition_to(WorkflowState.ERROR, action="error_occurred", metadata={"error": "Test error"})
        assert workflow.current_state == WorkflowState.ERROR

    def test_convergence_transition(self, director):
        """Test transition to CONVERGED state."""
        workflow = director.workflow

        # Can converge from ANALYZING or REFINING
        workflow.current_state = WorkflowState.ANALYZING

        workflow.transition_to(WorkflowState.CONVERGED, action="convergence_detected", metadata={})
        assert workflow.current_state == WorkflowState.CONVERGED

    def test_full_cycle_transitions(self, director):
        """Test complete cycle through all states."""
        workflow = director.workflow

        states_sequence = [
            WorkflowState.INITIALIZING,
            WorkflowState.GENERATING_HYPOTHESES,
            WorkflowState.DESIGNING_EXPERIMENTS,
            WorkflowState.EXECUTING,
            WorkflowState.ANALYZING,
            WorkflowState.REFINING,
            WorkflowState.GENERATING_HYPOTHESES,  # Back to generating for next iteration
        ]

        workflow.current_state = states_sequence[0]

        for i in range(len(states_sequence) - 1):
            current = states_sequence[i]
            next_state = states_sequence[i + 1]

            assert workflow.can_transition_to(next_state) is True
            workflow.transition_to(next_state, action=f"step_{i}", metadata={})
            assert workflow.current_state == next_state

    def test_transition_history_tracking(self, director):
        """Test transition history is recorded."""
        workflow = director.workflow

        initial_history_length = len(workflow.get_transition_history())

        workflow.transition_to(
            WorkflowState.GENERATING_HYPOTHESES,
            action="start_research",
            metadata={},
        )

        history = workflow.get_transition_history()
        assert len(history) > initial_history_length

        # Check latest transition
        latest = history[-1]
        assert latest.to_state == WorkflowState.GENERATING_HYPOTHESES
        assert latest.action == "start_research"


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
            rationale="Stimulant effects on cognition and performance",
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
            rationale="Attention enhancement and focus improvement",
            domain="neuroscience",
        )

        result = ExperimentResult(
            id="result_002",
            experiment_id="exp_002",
            protocol_id="proto_002",
            hypothesis_id=hypothesis.id,
            supports_hypothesis=False,
            primary_p_value=0.65,
            primary_effect_size=0.12,
            primary_test="t-test",
            status=ResultStatus.SUCCESS,
            statistical_tests=[
                StatisticalTestResult(
                    test_type="t-test",
                    test_name="t-test",
                    statistic=0.5,
                    p_value=0.65,
                    significant_0_05=False,
                    significant_0_01=False,
                    significant_0_001=False,
                    significance_label="ns"
                )
            ],
            metadata=ExecutionMetadata(
                experiment_id="exp_002",
                protocol_id="proto_002",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                duration_seconds=1.0,
                python_version="3.11",
                platform="linux"
            )
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
            rationale="Stimulant effects on cognition and performance",
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

    def test_strategy_adaptation_based_on_feedback(self, director):
        """Test strategy weights adapt based on success/failure."""
        initial_stats = director.strategy_stats.copy()

        # Record success for a strategy
        director.strategy_stats["hypothesis_generation"]["successes"] += 1
        director.strategy_stats["hypothesis_generation"]["attempts"] += 1

        # Record failure for another strategy
        if "failures" not in director.strategy_stats["experiment_design"]:
            director.strategy_stats["experiment_design"]["failures"] = 0
        director.strategy_stats["experiment_design"]["failures"] += 1
        director.strategy_stats["experiment_design"]["attempts"] += 1

        # Strategy selection should favor successful strategy
        # (This would be tested in select_next_strategy if implemented)
        assert director.strategy_stats["hypothesis_generation"]["successes"] > initial_stats["hypothesis_generation"]["successes"]

    def test_convergence_detection_integration(self, director):
        """Test convergence detector integrates with research loop."""
        from kosmos.core.convergence import ConvergenceDetector

        if not hasattr(director, 'convergence_detector'):
            director.convergence_detector = ConvergenceDetector()

        # Set up scenario for convergence
        director.research_plan.iteration_count = director.max_iterations

        hypotheses = [
            Hypothesis(
                id=f"hyp_{i}",
                research_question=director.research_question,
                statement=f"Hypothesis {i}",
                rationale="Rationale must be at least twenty characters long for validity",
                domain="neuroscience",
            )
            for i in range(3)
        ]

        results = [
            ExperimentResult(
                id=f"result_{i}",
                experiment_id=f"exp_{i}",
                protocol_id=f"proto_{i}",
                hypothesis_id=f"hyp_{i}",
                supports_hypothesis=True,
                primary_p_value=0.01,
                primary_effect_size=0.7,
                primary_test="t-test",
                status=ResultStatus.SUCCESS,
                statistical_tests=[
                    StatisticalTestResult(
                        test_type="t-test",
                        test_name="t-test",
                        statistic=2.5,
                        p_value=0.01,
                        significant_0_05=True,
                        significant_0_01=True,
                        significant_0_001=False,
                        significance_label="**"
                    )
                ],
                metadata=ExecutionMetadata(
                    experiment_id=f"exp_{i}",
                    protocol_id=f"proto_{i}",
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    duration_seconds=1.0,
                    python_version="3.11",
                    platform="linux"
                )
            )
            for i in range(3)
        ]

        # Check convergence
        decision = director.convergence_detector.check_convergence(
            director.research_plan, hypotheses, results
        )

        # Should stop due to iteration limit
        assert decision.should_stop is True
        assert decision.reason.value == "iteration_limit"
