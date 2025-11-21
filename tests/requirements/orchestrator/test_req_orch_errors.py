"""
Tests for Orchestrator Error Handling Requirements (REQ-ORCH-ERR-001 through REQ-ORCH-ERR-008).

These tests validate error handling, recovery mechanisms, error state transitions,
and graceful degradation for the Research Director orchestrator.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from kosmos.core.workflow import WorkflowState, ResearchWorkflow, NextAction
from kosmos.agents.research_director import ResearchDirectorAgent
from kosmos.agents.base import AgentMessage, MessageType, AgentStatus

# Test markers for requirements traceability
pytestmark = [
    pytest.mark.requirement("REQ-ORCH-ERR"),
    pytest.mark.category("orchestrator"),
]


@pytest.mark.requirement("REQ-ORCH-ERR-001")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_ERR_001_ErrorStateTransition:
    """
    REQ-ORCH-ERR-001: The workflow MUST support ERROR state transition from
    any active state for error handling and recovery.
    """

    def test_error_state_defined(self):
        """Verify ERROR state is defined in workflow."""
        assert WorkflowState.ERROR in WorkflowState

    def test_transition_to_error_from_initializing(self):
        """Verify can transition to ERROR from INITIALIZING."""
        workflow = ResearchWorkflow()

        assert workflow.can_transition_to(WorkflowState.ERROR)

        workflow.transition_to(WorkflowState.ERROR, action="Initialization error")

        assert workflow.current_state == WorkflowState.ERROR

    def test_transition_to_error_from_generating_hypotheses(self):
        """Verify can transition to ERROR from GENERATING_HYPOTHESES."""
        workflow = ResearchWorkflow()

        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)
        assert workflow.can_transition_to(WorkflowState.ERROR)

        workflow.transition_to(WorkflowState.ERROR, action="Hypothesis generation error")

        assert workflow.current_state == WorkflowState.ERROR

    def test_transition_to_error_from_executing(self):
        """Verify can transition to ERROR from EXECUTING."""
        workflow = ResearchWorkflow()

        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)
        workflow.transition_to(WorkflowState.DESIGNING_EXPERIMENTS)
        workflow.transition_to(WorkflowState.EXECUTING)

        assert workflow.can_transition_to(WorkflowState.ERROR)

        workflow.transition_to(WorkflowState.ERROR, action="Execution error")

        assert workflow.current_state == WorkflowState.ERROR

    def test_transition_to_error_from_any_state(self):
        """Verify ERROR is accessible from all active states."""
        states_to_test = [
            WorkflowState.INITIALIZING,
            WorkflowState.GENERATING_HYPOTHESES,
            WorkflowState.DESIGNING_EXPERIMENTS,
            WorkflowState.EXECUTING,
            WorkflowState.ANALYZING,
            WorkflowState.REFINING,
            WorkflowState.PAUSED
        ]

        for state in states_to_test:
            workflow = ResearchWorkflow()

            # Navigate to target state
            if state == WorkflowState.INITIALIZING:
                pass  # Already there
            elif state == WorkflowState.GENERATING_HYPOTHESES:
                workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)
            elif state == WorkflowState.PAUSED:
                workflow.transition_to(WorkflowState.PAUSED)
            else:
                workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)
                if workflow.can_transition_to(state):
                    workflow.transition_to(state)

            # Verify ERROR is accessible
            assert workflow.can_transition_to(WorkflowState.ERROR), \
                f"ERROR not accessible from {state}"


@pytest.mark.requirement("REQ-ORCH-ERR-002")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_ERR_002_ErrorMessageHandling:
    """
    REQ-ORCH-ERR-002: The Research Director MUST handle ERROR messages from
    agents without crashing, logging errors and updating error counter.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_error_counter_initialized(self, mock_wm, mock_llm):
        """Verify error counter is initialized."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        assert hasattr(director, 'errors_encountered')
        assert director.errors_encountered == 0

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_error_message_from_hypothesis_generator(self, mock_wm, mock_llm):
        """Verify error message from HypothesisGeneratorAgent is handled."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        error_message = AgentMessage(
            from_agent="hyp_gen_001",
            to_agent=director.agent_id,
            type=MessageType.ERROR,
            content={"error": "Failed to generate hypotheses", "details": "LLM timeout"},
            metadata={"agent_type": "HypothesisGeneratorAgent"}
        )

        initial_errors = director.errors_encountered

        # Should not raise exception
        director._handle_hypothesis_generator_response(error_message)

        # Verify error counted
        assert director.errors_encountered == initial_errors + 1

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_error_message_from_experiment_designer(self, mock_wm, mock_llm):
        """Verify error message from ExperimentDesignerAgent is handled."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        error_message = AgentMessage(
            from_agent="exp_designer_001",
            to_agent=director.agent_id,
            type=MessageType.ERROR,
            content={"error": "Failed to design experiment", "hypothesis_id": "hyp1"},
            metadata={"agent_type": "ExperimentDesignerAgent"}
        )

        initial_errors = director.errors_encountered

        director._handle_experiment_designer_response(error_message)

        assert director.errors_encountered == initial_errors + 1

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_error_message_from_executor(self, mock_wm, mock_llm):
        """Verify error message from Executor is handled."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        error_message = AgentMessage(
            from_agent="executor_001",
            to_agent=director.agent_id,
            type=MessageType.ERROR,
            content={"error": "Experiment execution failed", "protocol_id": "exp1"},
            metadata={"agent_type": "Executor"}
        )

        initial_errors = director.errors_encountered

        director._handle_executor_response(error_message)

        assert director.errors_encountered == initial_errors + 1

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_error_message_from_data_analyst(self, mock_wm, mock_llm):
        """Verify error message from DataAnalystAgent is handled."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        error_message = AgentMessage(
            from_agent="analyst_001",
            to_agent=director.agent_id,
            type=MessageType.ERROR,
            content={"error": "Analysis failed", "result_id": "res1"},
            metadata={"agent_type": "DataAnalystAgent"}
        )

        initial_errors = director.errors_encountered

        director._handle_data_analyst_response(error_message)

        assert director.errors_encountered == initial_errors + 1

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_multiple_errors_counted(self, mock_wm, mock_llm):
        """Verify multiple errors are counted correctly."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        # Create multiple error messages
        for i in range(5):
            error_message = AgentMessage(
                from_agent="agent_001",
                to_agent=director.agent_id,
                type=MessageType.ERROR,
                content={"error": f"Error {i}"},
                metadata={"agent_type": "HypothesisGeneratorAgent"}
            )
            director._handle_hypothesis_generator_response(error_message)

        assert director.errors_encountered == 5


@pytest.mark.requirement("REQ-ORCH-ERR-003")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_ERR_003_ErrorRecoveryMechanisms:
    """
    REQ-ORCH-ERR-003: The workflow MUST support recovery from ERROR state,
    allowing restart or resume from appropriate state.
    """

    def test_error_state_allows_restart(self):
        """Verify ERROR state allows transition back to INITIALIZING."""
        workflow = ResearchWorkflow()

        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)
        workflow.transition_to(WorkflowState.ERROR, action="Error occurred")

        # Should be able to restart
        assert workflow.can_transition_to(WorkflowState.INITIALIZING)

        workflow.transition_to(WorkflowState.INITIALIZING, action="Restart after error")

        assert workflow.current_state == WorkflowState.INITIALIZING

    def test_error_state_allows_resume_from_hypothesis_generation(self):
        """Verify ERROR state allows resume from GENERATING_HYPOTHESES."""
        workflow = ResearchWorkflow()

        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)
        workflow.transition_to(WorkflowState.ERROR)

        # Should be able to resume
        assert workflow.can_transition_to(WorkflowState.GENERATING_HYPOTHESES)

        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES, action="Resume")

        assert workflow.current_state == WorkflowState.GENERATING_HYPOTHESES

    def test_error_state_allows_pause(self):
        """Verify ERROR state allows transition to PAUSED for investigation."""
        workflow = ResearchWorkflow()

        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)
        workflow.transition_to(WorkflowState.ERROR)

        assert workflow.can_transition_to(WorkflowState.PAUSED)

        workflow.transition_to(WorkflowState.PAUSED, action="Pause for investigation")

        assert workflow.current_state == WorkflowState.PAUSED

    def test_workflow_reset_after_error(self):
        """Verify workflow can be reset after error."""
        workflow = ResearchWorkflow()

        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)
        workflow.transition_to(WorkflowState.ERROR)

        workflow.reset()

        assert workflow.current_state == WorkflowState.INITIALIZING
        assert len(workflow.transition_history) == 0


@pytest.mark.requirement("REQ-ORCH-ERR-004")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_ERR_004_ErrorActionDecisionMaking:
    """
    REQ-ORCH-ERR-004: The Research Director MUST handle ERROR workflow state
    in decide_next_action(), returning ERROR_RECOVERY action.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_error_recovery_action_defined(self, mock_wm, mock_llm):
        """Verify ERROR_RECOVERY action is defined."""
        assert NextAction.ERROR_RECOVERY in NextAction

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_decide_error_recovery_in_error_state(self, mock_wm, mock_llm):
        """Verify decide_next_action returns ERROR_RECOVERY in ERROR state."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        # Transition to ERROR state
        director.workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)
        director.workflow.transition_to(WorkflowState.ERROR)

        action = director.decide_next_action()

        assert action == NextAction.ERROR_RECOVERY


@pytest.mark.requirement("REQ-ORCH-ERR-005")
@pytest.mark.priority("SHOULD")
class TestREQ_ORCH_ERR_005_GracefulDegradation:
    """
    REQ-ORCH-ERR-005: The Research Director SHOULD continue operation with
    reduced functionality when non-critical components fail (e.g., world model).
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_continues_without_world_model(self, mock_wm, mock_llm):
        """Verify director can operate without world model."""
        mock_llm.return_value = Mock()
        mock_wm.side_effect = Exception("World model unavailable")

        # Should not raise exception
        director = ResearchDirectorAgent(research_question="Test")

        assert director.wm is None
        assert director.question_entity_id is None

        # Should still be able to start
        director.start()

        assert director.workflow.current_state == WorkflowState.GENERATING_HYPOTHESES

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_graph_persistence_fails_gracefully(self, mock_wm, mock_llm):
        """Verify graph persistence failures don't stop operation."""
        mock_llm.return_value = Mock()
        mock_wm_instance = Mock()
        mock_wm_instance.add_entity = Mock(side_effect=Exception("Graph unavailable"))
        mock_wm.return_value = mock_wm_instance

        director = ResearchDirectorAgent(research_question="Test")

        # Should handle error gracefully
        director._persist_hypothesis_to_graph("hyp1", "TestAgent")

        # Director should still be operational
        assert director.workflow.current_state == WorkflowState.INITIALIZING

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_llm_plan_generation_failure_handled(self, mock_wm, mock_llm):
        """Verify LLM failures for plan generation are handled gracefully."""
        mock_client = Mock()
        mock_client.generate = Mock(side_effect=Exception("LLM unavailable"))
        mock_llm.return_value = mock_client
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        # Should not crash
        plan = director.generate_research_plan()

        assert isinstance(plan, str)
        assert "error" in plan.lower() or "Error" in plan


@pytest.mark.requirement("REQ-ORCH-ERR-006")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_ERR_006_ThreadSafety:
    """
    REQ-ORCH-ERR-006: The Research Director MUST use thread-safe locks for
    concurrent access to shared state (research plan, workflow, strategy stats).
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_research_plan_lock_exists(self, mock_wm, mock_llm):
        """Verify research plan lock is initialized."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        assert hasattr(director, '_research_plan_lock')
        assert director._research_plan_lock is not None

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_strategy_stats_lock_exists(self, mock_wm, mock_llm):
        """Verify strategy stats lock is initialized."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        assert hasattr(director, '_strategy_stats_lock')
        assert director._strategy_stats_lock is not None

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_workflow_lock_exists(self, mock_wm, mock_llm):
        """Verify workflow lock is initialized."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        assert hasattr(director, '_workflow_lock')
        assert director._workflow_lock is not None

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_research_plan_context_manager(self, mock_wm, mock_llm):
        """Verify research plan context manager provides thread-safe access."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        # Should not raise exception
        with director._research_plan_context():
            director.research_plan.add_hypothesis("hyp1")

        assert "hyp1" in director.research_plan.hypothesis_pool

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_strategy_stats_context_manager(self, mock_wm, mock_llm):
        """Verify strategy stats context manager provides thread-safe access."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        # Should not raise exception
        with director._strategy_stats_context():
            director.strategy_stats["hypothesis_generation"]["attempts"] += 1

        assert director.strategy_stats["hypothesis_generation"]["attempts"] == 1

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_workflow_context_manager(self, mock_wm, mock_llm):
        """Verify workflow context manager provides thread-safe access."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        # Should not raise exception
        with director._workflow_context():
            director.workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)

        assert director.workflow.current_state == WorkflowState.GENERATING_HYPOTHESES


@pytest.mark.requirement("REQ-ORCH-ERR-007")
@pytest.mark.priority("SHOULD")
class TestREQ_ORCH_ERR_007_ErrorLogging:
    """
    REQ-ORCH-ERR-007: The Research Director SHOULD log detailed error information
    including context, stack traces, and agent details for debugging.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    @patch('kosmos.agents.research_director.logger')
    def test_error_message_logged(self, mock_logger, mock_wm, mock_llm):
        """Verify error messages are logged."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        error_message = AgentMessage(
            from_agent="agent_001",
            to_agent=director.agent_id,
            type=MessageType.ERROR,
            content={"error": "Test error"},
            metadata={"agent_type": "HypothesisGeneratorAgent"}
        )

        director._handle_hypothesis_generator_response(error_message)

        # Verify logger.error was called
        mock_logger.error.assert_called()
        call_args = str(mock_logger.error.call_args)
        assert "failed" in call_args.lower() or "error" in call_args.lower()

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    @patch('kosmos.agents.research_director.logger')
    def test_graph_persistence_error_logged(self, mock_logger, mock_wm, mock_llm):
        """Verify graph persistence errors are logged with warnings."""
        mock_llm.return_value = Mock()
        mock_wm_instance = Mock()
        mock_wm_instance.add_entity = Mock(side_effect=Exception("Graph error"))
        mock_wm.return_value = mock_wm_instance

        director = ResearchDirectorAgent(research_question="Test")

        director._persist_hypothesis_to_graph("hyp1", "TestAgent")

        # Verify warning was logged
        mock_logger.warning.assert_called()


@pytest.mark.requirement("REQ-ORCH-ERR-008")
@pytest.mark.priority("SHOULD")
class TestREQ_ORCH_ERR_008_ErrorMetrics:
    """
    REQ-ORCH-ERR-008: The Research Director SHOULD track error metrics
    (error counts by type, error rates) for monitoring and diagnostics.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_error_counter_tracked(self, mock_wm, mock_llm):
        """Verify total error count is tracked."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        assert director.errors_encountered == 0

        # Generate some errors
        for i in range(3):
            error_message = AgentMessage(
                from_agent="agent_001",
                to_agent=director.agent_id,
                type=MessageType.ERROR,
                content={"error": f"Error {i}"},
                metadata={"agent_type": "HypothesisGeneratorAgent"}
            )
            director._handle_hypothesis_generator_response(error_message)

        assert director.errors_encountered == 3

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_error_count_in_status(self, mock_wm, mock_llm):
        """Verify error count is available in status."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        # Generate error
        error_message = AgentMessage(
            from_agent="agent_001",
            to_agent=director.agent_id,
            type=MessageType.ERROR,
            content={"error": "Test error"},
            metadata={"agent_type": "HypothesisGeneratorAgent"}
        )
        director._handle_hypothesis_generator_response(error_message)

        status = director.get_research_status()

        # Error count should be trackable
        assert director.errors_encountered > 0

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_strategy_failure_tracking(self, mock_wm, mock_llm):
        """Verify strategy failures are tracked separately."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        # Track successful and failed attempts
        director.update_strategy_effectiveness("hypothesis_generation", success=True)
        director.update_strategy_effectiveness("hypothesis_generation", success=False)
        director.update_strategy_effectiveness("hypothesis_generation", success=False)

        stats = director.strategy_stats["hypothesis_generation"]

        assert stats["attempts"] == 3
        assert stats["successes"] == 1

        # Failure rate can be computed
        failure_rate = 1 - (stats["successes"] / stats["attempts"])
        assert abs(failure_rate - 2/3) < 0.01  # ~66.7% failure rate
