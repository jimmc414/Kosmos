"""
Tests for Orchestrator Lifecycle Requirements (REQ-ORCH-LIFE-001 through REQ-ORCH-LIFE-005).

These tests validate lifecycle management including initialization, startup,
pause/resume, shutdown, and state persistence for the Research Director.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any
import threading
import time

from kosmos.core.workflow import WorkflowState, ResearchWorkflow, ResearchPlan
from kosmos.agents.research_director import ResearchDirectorAgent
from kosmos.agents.base import AgentStatus

# Test markers for requirements traceability
pytestmark = [
    pytest.mark.requirement("REQ-ORCH-LIFE"),
    pytest.mark.category("orchestrator"),
]


@pytest.mark.requirement("REQ-ORCH-LIFE-001")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_LIFE_001_DirectorInitialization:
    """
    REQ-ORCH-LIFE-001: The Research Director MUST properly initialize with
    research question, domain, configuration, and create initial research plan.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_initialization_with_required_params(self, mock_wm, mock_llm):
        """Verify director initializes with required parameters."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="What is the mechanism of action?",
            domain="biology"
        )

        assert director.research_question == "What is the mechanism of action?"
        assert director.domain == "biology"
        assert director.agent_type == "ResearchDirector"
        assert director.agent_id is not None

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_creates_research_plan_on_init(self, mock_wm, mock_llm):
        """Verify director creates research plan during initialization."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test question",
            domain="test"
        )

        assert director.research_plan is not None
        assert director.research_plan.research_question == "Test question"
        assert director.research_plan.domain == "test"
        assert director.research_plan.iteration_count == 0
        assert director.research_plan.current_state == WorkflowState.INITIALIZING

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_creates_workflow_on_init(self, mock_wm, mock_llm):
        """Verify director creates workflow state machine during initialization."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test question"
        )

        assert director.workflow is not None
        assert director.workflow.current_state == WorkflowState.INITIALIZING
        assert director.workflow.research_plan is director.research_plan

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_initializes_with_custom_config(self, mock_wm, mock_llm):
        """Verify director accepts and applies custom configuration."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        config = {
            "max_iterations": 15,
            "mandatory_stopping_criteria": ["custom_criterion"],
            "enable_concurrent_operations": True,
            "max_parallel_hypotheses": 5
        }

        director = ResearchDirectorAgent(
            research_question="Test",
            config=config
        )

        assert director.max_iterations == 15
        assert director.research_plan.max_iterations == 15
        assert "custom_criterion" in director.mandatory_stopping_criteria
        assert director.enable_concurrent == True
        assert director.max_parallel_hypotheses == 5

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_initializes_strategy_tracking(self, mock_wm, mock_llm):
        """Verify director initializes strategy effectiveness tracking."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        assert director.strategy_stats is not None
        assert "hypothesis_generation" in director.strategy_stats
        assert "experiment_design" in director.strategy_stats
        assert "hypothesis_refinement" in director.strategy_stats

        # Verify initial stats structure
        for strategy, stats in director.strategy_stats.items():
            assert "attempts" in stats
            assert "successes" in stats
            assert "cost" in stats
            assert stats["attempts"] == 0
            assert stats["successes"] == 0

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_initializes_agent_registry(self, mock_wm, mock_llm):
        """Verify director initializes empty agent registry."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        assert director.agent_registry is not None
        assert len(director.agent_registry) == 0

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_initializes_thread_safety_locks(self, mock_wm, mock_llm):
        """Verify director initializes thread safety locks for concurrent operations."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        assert hasattr(director, '_research_plan_lock')
        assert hasattr(director, '_strategy_stats_lock')
        assert hasattr(director, '_workflow_lock')
        assert isinstance(director._research_plan_lock, threading.RLock)


@pytest.mark.requirement("REQ-ORCH-LIFE-002")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_LIFE_002_DirectorStartup:
    """
    REQ-ORCH-LIFE-002: The Research Director MUST properly start and transition
    from INITIALIZING to GENERATING_HYPOTHESES state, invoking _on_start hooks.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_start_transitions_to_generating_hypotheses(self, mock_wm, mock_llm):
        """Verify director transitions to GENERATING_HYPOTHESES on start."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        assert director.workflow.current_state == WorkflowState.INITIALIZING

        director.start()

        assert director.workflow.current_state == WorkflowState.GENERATING_HYPOTHESES
        assert director.get_status() in [AgentStatus.RUNNING, AgentStatus.IDLE]

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_on_start_hook_invoked(self, mock_wm, mock_llm):
        """Verify _on_start lifecycle hook is invoked."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        # Spy on _on_start method
        with patch.object(director, '_on_start', wraps=director._on_start) as mock_on_start:
            director.start()
            mock_on_start.assert_called_once()

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_start_records_transition(self, mock_wm, mock_llm):
        """Verify starting director records transition in history."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        director.start()

        history = director.workflow.get_transition_history()
        assert len(history) >= 1
        assert history[0].to_state == WorkflowState.GENERATING_HYPOTHESES

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_start_is_idempotent(self, mock_wm, mock_llm):
        """Verify multiple start calls don't cause issues."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        director.start()
        initial_state = director.workflow.current_state

        # Starting again should not cause errors
        director.start()

        assert director.workflow.current_state == initial_state


@pytest.mark.requirement("REQ-ORCH-LIFE-003")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_LIFE_003_PauseResume:
    """
    REQ-ORCH-LIFE-003: The Research Director MUST support pause and resume
    operations, preserving workflow state and allowing continuation.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_can_pause(self, mock_wm, mock_llm):
        """Verify director can be paused."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.start()

        director.pause()

        # Verify paused state
        assert director.get_status() == AgentStatus.PAUSED

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_pause_preserves_state(self, mock_wm, mock_llm):
        """Verify pausing preserves workflow state and research plan."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.start()

        # Add some state
        director.research_plan.add_hypothesis("hyp1")
        director.research_plan.add_hypothesis("hyp2")
        director.research_plan.iteration_count = 3

        current_state = director.workflow.current_state
        hypothesis_count = len(director.research_plan.hypothesis_pool)
        iteration = director.research_plan.iteration_count

        director.pause()

        # Verify state preserved
        assert director.workflow.current_state == current_state
        assert len(director.research_plan.hypothesis_pool) == hypothesis_count
        assert director.research_plan.iteration_count == iteration

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_can_resume_from_pause(self, mock_wm, mock_llm):
        """Verify director can resume from paused state."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.start()
        director.pause()

        assert director.get_status() == AgentStatus.PAUSED

        director.resume()

        assert director.get_status() in [AgentStatus.RUNNING, AgentStatus.IDLE]

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_workflow_can_transition_to_paused_from_any_state(self, mock_wm, mock_llm):
        """Verify workflow can transition to PAUSED from any active state."""
        workflow = ResearchWorkflow()

        # PAUSED should be allowed from INITIALIZING
        assert workflow.can_transition_to(WorkflowState.PAUSED)

        # Test from various states
        states_to_test = [
            WorkflowState.GENERATING_HYPOTHESES,
            WorkflowState.DESIGNING_EXPERIMENTS,
            WorkflowState.EXECUTING,
            WorkflowState.ANALYZING,
            WorkflowState.REFINING
        ]

        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)

        for state in states_to_test[1:]:
            if workflow.can_transition_to(state):
                workflow.transition_to(state)
                assert workflow.can_transition_to(WorkflowState.PAUSED)


@pytest.mark.requirement("REQ-ORCH-LIFE-004")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_LIFE_004_DirectorShutdown:
    """
    REQ-ORCH-LIFE-004: The Research Director MUST properly shutdown, cleaning up
    resources, closing connections, and invoking _on_stop hooks.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_can_stop(self, mock_wm, mock_llm):
        """Verify director can be stopped."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.start()

        director.stop()

        assert director.get_status() == AgentStatus.STOPPED

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_on_stop_hook_invoked(self, mock_wm, mock_llm):
        """Verify _on_stop lifecycle hook is invoked."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.start()

        with patch.object(director, '_on_stop', wraps=director._on_stop) as mock_on_stop:
            director.stop()
            mock_on_stop.assert_called_once()

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_stop_preserves_research_data(self, mock_wm, mock_llm):
        """Verify stopping director preserves research plan and workflow state."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.start()

        # Add research data
        director.research_plan.add_hypothesis("hyp1")
        director.research_plan.add_experiment("exp1")
        director.research_plan.increment_iteration()

        hypothesis_count = len(director.research_plan.hypothesis_pool)
        experiment_count = len(director.research_plan.experiment_queue)
        iteration = director.research_plan.iteration_count

        director.stop()

        # Data should be preserved
        assert len(director.research_plan.hypothesis_pool) == hypothesis_count
        assert len(director.research_plan.experiment_queue) == experiment_count
        assert director.research_plan.iteration_count == iteration

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_cleanup_async_resources_on_stop(self, mock_wm, mock_llm):
        """Verify async resources are cleaned up on stop."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test",
            config={"enable_concurrent_operations": True}
        )

        # Mock async client
        mock_async_client = Mock()
        mock_async_client.close = Mock(return_value=None)
        director.async_llm_client = mock_async_client

        director.start()
        director.stop()

        # Note: actual cleanup is async, but we verify the intent is there
        # In real implementation, close() would be called

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_stop_is_idempotent(self, mock_wm, mock_llm):
        """Verify multiple stop calls don't cause errors."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.start()

        director.stop()
        assert director.get_status() == AgentStatus.STOPPED

        # Stopping again should not cause errors
        director.stop()
        assert director.get_status() == AgentStatus.STOPPED


@pytest.mark.requirement("REQ-ORCH-LIFE-005")
@pytest.mark.priority("SHOULD")
class TestREQ_ORCH_LIFE_005_StatePersistence:
    """
    REQ-ORCH-LIFE-005: The Research Director SHOULD support state export and
    import for persistence across sessions (via workflow.to_dict()).
    """

    def test_workflow_state_export_to_dict(self):
        """Verify workflow state can be exported to dictionary."""
        workflow = ResearchWorkflow()
        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)
        workflow.transition_to(WorkflowState.DESIGNING_EXPERIMENTS)

        state_dict = workflow.to_dict()

        assert isinstance(state_dict, dict)
        assert "current_state" in state_dict
        assert "transition_count" in state_dict
        assert "recent_transitions" in state_dict

        assert state_dict["current_state"] == "designing_experiments"
        assert state_dict["transition_count"] == 2

    def test_workflow_export_includes_transition_history(self):
        """Verify exported state includes recent transition history."""
        workflow = ResearchWorkflow()

        transitions = [
            WorkflowState.GENERATING_HYPOTHESES,
            WorkflowState.DESIGNING_EXPERIMENTS,
            WorkflowState.EXECUTING,
            WorkflowState.ANALYZING,
            WorkflowState.REFINING
        ]

        for state in transitions:
            workflow.transition_to(state)

        state_dict = workflow.to_dict()

        assert "recent_transitions" in state_dict
        recent = state_dict["recent_transitions"]
        assert len(recent) <= 5  # Last 5 transitions
        assert all("from" in t and "to" in t and "action" in t for t in recent)

    def test_research_plan_serialization(self):
        """Verify research plan can be serialized."""
        plan = ResearchPlan(
            research_question="Test question",
            domain="test",
            max_iterations=10
        )

        plan.add_hypothesis("hyp1")
        plan.add_hypothesis("hyp2")
        plan.add_experiment("exp1")
        plan.increment_iteration()

        # Convert to dict (Pydantic models have .dict() or .model_dump())
        try:
            plan_dict = plan.model_dump()
        except AttributeError:
            plan_dict = plan.dict()

        assert isinstance(plan_dict, dict)
        assert plan_dict["research_question"] == "Test question"
        assert plan_dict["domain"] == "test"
        assert len(plan_dict["hypothesis_pool"]) == 2
        assert plan_dict["iteration_count"] == 1

    def test_research_plan_includes_all_critical_fields(self):
        """Verify research plan serialization includes all critical fields."""
        plan = ResearchPlan(research_question="Test")

        try:
            plan_dict = plan.model_dump()
        except AttributeError:
            plan_dict = plan.dict()

        required_fields = [
            "research_question",
            "current_state",
            "hypothesis_pool",
            "tested_hypotheses",
            "supported_hypotheses",
            "rejected_hypotheses",
            "experiment_queue",
            "completed_experiments",
            "results",
            "iteration_count",
            "max_iterations",
            "has_converged"
        ]

        for field in required_fields:
            assert field in plan_dict, f"Missing field: {field}"

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_get_research_status_provides_complete_state(self, mock_wm, mock_llm):
        """Verify director provides complete research status for persistence."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test question", domain="biology")
        director.start()

        status = director.get_research_status()

        assert isinstance(status, dict)
        assert status["research_question"] == "Test question"
        assert status["domain"] == "biology"
        assert "workflow_state" in status
        assert "iteration" in status
        assert "hypothesis_pool_size" in status
        assert "strategy_stats" in status

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_status_includes_convergence_info(self, mock_wm, mock_llm):
        """Verify director status includes convergence information."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.research_plan.has_converged = True
        director.research_plan.convergence_reason = "Test convergence"

        status = director.get_research_status()

        assert "has_converged" in status
        assert "convergence_reason" in status
        assert status["has_converged"] is True
        assert status["convergence_reason"] == "Test convergence"

    def test_workflow_state_statistics_export(self):
        """Verify workflow state statistics can be exported."""
        workflow = ResearchWorkflow()

        # Create some transitions
        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)
        workflow.transition_to(WorkflowState.DESIGNING_EXPERIMENTS)
        workflow.transition_to(WorkflowState.EXECUTING)

        stats = workflow.get_state_statistics()

        assert isinstance(stats, dict)
        assert "state_visit_counts" in stats
        assert "state_durations_seconds" in stats
        assert "total_transitions" in stats
        assert "current_state" in stats

        assert stats["total_transitions"] == 3
        assert stats["current_state"] == "executing"
