"""
Tests for Orchestrator 7-Phase Discovery Cycle Requirements (REQ-ORCH-CYCLE-* and REQ-ORCH-SYN-*).

These tests validate the 7-phase autonomous research cycle and synchronization
between workflow states as specified in the Kosmos research paper.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from kosmos.core.workflow import (
    WorkflowState,
    ResearchWorkflow,
    ResearchPlan,
    WorkflowTransition,
    NextAction
)
from kosmos.agents.research_director import ResearchDirectorAgent

# Test markers for requirements traceability
pytestmark = [
    pytest.mark.requirement("REQ-ORCH-CYCLE"),
    pytest.mark.category("orchestrator"),
]


@pytest.mark.requirement("REQ-ORCH-CYCLE-001")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_CYCLE_001_SevenPhaseDiscoveryCycle:
    """
    REQ-ORCH-CYCLE-001: The Research Director MUST orchestrate a complete 7-phase
    discovery cycle: INITIALIZING → GENERATING_HYPOTHESES → DESIGNING_EXPERIMENTS
    → EXECUTING → ANALYZING → REFINING → (loop or CONVERGED).
    """

    def test_seven_phase_workflow_states_defined(self):
        """Verify all 7 primary workflow states are defined."""
        expected_states = {
            WorkflowState.INITIALIZING,
            WorkflowState.GENERATING_HYPOTHESES,
            WorkflowState.DESIGNING_EXPERIMENTS,
            WorkflowState.EXECUTING,
            WorkflowState.ANALYZING,
            WorkflowState.REFINING,
            WorkflowState.CONVERGED
        }

        # All states should be in WorkflowState enum
        all_states = set(WorkflowState)
        assert expected_states.issubset(all_states), \
            f"Missing workflow states: {expected_states - all_states}"

    def test_workflow_initialization_state(self):
        """Verify workflow starts in INITIALIZING state."""
        workflow = ResearchWorkflow()

        assert workflow.current_state == WorkflowState.INITIALIZING
        assert len(workflow.transition_history) == 0

    def test_complete_cycle_transitions(self):
        """Verify workflow can transition through complete 7-phase cycle."""
        workflow = ResearchWorkflow(initial_state=WorkflowState.INITIALIZING)

        # Phase 1: INITIALIZING → GENERATING_HYPOTHESES
        assert workflow.can_transition_to(WorkflowState.GENERATING_HYPOTHESES)
        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES, action="Start hypothesis generation")
        assert workflow.current_state == WorkflowState.GENERATING_HYPOTHESES

        # Phase 2: GENERATING_HYPOTHESES → DESIGNING_EXPERIMENTS
        assert workflow.can_transition_to(WorkflowState.DESIGNING_EXPERIMENTS)
        workflow.transition_to(WorkflowState.DESIGNING_EXPERIMENTS, action="Design experiments")
        assert workflow.current_state == WorkflowState.DESIGNING_EXPERIMENTS

        # Phase 3: DESIGNING_EXPERIMENTS → EXECUTING
        assert workflow.can_transition_to(WorkflowState.EXECUTING)
        workflow.transition_to(WorkflowState.EXECUTING, action="Execute experiments")
        assert workflow.current_state == WorkflowState.EXECUTING

        # Phase 4: EXECUTING → ANALYZING
        assert workflow.can_transition_to(WorkflowState.ANALYZING)
        workflow.transition_to(WorkflowState.ANALYZING, action="Analyze results")
        assert workflow.current_state == WorkflowState.ANALYZING

        # Phase 5: ANALYZING → REFINING
        assert workflow.can_transition_to(WorkflowState.REFINING)
        workflow.transition_to(WorkflowState.REFINING, action="Refine hypotheses")
        assert workflow.current_state == WorkflowState.REFINING

        # Phase 6: REFINING → GENERATING_HYPOTHESES (loop back)
        assert workflow.can_transition_to(WorkflowState.GENERATING_HYPOTHESES)
        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES, action="Generate new hypotheses")
        assert workflow.current_state == WorkflowState.GENERATING_HYPOTHESES

        # Verify transition history
        assert len(workflow.transition_history) == 6

        # Phase 7: Eventually → CONVERGED
        workflow.transition_to(WorkflowState.DESIGNING_EXPERIMENTS)
        workflow.transition_to(WorkflowState.EXECUTING)
        workflow.transition_to(WorkflowState.ANALYZING)
        workflow.transition_to(WorkflowState.REFINING)
        workflow.transition_to(WorkflowState.CONVERGED, action="Research converged")
        assert workflow.current_state == WorkflowState.CONVERGED

    def test_allowed_transitions_per_phase(self):
        """Verify each phase has correct allowed transitions."""
        workflow = ResearchWorkflow()

        # INITIALIZING can go to GENERATING_HYPOTHESES, PAUSED, ERROR
        assert WorkflowState.GENERATING_HYPOTHESES in workflow.get_allowed_next_states()
        assert WorkflowState.PAUSED in workflow.get_allowed_next_states()
        assert WorkflowState.ERROR in workflow.get_allowed_next_states()

        # GENERATING_HYPOTHESES transitions
        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)
        allowed = workflow.get_allowed_next_states()
        assert WorkflowState.DESIGNING_EXPERIMENTS in allowed
        assert WorkflowState.CONVERGED in allowed

        # DESIGNING_EXPERIMENTS transitions
        workflow.transition_to(WorkflowState.DESIGNING_EXPERIMENTS)
        allowed = workflow.get_allowed_next_states()
        assert WorkflowState.EXECUTING in allowed
        assert WorkflowState.GENERATING_HYPOTHESES in allowed

        # EXECUTING transitions
        workflow.transition_to(WorkflowState.EXECUTING)
        allowed = workflow.get_allowed_next_states()
        assert WorkflowState.ANALYZING in allowed

        # ANALYZING transitions
        workflow.transition_to(WorkflowState.ANALYZING)
        allowed = workflow.get_allowed_next_states()
        assert WorkflowState.REFINING in allowed
        assert WorkflowState.DESIGNING_EXPERIMENTS in allowed

        # REFINING transitions (can loop back or converge)
        workflow.transition_to(WorkflowState.REFINING)
        allowed = workflow.get_allowed_next_states()
        assert WorkflowState.GENERATING_HYPOTHESES in allowed
        assert WorkflowState.DESIGNING_EXPERIMENTS in allowed
        assert WorkflowState.CONVERGED in allowed

    def test_invalid_transitions_rejected(self):
        """Verify invalid phase transitions are rejected."""
        workflow = ResearchWorkflow(initial_state=WorkflowState.INITIALIZING)

        # Cannot jump directly from INITIALIZING to EXECUTING
        assert not workflow.can_transition_to(WorkflowState.EXECUTING)
        with pytest.raises(ValueError, match="Invalid transition"):
            workflow.transition_to(WorkflowState.EXECUTING)

        # Cannot go from GENERATING_HYPOTHESES to ANALYZING
        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)
        assert not workflow.can_transition_to(WorkflowState.ANALYZING)
        with pytest.raises(ValueError, match="Invalid transition"):
            workflow.transition_to(WorkflowState.ANALYZING)

    def test_transition_history_tracking(self):
        """Verify transition history is maintained throughout cycle."""
        workflow = ResearchWorkflow()

        transitions = [
            (WorkflowState.GENERATING_HYPOTHESES, "Start"),
            (WorkflowState.DESIGNING_EXPERIMENTS, "Design"),
            (WorkflowState.EXECUTING, "Execute"),
            (WorkflowState.ANALYZING, "Analyze"),
            (WorkflowState.REFINING, "Refine"),
        ]

        for target_state, action in transitions:
            workflow.transition_to(target_state, action=action)

        history = workflow.get_transition_history()
        assert len(history) == 5

        # Verify each transition is recorded
        for i, (target_state, action) in enumerate(transitions):
            assert history[i].to_state == target_state
            assert action in history[i].action
            assert isinstance(history[i].timestamp, datetime)

    def test_state_duration_tracking(self):
        """Verify time spent in each phase is tracked."""
        workflow = ResearchWorkflow()

        # Transition through several states
        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)
        workflow.transition_to(WorkflowState.DESIGNING_EXPERIMENTS)
        workflow.transition_to(WorkflowState.EXECUTING)

        # Check duration tracking
        duration = workflow.get_state_duration(WorkflowState.GENERATING_HYPOTHESES)
        assert duration >= 0

        stats = workflow.get_state_statistics()
        assert "state_visit_counts" in stats
        assert "state_durations_seconds" in stats
        assert stats["current_state"] == "executing"

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_orchestrates_complete_cycle(self, mock_wm, mock_llm):
        """Verify ResearchDirector orchestrates complete 7-phase cycle."""
        # Mock dependencies
        mock_llm.return_value = Mock()
        mock_wm.return_value = None  # Disable world model for test

        director = ResearchDirectorAgent(
            research_question="Test question",
            domain="test",
            config={"max_iterations": 2}
        )

        # Start director (should transition to GENERATING_HYPOTHESES)
        director.start()
        assert director.workflow.current_state == WorkflowState.GENERATING_HYPOTHESES

        # Verify workflow is initialized
        assert director.workflow is not None
        assert director.research_plan is not None
        assert director.research_plan.iteration_count == 0


@pytest.mark.requirement("REQ-ORCH-SYN-001")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_SYN_001_WorkflowStateSynchronization:
    """
    REQ-ORCH-SYN-001: The Research Director MUST maintain synchronization between
    workflow state, research plan, and agent coordination throughout the research cycle.
    """

    def test_workflow_and_research_plan_sync(self):
        """Verify workflow state syncs with research plan."""
        plan = ResearchPlan(
            research_question="Test question",
            domain="test"
        )

        workflow = ResearchWorkflow(
            initial_state=WorkflowState.INITIALIZING,
            research_plan=plan
        )

        # Verify initial sync
        assert workflow.current_state == WorkflowState.INITIALIZING
        assert plan.current_state == WorkflowState.INITIALIZING

        # Transition workflow
        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)

        # Verify plan is updated
        assert workflow.current_state == WorkflowState.GENERATING_HYPOTHESES
        assert plan.current_state == WorkflowState.GENERATING_HYPOTHESES

    def test_research_plan_timestamp_updates(self):
        """Verify research plan timestamps update on state changes."""
        plan = ResearchPlan(research_question="Test")
        workflow = ResearchWorkflow(research_plan=plan)

        initial_timestamp = plan.updated_at

        # Transition should update timestamp
        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)

        assert plan.updated_at > initial_timestamp

    def test_hypothesis_tracking_sync_with_workflow(self):
        """Verify hypothesis pool syncs with workflow transitions."""
        plan = ResearchPlan(research_question="Test")
        workflow = ResearchWorkflow(research_plan=plan)

        # Add hypotheses in GENERATING_HYPOTHESES state
        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)
        plan.add_hypothesis("hyp1")
        plan.add_hypothesis("hyp2")

        assert len(plan.hypothesis_pool) == 2
        assert "hyp1" in plan.hypothesis_pool
        assert "hyp2" in plan.hypothesis_pool

        # Transition to DESIGNING_EXPERIMENTS
        workflow.transition_to(WorkflowState.DESIGNING_EXPERIMENTS)

        # Hypotheses should persist
        assert len(plan.hypothesis_pool) == 2

    def test_experiment_queue_sync_with_execution(self):
        """Verify experiment queue syncs with EXECUTING state."""
        plan = ResearchPlan(research_question="Test")
        workflow = ResearchWorkflow(research_plan=plan)

        # Add experiments
        plan.add_experiment("exp1")
        plan.add_experiment("exp2")

        assert len(plan.experiment_queue) == 2

        # Mark experiment complete
        plan.mark_experiment_complete("exp1")

        assert len(plan.experiment_queue) == 1
        assert "exp1" not in plan.experiment_queue
        assert "exp1" in plan.completed_experiments

    def test_iteration_counter_increments_per_cycle(self):
        """Verify iteration counter increments once per complete cycle."""
        plan = ResearchPlan(research_question="Test", max_iterations=5)

        assert plan.iteration_count == 0

        # Increment iteration
        plan.increment_iteration()
        assert plan.iteration_count == 1

        # Multiple increments
        for i in range(3):
            plan.increment_iteration()

        assert plan.iteration_count == 4
        assert plan.iteration_count < plan.max_iterations

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_maintains_sync_across_transitions(self, mock_wm, mock_llm):
        """Verify ResearchDirector maintains sync across all transitions."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test",
            config={"max_iterations": 3}
        )

        # Check initial sync
        assert director.workflow.current_state == director.research_plan.current_state

        # Transition workflow
        director.workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)

        # Verify sync maintained
        assert director.workflow.current_state == director.research_plan.current_state
        assert director.research_plan.current_state == WorkflowState.GENERATING_HYPOTHESES

    def test_convergence_state_propagation(self):
        """Verify convergence state propagates to all components."""
        plan = ResearchPlan(research_question="Test")
        workflow = ResearchWorkflow(research_plan=plan)

        # Mark as converged
        plan.has_converged = True
        plan.convergence_reason = "Iteration limit reached"

        # Transition to CONVERGED state
        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)
        workflow.transition_to(WorkflowState.CONVERGED, action="Converged")

        # Verify sync
        assert workflow.current_state == WorkflowState.CONVERGED
        assert plan.current_state == WorkflowState.CONVERGED
        assert plan.has_converged is True
        assert plan.convergence_reason == "Iteration limit reached"

    def test_metadata_preservation_across_transitions(self):
        """Verify transition metadata is preserved and accessible."""
        workflow = ResearchWorkflow()

        metadata1 = {"hypothesis_count": 3, "source": "generator"}
        workflow.transition_to(
            WorkflowState.GENERATING_HYPOTHESES,
            action="Generate hypotheses",
            metadata=metadata1
        )

        metadata2 = {"protocol_id": "exp1", "hypothesis_id": "hyp1"}
        workflow.transition_to(
            WorkflowState.DESIGNING_EXPERIMENTS,
            action="Design experiment",
            metadata=metadata2
        )

        # Verify metadata preserved
        history = workflow.get_transition_history()
        assert len(history) == 2
        assert history[0].metadata == metadata1
        assert history[1].metadata == metadata2

    def test_transition_rollback_prevention(self):
        """Verify workflow prevents invalid rollback transitions."""
        workflow = ResearchWorkflow()

        # Progress forward
        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)
        workflow.transition_to(WorkflowState.DESIGNING_EXPERIMENTS)
        workflow.transition_to(WorkflowState.EXECUTING)

        # Cannot rollback to INITIALIZING from EXECUTING
        assert not workflow.can_transition_to(WorkflowState.INITIALIZING)

        # ERROR state allows restart
        workflow.transition_to(WorkflowState.ERROR)
        assert workflow.can_transition_to(WorkflowState.INITIALIZING)
