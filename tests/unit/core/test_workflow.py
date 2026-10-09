"""
Unit tests for workflow state machine (Phase 7).
"""

import pytest
from datetime import datetime, timedelta, timezone

from kosmos.core.workflow import (
    WorkflowState,
    WorkflowTransition,
    ResearchPlan,
    ResearchWorkflow,
    NextAction
)


# Test WorkflowTransition

class TestWorkflowTransition:
    """Test workflow transition model."""

    def test_create_transition(self):
        """Test creating a transition."""
        transition = WorkflowTransition(
            from_state=WorkflowState.INITIALIZING,
            to_state=WorkflowState.GENERATING_HYPOTHESES,
            action="Start hypothesis generation"
        )

        assert transition.from_state == WorkflowState.INITIALIZING
        assert transition.to_state == WorkflowState.GENERATING_HYPOTHESES
        assert transition.action == "Start hypothesis generation"
        assert isinstance(transition.timestamp, datetime)

    def test_transition_with_metadata(self):
        """Test transition with metadata."""
        transition = WorkflowTransition(
            from_state=WorkflowState.EXECUTING,
            to_state=WorkflowState.ANALYZING,
            action="Execute experiment",
            metadata={"protocol_id": "proto-1", "duration": 5.3}
        )

        assert transition.metadata["protocol_id"] == "proto-1"
        assert transition.metadata["duration"] == 5.3


# Test ResearchPlan

class TestResearchPlan:
    """Test research plan model."""

    def test_create_plan(self):
        """Test creating research plan."""
        plan = ResearchPlan(
            research_question="Does X affect Y?",
            domain="biology",
            max_iterations=10
        )

        assert plan.research_question == "Does X affect Y?"
        assert plan.domain == "biology"
        assert plan.max_iterations == 10
        assert plan.current_state == WorkflowState.INITIALIZING
        assert plan.iteration_count == 0
        assert plan.has_converged is False

    def test_add_hypothesis(self):
        """Test adding hypothesis to pool."""
        plan = ResearchPlan(research_question="Test?")

        plan.add_hypothesis("hyp-1")
        plan.add_hypothesis("hyp-2")

        assert len(plan.hypothesis_pool) == 2
        assert "hyp-1" in plan.hypothesis_pool
        assert "hyp-2" in plan.hypothesis_pool

    def test_add_duplicate_hypothesis(self):
        """Test adding duplicate hypothesis (should be ignored)."""
        plan = ResearchPlan(research_question="Test?")

        plan.add_hypothesis("hyp-1")
        plan.add_hypothesis("hyp-1")  # Duplicate

        assert len(plan.hypothesis_pool) == 1

    def test_mark_tested(self):
        """Test marking hypothesis as tested."""
        plan = ResearchPlan(research_question="Test?")
        plan.add_hypothesis("hyp-1")

        plan.mark_tested("hyp-1")

        assert "hyp-1" in plan.tested_hypotheses

    def test_mark_supported(self):
        """Test marking hypothesis as supported."""
        plan = ResearchPlan(research_question="Test?")
        plan.add_hypothesis("hyp-1")

        plan.mark_supported("hyp-1")

        assert "hyp-1" in plan.supported_hypotheses
        assert "hyp-1" in plan.tested_hypotheses  # Automatically marked tested

    def test_mark_rejected(self):
        """Test marking hypothesis as rejected."""
        plan = ResearchPlan(research_question="Test?")
        plan.add_hypothesis("hyp-1")

        plan.mark_rejected("hyp-1")

        assert "hyp-1" in plan.rejected_hypotheses
        assert "hyp-1" in plan.tested_hypotheses  # Automatically marked tested

    def test_add_experiment(self):
        """Test adding experiment to queue."""
        plan = ResearchPlan(research_question="Test?")

        plan.add_experiment("proto-1")
        plan.add_experiment("proto-2")

        assert len(plan.experiment_queue) == 2
        assert "proto-1" in plan.experiment_queue

    def test_mark_experiment_complete(self):
        """Test marking experiment as completed."""
        plan = ResearchPlan(research_question="Test?")
        plan.add_experiment("proto-1")

        plan.mark_experiment_complete("proto-1")

        assert "proto-1" not in plan.experiment_queue
        assert "proto-1" in plan.completed_experiments

    def test_add_result(self):
        """Test adding result."""
        plan = ResearchPlan(research_question="Test?")

        plan.add_result("result-1")
        plan.add_result("result-2")

        assert len(plan.results) == 2
        assert "result-1" in plan.results

    def test_increment_iteration(self):
        """Test incrementing iteration counter."""
        plan = ResearchPlan(research_question="Test?", max_iterations=5)

        assert plan.iteration_count == 0

        plan.increment_iteration()
        assert plan.iteration_count == 1

        plan.increment_iteration()
        assert plan.iteration_count == 2

    def test_get_untested_hypotheses(self):
        """Test getting untested hypotheses."""
        plan = ResearchPlan(research_question="Test?")

        plan.add_hypothesis("hyp-1")
        plan.add_hypothesis("hyp-2")
        plan.add_hypothesis("hyp-3")
        plan.mark_tested("hyp-1")

        untested = plan.get_untested_hypotheses()

        assert len(untested) == 2
        assert "hyp-2" in untested
        assert "hyp-3" in untested
        assert "hyp-1" not in untested

    def test_get_testability_rate(self):
        """Test calculating testability rate."""
        plan = ResearchPlan(research_question="Test?")

        # Empty pool
        assert plan.get_testability_rate() == 0.0

        plan.add_hypothesis("hyp-1")
        plan.add_hypothesis("hyp-2")
        plan.add_hypothesis("hyp-3")
        plan.mark_tested("hyp-1")

        # 1/3 tested
        rate = plan.get_testability_rate()
        assert abs(rate - 0.333) < 0.01

    def test_get_support_rate(self):
        """Test calculating support rate."""
        plan = ResearchPlan(research_question="Test?")

        # No tested hypotheses
        assert plan.get_support_rate() == 0.0

        plan.add_hypothesis("hyp-1")
        plan.add_hypothesis("hyp-2")
        plan.add_hypothesis("hyp-3")
        plan.mark_supported("hyp-1")
        plan.mark_rejected("hyp-2")
        plan.mark_tested("hyp-3")  # Inconclusive

        # 1/3 supported
        rate = plan.get_support_rate()
        assert abs(rate - 0.333) < 0.01

    def test_update_timestamp(self):
        """Test updating timestamp."""
        plan = ResearchPlan(research_question="Test?")
        original_time = plan.updated_at

        # Wait a tiny bit
        import time
        time.sleep(0.01)

        plan.update_timestamp()

        assert plan.updated_at > original_time


# Test ResearchWorkflow

class TestResearchWorkflow:
    """Test workflow state machine."""

    def test_create_workflow(self):
        """Test creating workflow."""
        workflow = ResearchWorkflow(initial_state=WorkflowState.INITIALIZING)

        assert workflow.current_state == WorkflowState.INITIALIZING
        assert len(workflow.transition_history) == 0

    def test_create_workflow_with_plan(self):
        """Test creating workflow with research plan."""
        plan = ResearchPlan(research_question="Test?")
        workflow = ResearchWorkflow(
            initial_state=WorkflowState.INITIALIZING,
            research_plan=plan
        )

        assert workflow.research_plan == plan

    def test_can_transition_to_valid(self):
        """Test checking valid transitions."""
        workflow = ResearchWorkflow(initial_state=WorkflowState.INITIALIZING)

        # INITIALIZING can go to GENERATING_HYPOTHESES
        can_transition = workflow.can_transition_to(WorkflowState.GENERATING_HYPOTHESES)
        assert can_transition is True

    def test_can_transition_to_invalid(self):
        """Test checking invalid transitions."""
        workflow = ResearchWorkflow(initial_state=WorkflowState.INITIALIZING)

        # INITIALIZING cannot go directly to CONVERGED
        can_transition = workflow.can_transition_to(WorkflowState.CONVERGED)
        assert can_transition is False

    def test_transition_to_valid(self):
        """Test successful transition."""
        workflow = ResearchWorkflow(initial_state=WorkflowState.INITIALIZING)

        result = workflow.transition_to(
            WorkflowState.GENERATING_HYPOTHESES,
            action="Start generation"
        )

        assert result is True
        assert workflow.current_state == WorkflowState.GENERATING_HYPOTHESES
        assert len(workflow.transition_history) == 1

        transition = workflow.transition_history[0]
        assert transition.from_state == WorkflowState.INITIALIZING
        assert transition.to_state == WorkflowState.GENERATING_HYPOTHESES
        assert transition.action == "Start generation"

    def test_transition_to_invalid_raises(self):
        """Test invalid transition raises error."""
        workflow = ResearchWorkflow(initial_state=WorkflowState.INITIALIZING)

        with pytest.raises(ValueError, match="Invalid transition"):
            workflow.transition_to(WorkflowState.CONVERGED)

    def test_transition_updates_plan(self):
        """Test transition updates research plan state."""
        plan = ResearchPlan(research_question="Test?")
        workflow = ResearchWorkflow(
            initial_state=WorkflowState.INITIALIZING,
            research_plan=plan
        )

        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES, "Test")

        # Plan state should be updated
        assert plan.current_state == WorkflowState.GENERATING_HYPOTHESES

    def test_get_allowed_next_states(self):
        """Test getting allowed next states."""
        workflow = ResearchWorkflow(initial_state=WorkflowState.INITIALIZING)

        allowed = workflow.get_allowed_next_states()

        assert WorkflowState.GENERATING_HYPOTHESES in allowed
        assert WorkflowState.PAUSED in allowed
        assert WorkflowState.ERROR in allowed

    def test_get_transition_history(self):
        """Test getting transition history."""
        workflow = ResearchWorkflow(initial_state=WorkflowState.INITIALIZING)

        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES, "Gen")
        workflow.transition_to(WorkflowState.DESIGNING_EXPERIMENTS, "Design")

        history = workflow.get_transition_history()

        assert len(history) == 2
        assert history[0].to_state == WorkflowState.GENERATING_HYPOTHESES
        assert history[1].to_state == WorkflowState.DESIGNING_EXPERIMENTS

    def test_get_recent_transitions(self):
        """Test getting recent transitions."""
        workflow = ResearchWorkflow(initial_state=WorkflowState.INITIALIZING)

        # Create several transitions
        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES, "1")
        workflow.transition_to(WorkflowState.DESIGNING_EXPERIMENTS, "2")
        workflow.transition_to(WorkflowState.EXECUTING, "3")
        workflow.transition_to(WorkflowState.ANALYZING, "4")

        recent = workflow.get_recent_transitions(n=2)

        assert len(recent) == 2
        assert recent[0].to_state == WorkflowState.EXECUTING
        assert recent[1].to_state == WorkflowState.ANALYZING

    def test_reset_workflow(self):
        """Test resetting workflow to initial state."""
        plan = ResearchPlan(research_question="Test?")
        workflow = ResearchWorkflow(
            initial_state=WorkflowState.INITIALIZING,
            research_plan=plan
        )

        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES, "Gen")
        workflow.transition_to(WorkflowState.DESIGNING_EXPERIMENTS, "Design")

        workflow.reset()

        assert workflow.current_state == WorkflowState.INITIALIZING
        assert len(workflow.transition_history) == 0
        assert plan.current_state == WorkflowState.INITIALIZING

    def test_to_dict(self):
        """Test exporting workflow to dict."""
        workflow = ResearchWorkflow(initial_state=WorkflowState.INITIALIZING)

        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES, "Gen")
        workflow.transition_to(WorkflowState.DESIGNING_EXPERIMENTS, "Design")

        workflow_dict = workflow.to_dict()

        assert workflow_dict["current_state"] == WorkflowState.DESIGNING_EXPERIMENTS.value
        assert workflow_dict["transition_count"] == 2
        assert len(workflow_dict["recent_transitions"]) == 2
        first = workflow_dict["recent_transitions"][0]
        assert first["from"] == WorkflowState.INITIALIZING.value
        assert first["to"] == WorkflowState.GENERATING_HYPOTHESES.value
        assert first["action"] == "Gen"

    def test_get_state_duration(self):
        """Test calculating time spent in state."""
        workflow = ResearchWorkflow(initial_state=WorkflowState.INITIALIZING)

        # Transition through states
        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES, "Gen")

        # Manually set timestamp to simulate time passage
        workflow.transition_history[0].timestamp = datetime.now(timezone.utc) - timedelta(seconds=10)

        workflow.transition_to(WorkflowState.DESIGNING_EXPERIMENTS, "Design")

        duration = workflow.get_state_duration(WorkflowState.GENERATING_HYPOTHESES)

        # Should be around 10 seconds (allow some margin)
        assert 9 < duration < 11

    def test_get_state_statistics(self):
        """Test getting state statistics."""
        workflow = ResearchWorkflow(initial_state=WorkflowState.INITIALIZING)

        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES, "Gen")
        workflow.transition_to(WorkflowState.DESIGNING_EXPERIMENTS, "Design")
        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES, "Gen again")

        stats = workflow.get_state_statistics()

        assert stats["current_state"] == WorkflowState.GENERATING_HYPOTHESES.value
        assert stats["total_transitions"] == 3
        assert stats["state_visit_counts"][WorkflowState.GENERATING_HYPOTHESES.value] == 2
        assert stats["state_visit_counts"][WorkflowState.DESIGNING_EXPERIMENTS.value] == 1


# Test State Machine Transitions

class TestStateMachineTransitions:
    """Test all valid and invalid state transitions."""

    def test_full_research_cycle(self):
        """Test complete research cycle through states."""
        workflow = ResearchWorkflow(initial_state=WorkflowState.INITIALIZING)

        # Full cycle
        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES, "Gen")
        workflow.transition_to(WorkflowState.DESIGNING_EXPERIMENTS, "Design")
        workflow.transition_to(WorkflowState.EXECUTING, "Execute")
        workflow.transition_to(WorkflowState.ANALYZING, "Analyze")
        workflow.transition_to(WorkflowState.REFINING, "Refine")

        # Can go back to generating hypotheses for next iteration
        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES, "Next iteration")

        assert workflow.current_state == WorkflowState.GENERATING_HYPOTHESES
        assert len(workflow.transition_history) == 6

    def test_pause_and_resume(self):
        """Test pausing from any state."""
        workflow = ResearchWorkflow(initial_state=WorkflowState.INITIALIZING)

        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES, "Gen")
        workflow.transition_to(WorkflowState.PAUSED, "Pause")

        assert workflow.current_state == WorkflowState.PAUSED

        # Can resume to various states
        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES, "Resume")

        assert workflow.current_state == WorkflowState.GENERATING_HYPOTHESES

    def test_error_and_recovery(self):
        """Test error state and recovery."""
        workflow = ResearchWorkflow(initial_state=WorkflowState.EXECUTING)

        workflow.transition_to(WorkflowState.ERROR, "Execution error")

        assert workflow.current_state == WorkflowState.ERROR

        # Can reset or resume
        workflow.transition_to(WorkflowState.INITIALIZING, "Reset")

        assert workflow.current_state == WorkflowState.INITIALIZING

    def test_convergence_from_generation(self):
        """Test converging directly from hypothesis generation."""
        workflow = ResearchWorkflow(initial_state=WorkflowState.GENERATING_HYPOTHESES)

        # If no viable hypotheses generated, can converge
        workflow.transition_to(WorkflowState.CONVERGED, "No viable hypotheses")

        assert workflow.current_state == WorkflowState.CONVERGED

    def test_convergence_from_refining(self):
        """Test converging from refinement state."""
        workflow = ResearchWorkflow(initial_state=WorkflowState.REFINING)

        workflow.transition_to(WorkflowState.CONVERGED, "Research complete")

        assert workflow.current_state == WorkflowState.CONVERGED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
