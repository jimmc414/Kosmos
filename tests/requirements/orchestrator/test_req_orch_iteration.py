"""
Tests for Orchestrator Iteration Tracking Requirements (REQ-ORCH-ITER-001 through REQ-ORCH-ITER-008).

These tests validate iteration tracking, convergence detection, hypothesis management,
and cycle completion tracking for autonomous research.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from kosmos.core.workflow import ResearchPlan, WorkflowState, ResearchWorkflow
from kosmos.agents.research_director import ResearchDirectorAgent

# Test markers for requirements traceability
pytestmark = [
    pytest.mark.requirement("REQ-ORCH-ITER"),
    pytest.mark.category("orchestrator"),
]


@pytest.mark.requirement("REQ-ORCH-ITER-001")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_ITER_001_IterationCounting:
    """
    REQ-ORCH-ITER-001: The Research Director MUST track iteration count,
    incrementing once per complete research cycle (hypothesis → experiment → analysis → refinement).
    """

    def test_iteration_count_starts_at_zero(self):
        """Verify iteration count starts at zero."""
        plan = ResearchPlan(research_question="Test")

        assert plan.iteration_count == 0

    def test_increment_iteration(self):
        """Verify iteration can be incremented."""
        plan = ResearchPlan(research_question="Test")

        plan.increment_iteration()

        assert plan.iteration_count == 1

    def test_multiple_iteration_increments(self):
        """Verify multiple increments work correctly."""
        plan = ResearchPlan(research_question="Test", max_iterations=10)

        for i in range(1, 6):
            plan.increment_iteration()
            assert plan.iteration_count == i

    def test_iteration_increment_updates_timestamp(self):
        """Verify iteration increment updates research plan timestamp."""
        plan = ResearchPlan(research_question="Test")

        initial_timestamp = plan.updated_at

        plan.increment_iteration()

        assert plan.updated_at > initial_timestamp

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_increments_iteration_after_refinement(self, mock_wm, mock_llm):
        """Verify director increments iteration after completing refinement phase."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        initial_iteration = director.research_plan.iteration_count

        # Simulate completing a refinement cycle
        director.research_plan.add_hypothesis("hyp1")
        director.research_plan.mark_tested("hyp1")

        # Trigger refinement which should increment iteration
        from kosmos.core.workflow import NextAction
        with patch.object(director, '_send_to_hypothesis_refiner'):
            director._execute_next_action(NextAction.REFINE_HYPOTHESIS)

        # Iteration should be incremented
        assert director.research_plan.iteration_count == initial_iteration + 1

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_iteration_visible_in_research_status(self, mock_wm, mock_llm):
        """Verify iteration count is visible in research status."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.research_plan.increment_iteration()
        director.research_plan.increment_iteration()

        status = director.get_research_status()

        assert "iteration" in status
        assert status["iteration"] == 2


@pytest.mark.requirement("REQ-ORCH-ITER-002")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_ITER_002_MaxIterationLimit:
    """
    REQ-ORCH-ITER-002: The Research Director MUST enforce max_iterations limit
    and trigger convergence check when limit is reached.
    """

    def test_max_iterations_configurable(self):
        """Verify max_iterations can be configured."""
        plan1 = ResearchPlan(research_question="Test", max_iterations=5)
        plan2 = ResearchPlan(research_question="Test", max_iterations=20)

        assert plan1.max_iterations == 5
        assert plan2.max_iterations == 20

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_director_respects_max_iterations(self, mock_wm, mock_llm):
        """Verify director respects max_iterations configuration."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test",
            config={"max_iterations": 15}
        )

        assert director.max_iterations == 15
        assert director.research_plan.max_iterations == 15

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_convergence_triggered_at_iteration_limit(self, mock_wm, mock_llm):
        """Verify convergence check triggered when iteration limit reached."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test",
            config={"max_iterations": 3}
        )

        # Set iteration at limit
        director.research_plan.iteration_count = 3

        # Should check convergence
        should_converge = director._should_check_convergence()

        assert should_converge is True

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_iteration_limit_in_research_status(self, mock_wm, mock_llm):
        """Verify max_iterations visible in research status."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test",
            config={"max_iterations": 10}
        )

        status = director.get_research_status()

        assert "max_iterations" in status
        assert status["max_iterations"] == 10


@pytest.mark.requirement("REQ-ORCH-ITER-003")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_ITER_003_HypothesisPoolTracking:
    """
    REQ-ORCH-ITER-003: The Research Director MUST track hypothesis pool,
    including generated, tested, supported, and rejected hypotheses.
    """

    def test_hypothesis_pool_initialization(self):
        """Verify hypothesis pool starts empty."""
        plan = ResearchPlan(research_question="Test")

        assert len(plan.hypothesis_pool) == 0
        assert len(plan.tested_hypotheses) == 0
        assert len(plan.supported_hypotheses) == 0
        assert len(plan.rejected_hypotheses) == 0

    def test_add_hypothesis_to_pool(self):
        """Verify hypotheses can be added to pool."""
        plan = ResearchPlan(research_question="Test")

        plan.add_hypothesis("hyp1")
        plan.add_hypothesis("hyp2")

        assert len(plan.hypothesis_pool) == 2
        assert "hyp1" in plan.hypothesis_pool
        assert "hyp2" in plan.hypothesis_pool

    def test_duplicate_hypothesis_not_added_twice(self):
        """Verify duplicate hypotheses are not added multiple times."""
        plan = ResearchPlan(research_question="Test")

        plan.add_hypothesis("hyp1")
        plan.add_hypothesis("hyp1")

        assert len(plan.hypothesis_pool) == 1

    def test_mark_hypothesis_tested(self):
        """Verify hypothesis can be marked as tested."""
        plan = ResearchPlan(research_question="Test")

        plan.add_hypothesis("hyp1")
        plan.mark_tested("hyp1")

        assert "hyp1" in plan.tested_hypotheses
        assert len(plan.tested_hypotheses) == 1

    def test_mark_hypothesis_supported(self):
        """Verify hypothesis can be marked as supported."""
        plan = ResearchPlan(research_question="Test")

        plan.add_hypothesis("hyp1")
        plan.mark_supported("hyp1")

        assert "hyp1" in plan.supported_hypotheses
        assert "hyp1" in plan.tested_hypotheses  # Auto-marked as tested

    def test_mark_hypothesis_rejected(self):
        """Verify hypothesis can be marked as rejected."""
        plan = ResearchPlan(research_question="Test")

        plan.add_hypothesis("hyp1")
        plan.mark_rejected("hyp1")

        assert "hyp1" in plan.rejected_hypotheses
        assert "hyp1" in plan.tested_hypotheses  # Auto-marked as tested

    def test_get_untested_hypotheses(self):
        """Verify untested hypotheses can be retrieved."""
        plan = ResearchPlan(research_question="Test")

        plan.add_hypothesis("hyp1")
        plan.add_hypothesis("hyp2")
        plan.add_hypothesis("hyp3")

        plan.mark_tested("hyp1")

        untested = plan.get_untested_hypotheses()

        assert len(untested) == 2
        assert "hyp2" in untested
        assert "hyp3" in untested
        assert "hyp1" not in untested

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_hypothesis_counts_in_research_status(self, mock_wm, mock_llm):
        """Verify hypothesis counts visible in research status."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        director.research_plan.add_hypothesis("hyp1")
        director.research_plan.add_hypothesis("hyp2")
        director.research_plan.add_hypothesis("hyp3")

        director.research_plan.mark_supported("hyp1")
        director.research_plan.mark_rejected("hyp2")

        status = director.get_research_status()

        assert status["hypothesis_pool_size"] == 3
        assert status["hypotheses_tested"] == 2
        assert status["hypotheses_supported"] == 1
        assert status["hypotheses_rejected"] == 1


@pytest.mark.requirement("REQ-ORCH-ITER-004")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_ITER_004_ExperimentQueueTracking:
    """
    REQ-ORCH-ITER-004: The Research Director MUST track experiment queue,
    including queued and completed experiments.
    """

    def test_experiment_queue_initialization(self):
        """Verify experiment queue starts empty."""
        plan = ResearchPlan(research_question="Test")

        assert len(plan.experiment_queue) == 0
        assert len(plan.completed_experiments) == 0

    def test_add_experiment_to_queue(self):
        """Verify experiments can be added to queue."""
        plan = ResearchPlan(research_question="Test")

        plan.add_experiment("exp1")
        plan.add_experiment("exp2")

        assert len(plan.experiment_queue) == 2
        assert "exp1" in plan.experiment_queue
        assert "exp2" in plan.experiment_queue

    def test_duplicate_experiment_not_added_twice(self):
        """Verify duplicate experiments are not queued multiple times."""
        plan = ResearchPlan(research_question="Test")

        plan.add_experiment("exp1")
        plan.add_experiment("exp1")

        assert len(plan.experiment_queue) == 1

    def test_mark_experiment_complete(self):
        """Verify experiment can be marked as complete."""
        plan = ResearchPlan(research_question="Test")

        plan.add_experiment("exp1")
        plan.mark_experiment_complete("exp1")

        assert "exp1" not in plan.experiment_queue
        assert "exp1" in plan.completed_experiments

    def test_mark_experiment_complete_without_queuing(self):
        """Verify experiment can be marked complete even if not explicitly queued."""
        plan = ResearchPlan(research_question="Test")

        # Mark complete without adding to queue first
        plan.mark_experiment_complete("exp1")

        assert "exp1" in plan.completed_experiments
        assert "exp1" not in plan.experiment_queue

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_experiment_counts_in_research_status(self, mock_wm, mock_llm):
        """Verify experiment counts visible in research status."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        director.research_plan.add_experiment("exp1")
        director.research_plan.add_experiment("exp2")
        director.research_plan.add_experiment("exp3")

        director.research_plan.mark_experiment_complete("exp1")

        status = director.get_research_status()

        assert status["experiments_completed"] == 1


@pytest.mark.requirement("REQ-ORCH-ITER-005")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_ITER_005_ResultTracking:
    """
    REQ-ORCH-ITER-005: The Research Director MUST track experiment results
    and maintain result IDs for analysis.
    """

    def test_results_list_initialization(self):
        """Verify results list starts empty."""
        plan = ResearchPlan(research_question="Test")

        assert len(plan.results) == 0

    def test_add_result(self):
        """Verify results can be added."""
        plan = ResearchPlan(research_question="Test")

        plan.add_result("res1")
        plan.add_result("res2")

        assert len(plan.results) == 2
        assert "res1" in plan.results
        assert "res2" in plan.results

    def test_duplicate_result_not_added_twice(self):
        """Verify duplicate results are not added multiple times."""
        plan = ResearchPlan(research_question="Test")

        plan.add_result("res1")
        plan.add_result("res1")

        assert len(plan.results) == 1

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_result_count_in_research_status(self, mock_wm, mock_llm):
        """Verify result count visible in research status."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        director.research_plan.add_result("res1")
        director.research_plan.add_result("res2")
        director.research_plan.add_result("res3")

        status = director.get_research_status()

        assert status["results_count"] == 3


@pytest.mark.requirement("REQ-ORCH-ITER-006")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_ITER_006_TestabilityRate:
    """
    REQ-ORCH-ITER-006: The Research Director MUST calculate testability rate
    (ratio of tested to total hypotheses) for convergence analysis.
    """

    def test_testability_rate_with_no_hypotheses(self):
        """Verify testability rate is 0.0 when no hypotheses exist."""
        plan = ResearchPlan(research_question="Test")

        rate = plan.get_testability_rate()

        assert rate == 0.0

    def test_testability_rate_with_untested_hypotheses(self):
        """Verify testability rate calculation with some untested hypotheses."""
        plan = ResearchPlan(research_question="Test")

        plan.add_hypothesis("hyp1")
        plan.add_hypothesis("hyp2")
        plan.add_hypothesis("hyp3")
        plan.add_hypothesis("hyp4")

        plan.mark_tested("hyp1")
        plan.mark_tested("hyp2")

        rate = plan.get_testability_rate()

        assert rate == 0.5  # 2 tested out of 4 total

    def test_testability_rate_all_tested(self):
        """Verify testability rate is 1.0 when all hypotheses tested."""
        plan = ResearchPlan(research_question="Test")

        plan.add_hypothesis("hyp1")
        plan.add_hypothesis("hyp2")

        plan.mark_tested("hyp1")
        plan.mark_tested("hyp2")

        rate = plan.get_testability_rate()

        assert rate == 1.0

    def test_testability_rate_precision(self):
        """Verify testability rate handles fractional values correctly."""
        plan = ResearchPlan(research_question="Test")

        for i in range(10):
            plan.add_hypothesis(f"hyp{i}")

        for i in range(3):
            plan.mark_tested(f"hyp{i}")

        rate = plan.get_testability_rate()

        assert rate == 0.3  # 3 out of 10


@pytest.mark.requirement("REQ-ORCH-ITER-007")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_ITER_007_SupportRate:
    """
    REQ-ORCH-ITER-007: The Research Director MUST calculate support rate
    (ratio of supported to tested hypotheses) for convergence analysis.
    """

    def test_support_rate_with_no_tested_hypotheses(self):
        """Verify support rate is 0.0 when no hypotheses tested."""
        plan = ResearchPlan(research_question="Test")

        plan.add_hypothesis("hyp1")
        plan.add_hypothesis("hyp2")

        rate = plan.get_support_rate()

        assert rate == 0.0

    def test_support_rate_with_some_supported(self):
        """Verify support rate calculation with some supported hypotheses."""
        plan = ResearchPlan(research_question="Test")

        plan.add_hypothesis("hyp1")
        plan.add_hypothesis("hyp2")
        plan.add_hypothesis("hyp3")
        plan.add_hypothesis("hyp4")

        plan.mark_supported("hyp1")
        plan.mark_supported("hyp2")
        plan.mark_rejected("hyp3")
        plan.mark_tested("hyp4")  # Inconclusive

        rate = plan.get_support_rate()

        assert rate == 0.5  # 2 supported out of 4 tested

    def test_support_rate_all_supported(self):
        """Verify support rate is 1.0 when all tested hypotheses are supported."""
        plan = ResearchPlan(research_question="Test")

        plan.add_hypothesis("hyp1")
        plan.add_hypothesis("hyp2")

        plan.mark_supported("hyp1")
        plan.mark_supported("hyp2")

        rate = plan.get_support_rate()

        assert rate == 1.0

    def test_support_rate_all_rejected(self):
        """Verify support rate is 0.0 when all tested hypotheses are rejected."""
        plan = ResearchPlan(research_question="Test")

        plan.add_hypothesis("hyp1")
        plan.add_hypothesis("hyp2")

        plan.mark_rejected("hyp1")
        plan.mark_rejected("hyp2")

        rate = plan.get_support_rate()

        assert rate == 0.0

    def test_support_rate_precision(self):
        """Verify support rate handles fractional values correctly."""
        plan = ResearchPlan(research_question="Test")

        for i in range(10):
            plan.add_hypothesis(f"hyp{i}")

        for i in range(3):
            plan.mark_supported(f"hyp{i}")

        for i in range(3, 10):
            plan.mark_rejected(f"hyp{i}")

        rate = plan.get_support_rate()

        assert rate == 0.3  # 3 supported out of 10 tested


@pytest.mark.requirement("REQ-ORCH-ITER-008")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_ITER_008_ConvergenceState:
    """
    REQ-ORCH-ITER-008: The Research Director MUST track convergence state
    (has_converged, convergence_reason) and transition to CONVERGED state.
    """

    def test_convergence_state_initialization(self):
        """Verify convergence state starts as not converged."""
        plan = ResearchPlan(research_question="Test")

        assert plan.has_converged is False
        assert plan.convergence_reason is None

    def test_mark_converged_with_reason(self):
        """Verify convergence can be marked with reason."""
        plan = ResearchPlan(research_question="Test")

        plan.has_converged = True
        plan.convergence_reason = "Iteration limit reached"

        assert plan.has_converged is True
        assert plan.convergence_reason == "Iteration limit reached"

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_convergence_visible_in_research_status(self, mock_wm, mock_llm):
        """Verify convergence state visible in research status."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        director.research_plan.has_converged = True
        director.research_plan.convergence_reason = "No testable hypotheses remain"

        status = director.get_research_status()

        assert status["has_converged"] is True
        assert status["convergence_reason"] == "No testable hypotheses remain"

    def test_workflow_transition_to_converged(self):
        """Verify workflow can transition to CONVERGED state."""
        plan = ResearchPlan(research_question="Test")
        workflow = ResearchWorkflow(research_plan=plan)

        # Progress through workflow
        workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)
        workflow.transition_to(WorkflowState.DESIGNING_EXPERIMENTS)
        workflow.transition_to(WorkflowState.EXECUTING)
        workflow.transition_to(WorkflowState.ANALYZING)
        workflow.transition_to(WorkflowState.REFINING)

        # Transition to converged
        workflow.transition_to(WorkflowState.CONVERGED, action="Research complete")

        assert workflow.current_state == WorkflowState.CONVERGED
        assert plan.current_state == WorkflowState.CONVERGED

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_convergence_stops_director(self, mock_wm, mock_llm):
        """Verify convergence detection stops the director."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.start()

        # Simulate convergence message from ConvergenceDetector
        from kosmos.agents.base import AgentMessage, MessageType

        convergence_message = AgentMessage(
            from_agent="convergence_detector",
            to_agent=director.agent_id,
            type=MessageType.RESPONSE,
            content={
                "should_converge": True,
                "reason": "Iteration limit reached"
            },
            metadata={"agent_type": "ConvergenceDetector"}
        )

        with patch.object(director, 'stop') as mock_stop:
            director._handle_convergence_detector_response(convergence_message)

            # Verify director is stopped
            mock_stop.assert_called_once()

        # Verify convergence recorded
        assert director.research_plan.has_converged is True
        assert director.research_plan.convergence_reason == "Iteration limit reached"

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_should_check_convergence_conditions(self, mock_wm, mock_llm):
        """Verify convergence check is triggered under correct conditions."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test",
            config={"max_iterations": 5}
        )

        # Should not check convergence initially
        assert director._should_check_convergence() is False

        # Should check when iteration limit reached
        director.research_plan.iteration_count = 5
        assert director._should_check_convergence() is True

        # Reset
        director.research_plan.iteration_count = 2

        # Should check when no hypotheses in pool
        assert director._should_check_convergence() is True

        # Add hypothesis
        director.research_plan.add_hypothesis("hyp1")
        director.research_plan.mark_tested("hyp1")

        # Should check when no untested hypotheses and no queued experiments
        assert director._should_check_convergence() is True
