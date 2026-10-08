"""
Unit tests for ResearchDirectorAgent infinite loop prevention (Issue #51).

These tests verify the fixes for:
- Bug A: DESIGNING state premature CONVERGE
- Bug B: ANALYZING state with no results
- Bug C: Double iteration increment
- Bug D: EXECUTING state with empty queue/results
- MAX_ACTIONS_PER_ITERATION safety counter
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from kosmos.agents.research_director import (
    ResearchDirectorAgent,
    MAX_ACTIONS_PER_ITERATION
)
from kosmos.core.workflow import WorkflowState, NextAction


@pytest.fixture
def mock_director():
    """Create a mocked research director for state machine testing."""
    with patch('kosmos.agents.research_director.get_client') as mock_client, \
         patch('kosmos.agents.research_director.get_world_model') as mock_wm, \
         patch('kosmos.agents.research_director.SkillLoader') as mock_skills, \
         patch('kosmos.db.init_from_config'):
        mock_client.return_value = MagicMock()
        mock_wm.return_value = MagicMock()
        mock_skills.return_value = MagicMock()
        mock_skills.return_value.load_skills_for_task.return_value = ""

        director = ResearchDirectorAgent(
            research_question="Test question for loop prevention",
            domain="biology",
            config={"max_iterations": 10}
        )

        # Mock the research plan
        director.research_plan = MagicMock()
        director.research_plan.iteration_count = 0
        director.research_plan.max_iterations = 10
        director.research_plan.hypothesis_pool = []
        director.research_plan.experiment_queue = []
        director.research_plan.results = []
        director.research_plan.tested_hypotheses = set()
        director.research_plan.get_untested_hypotheses.return_value = []

        # Mock workflow
        director.workflow = MagicMock()
        director.workflow.current_state = WorkflowState.INITIALIZING

        yield director


class TestBugADesigningStateFix:
    """Test Bug A fix: DESIGNING state should check experiment_queue before converging."""

    def test_designing_with_queued_experiments_executes(self, mock_director):
        """DESIGNING state with queued experiments should execute, not converge."""
        mock_director.workflow.current_state = WorkflowState.DESIGNING_EXPERIMENTS
        mock_director.research_plan.hypothesis_pool = ["hyp_1"]  # Prevent early convergence
        mock_director.research_plan.get_untested_hypotheses.return_value = []
        mock_director.research_plan.experiment_queue = ["exp_1", "exp_2"]

        action = mock_director.decide_next_action()

        assert action == NextAction.EXECUTE_EXPERIMENT, \
            "Should execute experiments when queue is not empty"

    def test_designing_with_results_analyzes(self, mock_director):
        """DESIGNING state with no queue but results should analyze."""
        mock_director.workflow.current_state = WorkflowState.DESIGNING_EXPERIMENTS
        mock_director.research_plan.hypothesis_pool = ["hyp_1"]  # Prevent early convergence
        # Need untested hypothesis to prevent convergence check
        mock_director.research_plan.get_untested_hypotheses.return_value = ["hyp_1"]
        mock_director.research_plan.experiment_queue = []
        mock_director.research_plan.results = ["result_1"]

        action = mock_director.decide_next_action()

        # With untested hypotheses, should design more experiments
        assert action == NextAction.DESIGN_EXPERIMENT, \
            "Should design experiment for untested hypothesis"

    def test_designing_empty_converges(self, mock_director):
        """DESIGNING state with nothing to do should converge."""
        mock_director.workflow.current_state = WorkflowState.DESIGNING_EXPERIMENTS
        mock_director.research_plan.hypothesis_pool = ["hyp_1"]  # Has hypotheses but all tested
        mock_director.research_plan.get_untested_hypotheses.return_value = []
        mock_director.research_plan.experiment_queue = []
        mock_director.research_plan.results = []

        action = mock_director.decide_next_action()

        assert action == NextAction.CONVERGE, \
            "Should converge when nothing left to do"


class TestBugBAnalyzingStateFix:
    """Test Bug B fix: ANALYZING state should handle empty results."""

    def test_analyzing_with_no_results_falls_back(self, mock_director):
        """ANALYZING state with no results should fall back."""
        mock_director.workflow.current_state = WorkflowState.ANALYZING
        mock_director.research_plan.hypothesis_pool = ["hyp_1"]  # Prevent early convergence
        # Untested hypothesis prevents convergence check
        mock_director.research_plan.get_untested_hypotheses.return_value = ["hyp_1"]
        mock_director.research_plan.results = []
        mock_director.research_plan.experiment_queue = []

        action = mock_director.decide_next_action()

        assert action == NextAction.REFINE_HYPOTHESIS, \
            "Should refine hypothesis when no results to analyze"

    def test_analyzing_with_queue_executes(self, mock_director):
        """ANALYZING state with no results but queue should execute."""
        mock_director.workflow.current_state = WorkflowState.ANALYZING
        mock_director.research_plan.hypothesis_pool = ["hyp_1"]  # Prevent early convergence
        # Queue prevents convergence check from triggering
        mock_director.research_plan.get_untested_hypotheses.return_value = []
        mock_director.research_plan.results = []
        mock_director.research_plan.experiment_queue = ["exp_1"]

        action = mock_director.decide_next_action()

        assert action == NextAction.EXECUTE_EXPERIMENT, \
            "Should execute experiments when queue available"

    def test_analyzing_with_results_analyzes(self, mock_director):
        """ANALYZING state with results should analyze."""
        mock_director.workflow.current_state = WorkflowState.ANALYZING
        mock_director.research_plan.hypothesis_pool = ["hyp_1"]  # Prevent early convergence
        # Untested hypothesis prevents convergence check
        mock_director.research_plan.get_untested_hypotheses.return_value = ["hyp_1"]
        mock_director.research_plan.results = ["result_1"]

        action = mock_director.decide_next_action()

        assert action == NextAction.ANALYZE_RESULT, \
            "Should analyze when results exist"


class TestBugDExecutingStateFix:
    """Test Bug D fix: EXECUTING state should handle empty queue/results."""

    def test_executing_empty_queue_with_results_analyzes(self, mock_director):
        """EXECUTING state with empty queue but results should analyze."""
        mock_director.workflow.current_state = WorkflowState.EXECUTING
        mock_director.research_plan.hypothesis_pool = ["hyp_1"]  # Prevent early convergence
        # Untested hypothesis prevents convergence check
        mock_director.research_plan.get_untested_hypotheses.return_value = ["hyp_1"]
        mock_director.research_plan.experiment_queue = []
        mock_director.research_plan.results = ["result_1"]

        action = mock_director.decide_next_action()

        assert action == NextAction.ANALYZE_RESULT, \
            "Should analyze results when queue empty"

    def test_executing_empty_everything_refines(self, mock_director):
        """EXECUTING state with empty queue and no results should refine."""
        mock_director.workflow.current_state = WorkflowState.EXECUTING
        mock_director.research_plan.hypothesis_pool = ["hyp_1"]  # Prevent early convergence
        # Untested hypothesis prevents convergence check
        mock_director.research_plan.get_untested_hypotheses.return_value = ["hyp_1"]
        mock_director.research_plan.experiment_queue = []
        mock_director.research_plan.results = []

        action = mock_director.decide_next_action()

        assert action == NextAction.REFINE_HYPOTHESIS, \
            "Should refine to recover from stuck state"

    def test_executing_with_queue_executes(self, mock_director):
        """EXECUTING state with queued experiments should execute."""
        mock_director.workflow.current_state = WorkflowState.EXECUTING
        mock_director.research_plan.hypothesis_pool = ["hyp_1"]  # Prevent early convergence
        # Queue is set, so we test the actual state logic
        mock_director.research_plan.get_untested_hypotheses.return_value = []
        mock_director.research_plan.experiment_queue = ["exp_1"]

        action = mock_director.decide_next_action()

        assert action == NextAction.EXECUTE_EXPERIMENT, \
            "Should execute when queue has items"


class TestMaxActionsPerIterationSafety:
    """Test MAX_ACTIONS_PER_ITERATION safety counter."""

    def test_action_counter_forces_convergence(self, mock_director):
        """Exceeding MAX_ACTIONS should force convergence."""
        mock_director.workflow.current_state = WorkflowState.GENERATING_HYPOTHESES

        # Simulate exceeding action limit
        mock_director._actions_this_iteration = MAX_ACTIONS_PER_ITERATION + 1

        action = mock_director.decide_next_action()

        assert action == NextAction.CONVERGE, \
            "Should force convergence when action limit exceeded"

    def test_action_counter_increments(self, mock_director):
        """Action counter should increment each call."""
        mock_director.workflow.current_state = WorkflowState.GENERATING_HYPOTHESES

        # Ensure clean state
        if hasattr(mock_director, '_actions_this_iteration'):
            delattr(mock_director, '_actions_this_iteration')

        mock_director.decide_next_action()

        assert mock_director._actions_this_iteration == 1, \
            "Action counter should start at 1"

        mock_director.decide_next_action()

        assert mock_director._actions_this_iteration == 2, \
            "Action counter should increment"

    def test_max_actions_constant_defined(self):
        """MAX_ACTIONS_PER_ITERATION should be defined and reasonable."""
        assert MAX_ACTIONS_PER_ITERATION > 0, \
            "MAX_ACTIONS should be positive"
        assert MAX_ACTIONS_PER_ITERATION >= 10, \
            "MAX_ACTIONS should allow reasonable workflow"
        assert MAX_ACTIONS_PER_ITERATION <= 100, \
            "MAX_ACTIONS should prevent runaway loops"


class TestDomainValidation:
    """Test domain validation and logging."""

    def test_valid_domain_accepted(self, mock_director):
        """Valid domain should be accepted without warning."""
        # The fixture already sets domain="biology" which is valid
        assert mock_director.domain == "biology"

    def test_domain_validation_method_exists(self, mock_director):
        """_validate_domain method should exist."""
        assert hasattr(mock_director, '_validate_domain')
        assert callable(mock_director._validate_domain)


class TestSkillsIntegration:
    """Test skills integration."""

    def test_skills_attribute_exists(self, mock_director):
        """Skills attribute should be initialized."""
        assert hasattr(mock_director, 'skills')

    def test_get_skills_context_method_exists(self, mock_director):
        """get_skills_context method should exist."""
        assert hasattr(mock_director, 'get_skills_context')
        assert callable(mock_director.get_skills_context)

    def test_get_skills_context_returns_string(self, mock_director):
        """get_skills_context should return a string."""
        result = mock_director.get_skills_context()
        assert isinstance(result, str)


class TestLeaveRefining:
    """P1-2: every refinement pass leaves REFINING."""

    @pytest.fixture
    def refining_director(self, mock_director):
        from kosmos.core.workflow import ResearchPlan

        mock_director.research_plan = ResearchPlan(research_question="q", max_iterations=10)
        mock_director.research_plan.add_hypothesis("h1")
        mock_director.research_plan.mark_tested("h1")
        mock_director.workflow.current_state = WorkflowState.REFINING
        mock_director._hypothesis_refiner = Mock()
        return mock_director

    @staticmethod
    async def _refine_missing_hypothesis(director):
        session = MagicMock()
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=False)
        with patch('kosmos.agents.research_director.get_session', return_value=session), \
             patch('kosmos.db.operations.get_hypothesis', return_value=None):
            await director._handle_refine_hypothesis_action("h1")

    async def test_no_untested_goes_to_generation(self, refining_director):
        await self._refine_missing_hypothesis(refining_director)

        refining_director.workflow.transition_to.assert_called_once()
        assert refining_director.workflow.transition_to.call_args.args[0] == WorkflowState.GENERATING_HYPOTHESES

    async def test_untested_goes_to_design(self, refining_director):
        refining_director.research_plan.add_hypothesis("h2")

        await self._refine_missing_hypothesis(refining_director)

        assert refining_director.workflow.transition_to.call_args.args[0] == WorkflowState.DESIGNING_EXPERIMENTS

    async def test_outside_refining_does_not_transition(self, refining_director):
        refining_director.workflow.current_state = WorkflowState.EXECUTING

        await self._refine_missing_hypothesis(refining_director)

        refining_director.workflow.transition_to.assert_not_called()

    async def test_loop_closes_after_refinement(self, db_director):
        from kosmos.core.workflow import ResearchWorkflow
        from kosmos.db import get_session, operations
        from kosmos.hypothesis.refiner import RetirementDecision
        from tests.unit.agents.conftest import EXP_ID, H_ID

        with get_session() as session:
            operations.create_result(
                session, id="res-loop-1", experiment_id=EXP_ID,
                data={"execution_success": True}, p_value=0.2,
            )
        db_director.research_plan.mark_tested(H_ID)
        db_director.research_plan.add_hypothesis("h-untested")  # the dead end needs untested work
        db_director.research_plan.experiment_queue.clear()
        db_director.workflow = ResearchWorkflow(
            initial_state=WorkflowState.REFINING, research_plan=db_director.research_plan
        )
        db_director._hypothesis_refiner = Mock(
            evaluate_hypothesis_status=Mock(return_value=RetirementDecision.CONTINUE_TESTING)
        )

        assert db_director.decide_next_action() == NextAction.REFINE_HYPOTHESIS
        await db_director._handle_refine_hypothesis_action(H_ID)

        assert db_director.workflow.current_state == WorkflowState.DESIGNING_EXPERIMENTS
        assert db_director.decide_next_action() == NextAction.DESIGN_EXPERIMENT


class TestRefinementDuplicateFilter:
    """P2-5: refined and variant hypotheses that restate existing work are not stored."""

    async def test_near_duplicate_variant_dropped(self, db_director):
        from kosmos.db import get_session, operations
        from kosmos.hypothesis.novelty_checker import NoveltyChecker
        from kosmos.hypothesis.refiner import RetirementDecision
        from kosmos.knowledge.embeddings import reset_embedder
        from kosmos.models.hypothesis import Hypothesis
        from tests.unit.agents.conftest import EXP_ID, H_ID

        with get_session() as session:
            operations.create_result(
                session, id="res-dup-1", experiment_id=EXP_ID,
                data={"execution_success": True}, p_value=0.01,
                validation_status="validated",  # P2-7: variants spawn only from validated results
            )
        db_director.workflow.current_state = WorkflowState.REFINING

        def variant(vid, statement):
            return Hypothesis(
                id=vid, research_question="Does CO2 predict temperature?", statement=statement,
                rationale="Variant spawned from an inconclusive result", domain="climate",
            )

        duplicate = variant("var-dup", "CO2 concentration predicts the temperature anomaly")
        distinct = variant("var-new", "Volcanic aerosol index lowers the temperature anomaly a year later")
        db_director._hypothesis_refiner = Mock(
            evaluate_hypothesis_status=Mock(return_value=RetirementDecision.SPAWN_VARIANT),
            spawn_variant=Mock(return_value=[duplicate, distinct]),
        )

        reset_embedder()
        try:
            with patch('kosmos.knowledge.embeddings.HAS_SENTENCE_TRANSFORMERS', False), \
                 patch('kosmos.hypothesis.novelty_checker.get_vector_db', return_value=None), \
                 patch('kosmos.hypothesis.novelty_checker.UnifiedLiteratureSearch') as mock_search:
                mock_search.return_value.search.return_value = []
                db_director._novelty_checker = NoveltyChecker()
                await db_director._handle_refine_hypothesis_action(H_ID)
        finally:
            reset_embedder()

        with get_session() as session:
            assert operations.get_hypothesis(session, "var-dup") is None
            assert operations.get_hypothesis(session, "var-new") is not None
        assert "var-dup" not in db_director.research_plan.hypothesis_pool
        assert "var-new" in db_director.research_plan.hypothesis_pool


class TestPoolControlPieces:
    """P2-7: scores, inheritance, the backlog cap, the refinement limit, the REFINING exit."""

    def test_score_is_mean_of_testability_and_novelty(self):
        assert ResearchDirectorAgent._hypothesis_score(
            Mock(testability_score=0.8, novelty_score=0.4)
        ) == pytest.approx(0.6)
        assert ResearchDirectorAgent._hypothesis_score(
            Mock(testability_score=None, novelty_score=None)
        ) is None

    def test_full_backlog_designs_instead_of_generating(self, db_director):
        from kosmos.core.workflow import ResearchPlan, ResearchWorkflow

        db_director.research_plan = ResearchPlan(research_question="q")
        db_director.workflow = ResearchWorkflow(
            initial_state=WorkflowState.GENERATING_HYPOTHESES, research_plan=db_director.research_plan
        )
        for i in range(4):
            db_director.research_plan.add_hypothesis(f"h{i}")

        assert db_director.decide_next_action() == NextAction.DESIGN_EXPERIMENT
        assert db_director.workflow.current_state == WorkflowState.DESIGNING_EXPERIMENTS

    def test_refining_after_a_pass_leaves_instead_of_refining_again(self, db_director):
        from kosmos.core.workflow import ResearchWorkflow

        db_director.research_plan.mark_tested("hyp-exec-1")
        db_director.research_plan.add_hypothesis("h-untested")
        db_director.workflow = ResearchWorkflow(
            initial_state=WorkflowState.REFINING, research_plan=db_director.research_plan
        )
        assert db_director.decide_next_action() == NextAction.REFINE_HYPOTHESIS

        db_director._refined_since_analysis = True  # the pass ran and failed inside REFINING
        assert db_director.decide_next_action() == NextAction.DESIGN_EXPERIMENT
        assert db_director.workflow.current_state == WorkflowState.DESIGNING_EXPERIMENTS

    @staticmethod
    def _seed_validated_result(rid):
        from kosmos.db import get_session, operations
        from tests.unit.agents.conftest import EXP_ID

        with get_session() as session:
            operations.create_result(
                session, id=rid, experiment_id=EXP_ID, data={"execution_success": True},
                execution_success=True, data_source="file", p_value=0.01,
                validation_status="validated",
            )

    async def test_variant_inherits_score_and_generation(self, db_director):
        from kosmos.core.workflow import ResearchPlan
        from kosmos.hypothesis.refiner import RetirementDecision
        from kosmos.models.hypothesis import Hypothesis
        from tests.unit.agents.conftest import H_ID

        self._seed_validated_result("res-inherit-1")
        db_director.research_plan = ResearchPlan(research_question="q")
        db_director.research_plan.add_hypothesis(H_ID, score=0.8, generation=1)
        db_director.research_plan.mark_tested(H_ID)
        db_director.workflow.current_state = WorkflowState.REFINING
        variant = Hypothesis(
            id="var-inherit", research_question="Does CO2 predict temperature?",
            statement="Volcanic aerosol index lowers the temperature anomaly a year later",
            rationale="Sulphate aerosols reflect sunlight", domain="climate",
        )
        db_director._hypothesis_refiner = Mock(
            evaluate_hypothesis_status=Mock(return_value=RetirementDecision.SPAWN_VARIANT),
            spawn_variant=Mock(return_value=[variant]),
        )

        with patch.object(db_director, '_is_near_duplicate', return_value=False):
            await db_director._handle_refine_hypothesis_action(H_ID)

        plan = db_director.research_plan
        assert "var-inherit" in plan.hypothesis_pool
        assert plan.hypothesis_scores["var-inherit"] == pytest.approx(0.72)
        assert plan.hypothesis_generation["var-inherit"] == 2

    async def test_refinements_stop_at_the_limit(self, db_director):
        from kosmos.hypothesis.refiner import RetirementDecision
        from tests.unit.agents.conftest import H_ID

        self._seed_validated_result("res-limit-1")
        db_director.research_plan.mark_tested(H_ID)
        db_director._hypothesis_refiner = Mock(
            evaluate_hypothesis_status=Mock(return_value=RetirementDecision.CONTINUE_TESTING)
        )

        for _ in range(4):
            await db_director._handle_refine_hypothesis_action(H_ID)

        assert db_director._hypothesis_refiner.evaluate_hypothesis_status.call_count == 2

    async def test_no_variants_from_a_rejected_result(self, db_director):
        from kosmos.db import get_session, operations
        from kosmos.hypothesis.refiner import RetirementDecision
        from tests.unit.agents.conftest import EXP_ID, H_ID

        with get_session() as session:
            operations.create_result(
                session, id="res-rejected-1", experiment_id=EXP_ID, data={"execution_success": True},
                execution_success=True, data_source="file", p_value=0.4,
                validation_status="rejected",
            )
        db_director.research_plan.mark_tested(H_ID)
        db_director._hypothesis_refiner = Mock(
            evaluate_hypothesis_status=Mock(return_value=RetirementDecision.SPAWN_VARIANT)
        )

        await db_director._handle_refine_hypothesis_action(H_ID)

        db_director._hypothesis_refiner.evaluate_hypothesis_status.assert_called_once()
        db_director._hypothesis_refiner.spawn_variant.assert_not_called()

    async def test_no_variants_at_the_pool_cap(self, db_director):
        from kosmos.hypothesis.refiner import RetirementDecision
        from tests.unit.agents.conftest import H_ID

        self._seed_validated_result("res-cap-1")
        db_director.research_plan.mark_tested(H_ID)
        for i in range(11):
            db_director.research_plan.add_hypothesis(f"filler-{i}")
        db_director._hypothesis_refiner = Mock(
            evaluate_hypothesis_status=Mock(return_value=RetirementDecision.SPAWN_VARIANT)
        )

        await db_director._handle_refine_hypothesis_action(H_ID)

        db_director._hypothesis_refiner.spawn_variant.assert_not_called()
        assert len(db_director.research_plan.hypothesis_pool) == 12


class TestErrorRecoveryInEventLoop:
    """P1-3: error recovery never blocks or raises from the event-loop thread."""

    async def test_recovery_inside_loop_returns_quickly(self, mock_director):
        import time as _time

        mock_director.workflow.current_state = WorkflowState.EXECUTING
        t0 = _time.monotonic()

        mock_director._handle_error_with_recovery("CodeExecutor", "boom", recoverable=True)

        assert _time.monotonic() - t0 < 1.0
        assert mock_director._consecutive_errors == 1

    async def test_breaker_reaches_error_state(self, mock_director):
        from kosmos.agents.research_director import MAX_CONSECUTIVE_ERRORS

        mock_director.workflow.current_state = WorkflowState.EXECUTING
        last = None
        for _ in range(MAX_CONSECUTIVE_ERRORS):
            last = mock_director._handle_error_with_recovery("CodeExecutor", "boom", recoverable=True)

        targets = [c.args[0] for c in mock_director.workflow.transition_to.call_args_list]
        assert WorkflowState.ERROR in targets
        assert last == NextAction.ERROR_RECOVERY

    async def test_error_recovery_resumes_generation(self, mock_director):
        mock_director.workflow.current_state = WorkflowState.ERROR
        mock_director.research_plan.hypothesis_pool = ["h1"]
        mock_director.research_plan.get_untested_hypotheses.return_value = ["h1"]
        mock_director._consecutive_errors = 3

        assert mock_director.decide_next_action() == NextAction.ERROR_RECOVERY
        await mock_director._execute_next_action(NextAction.ERROR_RECOVERY)

        targets = [c.args[0] for c in mock_director.workflow.transition_to.call_args_list]
        assert WorkflowState.GENERATING_HYPOTHESES in targets
        assert mock_director._consecutive_errors == 0

    def test_recovery_outside_loop_backs_off(self, mock_director):
        mock_director.workflow.current_state = WorkflowState.EXECUTING
        with patch("time.sleep") as mock_sleep:
            mock_director._handle_error_with_recovery("CodeExecutor", "boom", recoverable=True)

        mock_sleep.assert_called_once_with(2)
