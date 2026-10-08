"""
Unit tests for hypothesis-pool control in ResearchDirectorAgent (viability plan P2-7).

The director is driven through decide_next_action and _do_execute_action for 60
actions: the generator returns 3 new hypotheses per call, design and execution
complete one experiment per DESIGN, analysis runs the real handler with a mocked
analyst and a fixed validation outcome, and refinement runs the real handler with
a mocked refiner whose variants restate existing hypotheses.
"""

from unittest.mock import Mock, patch

import pytest

from kosmos.agents.data_analyst import ResultInterpretation
from kosmos.core.workflow import ResearchPlan, ResearchWorkflow, WorkflowState
from kosmos.db import get_session
from kosmos.db import operations
from kosmos.hypothesis.refiner import RetirementDecision
from kosmos.models.hypothesis import Hypothesis

DOMAIN = "climate"
QUESTION = "Does CO2 predict temperature?"
TOPICS = [
    "solar irradiance", "volcanic aerosol", "ocean heat content", "arctic sea ice",
    "methane concentration", "el nino index", "urban heat island", "cloud cover",
    "snow albedo", "forest cover", "dust storms", "jet stream latitude",
    "soil moisture", "ozone column", "river discharge", "glacier mass",
    "permafrost thaw", "monsoon rainfall", "wildfire area", "sea level",
]


def _interpretation():
    return ResultInterpretation(
        experiment_id="exp", hypothesis_supported=None, confidence=0.5,
        summary="S", key_findings=[], significance_interpretation="",
        biological_significance=None, comparison_to_prior_work=None,
        potential_confounds=[], follow_up_experiments=[], anomalies_detected=[],
        patterns_detected=[], overall_assessment="",
    )


def _hypothesis(hid, statement, testability=0.6, novelty=0.6):
    return Hypothesis(
        id=hid, research_question=QUESTION, statement=statement,
        rationale="Physical mechanism links the driver to the temperature record",
        domain=DOMAIN, testability_score=testability, novelty_score=novelty,
    )


def _store(hyp):
    with get_session() as session:
        operations.create_hypothesis(
            session, id=hyp.id, research_question=hyp.research_question,
            statement=hyp.statement, rationale=hyp.rationale, domain=hyp.domain,
            novelty_score=hyp.novelty_score, testability_score=hyp.testability_score,
        )


class _Harness:
    """Mocks for every agent the loop calls; counts what they were asked to do."""

    def __init__(self, director, validation_status):
        self.d = director
        self.prefix = validation_status
        self.generated = []
        self.experiments = 0
        director.research_plan = ResearchPlan(research_question=QUESTION, max_iterations=20)
        director.workflow = ResearchWorkflow(
            initial_state=WorkflowState.GENERATING_HYPOTHESES,
            research_plan=director.research_plan,
        )
        director.research_plan.experiment_queue.clear()

        director._hypothesis_agent = Mock(generate_hypotheses=Mock(side_effect=self._generate))
        director._data_analyst = Mock(interpret_results=Mock(return_value=_interpretation()))
        director._hypothesis_refiner = Mock(
            evaluate_hypothesis_status=Mock(return_value=RetirementDecision.SPAWN_VARIANT),
            spawn_variant=Mock(side_effect=self._variants),
        )
        director._handle_design_experiment_action = self._design
        director._handle_execute_experiment_action = self._execute
        director._validate_result = Mock(return_value=(validation_status, {"reason": "test"}))

    def _generate(self, **kwargs):
        batch = []
        for _ in range(3):  # 3 new ids per call, whatever was asked for
            n = len(self.generated)
            hyp = _hypothesis(
                f"{self.prefix}-gen-{n}", f"{TOPICS[n]} predicts the global temperature anomaly"
            )
            _store(hyp)
            self.generated.append(hyp)
            batch.append(hyp)
        return Mock(hypotheses=batch)

    def _variants(self, parent, result, num_variants=2):
        # Two variants whose statements restate hypotheses already in the pool
        return [
            _hypothesis(f"var-{parent.id}-{i}", self.generated[i].statement)
            for i in range(2)
        ]

    async def _design(self, hypothesis_id):
        self.experiments += 1
        pid = f"{self.prefix}-exp-{self.experiments}"
        with get_session() as session:
            operations.create_experiment(
                session, id=pid, hypothesis_id=hypothesis_id, experiment_type="computational",
                description="d", protocol={"name": "Correlation"}, domain=DOMAIN,
            )
        self.d.research_plan.add_experiment(pid)
        self.d.workflow.transition_to(WorkflowState.EXECUTING, action=f"Designed {pid}")

    async def _execute(self, protocol_id):
        rid = f"res-{protocol_id}"
        with get_session() as session:
            operations.create_result(
                session, id=rid, experiment_id=protocol_id,
                data={"execution_success": True}, execution_success=True,
                data_source="file", p_value=0.01,
            )
        self.d.research_plan.mark_experiment_complete(protocol_id)
        self.d.research_plan.add_result(rid)
        self.d.workflow.transition_to(WorkflowState.ANALYZING, action=f"Analyze {rid}")

    async def drive(self, actions=60):
        """Run the loop; return the largest pool size seen."""
        peak = 0
        for _ in range(actions):
            if self.d.research_plan.has_converged:
                break
            await self.d._do_execute_action(self.d.decide_next_action())
            peak = max(peak, len(self.d.research_plan.hypothesis_pool))
        return peak


@pytest.fixture
def offline_novelty():
    """The real NoveltyChecker on its TF-IDF fallback, with no literature or vector search."""
    from kosmos.knowledge.embeddings import reset_embedder

    reset_embedder()
    with patch('kosmos.knowledge.embeddings.HAS_SENTENCE_TRANSFORMERS', False), \
         patch('kosmos.hypothesis.novelty_checker.get_vector_db', return_value=None), \
         patch('kosmos.hypothesis.novelty_checker.UnifiedLiteratureSearch') as mock_search:
        mock_search.return_value.search.return_value = []
        yield
    reset_embedder()


async def test_pool_stays_bounded_over_60_actions(db_director, offline_novelty):
    for validation_status in ("unvalidated", "validated"):
        harness = _Harness(db_director, validation_status)
        db_director._variants_dropped_duplicate = 0

        peak = await harness.drive(60)

        plan = db_director.research_plan
        refiner = db_director._hypothesis_refiner
        status = db_director.get_research_status()
        assert peak <= 12, validation_status
        assert len(plan.tested_hypotheses) >= 8, validation_status
        assert status["untested_backlog"] == len(plan.get_untested_hypotheses())
        assert status["untestable"] == 0
        if validation_status == "unvalidated":
            # Nothing can inform a refinement: no evaluation (LLM) call and no variants
            assert refiner.spawn_variant.call_count == 0
            assert refiner.evaluate_hypothesis_status.call_count == 0
            assert status["variants_dropped_duplicate"] == 0
        else:
            assert refiner.spawn_variant.call_count >= 1
            assert refiner.spawn_variant.call_args.kwargs["num_variants"] == 1
            assert status["variants_dropped_duplicate"] >= 1
            assert not any("-var-" in h or h.startswith("var-") for h in plan.hypothesis_pool)


def test_untested_hypotheses_ordered_by_score():
    plan = ResearchPlan(research_question=QUESTION)
    plan.add_hypothesis("low", score=0.3)
    plan.add_hypothesis("unscored")
    plan.add_hypothesis("high", score=0.9)
    plan.add_hypothesis("tie", score=0.9)
    plan.add_hypothesis("bound-to-nothing", score=1.0)
    plan.mark_untestable("bound-to-nothing")
    plan.add_hypothesis("done", score=0.95)
    plan.mark_tested("done")

    assert plan.get_untested_hypotheses() == ["high", "tie", "low", "unscored"]
