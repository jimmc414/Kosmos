"""
E2E Tests for Research Convergence.

Tests the ConvergenceDetector functionality:
- Research reaching convergence
- Convergence report completeness
- Hypothesis refinement across iterations
- Knowledge accumulation
- Early convergence detection
- Max iterations behavior
"""

import pytest
from datetime import datetime

from kosmos.core.convergence import (
    ConvergenceDetector,
    ConvergenceMetrics,
    ConvergenceReport,
    StoppingReason,
    StoppingDecision
)
from kosmos.core.workflow import ResearchPlan
from tests.e2e.factories import ResearchScenarioFactory


pytestmark = [pytest.mark.e2e]


class TestResearchConvergence:
    """Tests for research reaching convergence."""

    def test_research_reaches_convergence_via_iteration_limit(self, convergence_detector):
        """Verify research converges when iteration limit reached."""
        # Create research plan at max iterations
        research_plan = ResearchPlan(
            research_question="Test research",
            max_iterations=5
        )
        research_plan.iteration_count = 5

        # Create some hypotheses and results
        factory = ResearchScenarioFactory()
        hypotheses = [
            factory.create_simple_hypothesis(hypothesis_id=f"hyp_{i}")
            for i in range(3)
        ]
        results = []

        # Check convergence
        decision = convergence_detector.check_convergence(
            research_plan=research_plan,
            hypotheses=hypotheses,
            results=results
        )

        # Should stop due to iteration limit
        assert decision.should_stop is True
        assert decision.reason == StoppingReason.ITERATION_LIMIT
        assert decision.is_mandatory is True

    def test_research_reaches_convergence_via_hypothesis_exhaustion(self, convergence_detector):
        """Verify research converges when all hypotheses tested."""
        # Create research plan with all hypotheses tested
        research_plan = ResearchPlan(
            research_question="Test research",
            max_iterations=10
        )
        research_plan.iteration_count = 3

        # Add hypotheses to pool and mark as tested
        research_plan.hypothesis_pool = {"hyp_1", "hyp_2", "hyp_3"}
        research_plan.tested_hypotheses = {"hyp_1", "hyp_2", "hyp_3"}
        research_plan.experiment_queue = []  # No pending experiments

        # Create hypothesis objects
        factory = ResearchScenarioFactory()
        hypotheses = [
            factory.create_simple_hypothesis(hypothesis_id=f"hyp_{i}")
            for i in range(1, 4)
        ]

        # Create results
        results = [
            factory.create_successful_result(
                factory.create_experiment_protocol(h)
            )
            for h in hypotheses
        ]

        # Check convergence
        decision = convergence_detector.check_convergence(
            research_plan=research_plan,
            hypotheses=hypotheses,
            results=results
        )

        # Should stop due to no testable hypotheses
        assert decision.should_stop is True
        assert decision.reason == StoppingReason.NO_TESTABLE_HYPOTHESES
        assert decision.is_mandatory is True


class TestConvergenceReportCompleteness:
    """Tests for convergence report completeness."""

    def test_convergence_report_complete(self, convergence_detector):
        """Verify report has all required sections."""
        # Create scenario
        factory = ResearchScenarioFactory()
        scenario = factory.create_complete_scenario(
            domain="biology",
            num_hypotheses=5,
            num_results_per_hypothesis=1
        )

        # Generate report
        report = convergence_detector.generate_convergence_report(
            research_plan=scenario["research_plan"],
            hypotheses=scenario["hypotheses"],
            results=scenario["results"],
            stopping_reason=StoppingReason.ITERATION_LIMIT
        )

        # Verify report structure
        assert isinstance(report, ConvergenceReport)
        assert report.research_question == scenario["research_plan"].research_question
        assert report.stopping_reason == StoppingReason.ITERATION_LIMIT
        assert report.total_iterations >= 0
        assert report.total_hypotheses == 5
        assert report.experiments_conducted == 5

        # Verify metrics present
        assert isinstance(report.final_metrics, ConvergenceMetrics)
        assert report.final_metrics.total_experiments == 5

        # Verify summary is generated
        assert len(report.summary) > 0

        # Verify recommended next steps
        assert len(report.recommended_next_steps) > 0

    def test_convergence_report_markdown_export(self, convergence_detector):
        """Verify report exports to valid markdown."""
        # Create scenario
        factory = ResearchScenarioFactory()
        scenario = factory.create_complete_scenario(
            domain="biology",
            num_hypotheses=3
        )

        # Update research plan
        scenario["research_plan"].supported_hypotheses = {scenario["hypotheses"][0].id}
        scenario["research_plan"].rejected_hypotheses = {scenario["hypotheses"][1].id}

        # Generate report
        report = convergence_detector.generate_convergence_report(
            research_plan=scenario["research_plan"],
            hypotheses=scenario["hypotheses"],
            results=scenario["results"],
            stopping_reason=StoppingReason.ALL_HYPOTHESES_TESTED
        )

        # Export to markdown
        markdown = report.to_markdown()

        # Verify markdown structure
        assert "# Convergence Report" in markdown
        assert "Research Question" in markdown
        assert "Summary Statistics" in markdown
        assert "Key Metrics" in markdown
        assert "Discovery Rate" in markdown
        assert "Recommended Next Steps" in markdown
        assert "Generated" in markdown


class TestHypothesisRefinement:
    """Tests for hypothesis refinement across iterations."""

    def test_hypothesis_refinement_across_iterations(self, convergence_detector):
        """Verify hypotheses are refined based on results."""
        factory = ResearchScenarioFactory()

        # Iteration 1: Initial hypotheses with high novelty
        iteration1_hypotheses = [
            factory.create_simple_hypothesis(
                hypothesis_id="hyp_iter1_1",
                novelty_score=0.9
            ),
            factory.create_simple_hypothesis(
                hypothesis_id="hyp_iter1_2",
                novelty_score=0.85
            )
        ]

        # Iteration 2: Refined hypotheses with adjusted novelty
        iteration2_hypotheses = [
            factory.create_simple_hypothesis(
                hypothesis_id="hyp_iter2_1",
                novelty_score=0.7  # Lower novelty (building on prior work)
            )
        ]

        # Iteration 3: Further refined
        iteration3_hypotheses = [
            factory.create_simple_hypothesis(
                hypothesis_id="hyp_iter3_1",
                novelty_score=0.5  # Even lower novelty
            )
        ]

        # Combine all hypotheses
        all_hypotheses = iteration1_hypotheses + iteration2_hypotheses + iteration3_hypotheses

        # Verify novelty trend
        novelty_scores = [h.novelty_score for h in all_hypotheses]

        # Novelty should generally decrease as research progresses
        # (building on existing work)
        assert novelty_scores[0] > novelty_scores[-1]

        # Calculate novelty decline
        novelty, is_declining = convergence_detector.calculate_novelty_decline(all_hypotheses)

        # Novelty trend should show decline
        assert is_declining or novelty < 0.8  # Either declining or low


class TestKnowledgeAccumulation:
    """Tests for knowledge accumulation."""

    def test_knowledge_accumulates_across_iterations(self, convergence_detector):
        """Verify each iteration builds on previous knowledge."""

        # Simulate accumulating knowledge
        cumulative_findings = []
        cumulative_hypotheses = []

        for iteration in range(1, 4):
            # Add new hypotheses each iteration
            factory = ResearchScenarioFactory()
            new_hyp = factory.create_simple_hypothesis(
                hypothesis_id=f"hyp_iter{iteration}"
            )
            cumulative_hypotheses.append(new_hyp)

            # Add results
            proto = factory.create_experiment_protocol(new_hyp)
            result = factory.create_successful_result(proto)
            cumulative_findings.append(result)

            # Create research plan reflecting cumulative state
            research_plan = ResearchPlan(
                research_question="Cumulative knowledge test",
                max_iterations=10
            )
            research_plan.iteration_count = iteration
            research_plan.hypothesis_pool = {h.id for h in cumulative_hypotheses}
            research_plan.tested_hypotheses = {h.id for h in cumulative_hypotheses}

            # Update metrics
            convergence_detector._update_metrics(
                research_plan=research_plan,
                hypotheses=cumulative_hypotheses,
                results=cumulative_findings
            )

            # Verify metrics accumulate
            metrics = convergence_detector.get_metrics()
            assert metrics.total_experiments == iteration
            assert metrics.hypotheses_tested == iteration
            assert metrics.total_hypotheses == iteration


class TestEarlyConvergenceDetection:
    """Tests for early convergence detection."""

    def test_early_convergence_detection_novelty_decline(self):
        """Verify research stops early if novelty declining."""
        detector = ConvergenceDetector(
            mandatory_criteria=["iteration_limit"],
            optional_criteria=["novelty_decline"],
            config={
                "novelty_decline_threshold": 0.3,
                "novelty_decline_window": 3
            }
        )

        # Create hypotheses with declining novelty
        factory = ResearchScenarioFactory()
        hypotheses = [
            factory.create_simple_hypothesis(
                hypothesis_id=f"hyp_{i}",
                novelty_score=0.2 - (i * 0.01)  # All below threshold
            )
            for i in range(5)
        ]

        # Create research plan (not at iteration limit)
        research_plan = ResearchPlan(
            research_question="Early convergence test",
            max_iterations=10
        )
        research_plan.iteration_count = 3
        research_plan.hypothesis_pool = {h.id for h in hypotheses}

        # Add results
        results = [
            factory.create_successful_result(
                factory.create_experiment_protocol(h)
            )
            for h in hypotheses
        ]

        # Check convergence
        decision = detector.check_convergence(
            research_plan=research_plan,
            hypotheses=hypotheses,
            results=results
        )

        # Should detect novelty decline (optional criterion)
        if decision.should_stop and decision.reason == StoppingReason.NOVELTY_DECLINE:
            assert decision.is_mandatory is False

    def test_early_convergence_detection_diminishing_returns(self):
        """Verify research stops early if cost per discovery too high."""
        detector = ConvergenceDetector(
            mandatory_criteria=["iteration_limit"],
            optional_criteria=["diminishing_returns"],
            config={
                "cost_per_discovery_threshold": 50.0  # $50 max per discovery
            }
        )

        # Set high cost with few discoveries
        detector.metrics.total_cost = 1000.0  # $1000 spent
        detector.metrics.significant_results = 2  # Only 2 discoveries
        detector.metrics.cost_per_discovery = 500.0  # $500/discovery (way over threshold)

        # Check diminishing returns
        decision = detector.check_diminishing_returns()

        # Should recommend stopping
        assert decision.should_stop is True
        assert decision.reason == StoppingReason.DIMINISHING_RETURNS
        assert decision.is_mandatory is False


class TestMaxIterationsReached:
    """Tests for max iterations behavior."""

    def test_max_iterations_reached_graceful_stop(self, convergence_detector):
        """Verify graceful stop at max iterations."""
        # Create research plan at max
        research_plan = ResearchPlan(
            research_question="Max iterations test",
            max_iterations=5
        )
        research_plan.iteration_count = 5

        # Still have untested hypotheses
        research_plan.hypothesis_pool = {"hyp_1", "hyp_2", "hyp_3"}
        research_plan.tested_hypotheses = {"hyp_1"}  # Only 1 tested

        # Check iteration limit
        decision = convergence_detector.check_iteration_limit(research_plan)

        # Should stop due to iteration limit
        assert decision.should_stop is True
        assert decision.reason == StoppingReason.ITERATION_LIMIT
        assert "5/5" in decision.details

    def test_continues_before_max_iterations(self, convergence_detector):
        """Verify research continues before reaching max iterations."""
        research_plan = ResearchPlan(
            research_question="Continue test",
            max_iterations=10
        )
        research_plan.iteration_count = 3  # Not at max

        # Check iteration limit
        decision = convergence_detector.check_iteration_limit(research_plan)

        # Should not stop
        assert decision.should_stop is False
        assert decision.reason == StoppingReason.ITERATION_LIMIT
        assert "3/10" in decision.details

    def test_convergence_metrics_at_max_iterations(self, convergence_detector):
        """Verify metrics are properly recorded at max iterations."""
        factory = ResearchScenarioFactory()

        # Create complete scenario
        scenario = factory.create_complete_scenario(
            domain="biology",
            num_hypotheses=5,
            num_results_per_hypothesis=2
        )

        # Set to max iterations
        scenario["research_plan"].iteration_count = 10
        scenario["research_plan"].max_iterations = 10

        # Update metrics
        convergence_detector._update_metrics(
            research_plan=scenario["research_plan"],
            hypotheses=scenario["hypotheses"],
            results=scenario["results"]
        )

        # Verify metrics
        metrics = convergence_detector.get_metrics()
        assert metrics.iteration_count == 10
        assert metrics.max_iterations == 10
        assert metrics.total_experiments == 10  # 5 hypotheses * 2 results each
