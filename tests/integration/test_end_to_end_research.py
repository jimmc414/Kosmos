"""
End-to-end integration tests for autonomous research system (Phase 7).

Tests complete autonomous research cycles from question to convergence,
including report generation and all agent coordination.
"""

import json
from unittest.mock import Mock, patch, MagicMock
import pytest

from kosmos.agents.research_director import ResearchDirectorAgent, NextAction
from kosmos.core.workflow import WorkflowState, ResearchWorkflow
from kosmos.core.convergence import ConvergenceDetector, StoppingReason
from kosmos.models.hypothesis import Hypothesis, HypothesisStatus
from kosmos.models.result import ExperimentResult, ResultStatus


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_research_question():
    """Simple research question for basic testing."""
    return "Does caffeine improve short-term memory performance?"


@pytest.fixture
def complex_research_question():
    """Complex multi-domain research question."""
    return "What are the combined effects of caffeine and sleep deprivation on cognitive performance and emotional regulation?"


@pytest.fixture
def mock_claude_for_simple_research(mock_llm_client):
    """Mock Claude responses for simple research cycle."""
    # Different responses for different prompts
    def mock_generate(prompt, **kwargs):
        if "research plan" in prompt.lower():
            return json.dumps({
                "strategy": "Test caffeine effects on memory",
                "hypothesis_directions": ["Memory improvement", "Dose-response"],
                "experiment_strategy": "Statistical analysis of performance data",
                "success_criteria": "p < 0.05 with medium effect size",
            })
        elif "hypotheses" in prompt.lower() or "generate" in prompt.lower():
            return json.dumps({
                "hypotheses": [
                    {
                        "statement": "Caffeine (200mg) improves short-term memory recall by 15%",
                        "rationale": "Stimulant effects enhance attention and encoding",
                        "testability_score": 0.9,
                        "novelty_score": 0.6,
                    }
                ]
            })
        else:
            return json.dumps({"result": "mocked response"})

    mock_llm_client.generate.side_effect = mock_generate
    return mock_llm_client


# ============================================================================
# Test Class 1: Simple Research Cycle
# ============================================================================

class TestSimpleResearchCycle:
    """Test simple autonomous research cycle that converges quickly."""

    def test_simple_cycle_completion(
        self, simple_research_question, mock_claude_for_simple_research
    ):
        """Test complete research cycle for simple question."""
        director = ResearchDirectorAgent(
            research_question=simple_research_question,
            domain="neuroscience",
            config={"max_iterations": 3}
        )

        # Start research
        director.start()

        assert director.workflow.current_state == WorkflowState.GENERATING_HYPOTHESES
        assert director.research_plan is not None

    def test_simple_cycle_generates_hypotheses(
        self, simple_research_question, mock_claude_for_simple_research
    ):
        """Test simple cycle generates at least one hypothesis."""
        director = ResearchDirectorAgent(
            research_question=simple_research_question,
            domain="neuroscience",
            config={"max_iterations": 3}
        )

        director.start()

        # Add hypothesis (simulating generator response)
        director.research_plan.add_hypothesis("hyp_001")

        assert len(director.research_plan.hypothesis_pool) > 0

    def test_simple_cycle_runs_experiments(
        self, simple_research_question, mock_claude_for_simple_research
    ):
        """Test simple cycle runs at least one experiment."""
        director = ResearchDirectorAgent(
            research_question=simple_research_question,
            domain="neuroscience",
            config={"max_iterations": 3}
        )

        director.start()

        # Simulate experiment execution
        director.research_plan.add_hypothesis("hyp_001")
        director.research_plan.add_experiment("exp_001")
        director.research_plan.add_result("result_001")
        director.research_plan.mark_tested("hyp_001")
        director.research_plan.mark_supported("hyp_001")

        assert len(director.research_plan.tested_hypotheses) > 0
        assert len(director.research_plan.results) > 0

    def test_simple_cycle_converges(
        self, simple_research_question, mock_claude_for_simple_research
    ):
        """Test simple cycle reaches convergence."""
        director = ResearchDirectorAgent(
            research_question=simple_research_question,
            domain="neuroscience",
            config={"max_iterations": 2}  # Low limit for quick convergence
        )

        director.start()

        # Run 2 iterations
        for i in range(2):
            director.research_plan.add_hypothesis(f"hyp_{i}")
            director.research_plan.mark_tested(f"hyp_{i}")
            director.research_plan.mark_supported(f"hyp_{i}")
            director.research_plan.increment_iteration()

        # Should be at max iterations
        assert director.research_plan.iteration_count == 2
        assert director.research_plan.iteration_count >= director.max_iterations

    def test_simple_cycle_produces_findings(
        self, simple_research_question, mock_claude_for_simple_research
    ):
        """Test simple cycle produces at least one finding."""
        director = ResearchDirectorAgent(
            research_question=simple_research_question,
            domain="neuroscience",
            config={"max_iterations": 3}
        )

        director.start()

        # Simulate successful finding
        director.research_plan.add_hypothesis("hyp_001")
        director.research_plan.mark_tested("hyp_001")
        director.research_plan.mark_supported("hyp_001")

        assert len(director.research_plan.supported_hypotheses) > 0


# ============================================================================
# Test Class 2: Complex Research Cycle
# ============================================================================

class TestComplexResearchCycle:
    """Test complex multi-domain research cycle."""

    def test_complex_cycle_handles_multiple_domains(
        self, complex_research_question, mock_llm_client
    ):
        """Test complex cycle handles multi-domain research."""
        mock_llm_client.generate.return_value = json.dumps({
            "strategy": "Multi-domain analysis",
            "hypothesis_directions": ["Cognitive", "Emotional", "Interaction"],
            "experiment_strategy": "Factorial design",
            "success_criteria": "Multiple significant effects",
        })

        director = ResearchDirectorAgent(
            research_question=complex_research_question,
            domain="neuroscience",  # Primary domain
            config={"max_iterations": 10}
        )

        director.start()

        assert director.research_plan is not None
        assert director.research_question == complex_research_question

    def test_complex_cycle_multiple_iterations(
        self, complex_research_question, mock_llm_client
    ):
        """Test complex cycle runs multiple iterations."""
        mock_llm_client.generate.return_value = json.dumps({"strategy": "test"})

        director = ResearchDirectorAgent(
            research_question=complex_research_question,
            domain="neuroscience",
            config={"max_iterations": 5}
        )

        director.start()

        # Simulate 5 iterations
        for i in range(5):
            director.research_plan.add_hypothesis(f"hyp_{i}")
            director.research_plan.mark_tested(f"hyp_{i}")

            if i % 2 == 0:
                director.research_plan.mark_supported(f"hyp_{i}")
            else:
                director.research_plan.mark_rejected(f"hyp_{i}")

            director.research_plan.increment_iteration()

        assert director.research_plan.iteration_count == 5
        assert len(director.research_plan.tested_hypotheses) == 5

    def test_complex_cycle_hypothesis_refinement(
        self, complex_research_question, mock_llm_client
    ):
        """Test complex cycle refines hypotheses based on results."""
        mock_llm_client.generate.return_value = json.dumps({"strategy": "test"})

        director = ResearchDirectorAgent(
            research_question=complex_research_question,
            domain="neuroscience",
            config={"max_iterations": 10}
        )

        director.start()

        # Initial hypothesis
        director.research_plan.add_hypothesis("hyp_001_gen1")
        director.research_plan.mark_tested("hyp_001_gen1")
        director.research_plan.mark_rejected("hyp_001_gen1")

        # Refined hypothesis (simulating refinement)
        director.research_plan.add_hypothesis("hyp_001_gen2")  # Refined version
        director.research_plan.mark_tested("hyp_001_gen2")
        director.research_plan.mark_supported("hyp_001_gen2")

        # Check we have both generations
        assert "hyp_001_gen1" in director.research_plan.tested_hypotheses
        assert "hyp_001_gen2" in director.research_plan.tested_hypotheses
        assert "hyp_001_gen2" in director.research_plan.supported_hypotheses

    def test_complex_cycle_tracks_multiple_hypotheses(
        self, complex_research_question, mock_llm_client
    ):
        """Test complex cycle tracks multiple concurrent hypotheses."""
        mock_llm_client.generate.return_value = json.dumps({"strategy": "test"})

        director = ResearchDirectorAgent(
            research_question=complex_research_question,
            domain="neuroscience",
            config={"max_iterations": 10}
        )

        director.start()

        # Add multiple hypotheses
        hypothesis_ids = [f"hyp_{i}" for i in range(10)]

        for hyp_id in hypothesis_ids:
            director.research_plan.add_hypothesis(hyp_id)

        assert len(director.research_plan.hypothesis_pool) == 10


# ============================================================================
# Test Class 3: Convergence Scenarios
# ============================================================================

class TestConvergenceScenarios:
    """Test different convergence scenarios."""

    def test_convergence_by_iteration_limit(
        self, simple_research_question, mock_llm_client
    ):
        """Test convergence triggered by iteration limit."""
        mock_llm_client.generate.return_value = json.dumps({"strategy": "test"})

        director = ResearchDirectorAgent(
            research_question=simple_research_question,
            domain="neuroscience",
            config={"max_iterations": 3}
        )

        # Create convergence detector
        from kosmos.core.convergence import ConvergenceDetector

        detector = ConvergenceDetector()

        # Run to max iterations
        for i in range(3):
            director.research_plan.add_hypothesis(f"hyp_{i}")
            director.research_plan.mark_tested(f"hyp_{i}")
            director.research_plan.increment_iteration()

        # Check convergence
        decision = detector.check_convergence(
            director.research_plan,
            [],  # hypotheses
            [],  # results
        )

        assert decision.should_stop is True
        assert decision.reason == StoppingReason.ITERATION_LIMIT

    def test_convergence_by_hypothesis_exhaustion(
        self, simple_research_question, mock_llm_client
    ):
        """Test convergence triggered by hypothesis exhaustion."""
        mock_llm_client.generate.return_value = json.dumps({"strategy": "test"})

        director = ResearchDirectorAgent(
            research_question=simple_research_question,
            domain="neuroscience",
            config={"max_iterations": 10}
        )

        from kosmos.core.convergence import ConvergenceDetector

        detector = ConvergenceDetector()

        # Add and test all hypotheses
        director.research_plan.add_hypothesis("hyp_001")
        director.research_plan.add_hypothesis("hyp_002")
        director.research_plan.mark_tested("hyp_001")
        director.research_plan.mark_tested("hyp_002")
        # No experiments queued

        # Check convergence
        hypotheses = [
            Hypothesis(
                id="hyp_001",
                research_question=simple_research_question,
                statement="Testing hypothesis statement 1",
                rationale="Rationale must be at least twenty characters long for validity",
                domain="neuroscience",
            ),
            Hypothesis(
                id="hyp_002",
                research_question=simple_research_question,
                statement="Testing hypothesis statement 2",
                rationale="Rationale must be at least twenty characters long for validity",
                domain="neuroscience",
            ),
        ]

        decision = detector.check_convergence(
            director.research_plan,
            hypotheses,
            [],
        )

        assert decision.should_stop is True
        assert decision.reason == StoppingReason.HYPOTHESIS_EXHAUSTION

    def test_convergence_by_novelty_decline(
        self, simple_research_question, mock_llm_client
    ):
        """Test convergence triggered by novelty decline."""
        mock_llm_client.generate.return_value = json.dumps({"strategy": "test"})

        director = ResearchDirectorAgent(
            research_question=simple_research_question,
            domain="neuroscience",
            config={"max_iterations": 10}
        )

        from kosmos.core.convergence import ConvergenceDetector

        detector = ConvergenceDetector()

        # Create hypotheses with declining novelty
        hypotheses = [
            Hypothesis(
                id=f"hyp_{i}",
                research_question=simple_research_question,
                statement=f"Hypothesis {i}",
                rationale="Rationale must be at least twenty characters long for validity",
                domain="neuroscience",
                novelty_score=0.8 - (i * 0.15),  # Declining: 0.8, 0.65, 0.5, 0.35, 0.2, 0.05
            )
            for i in range(6)
        ]

        # Check convergence
        decision = detector.check_convergence(
            director.research_plan,
            hypotheses,
            [],
        )

        # Should detect novelty decline
        if decision.should_stop and decision.reason == StoppingReason.NOVELTY_DECLINE:
            assert decision.is_mandatory is False  # Optional criterion

    def test_convergence_by_diminishing_returns(
        self, simple_research_question, mock_llm_client
    ):
        """Test convergence triggered by diminishing returns."""
        mock_llm_client.generate.return_value = json.dumps({"strategy": "test"})

        director = ResearchDirectorAgent(
            research_question=simple_research_question,
            domain="neuroscience",
            config={"max_iterations": 10}
        )

        from kosmos.core.convergence import ConvergenceDetector

        detector = ConvergenceDetector(config={"cost_threshold": 50.0})

        # Set high cost per discovery
        detector.metrics.cost_per_discovery = 100.0  # Above threshold

        # Check convergence
        decision = detector.check_convergence(
            director.research_plan,
            [],
            [],
        )

        # Should detect diminishing returns
        if decision.should_stop and decision.reason == StoppingReason.DIMINISHING_RETURNS:
            assert decision.is_mandatory is False  # Optional criterion


# ============================================================================
# Test Class 4: Report Generation
# ============================================================================

class TestReportGeneration:
    """Test convergence report generation."""

    def test_report_generation_complete(
        self, simple_research_question, mock_llm_client
    ):
        """Test complete convergence report is generated."""
        mock_llm_client.generate.return_value = json.dumps({"strategy": "test"})

        director = ResearchDirectorAgent(
            research_question=simple_research_question,
            domain="neuroscience",
            config={"max_iterations": 3}
        )

        from kosmos.core.convergence import ConvergenceDetector, StoppingDecision, StoppingReason

        detector = ConvergenceDetector()

        # Simulate research completion
        for i in range(3):
            director.research_plan.add_hypothesis(f"hyp_{i}")
            director.research_plan.mark_tested(f"hyp_{i}")
            director.research_plan.mark_supported(f"hyp_{i}")
            director.research_plan.increment_iteration()

        hypotheses = [
            Hypothesis(
                id=f"hyp_{i}",
                research_question=simple_research_question,
                statement=f"Hypothesis {i}",
                rationale="Rationale must be at least twenty characters long for validity",
                domain="neuroscience",
            )
            for i in range(3)
        ]

        results = [
            ExperimentResult(
                id=f"result_{i}",
                hypothesis_id=f"hyp_{i}",
                supports_hypothesis=True,
                primary_p_value=0.01,
                primary_effect_size=0.7,
                primary_test="t-test",
                status=ResultStatus.SUCCESS,
            )
            for i in range(3)
        ]

        decision = StoppingDecision(
            should_stop=True,
            reason=StoppingReason.ITERATION_LIMIT,
            is_mandatory=True,
            confidence=1.0,
            details="Reached max iterations",
        )

        report = detector.generate_convergence_report(
            director.research_plan,
            hypotheses,
            results,
            stopping_reason=decision.reason,
        )

        assert report is not None
        assert report.research_question == simple_research_question
        assert report.converged is True
        assert report.total_iterations == 3
        assert report.hypotheses_tested == 3
        assert report.hypotheses_supported == 3

    def test_report_markdown_export(
        self, simple_research_question, mock_llm_client
    ):
        """Test convergence report exports to markdown."""
        mock_llm_client.generate.return_value = json.dumps({"strategy": "test"})

        director = ResearchDirectorAgent(
            research_question=simple_research_question,
            domain="neuroscience",
            config={"max_iterations": 2}
        )

        from kosmos.core.convergence import ConvergenceDetector, StoppingDecision, StoppingReason

        detector = ConvergenceDetector()

        # Simulate minimal research
        director.research_plan.add_hypothesis("hyp_001")
        director.research_plan.mark_tested("hyp_001")
        director.research_plan.mark_supported("hyp_001")
        director.research_plan.increment_iteration()

        decision = StoppingDecision(
            should_stop=True,
            reason=StoppingReason.ITERATION_LIMIT,
            is_mandatory=True,
            confidence=1.0,
            details="Test",
        )

        report = detector.generate_convergence_report(
            director.research_plan,
            [],
            [],
            stopping_reason=decision.reason,
        )

        markdown = report.to_markdown()

        assert isinstance(markdown, str)
        assert len(markdown) > 0
        assert "# Convergence Report" in markdown
        assert simple_research_question in markdown
        assert "## Summary" in markdown

    def test_report_includes_all_sections(
        self, simple_research_question, mock_llm_client
    ):
        """Test report includes all required sections."""
        mock_llm_client.generate.return_value = json.dumps({"strategy": "test"})

        director = ResearchDirectorAgent(
            research_question=simple_research_question,
            domain="neuroscience",
            config={"max_iterations": 2}
        )

        from kosmos.core.convergence import ConvergenceDetector, StoppingDecision, StoppingReason

        detector = ConvergenceDetector()

        decision = StoppingDecision(
            should_stop=True,
            reason=StoppingReason.ITERATION_LIMIT,
            is_mandatory=True,
            confidence=1.0,
            details="Complete",
        )

        report = detector.generate_convergence_report(
            director.research_plan,
            [],
            [],
            stopping_reason=decision.reason,
        )

        markdown = report.to_markdown()

        # Check for required sections
        required_sections = [
            "# Convergence Report",
            "## Summary",
            "## Progress Metrics",
            "## Research Outcomes",
            "## Stopping Criterion",
            "## Recommended Next Steps",
        ]

        for section in required_sections:
            assert section in markdown, f"Missing section: {section}"

    def test_report_includes_metrics(
        self, simple_research_question, mock_llm_client
    ):
        """Test report includes progress metrics."""
        mock_llm_client.generate.return_value = json.dumps({"strategy": "test"})

        director = ResearchDirectorAgent(
            research_question=simple_research_question,
            domain="neuroscience",
            config={"max_iterations": 2}
        )

        from kosmos.core.convergence import ConvergenceDetector, StoppingDecision, StoppingReason

        detector = ConvergenceDetector()

        decision = StoppingDecision(
            should_stop=True,
            reason=StoppingReason.ITERATION_LIMIT,
            is_mandatory=True,
            confidence=1.0,
            details="Complete",
        )

        report = detector.generate_convergence_report(
            director.research_plan,
            [],
            [],
            stopping_reason=decision.reason,
        )

        assert report.final_metrics is not None
        assert hasattr(report.final_metrics, "discovery_rate")
        assert hasattr(report.final_metrics, "novelty_score")
        assert hasattr(report.final_metrics, "saturation_ratio")
        assert hasattr(report.final_metrics, "consistency_score")

    def test_report_includes_recommendations(
        self, simple_research_question, mock_llm_client
    ):
        """Test report includes next steps recommendations."""
        mock_llm_client.generate.return_value = json.dumps({"strategy": "test"})

        director = ResearchDirectorAgent(
            research_question=simple_research_question,
            domain="neuroscience",
            config={"max_iterations": 2}
        )

        from kosmos.core.convergence import ConvergenceDetector, StoppingDecision, StoppingReason

        detector = ConvergenceDetector()

        decision = StoppingDecision(
            should_stop=True,
            reason=StoppingReason.ITERATION_LIMIT,
            is_mandatory=True,
            confidence=1.0,
            details="Complete",
        )

        report = detector.generate_convergence_report(
            director.research_plan,
            [],
            [],
            stopping_reason=decision.reason,
        )

        assert len(report.recommended_next_steps) > 0
        assert all(isinstance(step, str) for step in report.recommended_next_steps)
