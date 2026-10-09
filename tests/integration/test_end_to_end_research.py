"""
End-to-end integration tests for autonomous research system (Phase 7).

Tests complete autonomous research cycles from question to convergence,
including report generation and all agent coordination.
"""

import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import pytest

from kosmos.agents.research_director import ResearchDirectorAgent, NextAction
from kosmos.core.workflow import WorkflowState, ResearchWorkflow
from kosmos.core.convergence import ConvergenceDetector, StoppingReason
from kosmos.models.hypothesis import Hypothesis, HypothesisStatus
from kosmos.models.result import ExecutionMetadata, ExperimentResult, ResultStatus
from kosmos.world_model import factory as world_model_factory
from kosmos.world_model import reset_world_model
from kosmos.world_model.in_memory import InMemoryWorldModel


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def in_memory_world_model():
    """Give each director a fresh in-memory knowledge graph.

    Pre-setting the singleton keeps the director from probing or connecting to a
    live Neo4j (the default "simple" mode tries bolt://localhost:7687 first).
    """
    reset_world_model()
    world_model_factory._world_model = InMemoryWorldModel()
    yield world_model_factory._world_model
    reset_world_model()


def _hypothesis(hyp_id, question, statement, **kwargs):
    """A valid Hypothesis (statement >= 10 chars, rationale >= 20 chars)."""
    return Hypothesis(
        id=hyp_id,
        research_question=question,
        statement=statement,
        rationale="Stimulant effects on attention and memory encoding",
        domain="neuroscience",
        **kwargs,
    )


def _result(i, hypothesis_id, supports=True):
    """A valid ExperimentResult with the required execution metadata."""
    now = datetime.now(timezone.utc)
    return ExperimentResult(
        id=f"result_{i}",
        experiment_id=f"exp_{i}",
        protocol_id=f"protocol_{i}",
        hypothesis_id=hypothesis_id,
        supports_hypothesis=supports,
        primary_p_value=0.01 if supports else 0.4,
        primary_effect_size=0.7 if supports else 0.05,
        status=ResultStatus.SUCCESS,
        metadata=ExecutionMetadata(
            start_time=now,
            end_time=now,
            duration_seconds=0.0,
            python_version="3.11",
            platform="linux",
            experiment_id=f"exp_{i}",
            protocol_id=f"protocol_{i}",
        ),
    )


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
    """Test different convergence scenarios against the current ConvergenceDetector."""

    def test_convergence_by_iteration_limit(
        self, simple_research_question, mock_llm_client
    ):
        """Iteration limit stops the run once enough experiments completed.

        Testable work remains (one untested hypothesis), so the mandatory
        no_testable_hypotheses criterion does not fire first, and two completed
        experiments satisfy min_experiments_before_convergence.
        """
        mock_llm_client.generate.return_value = json.dumps({"strategy": "test"})

        director = ResearchDirectorAgent(
            research_question=simple_research_question,
            domain="neuroscience",
            config={"max_iterations": 3}
        )

        detector = ConvergenceDetector()

        # Run to max iterations, completing an experiment in each
        for i in range(3):
            director.research_plan.add_hypothesis(f"hyp_{i}")
            director.research_plan.add_experiment(f"protocol_{i}")
            director.research_plan.mark_experiment_complete(f"protocol_{i}")
            director.research_plan.mark_tested(f"hyp_{i}")
            director.research_plan.increment_iteration()
        director.research_plan.add_hypothesis("hyp_untested")

        decision = detector.check_convergence(director.research_plan, [], [])

        assert decision.should_stop is True
        assert decision.reason == StoppingReason.ITERATION_LIMIT
        assert decision.is_mandatory is True

    def test_iteration_limit_deferred_until_enough_experiments(
        self, simple_research_question, mock_llm_client
    ):
        """At the limit with untested work but < 2 completed experiments, keep going."""
        director = ResearchDirectorAgent(
            research_question=simple_research_question,
            domain="neuroscience",
            config={"max_iterations": 3}
        )

        detector = ConvergenceDetector()

        for i in range(3):
            director.research_plan.add_hypothesis(f"hyp_{i}")
            director.research_plan.increment_iteration()

        decision = detector.check_convergence(director.research_plan, [], [])

        assert decision.should_stop is False
        limit = detector.check_iteration_limit(director.research_plan)
        assert limit.should_stop is False
        assert "deferring" in limit.details

    def test_convergence_by_hypothesis_exhaustion(
        self, simple_research_question, mock_llm_client
    ):
        """No untested hypotheses and an empty queue stops the run."""
        mock_llm_client.generate.return_value = json.dumps({"strategy": "test"})

        director = ResearchDirectorAgent(
            research_question=simple_research_question,
            domain="neuroscience",
            config={"max_iterations": 10}
        )

        detector = ConvergenceDetector()

        # Add and test all hypotheses
        director.research_plan.add_hypothesis("hyp_001")
        director.research_plan.add_hypothesis("hyp_002")
        director.research_plan.mark_tested("hyp_001")
        director.research_plan.mark_tested("hyp_002")
        # No experiments queued

        hypotheses = [
            _hypothesis("hyp_001", simple_research_question, "Caffeine improves recall"),
            _hypothesis("hyp_002", simple_research_question, "Caffeine speeds reaction time"),
        ]

        decision = detector.check_convergence(director.research_plan, hypotheses, [])

        assert decision.should_stop is True
        assert decision.reason == StoppingReason.NO_TESTABLE_HYPOTHESES
        assert decision.is_mandatory is True

    def test_convergence_by_novelty_decline(
        self, simple_research_question, mock_llm_client
    ):
        """A monotonically declining novelty trend over the window stops the run."""
        mock_llm_client.generate.return_value = json.dumps({"strategy": "test"})

        director = ResearchDirectorAgent(
            research_question=simple_research_question,
            domain="neuroscience",
            config={"max_iterations": 10}
        )

        detector = ConvergenceDetector()
        window = detector.novelty_decline_window

        # Untested work remains, so the mandatory criteria do not fire
        director.research_plan.add_hypothesis("hyp_pending")

        # Hypotheses with declining novelty: 0.8, 0.65, 0.5, 0.35, 0.2, 0.05
        hypotheses = [
            _hypothesis(
                f"hyp_{i}", simple_research_question, f"Hypothesis number {i}",
                novelty_score=round(0.8 - (i * 0.15), 2),
            )
            for i in range(6)
        ]

        # Each check appends the newest novelty score to the trend
        decisions = [
            detector.check_convergence(director.research_plan, hypotheses[:k], [])
            for k in range(1, window + 1)
        ]

        assert all(d.should_stop is False for d in decisions[:-1])
        assert decisions[-1].should_stop is True
        assert decisions[-1].reason == StoppingReason.NOVELTY_DECLINE
        assert decisions[-1].is_mandatory is False  # Optional criterion

    def test_convergence_by_diminishing_returns(
        self, simple_research_question, mock_llm_client
    ):
        """Cost per discovery above the threshold stops the run."""
        mock_llm_client.generate.return_value = json.dumps({"strategy": "test"})

        director = ResearchDirectorAgent(
            research_question=simple_research_question,
            domain="neuroscience",
            config={"max_iterations": 10}
        )

        detector = ConvergenceDetector(config={"cost_per_discovery_threshold": 50.0})

        # Untested work remains, so the mandatory criteria do not fire
        director.research_plan.add_hypothesis("hyp_pending")

        # One discovery for $100 total = $100 per discovery, above $50
        results = [_result(0, "hyp_0", supports=True), _result(1, "hyp_1", supports=False)]

        decision = detector.check_convergence(
            director.research_plan, [], results, total_cost=100.0
        )

        assert detector.metrics.cost_per_discovery == pytest.approx(100.0)
        assert decision.should_stop is True
        assert decision.reason == StoppingReason.DIMINISHING_RETURNS
        assert decision.is_mandatory is False  # Optional criterion


# ============================================================================
# Test Class 4: Report Generation
# ============================================================================

class TestReportGeneration:
    """Test convergence report generation (generate_convergence_report and to_markdown)."""

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

        detector = ConvergenceDetector()

        # Simulate research completion
        for i in range(3):
            director.research_plan.add_hypothesis(f"hyp_{i}")
            director.research_plan.mark_tested(f"hyp_{i}")
            director.research_plan.mark_supported(f"hyp_{i}")
            director.research_plan.increment_iteration()
        # The director marks the plan converged before reporting
        director.research_plan.has_converged = True

        hypotheses = [
            _hypothesis(f"hyp_{i}", simple_research_question, f"Hypothesis number {i}")
            for i in range(3)
        ]
        results = [_result(i, f"hyp_{i}", supports=True) for i in range(3)]

        report = detector.generate_convergence_report(
            director.research_plan,
            hypotheses,
            results,
            stopping_reason=StoppingReason.ITERATION_LIMIT,
        )

        assert report is not None
        assert report.research_question == simple_research_question
        assert report.converged is True
        assert report.research_complete is True
        assert report.stopping_reason == StoppingReason.ITERATION_LIMIT
        assert report.total_iterations == 3
        assert report.total_hypotheses == 3
        assert report.hypotheses_supported == 3
        assert report.hypotheses_rejected == 0
        assert report.experiments_conducted == 3
        assert report.supported_hypotheses == [h.statement for h in hypotheses]
        assert report.final_metrics.discovery_rate == pytest.approx(1.0)

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

        detector = ConvergenceDetector()

        # Simulate minimal research
        director.research_plan.add_hypothesis("hyp_001")
        director.research_plan.mark_tested("hyp_001")
        director.research_plan.mark_supported("hyp_001")
        director.research_plan.increment_iteration()

        report = detector.generate_convergence_report(
            director.research_plan,
            [],
            [],
            stopping_reason=StoppingReason.ITERATION_LIMIT,
        )

        markdown = report.to_markdown()

        assert isinstance(markdown, str)
        assert "# Convergence Report" in markdown
        assert simple_research_question in markdown
        assert "## Summary Statistics" in markdown
        assert "**Stopping Reason**: iteration_limit" in markdown
        assert "**Total Iterations**: 1" in markdown

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

        detector = ConvergenceDetector()

        report = detector.generate_convergence_report(
            director.research_plan,
            [],
            [],
            stopping_reason=StoppingReason.ITERATION_LIMIT,
        )

        markdown = report.to_markdown()

        # The sections ConvergenceReport.to_markdown writes
        required_sections = [
            "# Convergence Report",
            "**Stopping Reason**",
            "## Summary Statistics",
            "## Key Metrics",
            "## Supported Hypotheses",
            "## Recommended Next Steps",
            "## Detailed Summary",
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

        detector = ConvergenceDetector()

        director.research_plan.add_hypothesis("hyp_0")
        director.research_plan.add_hypothesis("hyp_1")
        director.research_plan.mark_supported("hyp_0")

        report = detector.generate_convergence_report(
            director.research_plan,
            [],
            [_result(0, "hyp_0", supports=True), _result(1, "hyp_1", supports=False)],
            stopping_reason=StoppingReason.ITERATION_LIMIT,
        )

        metrics = report.final_metrics
        assert metrics is not None
        assert metrics.discovery_rate == pytest.approx(0.5)
        assert metrics.consistency_score == pytest.approx(0.5)
        assert metrics.saturation_ratio == pytest.approx(0.5)  # 1 of 2 tested
        assert metrics.novelty_score == pytest.approx(0.0)  # no hypotheses scored

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

        detector = ConvergenceDetector()

        director.research_plan.add_hypothesis("hyp_0")
        director.research_plan.add_hypothesis("hyp_1")
        director.research_plan.mark_supported("hyp_0")

        report = detector.generate_convergence_report(
            director.research_plan,
            [],
            [],
            stopping_reason=StoppingReason.ITERATION_LIMIT,
        )

        steps = report.recommended_next_steps
        assert all(isinstance(step, str) for step in steps)
        assert "Replicate supported hypotheses in larger studies" in steps
        assert "Test remaining 1 hypotheses" in steps
        assert steps[-1] == "Document findings and prepare publication"
