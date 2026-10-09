"""
Tests for ConvergenceDetector (Phase 7).

Tests convergence metrics, stopping criteria, and convergence reporting.
"""

from datetime import datetime, timezone

import pytest

from kosmos.core.convergence import (
    ConvergenceDetector,
    ConvergenceMetrics,
    StoppingReason,
    ConvergenceReport,
)
from kosmos.core.workflow import ResearchPlan
from kosmos.models.hypothesis import Hypothesis, HypothesisStatus
from kosmos.models.result import (
    ExecutionMetadata,
    ExperimentResult,
    ResultStatus,
    StatisticalTestResult,
)


# ============================================================================
# Fixtures
# ============================================================================

def _make_result(result_id, hypothesis_id, supports, p_value, effect_size):
    """Build a valid ExperimentResult (experiment/protocol ids, metadata, primary test)."""
    now = datetime.now(timezone.utc)
    experiment_id = f"exp_{result_id}"
    return ExperimentResult(
        id=result_id,
        experiment_id=experiment_id,
        protocol_id="protocol_001",
        hypothesis_id=hypothesis_id,
        supports_hypothesis=supports,
        primary_p_value=p_value,
        primary_effect_size=effect_size,
        statistical_tests=[
            StatisticalTestResult(
                test_type="t-test",
                test_name="t-test",
                statistic=2.5,
                p_value=p_value,
                effect_size=effect_size,
                significant_0_05=p_value < 0.05,
                significant_0_01=p_value < 0.01,
                significant_0_001=p_value < 0.001,
                significance_label="*" if p_value < 0.05 else "ns",
                is_primary=True,
            )
        ],
        primary_test="t-test",
        status=ResultStatus.SUCCESS,
        metadata=ExecutionMetadata(
            start_time=now,
            end_time=now,
            duration_seconds=1.0,
            python_version="3.11",
            platform="linux",
            experiment_id=experiment_id,
            protocol_id="protocol_001",
            hypothesis_id=hypothesis_id,
        ),
    )


def _with_novelty(hypotheses, novelty):
    """Copy hypotheses with every novelty_score set to `novelty`."""
    return [h.model_copy(update={"novelty_score": novelty}) for h in hypotheses]


@pytest.fixture
def convergence_detector():
    """Create a ConvergenceDetector instance."""
    return ConvergenceDetector(
        mandatory_criteria=["iteration_limit", "no_testable_hypotheses"],
        optional_criteria=["novelty_decline", "diminishing_returns"],
        config={
            "novelty_decline_threshold": 0.3,
            "novelty_decline_window": 5,
            "cost_per_discovery_threshold": 100.0,
        },
    )


@pytest.fixture
def research_plan():
    """Create a sample research plan."""
    plan = ResearchPlan(
        research_question="Does caffeine improve cognitive performance?",
        max_iterations=10,
    )
    # Add some hypotheses
    plan.hypothesis_pool = ["hyp_001", "hyp_002", "hyp_003"]
    plan.tested_hypotheses = ["hyp_001"]
    plan.supported_hypotheses = ["hyp_001"]
    plan.rejected_hypotheses = []
    # Two completed experiments meet min_experiments_before_convergence (default 2),
    # so reaching the iteration limit is not deferred.
    plan.completed_experiments = ["exp_result_001", "exp_result_002"]
    plan.iteration_count = 5
    return plan


@pytest.fixture
def sample_hypotheses():
    """Create sample hypotheses."""
    return [
        Hypothesis(
            id="hyp_001",
            research_question="Question",
            statement="Caffeine improves memory",
            rationale="Adenosine antagonism is linked to improved memory encoding",
            domain="neuroscience",
            novelty_score=0.8,
            status=HypothesisStatus.SUPPORTED,
        ),
        Hypothesis(
            id="hyp_002",
            research_question="Question",
            statement="Caffeine enhances attention",
            rationale="Caffeine raises arousal, which supports sustained attention",
            domain="neuroscience",
            novelty_score=0.6,
            status=HypothesisStatus.GENERATED,
        ),
        Hypothesis(
            id="hyp_003",
            research_question="Question",
            statement="Caffeine reduces fatigue",
            rationale="Adenosine receptor blockade delays the onset of fatigue",
            domain="neuroscience",
            novelty_score=0.5,
            status=HypothesisStatus.GENERATED,
        ),
    ]


@pytest.fixture
def sample_results():
    """Create sample experiment results."""
    return [
        _make_result(
            "result_001",
            "hyp_001",
            supports=True,
            p_value=0.01,
            effect_size=0.75,
        ),
        _make_result(
            "result_002",
            "hyp_002",
            supports=False,
            p_value=0.65,
            effect_size=0.12,
        ),
    ]


# ============================================================================
# Test Class 1: Initialization
# ============================================================================

class TestConvergenceDetectorInitialization:
    """Test ConvergenceDetector initialization."""

    def test_initialization_default_config(self):
        """Test detector initializes with default configuration."""
        detector = ConvergenceDetector()

        assert detector.mandatory_criteria == ["iteration_limit", "no_testable_hypotheses"]
        assert detector.optional_criteria == ["novelty_decline", "diminishing_returns"]
        assert detector.novelty_decline_threshold == 0.3
        assert detector.novelty_decline_window == 5
        assert detector.cost_per_discovery_threshold == 1000.0
        assert detector.min_experiments_before_convergence == 2

    def test_initialization_custom_criteria(self):
        """Test detector initializes with custom criteria."""
        detector = ConvergenceDetector(
            mandatory_criteria=["iteration_limit"],
            optional_criteria=["novelty_decline"],
        )

        assert detector.mandatory_criteria == ["iteration_limit"]
        assert detector.optional_criteria == ["novelty_decline"]

    def test_initialization_custom_config(self):
        """Test detector initializes with custom configuration."""
        config = {
            "novelty_decline_threshold": 0.5,
            "novelty_decline_window": 10,
            "cost_per_discovery_threshold": 200.0,
            "min_experiments_before_convergence": 4,
        }

        detector = ConvergenceDetector(config=config)

        assert detector.novelty_decline_threshold == 0.5
        assert detector.novelty_decline_window == 10
        assert detector.cost_per_discovery_threshold == 200.0
        assert detector.min_experiments_before_convergence == 4

    def test_metrics_initialization(self):
        """Test metrics are initialized."""
        detector = ConvergenceDetector()

        assert detector.metrics is not None
        assert isinstance(detector.metrics, ConvergenceMetrics)


# ============================================================================
# Test Class 2: Progress Metrics
# ============================================================================

class TestProgressMetrics:
    """Test progress metrics calculation."""

    def test_calculate_discovery_rate_all_supported(
        self, convergence_detector
    ):
        """Test discovery rate when all results support hypotheses."""
        results = [
            _make_result(
                f"result_{i}",
                f"hyp_{i}",
                supports=True,
                p_value=0.01,
                effect_size=0.7,
            )
            for i in range(5)
        ]

        rate = convergence_detector.calculate_discovery_rate(results)

        assert rate == 1.0  # 100% discovery rate

    def test_calculate_discovery_rate_mixed(
        self, convergence_detector
    ):
        """Test discovery rate with mixed results."""
        results = [
            _make_result(
                "result_1",
                "hyp_1",
                supports=True,
                p_value=0.01,
                effect_size=0.7,
            ),
            _make_result(
                "result_2",
                "hyp_2",
                supports=False,
                p_value=0.65,
                effect_size=0.1,
            ),
        ]

        rate = convergence_detector.calculate_discovery_rate(results)

        assert rate == 0.5  # 1 out of 2

    def test_calculate_discovery_rate_empty(
        self, convergence_detector
    ):
        """Test discovery rate with no results."""
        rate = convergence_detector.calculate_discovery_rate([])

        assert rate == 0.0

    def test_calculate_novelty_decline(
        self, convergence_detector
    ):
        """Test novelty decline calculation."""
        # Decreasing novelty scores
        hypotheses = [
            Hypothesis(
                id=f"hyp_{i}",
                research_question="Question",
                statement=f"Statement {i}",
                rationale="Rationale long enough to pass model validation",
                domain="test",
                novelty_score=1.0 - (i * 0.1),  # 1.0, 0.9, 0.8, ...
            )
            for i in range(6)
        ]

        current_novelty, is_declining = convergence_detector.calculate_novelty_decline(hypotheses)

        assert 0.0 <= current_novelty <= 1.0
        assert isinstance(is_declining, bool)
        assert is_declining is True  # Should be declining

    def test_calculate_novelty_decline_increasing(
        self, convergence_detector
    ):
        """Test novelty decline with increasing novelty."""
        # Increasing novelty scores
        hypotheses = [
            Hypothesis(
                id=f"hyp_{i}",
                research_question="Question",
                statement=f"Statement {i}",
                rationale="Rationale long enough to pass model validation",
                domain="test",
                novelty_score=0.5 + (i * 0.1),  # 0.5, 0.6, 0.7, ...
            )
            for i in range(6)
        ]

        current_novelty, is_declining = convergence_detector.calculate_novelty_decline(hypotheses)

        assert is_declining is False

    def test_calculate_saturation(
        self, convergence_detector, research_plan
    ):
        """Test saturation calculation."""
        # 1 tested out of 3 total
        saturation = convergence_detector.calculate_saturation(research_plan)

        assert saturation == pytest.approx(1.0 / 3.0)

    def test_calculate_saturation_all_tested(
        self, convergence_detector, research_plan
    ):
        """Test saturation when all hypotheses tested."""
        research_plan.tested_hypotheses = ["hyp_001", "hyp_002", "hyp_003"]

        saturation = convergence_detector.calculate_saturation(research_plan)

        assert saturation == 1.0

    def test_calculate_consistency(
        self, convergence_detector
    ):
        """Test consistency calculation."""
        # Mixed results
        results = [
            _make_result(
                "result_1",
                "hyp_1",
                supports=True,
                p_value=0.01,
                effect_size=0.7,
            ),
            _make_result(
                "result_2",
                "hyp_2",
                supports=True,
                p_value=0.02,
                effect_size=0.6,
            ),
            _make_result(
                "result_3",
                "hyp_3",
                supports=False,
                p_value=0.65,
                effect_size=0.1,
            ),
        ]

        consistency = convergence_detector.calculate_consistency(results)

        # 2 supported out of 3 = 2/3
        assert consistency == pytest.approx(2.0 / 3.0)

    def test_calculate_consistency_empty(
        self, convergence_detector
    ):
        """Test consistency with no results."""
        consistency = convergence_detector.calculate_consistency([])

        assert consistency == 0.0


# ============================================================================
# Test Class 3: Mandatory Criteria
# ============================================================================

class TestMandatoryCriteria:
    """Test mandatory stopping criteria."""

    def test_check_iteration_limit_not_reached(
        self, convergence_detector, research_plan
    ):
        """Test iteration limit check when not reached."""
        research_plan.iteration_count = 5
        research_plan.max_iterations = 10

        decision = convergence_detector.check_iteration_limit(research_plan)

        assert decision.should_stop is False
        assert decision.reason == StoppingReason.ITERATION_LIMIT
        assert decision.is_mandatory is True

    def test_check_iteration_limit_reached(
        self, convergence_detector, research_plan
    ):
        """Test iteration limit check when reached."""
        research_plan.iteration_count = 10
        research_plan.max_iterations = 10

        decision = convergence_detector.check_iteration_limit(research_plan)

        assert decision.should_stop is True
        assert decision.reason == StoppingReason.ITERATION_LIMIT

    def test_check_iteration_limit_exceeded(
        self, convergence_detector, research_plan
    ):
        """Test iteration limit check when exceeded."""
        research_plan.iteration_count = 12
        research_plan.max_iterations = 10

        decision = convergence_detector.check_iteration_limit(research_plan)

        assert decision.should_stop is True

    def test_check_iteration_limit_deferred_when_too_few_experiments(
        self, convergence_detector, research_plan
    ):
        """At the limit, convergence is deferred while fewer than
        min_experiments_before_convergence experiments ran and work remains."""
        research_plan.iteration_count = 10
        research_plan.max_iterations = 10
        research_plan.completed_experiments = ["exp_result_001"]  # 1 < 2

        decision = convergence_detector.check_iteration_limit(research_plan)

        assert decision.should_stop is False
        assert decision.reason == StoppingReason.ITERATION_LIMIT
        assert decision.is_mandatory is True
        assert "deferring" in decision.details

    def test_check_iteration_limit_not_deferred_without_testable_work(
        self, convergence_detector, research_plan
    ):
        """Too few experiments but nothing left to test: stop at the limit."""
        research_plan.iteration_count = 10
        research_plan.max_iterations = 10
        research_plan.completed_experiments = []
        research_plan.tested_hypotheses = ["hyp_001", "hyp_002", "hyp_003"]
        research_plan.experiment_queue = []

        decision = convergence_detector.check_iteration_limit(research_plan)

        assert decision.should_stop is True

    def test_check_hypothesis_exhaustion_not_exhausted(
        self, convergence_detector, research_plan, sample_hypotheses
    ):
        """Test hypothesis exhaustion when hypotheses remain."""
        # Has untested hypotheses
        research_plan.hypothesis_pool = ["hyp_001", "hyp_002", "hyp_003"]
        research_plan.tested_hypotheses = ["hyp_001"]

        decision = convergence_detector.check_hypothesis_exhaustion(research_plan, sample_hypotheses)

        assert decision.should_stop is False

    def test_check_hypothesis_exhaustion_all_tested(
        self, convergence_detector, research_plan, sample_hypotheses
    ):
        """Test hypothesis exhaustion when all tested."""
        # All hypotheses tested, no experiments queued
        research_plan.hypothesis_pool = ["hyp_001", "hyp_002", "hyp_003"]
        research_plan.tested_hypotheses = ["hyp_001", "hyp_002", "hyp_003"]
        research_plan.experiment_queue = []

        decision = convergence_detector.check_hypothesis_exhaustion(research_plan, sample_hypotheses)

        assert decision.should_stop is True
        assert decision.reason == StoppingReason.NO_TESTABLE_HYPOTHESES

    def test_check_hypothesis_exhaustion_with_queued_experiments(
        self, convergence_detector, research_plan, sample_hypotheses
    ):
        """Test hypothesis exhaustion with experiments still queued."""
        research_plan.hypothesis_pool = ["hyp_001"]
        research_plan.tested_hypotheses = ["hyp_001"]
        research_plan.experiment_queue = ["exp_001"]  # Has queued experiment

        decision = convergence_detector.check_hypothesis_exhaustion(research_plan, sample_hypotheses)

        assert decision.should_stop is False


# ============================================================================
# Test Class 4: Optional Criteria
# ============================================================================

class TestOptionalCriteria:
    """Test optional stopping criteria."""

    def test_check_novelty_decline_not_declining(
        self, convergence_detector
    ):
        """Test novelty decline when novelty is stable."""
        # Set metrics with high novelty
        convergence_detector.metrics.novelty_trend = [0.8, 0.75, 0.78, 0.77, 0.76]
        convergence_detector.metrics.novelty_score = 0.76

        decision = convergence_detector.check_novelty_decline()

        assert decision.should_stop is False

    def test_check_novelty_decline_all_below_threshold(
        self, convergence_detector
    ):
        """Test novelty decline when all recent values below threshold."""
        # All below 0.3 threshold
        convergence_detector.metrics.novelty_trend = [0.2, 0.25, 0.22, 0.23, 0.21]

        decision = convergence_detector.check_novelty_decline()

        assert decision.should_stop is True
        assert decision.reason == StoppingReason.NOVELTY_DECLINE

    def test_check_novelty_decline_strictly_declining(
        self, convergence_detector
    ):
        """Test novelty decline when strictly decreasing."""
        # Strictly declining
        convergence_detector.metrics.novelty_trend = [0.8, 0.7, 0.6, 0.5, 0.4]

        decision = convergence_detector.check_novelty_decline()

        assert decision.should_stop is True

    def test_check_novelty_decline_insufficient_data(
        self, convergence_detector
    ):
        """Test novelty decline with insufficient data."""
        # Not enough data points
        convergence_detector.metrics.novelty_trend = [0.8, 0.7]

        decision = convergence_detector.check_novelty_decline()

        assert decision.should_stop is False

    def test_check_diminishing_returns_below_threshold(
        self, convergence_detector
    ):
        """Test diminishing returns when below threshold."""
        convergence_detector.metrics.cost_per_discovery = 50.0  # Below 100 threshold

        decision = convergence_detector.check_diminishing_returns()

        assert decision.should_stop is False

    def test_check_diminishing_returns_above_threshold(
        self, convergence_detector
    ):
        """Test diminishing returns when above threshold."""
        convergence_detector.metrics.cost_per_discovery = 150.0  # Above 100 threshold

        decision = convergence_detector.check_diminishing_returns()

        assert decision.should_stop is True
        assert decision.reason == StoppingReason.DIMINISHING_RETURNS

    def test_check_diminishing_returns_no_cost_data(
        self, convergence_detector
    ):
        """Test diminishing returns with no cost data."""
        convergence_detector.metrics.cost_per_discovery = None

        decision = convergence_detector.check_diminishing_returns()

        assert decision.should_stop is False


# ============================================================================
# Test Class 5: Convergence Decision
# ============================================================================

class TestConvergenceDecision:
    """Test overall convergence decision logic."""

    def test_check_convergence_not_converged(
        self, convergence_detector, research_plan, sample_hypotheses, sample_results
    ):
        """Test convergence check when not converged."""
        # Normal state: not at limits
        research_plan.iteration_count = 5
        research_plan.max_iterations = 10

        decision = convergence_detector.check_convergence(research_plan, sample_hypotheses, sample_results)

        assert decision.should_stop is False

    def test_check_convergence_iteration_limit(
        self, convergence_detector, research_plan, sample_hypotheses, sample_results
    ):
        """Test convergence due to iteration limit."""
        research_plan.iteration_count = 10
        research_plan.max_iterations = 10

        decision = convergence_detector.check_convergence(research_plan, sample_hypotheses, sample_results)

        assert decision.should_stop is True
        assert decision.reason == StoppingReason.ITERATION_LIMIT
        assert decision.is_mandatory is True

    def test_check_convergence_hypothesis_exhaustion(
        self, convergence_detector, research_plan, sample_hypotheses, sample_results
    ):
        """Test convergence due to hypothesis exhaustion."""
        research_plan.hypothesis_pool = ["hyp_001", "hyp_002", "hyp_003"]
        research_plan.tested_hypotheses = ["hyp_001", "hyp_002", "hyp_003"]
        research_plan.experiment_queue = []

        decision = convergence_detector.check_convergence(research_plan, sample_hypotheses, sample_results)

        assert decision.should_stop is True
        assert decision.reason == StoppingReason.NO_TESTABLE_HYPOTHESES

    def test_check_convergence_optional_criteria(
        self, convergence_detector, research_plan, sample_hypotheses, sample_results
    ):
        """Test convergence due to optional criteria."""
        # Set up novelty decline: prior trend plus the low current novelty
        # appended by check_convergence are all below the 0.3 threshold.
        convergence_detector.metrics.novelty_trend = [0.2, 0.22, 0.21, 0.19]
        low_novelty = _with_novelty(sample_hypotheses, 0.1)

        decision = convergence_detector.check_convergence(research_plan, low_novelty, sample_results)

        assert decision.should_stop is True
        assert decision.reason == StoppingReason.NOVELTY_DECLINE
        assert decision.is_mandatory is False

    def test_check_convergence_mandatory_takes_precedence(
        self, convergence_detector, research_plan, sample_hypotheses, sample_results
    ):
        """Hard-stop mandatory criteria (no_testable_hypotheses) are checked before optional."""
        # Set both a hard-stop mandatory and an optional criterion to trigger
        research_plan.tested_hypotheses = ["hyp_001", "hyp_002", "hyp_003"]
        research_plan.experiment_queue = []
        convergence_detector.metrics.novelty_trend = [0.2, 0.22, 0.21, 0.19]
        low_novelty = _with_novelty(sample_hypotheses, 0.1)

        decision = convergence_detector.check_convergence(research_plan, low_novelty, sample_results)

        assert decision.should_stop is True
        # Should be mandatory (hypothesis exhaustion), not optional (novelty)
        assert decision.reason == StoppingReason.NO_TESTABLE_HYPOTHESES
        assert decision.is_mandatory is True

    def test_check_convergence_optional_reported_before_iteration_limit(
        self, convergence_detector, research_plan, sample_hypotheses, sample_results
    ):
        """iteration_limit is checked last, so a scientific (optional) reason is
        reported when both would fire (commit 489d8cd)."""
        research_plan.iteration_count = 10
        research_plan.max_iterations = 10
        convergence_detector.metrics.novelty_trend = [0.2, 0.22, 0.21, 0.19]
        low_novelty = _with_novelty(sample_hypotheses, 0.1)

        decision = convergence_detector.check_convergence(research_plan, low_novelty, sample_results)

        assert decision.should_stop is True
        assert decision.reason == StoppingReason.NOVELTY_DECLINE
        assert decision.is_mandatory is False

    def test_check_convergence_updates_metrics(
        self, convergence_detector, research_plan, sample_hypotheses, sample_results
    ):
        """Test convergence check updates metrics."""
        initial_timestamp = convergence_detector.metrics.last_update

        convergence_detector.check_convergence(
            research_plan, sample_hypotheses, sample_results, total_cost=10.0
        )

        # Metrics should be updated
        metrics = convergence_detector.metrics
        assert metrics.last_update >= initial_timestamp
        assert metrics.iteration_count == research_plan.iteration_count
        assert metrics.total_experiments == 2
        assert metrics.significant_results == 1
        assert metrics.discovery_rate == pytest.approx(0.5)
        assert metrics.hypotheses_tested == 1
        assert metrics.total_hypotheses == 3
        assert metrics.novelty_trend == [0.5]  # novelty of the last hypothesis
        assert metrics.total_cost == pytest.approx(10.0)
        assert metrics.cost_per_discovery == pytest.approx(10.0)


# ============================================================================
# Test Class 6: Convergence Report
# ============================================================================

class TestConvergenceReport:
    """Test convergence report generation."""

    def test_generate_convergence_report(
        self, convergence_detector, research_plan, sample_hypotheses, sample_results
    ):
        """Test generating convergence report."""
        research_plan.has_converged = True

        report = convergence_detector.generate_convergence_report(
            research_plan,
            sample_hypotheses,
            sample_results,
            stopping_reason=StoppingReason.ITERATION_LIMIT,
        )

        assert isinstance(report, ConvergenceReport)
        assert report.research_question == research_plan.research_question
        assert report.converged is True
        assert report.research_complete is True
        assert report.stopping_reason == StoppingReason.ITERATION_LIMIT
        assert report.total_iterations == research_plan.iteration_count
        assert report.total_hypotheses == 3
        assert report.hypotheses_supported == 1
        assert report.hypotheses_rejected == 0
        assert report.experiments_conducted == 2
        assert report.supported_hypotheses == ["Caffeine improves memory"]
        assert report.final_metrics.hypotheses_tested > 0
        assert research_plan.research_question in report.summary

    def test_convergence_report_to_markdown(
        self, convergence_detector, research_plan, sample_hypotheses, sample_results
    ):
        """Test exporting convergence report to markdown."""
        research_plan.has_converged = True

        report = convergence_detector.generate_convergence_report(
            research_plan,
            sample_hypotheses,
            sample_results,
            stopping_reason=StoppingReason.ITERATION_LIMIT,
        )

        markdown = report.to_markdown()

        assert isinstance(markdown, str)
        assert "# Convergence Report" in markdown
        assert research_plan.research_question in markdown
        assert "**Stopping Reason**: iteration_limit" in markdown
        assert "## Summary Statistics" in markdown
        assert "## Key Metrics" in markdown
        assert "- Caffeine improves memory" in markdown

    def test_report_includes_metrics(
        self, convergence_detector, research_plan, sample_hypotheses, sample_results
    ):
        """Test report includes final metrics."""
        report = convergence_detector.generate_convergence_report(
            research_plan,
            sample_hypotheses,
            sample_results,
            stopping_reason=StoppingReason.ITERATION_LIMIT,
        )

        assert report.final_metrics is not None
        assert isinstance(report.final_metrics, ConvergenceMetrics)
        assert report.final_metrics.total_experiments == 2

    def test_report_includes_recommendations(
        self, convergence_detector, research_plan, sample_hypotheses, sample_results
    ):
        """Test report includes recommended next steps."""
        report = convergence_detector.generate_convergence_report(
            research_plan,
            sample_hypotheses,
            sample_results,
            stopping_reason=StoppingReason.NOVELTY_DECLINE,
        )

        assert len(report.recommended_next_steps) > 0
        assert report.recommended_next_steps[-1] == "Document findings and prepare publication"

    def test_report_not_converged(
        self, convergence_detector, research_plan, sample_hypotheses, sample_results
    ):
        """Test report generation when not converged."""
        research_plan.has_converged = False

        report = convergence_detector.generate_convergence_report(
            research_plan, sample_hypotheses, sample_results
        )

        assert report.converged is False
        assert report.research_complete is False
        assert report.stopping_reason is None
        assert "N/A" in report.to_markdown()


# ============================================================================
# Test Class 7: Recommended Next Steps
# ============================================================================

class TestRecommendedNextSteps:
    """Test recommended next steps generation (driven by plan state and metrics)."""

    def test_recommend_next_steps_supported_and_untested(
        self, convergence_detector, research_plan, sample_hypotheses, sample_results
    ):
        """Supported hypotheses -> replicate; untested hypotheses -> test them."""
        steps = convergence_detector._recommend_next_steps(
            research_plan, sample_hypotheses, sample_results
        )

        assert isinstance(steps, list)
        assert "Replicate supported hypotheses in larger studies" in steps
        assert "Test remaining 2 hypotheses" in steps
        assert steps[-1] == "Document findings and prepare publication"

    def test_recommend_next_steps_all_tested_none_supported(
        self, convergence_detector, research_plan, sample_hypotheses, sample_results
    ):
        """No supported and no untested hypotheses -> neither recommendation."""
        research_plan.supported_hypotheses = []
        research_plan.tested_hypotheses = ["hyp_001", "hyp_002", "hyp_003"]

        steps = convergence_detector._recommend_next_steps(
            research_plan, sample_hypotheses, sample_results
        )

        recommendations_text = " ".join(steps).lower()
        assert "replicate" not in recommendations_text
        assert "test remaining" not in recommendations_text
        assert steps[-1] == "Document findings and prepare publication"

    def test_recommend_next_steps_high_novelty(
        self, convergence_detector, research_plan, sample_hypotheses, sample_results
    ):
        """Novelty score above 0.7 -> explore related high-novelty areas."""
        convergence_detector.metrics.novelty_score = 0.8

        steps = convergence_detector._recommend_next_steps(
            research_plan, sample_hypotheses, sample_results
        )

        assert "Explore related high-novelty areas" in steps

    def test_recommend_next_steps_low_discovery_rate(
        self, convergence_detector, research_plan, sample_hypotheses, sample_results
    ):
        """Discovery rate below 0.2 -> refine the experimental approach."""
        convergence_detector.metrics.discovery_rate = 0.1
        convergence_detector.metrics.novelty_score = 0.5

        steps = convergence_detector._recommend_next_steps(
            research_plan, sample_hypotheses, sample_results
        )

        assert "Refine experimental approach to increase discovery rate" in steps
        assert "Explore related high-novelty areas" not in steps

    def test_get_metrics(self, convergence_detector):
        """Test getting current metrics."""
        metrics = convergence_detector.get_metrics()

        assert isinstance(metrics, ConvergenceMetrics)

    def test_get_metrics_dict(self, convergence_detector):
        """Test getting metrics as dictionary."""
        metrics_dict = convergence_detector.get_metrics_dict()

        assert isinstance(metrics_dict, dict)
        assert "discovery_rate" in metrics_dict
        assert "novelty_score" in metrics_dict
        assert "saturation_ratio" in metrics_dict
        assert "consistency_score" in metrics_dict
