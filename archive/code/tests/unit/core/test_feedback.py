"""
Tests for FeedbackLoop (Phase 7).

Tests success/failure pattern extraction, feedback signal generation, and learning.
"""

from datetime import datetime
import pytest

from kosmos.core.feedback import (
    FeedbackLoop,
    FeedbackSignalType,
    FeedbackSignal,
    SuccessPattern,
    FailurePattern,
)
from kosmos.models.hypothesis import Hypothesis, HypothesisStatus
from kosmos.models.result import ExperimentResult, ResultStatus


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def feedback_loop():
    """Create a FeedbackLoop instance."""
    return FeedbackLoop(config={
        "success_learning_rate": 0.3,
        "failure_learning_rate": 0.4,
    })


@pytest.fixture
def sample_hypothesis():
    """Create a sample hypothesis."""
    return Hypothesis(
        id="hyp_001",
        research_question="Does caffeine improve cognitive performance?",
        statement="Caffeine improves working memory",
        rationale="Stimulant effects",
        domain="neuroscience",
        testability_score=0.9,
        novelty_score=0.6,
        confidence_score=0.5,
    )


@pytest.fixture
def successful_result():
    """Create a successful experiment result."""
    return ExperimentResult(
        id="result_success_001",
        hypothesis_id="hyp_001",
        supports_hypothesis=True,
        primary_p_value=0.01,
        primary_effect_size=0.75,
        primary_test="t-test",
        status=ResultStatus.SUCCESS,
    )


@pytest.fixture
def failed_result():
    """Create a failed experiment result (rejected hypothesis)."""
    return ExperimentResult(
        id="result_fail_001",
        hypothesis_id="hyp_001",
        supports_hypothesis=False,
        primary_p_value=0.65,
        primary_effect_size=0.12,
        primary_test="t-test",
        status=ResultStatus.SUCCESS,
    )


@pytest.fixture
def execution_error_result():
    """Create an execution error result."""
    return ExperimentResult(
        id="result_error_001",
        hypothesis_id="hyp_001",
        supports_hypothesis=None,
        primary_p_value=None,
        primary_effect_size=None,
        primary_test="t-test",
        status=ResultStatus.FAILURE,
    )


@pytest.fixture
def underpowered_result():
    """Create an underpowered result (not significant, small effect)."""
    return ExperimentResult(
        id="result_underpowered_001",
        hypothesis_id="hyp_001",
        supports_hypothesis=False,
        primary_p_value=0.08,  # Not significant
        primary_effect_size=0.15,  # Small effect
        primary_test="t-test",
        status=ResultStatus.SUCCESS,
    )


@pytest.fixture
def statistical_failure_result():
    """Create a statistical failure (large effect but not significant)."""
    return ExperimentResult(
        id="result_statistical_001",
        hypothesis_id="hyp_001",
        supports_hypothesis=False,
        primary_p_value=0.12,
        primary_effect_size=0.6,  # Large effect
        primary_test="t-test",
        status=ResultStatus.SUCCESS,
    )


# ============================================================================
# Test Class 1: Initialization
# ============================================================================

class TestFeedbackLoopInitialization:
    """Test FeedbackLoop initialization."""

    def test_initialization_default_config(self):
        """Test feedback loop initializes with default configuration."""
        loop = FeedbackLoop()

        assert loop.success_patterns == {}
        assert loop.failure_patterns == {}
        assert loop.pending_signals == []
        assert loop.applied_signals == []
        assert loop.success_learning_rate == 0.3
        assert loop.failure_learning_rate == 0.4

    def test_initialization_custom_config(self):
        """Test feedback loop initializes with custom configuration."""
        custom_config = {
            "success_learning_rate": 0.5,
            "failure_learning_rate": 0.6,
        }

        loop = FeedbackLoop(config=custom_config)

        assert loop.success_learning_rate == 0.5
        assert loop.failure_learning_rate == 0.6

    def test_initialization_empty_state(self):
        """Test feedback loop starts with empty state."""
        loop = FeedbackLoop()

        assert len(loop.success_patterns) == 0
        assert len(loop.failure_patterns) == 0
        assert len(loop.pending_signals) == 0
        assert len(loop.applied_signals) == 0


# ============================================================================
# Test Class 2: Success Pattern Extraction
# ============================================================================

class TestSuccessPatternExtraction:
    """Test success pattern extraction and storage."""

    def test_extract_success_pattern(
        self, feedback_loop, sample_hypothesis, successful_result
    ):
        """Test extracting success pattern from result."""
        pattern = feedback_loop._extract_success_pattern(successful_result, sample_hypothesis)

        assert pattern is not None
        assert isinstance(pattern, SuccessPattern)
        assert "t-test" in pattern.description
        assert "0.75" in pattern.description or "effect" in pattern.description.lower()
        assert pattern.hypothesis_characteristics["domain"] == "neuroscience"
        assert pattern.experiment_design["test_type"] == "t-test"
        assert pattern.statistical_approach["p_value"] == 0.01
        assert pattern.statistical_approach["effect_size"] == 0.75
        assert successful_result.id in pattern.examples

    def test_analyze_success_creates_pattern(
        self, feedback_loop, sample_hypothesis, successful_result
    ):
        """Test _analyze_success creates and stores pattern."""
        signals = feedback_loop._analyze_success(successful_result, sample_hypothesis)

        assert len(signals) > 0
        assert signals[0].signal_type == FeedbackSignalType.SUCCESS_PATTERN
        assert len(feedback_loop.success_patterns) == 1

    def test_analyze_success_updates_existing_pattern(
        self, feedback_loop, sample_hypothesis, successful_result
    ):
        """Test _analyze_success updates existing pattern instead of creating new one."""
        # First success
        signals1 = feedback_loop._analyze_success(successful_result, sample_hypothesis)
        pattern_id_1 = signals1[0].data["pattern_id"]

        # Second success with same test type
        successful_result_2 = ExperimentResult(
            id="result_success_002",
            hypothesis_id="hyp_001",
            supports_hypothesis=True,
            primary_p_value=0.02,
            primary_effect_size=0.65,
            primary_test="t-test",  # Same test type
            status=ResultStatus.SUCCESS,
        )

        signals2 = feedback_loop._analyze_success(successful_result_2, sample_hypothesis)
        pattern_id_2 = signals2[0].data["pattern_id"]

        # Should update existing pattern
        assert pattern_id_1 == pattern_id_2
        assert len(feedback_loop.success_patterns) == 1

        # Check pattern was updated
        pattern = feedback_loop.success_patterns[pattern_id_1]
        assert pattern.occurrences == 2
        assert len(pattern.examples) == 2

    def test_success_pattern_confidence_increases(
        self, feedback_loop, sample_hypothesis, successful_result
    ):
        """Test success pattern confidence increases with occurrences."""
        # First success
        feedback_loop._analyze_success(successful_result, sample_hypothesis)
        initial_confidence = list(feedback_loop.success_patterns.values())[0].confidence

        # Second success
        successful_result_2 = ExperimentResult(
            id="result_success_002",
            hypothesis_id="hyp_001",
            supports_hypothesis=True,
            primary_p_value=0.02,
            primary_effect_size=0.65,
            primary_test="t-test",
            status=ResultStatus.SUCCESS,
        )

        feedback_loop._analyze_success(successful_result_2, sample_hypothesis)
        updated_confidence = list(feedback_loop.success_patterns.values())[0].confidence

        # Confidence should increase (up to max of 1.0)
        assert updated_confidence >= initial_confidence


# ============================================================================
# Test Class 3: Failure Pattern Extraction
# ============================================================================

class TestFailurePatternExtraction:
    """Test failure pattern extraction and storage."""

    def test_extract_failure_pattern_underpowered(
        self, feedback_loop, sample_hypothesis, underpowered_result
    ):
        """Test extracting underpowered failure pattern."""
        failure_type = feedback_loop._categorize_failure(underpowered_result)
        pattern = feedback_loop._extract_failure_pattern(
            underpowered_result, sample_hypothesis, failure_type
        )

        assert pattern is not None
        assert isinstance(pattern, FailurePattern)
        assert pattern.failure_type == "underpowered"
        assert "Increase sample size" in pattern.recommended_fixes
        assert pattern.common_characteristics["domain"] == "neuroscience"
        assert underpowered_result.id in pattern.examples

    def test_extract_failure_pattern_statistical(
        self, feedback_loop, sample_hypothesis, statistical_failure_result
    ):
        """Test extracting statistical failure pattern."""
        failure_type = feedback_loop._categorize_failure(statistical_failure_result)
        pattern = feedback_loop._extract_failure_pattern(
            statistical_failure_result, sample_hypothesis, failure_type
        )

        assert pattern is not None
        assert pattern.failure_type == "statistical"
        assert any("outliers" in fix.lower() or "assumptions" in fix.lower() for fix in pattern.recommended_fixes)

    def test_extract_failure_pattern_conceptual(
        self, feedback_loop, sample_hypothesis, failed_result
    ):
        """Test extracting conceptual failure pattern."""
        failure_type = feedback_loop._categorize_failure(failed_result)
        pattern = feedback_loop._extract_failure_pattern(
            failed_result, sample_hypothesis, failure_type
        )

        assert pattern is not None
        assert pattern.failure_type == "conceptual"
        assert any("hypothesis" in fix.lower() or "refine" in fix.lower() for fix in pattern.recommended_fixes)

    def test_analyze_failure_creates_pattern(
        self, feedback_loop, sample_hypothesis, failed_result
    ):
        """Test _analyze_failure creates and stores pattern."""
        signals = feedback_loop._analyze_failure(failed_result, sample_hypothesis)

        assert len(signals) > 0
        assert signals[0].signal_type == FeedbackSignalType.FAILURE_PATTERN
        assert len(feedback_loop.failure_patterns) == 1

    def test_analyze_failure_updates_existing_pattern(
        self, feedback_loop, sample_hypothesis, failed_result
    ):
        """Test _analyze_failure updates existing pattern."""
        # First failure
        signals1 = feedback_loop._analyze_failure(failed_result, sample_hypothesis)
        pattern_id_1 = signals1[0].data["pattern_id"]

        # Second failure of same type
        failed_result_2 = ExperimentResult(
            id="result_fail_002",
            hypothesis_id="hyp_002",
            supports_hypothesis=False,
            primary_p_value=0.70,
            primary_effect_size=0.10,
            primary_test="t-test",
            status=ResultStatus.SUCCESS,
        )

        sample_hypothesis_2 = Hypothesis(
            id="hyp_002",
            research_question="Another question",
            statement="Another statement",
            rationale="Rationale",
            domain="neuroscience",
        )

        signals2 = feedback_loop._analyze_failure(failed_result_2, sample_hypothesis_2)
        pattern_id_2 = signals2[0].data["pattern_id"]

        # Should update existing pattern (same failure type)
        assert pattern_id_1 == pattern_id_2
        assert len(feedback_loop.failure_patterns) == 1

        # Check pattern was updated
        pattern = feedback_loop.failure_patterns[pattern_id_1]
        assert pattern.occurrences == 2
        assert len(pattern.examples) == 2


# ============================================================================
# Test Class 4: Failure Categorization
# ============================================================================

class TestFailureCategorization:
    """Test failure categorization logic."""

    def test_categorize_execution_error(
        self, feedback_loop, execution_error_result
    ):
        """Test categorizing execution errors."""
        category = feedback_loop._categorize_failure(execution_error_result)

        assert category == "execution_error"

    def test_categorize_underpowered(
        self, feedback_loop, underpowered_result
    ):
        """Test categorizing underpowered studies."""
        category = feedback_loop._categorize_failure(underpowered_result)

        # p > 0.05 and effect < 0.2 = underpowered
        assert category == "underpowered"

    def test_categorize_statistical(
        self, feedback_loop, statistical_failure_result
    ):
        """Test categorizing statistical failures."""
        category = feedback_loop._categorize_failure(statistical_failure_result)

        # p > 0.05 and effect >= 0.2 = statistical issue
        assert category == "statistical"

    def test_categorize_conceptual(
        self, feedback_loop, failed_result
    ):
        """Test categorizing conceptual failures."""
        category = feedback_loop._categorize_failure(failed_result)

        # High p-value with small effect = conceptual issue
        assert category in ["conceptual", "underpowered"]

    def test_categorize_with_none_values(
        self, feedback_loop
    ):
        """Test categorization handles None p-value and effect size."""
        result_no_stats = ExperimentResult(
            id="result_no_stats",
            hypothesis_id="hyp_001",
            supports_hypothesis=False,
            primary_p_value=None,
            primary_effect_size=None,
            primary_test="t-test",
            status=ResultStatus.SUCCESS,
        )

        category = feedback_loop._categorize_failure(result_no_stats)

        # Should default to conceptual
        assert category == "conceptual"


# ============================================================================
# Test Class 5: Feedback Signal Generation
# ============================================================================

class TestFeedbackSignalGeneration:
    """Test feedback signal generation."""

    def test_generate_hypothesis_update_signal_supported(
        self, feedback_loop, sample_hypothesis, successful_result
    ):
        """Test generating update signal for supported hypothesis."""
        signal = feedback_loop._generate_hypothesis_update_signal(
            successful_result, sample_hypothesis
        )

        assert signal.signal_type == FeedbackSignalType.HYPOTHESIS_UPDATE
        assert signal.data["hypothesis_id"] == sample_hypothesis.id
        assert signal.data["action"] == "increase_confidence"
        assert signal.data["update_value"] == 0.3  # success_learning_rate
        assert signal.data["result_summary"]["supports"] is True
        assert signal.confidence == 1.0

    def test_generate_hypothesis_update_signal_rejected(
        self, feedback_loop, sample_hypothesis, failed_result
    ):
        """Test generating update signal for rejected hypothesis."""
        signal = feedback_loop._generate_hypothesis_update_signal(
            failed_result, sample_hypothesis
        )

        assert signal.signal_type == FeedbackSignalType.HYPOTHESIS_UPDATE
        assert signal.data["action"] == "decrease_confidence"
        assert signal.data["update_value"] == 0.4  # failure_learning_rate

    def test_generate_hypothesis_update_signal_inconclusive(
        self, feedback_loop, sample_hypothesis
    ):
        """Test generating update signal for inconclusive result."""
        inconclusive_result = ExperimentResult(
            id="result_inconclusive",
            hypothesis_id="hyp_001",
            supports_hypothesis=None,
            primary_p_value=0.08,
            primary_effect_size=0.3,
            primary_test="t-test",
            status=ResultStatus.SUCCESS,
        )

        signal = feedback_loop._generate_hypothesis_update_signal(
            inconclusive_result, sample_hypothesis
        )

        assert signal.data["action"] == "no_change"
        assert signal.data["update_value"] == 0.0

    def test_process_result_feedback_generates_multiple_signals(
        self, feedback_loop, sample_hypothesis, successful_result
    ):
        """Test process_result_feedback generates all appropriate signals."""
        signals = feedback_loop.process_result_feedback(
            successful_result, sample_hypothesis
        )

        # Should generate at least:
        # 1. Success pattern signal
        # 2. Hypothesis update signal
        assert len(signals) >= 2

        signal_types = [s.signal_type for s in signals]
        assert FeedbackSignalType.SUCCESS_PATTERN in signal_types
        assert FeedbackSignalType.HYPOTHESIS_UPDATE in signal_types

        # Signals should be in pending queue
        assert len(feedback_loop.pending_signals) >= 2

    def test_process_result_feedback_for_failure(
        self, feedback_loop, sample_hypothesis, failed_result
    ):
        """Test process_result_feedback for failures."""
        signals = feedback_loop.process_result_feedback(
            failed_result, sample_hypothesis
        )

        signal_types = [s.signal_type for s in signals]
        assert FeedbackSignalType.FAILURE_PATTERN in signal_types
        assert FeedbackSignalType.HYPOTHESIS_UPDATE in signal_types


# ============================================================================
# Test Class 6: Feedback Application
# ============================================================================

class TestFeedbackApplication:
    """Test applying feedback signals to update system state."""

    def test_apply_hypothesis_update_increase_confidence(
        self, feedback_loop, sample_hypothesis
    ):
        """Test applying hypothesis update signal (increase confidence)."""
        signal = FeedbackSignal(
            signal_type=FeedbackSignalType.HYPOTHESIS_UPDATE,
            source="result_001",
            data={
                "hypothesis_id": sample_hypothesis.id,
                "action": "increase_confidence",
                "update_value": 0.3,
                "result_summary": {"supports": True, "p_value": 0.01, "effect_size": 0.75},
            },
            confidence=1.0,
        )

        initial_confidence = sample_hypothesis.confidence_score  # 0.5

        changes = feedback_loop.apply_feedback(signal, [sample_hypothesis])

        assert sample_hypothesis.id in changes["hypotheses_updated"]
        assert sample_hypothesis.confidence_score > initial_confidence
        assert sample_hypothesis.confidence_score <= 1.0

        # Signal should be marked as applied
        assert signal.applied is True
        assert signal in feedback_loop.applied_signals
        assert signal not in feedback_loop.pending_signals

    def test_apply_hypothesis_update_decrease_confidence(
        self, feedback_loop, sample_hypothesis
    ):
        """Test applying hypothesis update signal (decrease confidence)."""
        signal = FeedbackSignal(
            signal_type=FeedbackSignalType.HYPOTHESIS_UPDATE,
            source="result_002",
            data={
                "hypothesis_id": sample_hypothesis.id,
                "action": "decrease_confidence",
                "update_value": 0.4,
                "result_summary": {"supports": False, "p_value": 0.65, "effect_size": 0.12},
            },
            confidence=1.0,
        )

        initial_confidence = sample_hypothesis.confidence_score  # 0.5

        changes = feedback_loop.apply_feedback(signal, [sample_hypothesis])

        assert sample_hypothesis.id in changes["hypotheses_updated"]
        assert sample_hypothesis.confidence_score < initial_confidence
        assert sample_hypothesis.confidence_score >= 0.0

    def test_apply_success_pattern_signal(
        self, feedback_loop, sample_hypothesis
    ):
        """Test applying success pattern signal."""
        pattern = SuccessPattern(
            pattern_id="success_1",
            description="Successful t-test",
            hypothesis_characteristics={"domain": "neuroscience"},
            experiment_design={"test_type": "t-test"},
            statistical_approach={"p_value": 0.01, "effect_size": 0.75},
        )

        signal = FeedbackSignal(
            signal_type=FeedbackSignalType.SUCCESS_PATTERN,
            source="result_001",
            data={
                "pattern_id": "success_1",
                "pattern": pattern.model_dump(),
                "action": "increase_priority",
                "target": "similar_hypotheses",
            },
            confidence=0.9,
        )

        changes = feedback_loop.apply_feedback(signal, [sample_hypothesis])

        assert "success_pattern_applied" in changes["strategies_adjusted"]

    def test_apply_failure_pattern_signal(
        self, feedback_loop, sample_hypothesis
    ):
        """Test applying failure pattern signal."""
        pattern = FailurePattern(
            pattern_id="failure_1",
            description="Underpowered failure",
            failure_type="underpowered",
            recommended_fixes=["Increase sample size"],
        )

        signal = FeedbackSignal(
            signal_type=FeedbackSignalType.FAILURE_PATTERN,
            source="result_002",
            data={
                "pattern_id": "failure_1",
                "pattern": pattern.model_dump(),
                "action": "avoid_pattern",
                "recommended_fixes": pattern.recommended_fixes,
            },
            confidence=0.8,
        )

        changes = feedback_loop.apply_feedback(signal, [sample_hypothesis])

        assert "failure_pattern_avoided" in changes["strategies_adjusted"]

    def test_apply_feedback_hypothesis_not_found(
        self, feedback_loop, sample_hypothesis
    ):
        """Test applying feedback when hypothesis not found in list."""
        signal = FeedbackSignal(
            signal_type=FeedbackSignalType.HYPOTHESIS_UPDATE,
            source="result_001",
            data={
                "hypothesis_id": "non_existent_hyp",
                "action": "increase_confidence",
                "update_value": 0.3,
                "result_summary": {},
            },
            confidence=1.0,
        )

        changes = feedback_loop.apply_feedback(signal, [sample_hypothesis])

        # Should not update any hypotheses
        assert len(changes["hypotheses_updated"]) == 0

    def test_apply_feedback_confidence_bounds(
        self, feedback_loop, sample_hypothesis
    ):
        """Test confidence updates respect [0, 1] bounds."""
        # Set to very high confidence
        sample_hypothesis.confidence_score = 0.95

        signal_increase = FeedbackSignal(
            signal_type=FeedbackSignalType.HYPOTHESIS_UPDATE,
            source="result_001",
            data={
                "hypothesis_id": sample_hypothesis.id,
                "action": "increase_confidence",
                "update_value": 0.3,
                "result_summary": {},
            },
            confidence=1.0,
        )

        feedback_loop.apply_feedback(signal_increase, [sample_hypothesis])

        # Should not exceed 1.0
        assert sample_hypothesis.confidence_score <= 1.0

        # Set to very low confidence
        sample_hypothesis.confidence_score = 0.05

        signal_decrease = FeedbackSignal(
            signal_type=FeedbackSignalType.HYPOTHESIS_UPDATE,
            source="result_002",
            data={
                "hypothesis_id": sample_hypothesis.id,
                "action": "decrease_confidence",
                "update_value": 0.4,
                "result_summary": {},
            },
            confidence=1.0,
        )

        feedback_loop.apply_feedback(signal_decrease, [sample_hypothesis])

        # Should not go below 0.0
        assert sample_hypothesis.confidence_score >= 0.0


# ============================================================================
# Test Class 7: Learning Summary
# ============================================================================

class TestLearningSummary:
    """Test learning summary and reporting."""

    def test_get_success_patterns(
        self, feedback_loop, sample_hypothesis, successful_result
    ):
        """Test retrieving success patterns."""
        feedback_loop.process_result_feedback(successful_result, sample_hypothesis)

        patterns = feedback_loop.get_success_patterns()

        assert len(patterns) > 0
        assert all(isinstance(p, SuccessPattern) for p in patterns)

    def test_get_failure_patterns(
        self, feedback_loop, sample_hypothesis, failed_result
    ):
        """Test retrieving failure patterns."""
        feedback_loop.process_result_feedback(failed_result, sample_hypothesis)

        patterns = feedback_loop.get_failure_patterns()

        assert len(patterns) > 0
        assert all(isinstance(p, FailurePattern) for p in patterns)

    def test_get_pending_signals(
        self, feedback_loop, sample_hypothesis, successful_result
    ):
        """Test retrieving pending signals."""
        feedback_loop.process_result_feedback(successful_result, sample_hypothesis)

        pending = feedback_loop.get_pending_signals()

        assert len(pending) > 0
        assert all(isinstance(s, FeedbackSignal) for s in pending)

    def test_get_learning_summary(
        self, feedback_loop, sample_hypothesis, successful_result, failed_result
    ):
        """Test getting comprehensive learning summary."""
        # Process some results
        feedback_loop.process_result_feedback(successful_result, sample_hypothesis)
        feedback_loop.process_result_feedback(failed_result, sample_hypothesis)

        summary = feedback_loop.get_learning_summary()

        assert "success_patterns_learned" in summary
        assert "failure_patterns_learned" in summary
        assert "pending_signals" in summary
        assert "applied_signals" in summary
        assert summary["success_patterns_learned"] > 0
        assert summary["failure_patterns_learned"] > 0
        assert summary["pending_signals"] > 0

    def test_most_common_success_pattern(
        self, feedback_loop, sample_hypothesis, successful_result
    ):
        """Test identifying most common success pattern."""
        # Create pattern with multiple occurrences
        feedback_loop.process_result_feedback(successful_result, sample_hypothesis)

        # Add another occurrence
        successful_result_2 = ExperimentResult(
            id="result_success_002",
            hypothesis_id="hyp_001",
            supports_hypothesis=True,
            primary_p_value=0.02,
            primary_effect_size=0.65,
            primary_test="t-test",
            status=ResultStatus.SUCCESS,
        )
        feedback_loop.process_result_feedback(successful_result_2, sample_hypothesis)

        most_common = feedback_loop._get_most_common_success_pattern()

        assert most_common is not None
        assert "t-test" in most_common

    def test_most_common_failure_pattern(
        self, feedback_loop, sample_hypothesis, failed_result
    ):
        """Test identifying most common failure pattern."""
        feedback_loop.process_result_feedback(failed_result, sample_hypothesis)

        most_common = feedback_loop._get_most_common_failure_pattern()

        assert most_common is not None

    def test_learning_summary_empty_state(
        self, feedback_loop
    ):
        """Test learning summary with no patterns yet."""
        summary = feedback_loop.get_learning_summary()

        assert summary["success_patterns_learned"] == 0
        assert summary["failure_patterns_learned"] == 0
        assert summary["pending_signals"] == 0
        assert summary["applied_signals"] == 0
        assert summary["most_common_success"] is None
        assert summary["most_common_failure"] is None
