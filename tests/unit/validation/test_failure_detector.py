"""
Unit tests for FailureDetector module.

Tests failure mode detection for:
1. Over-interpretation (claims exceed statistical support)
2. Invented metrics (metrics don't exist in data/standards)
3. Rabbit hole (research drifted from original question)

Issue: #63 (GAP-010)
"""

import pytest
from typing import Dict, Any

from kosmos.validation.failure_detector import (
    FailureDetector,
    FailureDetectionResult,
    FailureModeScore,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def failure_detector():
    """Create FailureDetector with default settings."""
    return FailureDetector()


@pytest.fixture
def custom_threshold_detector():
    """Create FailureDetector with custom thresholds."""
    return FailureDetector(
        over_interpretation_threshold=0.5,
        invented_metrics_threshold=0.4,
        rabbit_hole_threshold=0.6,
    )


@pytest.fixture
def clean_finding():
    """A finding that should pass all failure checks."""
    return {
        'finding_id': 'clean_001',
        'summary': 'Genetic variants in BRCA1 are associated with increased cancer susceptibility in human populations',
        'statistics': {
            'p_value': 0.001,
            'effect_size': 0.7,
            'sample_size': 150,
            'test_type': 't_test',
        },
        'interpretation': 'These results suggest genetic factors contribute to cancer risk through BRCA1 variants.',
    }


@pytest.fixture
def over_interpretation_finding():
    """A finding with strong claims but weak statistics."""
    return {
        'finding_id': 'over_interp_001',
        'summary': 'This proves that Gene X definitively causes cancer',
        'statistics': {
            'p_value': 0.08,  # Not significant
            'effect_size': 0.1,  # Small effect
            'sample_size': 20,  # Small sample
        },
        'interpretation': 'This conclusively demonstrates and confirms that Gene X is the cause of cancer. The results clearly establish this relationship.',
    }


@pytest.fixture
def invented_metrics_finding():
    """A finding with metrics that don't exist in standard statistics."""
    return {
        'finding_id': 'invented_001',
        'summary': 'The synergy_index = 0.95 and novelty_score = 0.88',
        'statistics': {
            'p_value': 0.01,
            'effect_size': 0.5,
        },
        'interpretation': 'The quantum_coherence_ratio = 0.72 indicates strong effect. The mystical_significance = 0.99 confirms our hypothesis.',
    }


@pytest.fixture
def rabbit_hole_finding():
    """A finding that has drifted from the research question."""
    return {
        'finding_id': 'rabbit_001',
        'summary': 'We found that pizza consumption correlates with economic indicators',
        'statistics': {
            'p_value': 0.001,
            'effect_size': 0.8,
            'sample_size': 100,
        },
        'interpretation': 'This economic analysis suggests pizza is important.',
    }


@pytest.fixture
def research_question():
    """The original research question (for rabbit hole detection)."""
    return "What genetic factors contribute to cancer susceptibility in human populations?"


@pytest.fixture
def hypothesis_with_deep_generation():
    """A hypothesis that has been refined many times."""
    return {
        'hypothesis_id': 'hyp_001',
        'statement': 'Gene X variants affect metabolism',
        'generation': 5,
        'refinement_count': 6,
        'research_question': 'What genetic factors contribute to cancer susceptibility?',
    }


# =============================================================================
# FailureModeScore Tests
# =============================================================================


class TestFailureModeScore:
    """Tests for FailureModeScore dataclass."""

    def test_create_failure_mode_score(self):
        """Test creating FailureModeScore with all fields."""
        score = FailureModeScore(
            score=0.75,
            detected=True,
            confidence=0.8,
            evidence=['High claim strength'],
            recommendations=['Reduce claim strength'],
        )
        assert score.score == 0.75
        assert score.detected is True
        assert score.confidence == 0.8
        assert len(score.evidence) == 1
        assert len(score.recommendations) == 1

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        score = FailureModeScore(
            score=0.5,
            detected=False,
            confidence=0.7,
        )
        d = score.to_dict()
        assert d['score'] == 0.5
        assert d['detected'] is False
        assert d['confidence'] == 0.7

    def test_from_dict_deserialization(self):
        """Test deserialization from dictionary."""
        data = {
            'score': 0.6,
            'detected': True,
            'confidence': 0.9,
            'evidence': ['Test evidence'],
            'recommendations': [],
        }
        score = FailureModeScore.from_dict(data)
        assert score.score == 0.6
        assert score.detected is True
        assert score.evidence == ['Test evidence']


# =============================================================================
# FailureDetectionResult Tests
# =============================================================================


class TestFailureDetectionResult:
    """Tests for FailureDetectionResult dataclass."""

    def test_create_failure_detection_result(self):
        """Test creating FailureDetectionResult with all fields."""
        over_interp = FailureModeScore(score=0.3, detected=False, confidence=0.8)
        invented = FailureModeScore(score=0.2, detected=False, confidence=0.7)
        rabbit_hole = FailureModeScore(score=0.4, detected=False, confidence=0.6)

        result = FailureDetectionResult(
            over_interpretation=over_interp,
            invented_metrics=invented,
            rabbit_hole=rabbit_hole,
            overall_score=0.3,
            passes_validation=True,
            finding_id='test_001',
        )

        assert result.over_interpretation.score == 0.3
        assert result.invented_metrics.score == 0.2
        assert result.rabbit_hole.score == 0.4
        assert result.passes_validation is True

    def test_has_failures_property_no_failures(self):
        """Test has_failures returns False when no failures detected."""
        over_interp = FailureModeScore(score=0.3, detected=False, confidence=0.8)
        invented = FailureModeScore(score=0.2, detected=False, confidence=0.7)
        rabbit_hole = FailureModeScore(score=0.4, detected=False, confidence=0.6)

        result = FailureDetectionResult(
            over_interpretation=over_interp,
            invented_metrics=invented,
            rabbit_hole=rabbit_hole,
            overall_score=0.3,
            passes_validation=True,
            finding_id='test_001',
        )

        assert result.has_failures is False

    def test_has_failures_property_with_failures(self):
        """Test has_failures returns True when failures detected."""
        over_interp = FailureModeScore(score=0.8, detected=True, confidence=0.8)
        invented = FailureModeScore(score=0.2, detected=False, confidence=0.7)
        rabbit_hole = FailureModeScore(score=0.4, detected=False, confidence=0.6)

        result = FailureDetectionResult(
            over_interpretation=over_interp,
            invented_metrics=invented,
            rabbit_hole=rabbit_hole,
            overall_score=0.5,
            passes_validation=False,
            finding_id='test_001',
        )

        assert result.has_failures is True

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        over_interp = FailureModeScore(score=0.3, detected=False, confidence=0.8)
        invented = FailureModeScore(score=0.2, detected=False, confidence=0.7)
        rabbit_hole = FailureModeScore(score=0.4, detected=False, confidence=0.6)

        result = FailureDetectionResult(
            over_interpretation=over_interp,
            invented_metrics=invented,
            rabbit_hole=rabbit_hole,
            overall_score=0.3,
            passes_validation=True,
            finding_id='test_001',
            research_question='Test RQ',
        )

        d = result.to_dict()
        assert d['overall_score'] == 0.3
        assert d['passes_validation'] is True
        assert d['finding_id'] == 'test_001'
        assert 'over_interpretation' in d
        assert d['over_interpretation']['score'] == 0.3

    def test_from_dict_deserialization(self):
        """Test deserialization from dictionary."""
        data = {
            'over_interpretation': {'score': 0.5, 'detected': False, 'confidence': 0.8, 'evidence': [], 'recommendations': []},
            'invented_metrics': {'score': 0.3, 'detected': False, 'confidence': 0.7, 'evidence': [], 'recommendations': []},
            'rabbit_hole': {'score': 0.4, 'detected': False, 'confidence': 0.6, 'evidence': [], 'recommendations': []},
            'overall_score': 0.4,
            'passes_validation': True,
            'finding_id': 'test_002',
        }
        result = FailureDetectionResult.from_dict(data)
        assert result.overall_score == 0.4
        assert result.finding_id == 'test_002'

    def test_get_summary_no_failures(self):
        """Test summary generation with no failures."""
        over_interp = FailureModeScore(score=0.3, detected=False, confidence=0.8)
        invented = FailureModeScore(score=0.2, detected=False, confidence=0.7)
        rabbit_hole = FailureModeScore(score=0.4, detected=False, confidence=0.6)

        result = FailureDetectionResult(
            over_interpretation=over_interp,
            invented_metrics=invented,
            rabbit_hole=rabbit_hole,
            overall_score=0.3,
            passes_validation=True,
            finding_id='test_001',
        )

        summary = result.get_summary()
        assert 'No failure modes detected' in summary

    def test_get_summary_with_failures(self):
        """Test summary generation with failures."""
        over_interp = FailureModeScore(score=0.8, detected=True, confidence=0.8)
        invented = FailureModeScore(score=0.2, detected=False, confidence=0.7)
        rabbit_hole = FailureModeScore(score=0.9, detected=True, confidence=0.6)

        result = FailureDetectionResult(
            over_interpretation=over_interp,
            invented_metrics=invented,
            rabbit_hole=rabbit_hole,
            overall_score=0.6,
            passes_validation=False,
            finding_id='test_001',
        )

        summary = result.get_summary()
        assert 'Over-interpretation' in summary
        assert 'Rabbit hole' in summary


# =============================================================================
# FailureDetector Initialization Tests
# =============================================================================


class TestFailureDetectorInit:
    """Tests for FailureDetector initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        detector = FailureDetector()
        assert detector.over_interpretation_threshold == 0.6
        assert detector.invented_metrics_threshold == 0.5
        assert detector.rabbit_hole_threshold == 0.7
        assert detector.similarity_threshold == 0.3
        assert detector.client is None

    def test_custom_thresholds(self):
        """Test initialization with custom thresholds."""
        detector = FailureDetector(
            over_interpretation_threshold=0.4,
            invented_metrics_threshold=0.3,
            rabbit_hole_threshold=0.5,
        )
        assert detector.over_interpretation_threshold == 0.4
        assert detector.invented_metrics_threshold == 0.3
        assert detector.rabbit_hole_threshold == 0.5

    def test_with_llm_client(self):
        """Test initialization with LLM client."""
        mock_client = object()
        detector = FailureDetector(anthropic_client=mock_client)
        assert detector.client is mock_client

    def test_standard_metrics_populated(self):
        """Test that standard metrics set is populated."""
        detector = FailureDetector()
        assert 'p_value' in detector.STANDARD_METRICS
        assert 'effect_size' in detector.STANDARD_METRICS
        assert 'cohens_d' in detector.STANDARD_METRICS
        assert 'sample_size' in detector.STANDARD_METRICS


# =============================================================================
# Over-Interpretation Detection Tests
# =============================================================================


class TestOverInterpretationDetection:
    """Tests for over-interpretation detection."""

    def test_strong_claims_weak_stats_detected(
        self, failure_detector, over_interpretation_finding
    ):
        """Test that strong claims with weak statistics are detected."""
        result = failure_detector.detect_over_interpretation(over_interpretation_finding)
        assert result.detected is True
        assert result.score >= 0.6

    def test_hedged_claims_strong_stats_passes(self, failure_detector, clean_finding):
        """Test that hedged claims with strong statistics pass."""
        result = failure_detector.detect_over_interpretation(clean_finding)
        assert result.detected is False
        assert result.score < 0.6

    def test_no_interpretation_passes(self, failure_detector):
        """Test that finding without interpretation passes."""
        finding = {
            'finding_id': 'test_001',
            'summary': '',
            'statistics': {'p_value': 0.05},
            'interpretation': '',
        }
        result = failure_detector.detect_over_interpretation(finding)
        assert result.detected is False
        assert result.score == 0.0

    def test_no_statistics_flags_any_claim(self, failure_detector):
        """Test that claims without statistics get flagged."""
        finding = {
            'finding_id': 'test_001',
            'summary': 'This proves the hypothesis',
            'statistics': {},
            'interpretation': 'This conclusively demonstrates the effect.',
        }
        result = failure_detector.detect_over_interpretation(finding)
        # Should have elevated score due to claims without stats
        assert result.score > 0.3

    def test_claim_strength_measurement(self, failure_detector):
        """Test claim strength measurement."""
        strong_text = 'This proves and demonstrates and confirms the hypothesis'
        weak_text = 'This suggests and indicates a possible trend'

        strong_strength = failure_detector._measure_claim_strength(strong_text)
        weak_strength = failure_detector._measure_claim_strength(weak_text)

        assert strong_strength > weak_strength

    def test_statistical_strength_from_p_value(self, failure_detector):
        """Test statistical strength from p-value."""
        # Very significant
        strong_stats = {'p_value': 0.0001}
        strong_strength = failure_detector._measure_statistical_strength(strong_stats)

        # Not significant
        weak_stats = {'p_value': 0.1}
        weak_strength = failure_detector._measure_statistical_strength(weak_stats)

        assert strong_strength > weak_strength

    def test_statistical_strength_from_effect_size(self, failure_detector):
        """Test statistical strength from effect size."""
        large_effect = {'effect_size': 1.0}
        small_effect = {'effect_size': 0.1}

        large_strength = failure_detector._measure_statistical_strength(large_effect)
        small_strength = failure_detector._measure_statistical_strength(small_effect)

        assert large_strength > small_strength

    def test_statistical_strength_from_sample_size(self, failure_detector):
        """Test statistical strength from sample size."""
        large_sample = {'sample_size': 500}
        small_sample = {'sample_size': 10}

        large_strength = failure_detector._measure_statistical_strength(large_sample)
        small_strength = failure_detector._measure_statistical_strength(small_sample)

        assert large_strength > small_strength

    def test_combined_statistical_strength(self, failure_detector):
        """Test combined statistical strength from multiple metrics."""
        strong_stats = {
            'p_value': 0.001,
            'effect_size': 0.8,
            'sample_size': 200,
        }
        weak_stats = {
            'p_value': 0.1,
            'effect_size': 0.1,
            'sample_size': 15,
        }

        strong_strength = failure_detector._measure_statistical_strength(strong_stats)
        weak_strength = failure_detector._measure_statistical_strength(weak_stats)

        assert strong_strength > 0.7
        assert weak_strength < 0.5

    def test_evidence_generation(self, failure_detector, over_interpretation_finding):
        """Test that evidence is generated for over-interpretation."""
        result = failure_detector.detect_over_interpretation(over_interpretation_finding)
        assert len(result.evidence) > 0
        assert any('Claim strength' in e for e in result.evidence)

    def test_recommendations_generation(self, failure_detector, over_interpretation_finding):
        """Test that recommendations are generated for over-interpretation."""
        result = failure_detector.detect_over_interpretation(over_interpretation_finding)
        assert len(result.recommendations) > 0

    def test_significant_p_value_with_moderate_claims(self, failure_detector):
        """Test that significant p-value with moderate claims passes."""
        finding = {
            'finding_id': 'test_001',
            'summary': 'Results indicate a significant association',
            'statistics': {'p_value': 0.01, 'effect_size': 0.5},
            'interpretation': 'These results suggest a relationship.',
        }
        result = failure_detector.detect_over_interpretation(finding)
        # Should pass since claims are moderate and stats are good
        assert result.score < 0.6


# =============================================================================
# Invented Metrics Detection Tests
# =============================================================================


class TestInventedMetricsDetection:
    """Tests for invented metrics detection."""

    def test_standard_metrics_pass(self, failure_detector, clean_finding):
        """Test that standard metrics pass validation."""
        result = failure_detector.detect_invented_metrics(clean_finding)
        assert result.detected is False

    def test_invented_metric_detected(self, failure_detector, invented_metrics_finding):
        """Test that invented metrics are detected."""
        result = failure_detector.detect_invented_metrics(invented_metrics_finding)
        assert result.detected is True
        assert result.score >= 0.5

    def test_custom_schema_metrics_pass(self, failure_detector):
        """Test that metrics from dataset schema pass."""
        finding = {
            'summary': 'The gene_expression = 0.85',
            'statistics': {'p_value': 0.01},
            'interpretation': 'The gene_expression metric shows...',
        }
        # Provide gene_expression in schema
        result = failure_detector.detect_invented_metrics(
            finding, dataset_schema=['gene_expression', 'phenotype']
        )
        assert result.detected is False

    def test_metric_extraction_equals_format(self, failure_detector):
        """Test metric extraction from 'metric = value' format."""
        text = 'p_value = 0.05 and effect_size = 0.8'
        metrics = failure_detector._extract_metrics_from_text(text)
        assert 'p_value' in metrics or 'p' in metrics
        assert 'effect_size' in metrics

    def test_metric_extraction_colon_format(self, failure_detector):
        """Test metric extraction from 'metric: value' format."""
        text = 'p_value: 0.05 and effect_size: 0.8'
        metrics = failure_detector._extract_metrics_from_text(text)
        assert 'p_value' in metrics or 'p' in metrics
        assert 'effect_size' in metrics

    def test_partial_match_passes(self, failure_detector):
        """Test that partial matches to standard metrics pass."""
        finding = {
            'summary': 'The pvalue = 0.05',  # Variant of p_value
            'statistics': {'p_value': 0.05},
            'interpretation': '',
        }
        result = failure_detector.detect_invented_metrics(finding)
        # Should pass due to partial match
        assert result.score < 0.5

    def test_multiple_invented_metrics(self, failure_detector):
        """Test detection of multiple invented metrics."""
        finding = {
            'summary': 'novelty_index = 0.9, synergy_coefficient = 0.8',
            'statistics': {'p_value': 0.01},
            'interpretation': 'The magic_number = 0.7 confirms our quantum_significance = 0.95',
        }
        result = failure_detector.detect_invented_metrics(finding)
        assert result.detected is True
        assert len(result.evidence) > 0

    def test_empty_text_passes(self, failure_detector):
        """Test that empty text passes."""
        finding = {
            'summary': '',
            'statistics': {},
            'interpretation': '',
        }
        result = failure_detector.detect_invented_metrics(finding)
        assert result.detected is False
        assert result.score == 0.0

    def test_case_insensitivity(self, failure_detector):
        """Test that metric matching is case-insensitive."""
        finding = {
            'summary': 'The P_VALUE = 0.05',
            'statistics': {'p_value': 0.05},
            'interpretation': '',
        }
        result = failure_detector.detect_invented_metrics(finding)
        assert result.detected is False

    def test_with_dataset_schema(self, failure_detector):
        """Test metrics validation with dataset schema."""
        finding = {
            'summary': 'The custom_metric = 0.9',
            'statistics': {},
            'interpretation': '',
        }
        # Without schema, custom_metric should be flagged
        result1 = failure_detector.detect_invented_metrics(finding)

        # With schema including custom_metric, should pass
        result2 = failure_detector.detect_invented_metrics(
            finding, dataset_schema=['custom_metric']
        )

        assert result1.score >= result2.score


# =============================================================================
# Rabbit Hole Detection Tests
# =============================================================================


class TestRabbitHoleDetection:
    """Tests for rabbit hole detection."""

    def test_relevant_finding_passes(
        self, failure_detector, clean_finding, research_question
    ):
        """Test that relevant finding passes."""
        result = failure_detector.detect_rabbit_hole(
            clean_finding, research_question
        )
        # Cancer-related finding matches cancer-related RQ
        assert result.score < 0.7

    def test_irrelevant_finding_detected(
        self, failure_detector, rabbit_hole_finding, research_question
    ):
        """Test that irrelevant finding is detected."""
        result = failure_detector.detect_rabbit_hole(
            rabbit_hole_finding, research_question
        )
        # Pizza/economics doesn't match cancer genetics
        assert result.score >= 0.5

    def test_high_generation_penalty(
        self, failure_detector, clean_finding, research_question, hypothesis_with_deep_generation
    ):
        """Test that high hypothesis generation adds penalty."""
        # Without hypothesis
        result1 = failure_detector.detect_rabbit_hole(
            clean_finding, research_question
        )

        # With deeply evolved hypothesis
        result2 = failure_detector.detect_rabbit_hole(
            clean_finding, research_question, hypothesis_with_deep_generation
        )

        # Score should be higher with generation penalty
        assert result2.score > result1.score

    def test_semantic_similarity_calculation(self, failure_detector):
        """Test semantic similarity calculation."""
        text1 = 'Gene expression analysis in cancer cells'
        text2 = 'Genetic factors in cancer susceptibility'

        sim = failure_detector._compute_relevance_similarity(text1, text2)
        assert sim > 0.0  # Some similarity expected

    def test_keyword_similarity_fallback(self, failure_detector):
        """Test keyword similarity calculation."""
        text1 = 'Gene expression in cancer'
        text2 = 'Gene analysis for cancer'

        sim = failure_detector._keyword_similarity(text1, text2)
        assert sim > 0.3  # Shared keywords: gene, cancer

    def test_empty_research_question(self, failure_detector, clean_finding):
        """Test handling of empty research question."""
        result = failure_detector.detect_rabbit_hole(clean_finding, '')
        # Should not flag as rabbit hole without RQ to compare
        assert result.detected is False
        assert result.confidence < 0.5

    def test_evolution_history_tracking(self, failure_detector, clean_finding):
        """Test that hypothesis evolution is considered."""
        hypothesis = {
            'generation': 2,
            'refinement_count': 1,
        }
        result1 = failure_detector.detect_rabbit_hole(
            clean_finding, 'Cancer genetics', hypothesis
        )

        # Higher generation
        hypothesis['generation'] = 5
        result2 = failure_detector.detect_rabbit_hole(
            clean_finding, 'Cancer genetics', hypothesis
        )

        assert result2.score > result1.score

    def test_combined_score_calculation(self, failure_detector, rabbit_hole_finding):
        """Test combined score calculation."""
        hypothesis = {'generation': 4}
        result = failure_detector.detect_rabbit_hole(
            rabbit_hole_finding,
            'Cancer genetics research',
            hypothesis
        )
        # Should have both low similarity AND generation penalty
        assert result.score >= 0.5

    def test_recommendations_for_drift(
        self, failure_detector, rabbit_hole_finding, research_question
    ):
        """Test recommendations generated for research drift."""
        result = failure_detector.detect_rabbit_hole(
            rabbit_hole_finding, research_question
        )
        if result.detected:
            assert len(result.recommendations) > 0


# =============================================================================
# Main Detection Method Tests
# =============================================================================


class TestMainDetectionMethod:
    """Tests for main detect_failures() method."""

    def test_detect_all_failures(
        self, failure_detector, over_interpretation_finding, research_question
    ):
        """Test detection of all failure modes."""
        context = {
            'research_question': research_question,
            'dataset_schema': ['p_value', 'effect_size'],
        }
        result = failure_detector.detect_failures(over_interpretation_finding, context)

        assert isinstance(result, FailureDetectionResult)
        assert result.over_interpretation is not None
        assert result.invented_metrics is not None
        assert result.rabbit_hole is not None

    def test_overall_score_calculation(self, failure_detector, clean_finding):
        """Test overall score is weighted average."""
        context = {'research_question': 'Gene expression in cancer'}
        result = failure_detector.detect_failures(clean_finding, context)

        # Manually calculate expected
        expected = (
            0.4 * result.over_interpretation.score +
            0.3 * result.invented_metrics.score +
            0.3 * result.rabbit_hole.score
        )

        assert abs(result.overall_score - expected) < 0.01

    def test_passes_validation_logic(self, failure_detector, clean_finding):
        """Test passes_validation is True when no failures."""
        context = {'research_question': 'Gene expression in cancer'}
        result = failure_detector.detect_failures(clean_finding, context)

        if not result.has_failures:
            assert result.passes_validation is True
        else:
            assert result.passes_validation is False

    def test_computation_time_tracked(self, failure_detector, clean_finding):
        """Test that computation time is tracked."""
        result = failure_detector.detect_failures(clean_finding)
        assert result.computation_time_seconds >= 0


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_finding(self, failure_detector):
        """Test handling of empty finding."""
        finding = {
            'finding_id': 'empty_001',
            'summary': '',
            'statistics': {},
            'interpretation': '',
        }
        result = failure_detector.detect_failures(finding)
        assert isinstance(result, FailureDetectionResult)

    def test_none_values(self, failure_detector):
        """Test handling of None values."""
        finding = {
            'finding_id': 'none_001',
            'summary': None,
            'statistics': None,
            'interpretation': None,
        }
        result = failure_detector.detect_failures(finding)
        assert isinstance(result, FailureDetectionResult)

    def test_very_long_text(self, failure_detector):
        """Test handling of very long text."""
        long_text = 'This suggests ' * 1000
        finding = {
            'finding_id': 'long_001',
            'summary': long_text,
            'statistics': {'p_value': 0.01},
            'interpretation': long_text,
        }
        result = failure_detector.detect_failures(finding)
        assert isinstance(result, FailureDetectionResult)

    def test_unicode_content(self, failure_detector):
        """Test handling of unicode content."""
        finding = {
            'finding_id': 'unicode_001',
            'summary': 'Gene expression in cancer',
            'statistics': {'p_value': 0.01},
            'interpretation': 'Results suggest a relationship',
        }
        result = failure_detector.detect_failures(finding)
        assert isinstance(result, FailureDetectionResult)

    def test_special_characters(self, failure_detector):
        """Test handling of special characters."""
        finding = {
            'finding_id': 'special_001',
            'summary': 'p-value = 0.05; effect_size = 0.8',
            'statistics': {'p_value': 0.05},
            'interpretation': 'Results (p < 0.05) suggest...',
        }
        result = failure_detector.detect_failures(finding)
        assert isinstance(result, FailureDetectionResult)


# =============================================================================
# Batch Processing Tests
# =============================================================================


class TestBatchProcessing:
    """Tests for batch processing methods."""

    def test_batch_detect(self, failure_detector, clean_finding, over_interpretation_finding):
        """Test batch detection of multiple findings."""
        findings = [clean_finding, over_interpretation_finding]
        results = failure_detector.batch_detect(findings)

        assert len(results) == 2
        assert all(isinstance(r, FailureDetectionResult) for r in results)

    def test_get_failure_statistics(self, failure_detector, clean_finding, over_interpretation_finding):
        """Test aggregate failure statistics."""
        findings = [clean_finding, over_interpretation_finding]
        results = failure_detector.batch_detect(findings)
        stats = failure_detector.get_failure_statistics(results)

        assert stats['count'] == 2
        assert 'pass_rate' in stats
        assert 'failure_rates' in stats
        assert 'avg_scores' in stats

    def test_empty_batch_statistics(self, failure_detector):
        """Test statistics for empty batch."""
        stats = failure_detector.get_failure_statistics([])
        assert stats['count'] == 0
        assert stats['pass_rate'] == 0.0


# =============================================================================
# Threshold Tests
# =============================================================================


class TestThresholds:
    """Tests for threshold behavior."""

    def test_custom_over_interpretation_threshold(
        self, custom_threshold_detector, clean_finding
    ):
        """Test custom over-interpretation threshold."""
        result = custom_threshold_detector.detect_over_interpretation(clean_finding)
        # With lower threshold, more likely to detect
        assert custom_threshold_detector.over_interpretation_threshold == 0.5

    def test_threshold_boundary(self, failure_detector):
        """Test behavior at exact threshold boundary."""
        # Create finding that should be just at threshold
        finding = {
            'finding_id': 'boundary_001',
            'summary': 'This demonstrates the effect',
            'statistics': {'p_value': 0.04, 'effect_size': 0.3},
            'interpretation': 'Results show a clear pattern.',
        }
        result = failure_detector.detect_over_interpretation(finding)
        # Should have a score close to threshold
        assert 0.0 <= result.score <= 1.0

    def test_all_thresholds_configurable(self):
        """Test that all thresholds are configurable."""
        detector = FailureDetector(
            over_interpretation_threshold=0.1,
            invented_metrics_threshold=0.2,
            rabbit_hole_threshold=0.3,
            similarity_threshold=0.4,
        )
        assert detector.over_interpretation_threshold == 0.1
        assert detector.invented_metrics_threshold == 0.2
        assert detector.rabbit_hole_threshold == 0.3
        assert detector.similarity_threshold == 0.4
