"""
Integration tests for FailureDetector in the validation pipeline.

Tests end-to-end failure detection scenarios and integration with
ScholarEval and the research loop.

Issue: #63 (GAP-010)
"""

import pytest
import time
from typing import Dict, Any, List

from kosmos.validation import (
    FailureDetector,
    FailureDetectionResult,
    FailureModeScore,
    ScholarEvalValidator,
    ScholarEvalScore,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def failure_detector():
    """Create FailureDetector for integration testing."""
    return FailureDetector()


@pytest.fixture
def scholar_validator():
    """Create ScholarEvalValidator without LLM client."""
    return ScholarEvalValidator(anthropic_client=None)


@pytest.fixture
def real_world_finding():
    """A realistic research finding from gene expression analysis."""
    return {
        'finding_id': 'gex_001',
        'summary': 'Genetic analysis identified 42 significant variants associated with breast cancer susceptibility in European populations.',
        'statistics': {
            'test_type': 'GWAS',
            'p_value': 0.0001,
            'effect_size': 0.72,
            'sample_size': 200,
            'fdr': 0.05,
            'significant_variants': 42,
            'total_variants_tested': 20000,
        },
        'methods': 'Genome-wide association study was performed to identify genetic factors contributing to breast cancer risk.',
        'interpretation': 'These results suggest that the identified genetic variants may contribute to breast cancer susceptibility. The moderate effect size indicates a meaningful biological signal.',
        'hypothesis_id': 'hyp_cancer_001',
    }


@pytest.fixture
def over_interpreted_finding():
    """A finding with over-interpretation issues."""
    return {
        'finding_id': 'over_001',
        'summary': 'Analysis definitively proves that Gene X causes cancer.',
        'statistics': {
            'p_value': 0.07,  # Not significant
            'effect_size': 0.15,  # Small effect
            'sample_size': 25,  # Small sample
        },
        'methods': 'Basic correlation analysis',
        'interpretation': 'This conclusively establishes Gene X as the primary driver of cancer. The results undoubtedly confirm our hypothesis and clearly demonstrate causality.',
        'hypothesis_id': 'hyp_genex_001',
    }


@pytest.fixture
def invented_metrics_finding():
    """A finding with invented metrics issues."""
    return {
        'finding_id': 'invented_001',
        'summary': 'The synergy_coefficient = 0.92 indicates strong interaction',
        'statistics': {
            'p_value': 0.01,
            'effect_size': 0.5,
        },
        'interpretation': 'The quantum_significance = 0.99 confirms our findings. Our novel impact_factor = 0.87 shows high importance.',
    }


@pytest.fixture
def rabbit_hole_finding():
    """A finding that has drifted from the research question."""
    return {
        'finding_id': 'drift_001',
        'summary': 'Analysis of social media sentiment shows positive trends for tech companies.',
        'statistics': {
            'p_value': 0.001,
            'correlation': 0.65,
            'sample_size': 10000,
        },
        'interpretation': 'Twitter sentiment analysis reveals investor confidence in technology sector.',
    }


@pytest.fixture
def cancer_research_question():
    """Original research question about cancer genetics."""
    return "What genetic factors contribute to breast cancer susceptibility in populations of European ancestry?"


@pytest.fixture
def research_context(cancer_research_question):
    """Full research context for failure detection."""
    return {
        'research_question': cancer_research_question,
        'dataset_schema': [
            'gene_id', 'gene_name', 'log_fold_change', 'p_value', 'fdr',
            'sample_id', 'tissue_type', 'patient_age', 'tumor_stage',
        ],
        'hypothesis': {
            'hypothesis_id': 'hyp_001',
            'statement': 'Specific gene mutations increase breast cancer risk',
            'generation': 1,
            'refinement_count': 0,
        },
        'prior_findings': [
            {'finding_id': 'prev_001', 'summary': 'Initial gene expression patterns identified'},
        ],
    }


# =============================================================================
# End-to-End Detection Tests
# =============================================================================


class TestEndToEndDetection:
    """End-to-end tests for failure detection scenarios."""

    def test_real_world_finding_passes(
        self, failure_detector, real_world_finding, research_context
    ):
        """Test that a realistic, well-formed finding passes validation."""
        result = failure_detector.detect_failures(real_world_finding, research_context)

        assert result.passes_validation is True
        assert result.over_interpretation.detected is False
        assert result.invented_metrics.detected is False
        # May or may not be rabbit hole depending on RQ match

    def test_over_interpreted_finding_detected(
        self, failure_detector, over_interpreted_finding, research_context
    ):
        """Test that over-interpreted finding is detected."""
        result = failure_detector.detect_failures(over_interpreted_finding, research_context)

        assert result.over_interpretation.detected is True
        assert result.passes_validation is False
        assert 'Over-interpretation' in ' '.join(result.warnings)

    def test_invented_metrics_finding_detected(
        self, failure_detector, invented_metrics_finding, research_context
    ):
        """Test that invented metrics are detected."""
        result = failure_detector.detect_failures(invented_metrics_finding, research_context)

        assert result.invented_metrics.detected is True
        assert result.passes_validation is False

    def test_rabbit_hole_finding_detected(
        self, failure_detector, rabbit_hole_finding, cancer_research_question
    ):
        """Test that rabbit hole (research drift) is detected."""
        context = {
            'research_question': cancer_research_question,
        }
        result = failure_detector.detect_failures(rabbit_hole_finding, context)

        # Social media analysis is unrelated to cancer genetics
        assert result.rabbit_hole.score > 0.5

    def test_multiple_failures_detected(self, failure_detector, cancer_research_question):
        """Test detection of multiple simultaneous failure modes."""
        problematic_finding = {
            'finding_id': 'multi_001',
            'summary': 'The magic_index = 0.99 proves pizza consumption causes happiness.',
            'statistics': {'p_value': 0.2},  # Not significant
            'interpretation': 'This conclusively establishes the synergy_factor = 0.88.',
        }
        context = {'research_question': cancer_research_question}
        result = failure_detector.detect_failures(problematic_finding, context)

        failures_detected = sum([
            result.over_interpretation.detected,
            result.invented_metrics.detected,
            result.rabbit_hole.detected,
        ])
        # Should detect at least 2 failure modes
        assert failures_detected >= 2


# =============================================================================
# ScholarEval Integration Tests
# =============================================================================


class TestScholarEvalIntegration:
    """Tests for integration with ScholarEvalValidator."""

    def test_failure_detection_after_scholar_eval(
        self, failure_detector, scholar_validator, real_world_finding, research_context
    ):
        """Test failure detection works alongside ScholarEval."""
        # First run ScholarEval
        scholar_score = scholar_validator.evaluate_finding(real_world_finding)
        assert isinstance(scholar_score, ScholarEvalScore)

        # Then run failure detection
        failure_result = failure_detector.detect_failures(real_world_finding, research_context)
        assert isinstance(failure_result, FailureDetectionResult)

        # Both should pass for good finding
        if scholar_score.passes_threshold:
            # If ScholarEval passes, failure detection likely passes too
            assert failure_result.over_interpretation.score < 0.8

    def test_combined_validation_workflow(
        self, failure_detector, scholar_validator, over_interpreted_finding, research_context
    ):
        """Test combined ScholarEval + Failure Detection workflow."""
        # ScholarEval
        scholar_score = scholar_validator.evaluate_finding(over_interpreted_finding)

        # Failure Detection
        failure_result = failure_detector.detect_failures(
            over_interpreted_finding, research_context
        )

        # Either or both should catch the problematic finding
        validation_failed = (
            not scholar_score.passes_threshold or
            failure_result.has_failures
        )
        assert validation_failed

    def test_failure_result_can_be_stored(
        self, failure_detector, real_world_finding, research_context
    ):
        """Test that failure result can be serialized for storage."""
        result = failure_detector.detect_failures(real_world_finding, research_context)
        result_dict = result.to_dict()

        # Should be JSON-serializable
        import json
        json_str = json.dumps(result_dict)
        assert len(json_str) > 0

        # Should be reconstructible
        reconstructed = FailureDetectionResult.from_dict(result_dict)
        assert reconstructed.finding_id == result.finding_id
        assert reconstructed.overall_score == result.overall_score

    def test_scholar_eval_score_structure_compatible(
        self, scholar_validator, real_world_finding
    ):
        """Test ScholarEvalScore structure is compatible for extension."""
        score = scholar_validator.evaluate_finding(real_world_finding)
        score_dict = score.to_dict()

        # ScholarEvalScore has fields that could include failure_detection_result
        assert 'overall_score' in score_dict
        assert 'passes_threshold' in score_dict


# =============================================================================
# Research Loop Integration Tests
# =============================================================================


class TestResearchLoopIntegration:
    """Tests simulating integration with research loop."""

    def test_validation_pipeline_flow(
        self, failure_detector, scholar_validator, real_world_finding, research_context
    ):
        """Test complete validation pipeline as used in research loop."""
        # Simulate task result from research loop
        task_result = {
            'task_id': 1,
            'cycle': 1,
            'finding': real_world_finding,
            'status': 'completed',
        }

        # Step 1: ScholarEval validation
        scholar_score = scholar_validator.evaluate_finding(task_result['finding'])

        # Step 2: Failure mode detection (only if ScholarEval passes or always)
        failure_result = failure_detector.detect_failures(
            task_result['finding'], research_context
        )

        # Step 3: Combined decision
        should_save = (
            scholar_score.passes_threshold and
            failure_result.passes_validation
        )

        # For good finding, should save
        # (depends on mock ScholarEval behavior)
        assert isinstance(should_save, bool)

    def test_context_building_from_cycle(self, failure_detector, real_world_finding):
        """Test building context from research cycle data."""
        # Simulate cycle context (as would come from ArtifactStateManager)
        cycle_context = {
            'cycle': 5,
            'findings_count': 10,
            'recent_findings': [
                {'finding_id': 'prev_001', 'summary': 'Previous result'},
            ],
            'unsupported_hypotheses': [],
            'statistics': {
                'total_findings': 50,
                'validated_findings': 40,
            },
        }

        # Build failure detection context from cycle context
        detection_context = {
            'research_question': 'Cancer genetics research',
            'prior_findings': cycle_context['recent_findings'],
        }

        result = failure_detector.detect_failures(real_world_finding, detection_context)
        assert isinstance(result, FailureDetectionResult)

    def test_finding_augmentation_with_failure_result(
        self, failure_detector, real_world_finding, research_context
    ):
        """Test augmenting finding with failure detection result."""
        # Run failure detection
        result = failure_detector.detect_failures(real_world_finding, research_context)

        # Augment finding with result (as would be done before saving)
        augmented_finding = real_world_finding.copy()
        augmented_finding['failure_detection_result'] = result.to_dict()

        # Verify augmented finding has all expected fields
        assert 'failure_detection_result' in augmented_finding
        assert 'over_interpretation' in augmented_finding['failure_detection_result']
        assert 'invented_metrics' in augmented_finding['failure_detection_result']
        assert 'rabbit_hole' in augmented_finding['failure_detection_result']


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Tests for performance characteristics."""

    def test_single_detection_performance(
        self, failure_detector, real_world_finding, research_context
    ):
        """Test that single detection completes in reasonable time."""
        start = time.time()
        result = failure_detector.detect_failures(real_world_finding, research_context)
        elapsed = time.time() - start

        # Should complete in under 1 second (no LLM calls)
        assert elapsed < 1.0
        assert result.computation_time_seconds < 1.0

    def test_batch_detection_performance(self, failure_detector, research_context):
        """Test batch detection performance."""
        # Create 50 findings
        findings = []
        for i in range(50):
            findings.append({
                'finding_id': f'perf_{i}',
                'summary': f'Finding {i} about gene expression',
                'statistics': {'p_value': 0.01, 'effect_size': 0.5},
                'interpretation': 'Results suggest a relationship.',
            })

        start = time.time()
        results = failure_detector.batch_detect(findings, research_context)
        elapsed = time.time() - start

        assert len(results) == 50
        # 50 findings should complete in under 5 seconds
        assert elapsed < 5.0

    def test_memory_usage_large_text(self, failure_detector):
        """Test memory handling with large text content."""
        large_text = 'This suggests a relationship. ' * 5000  # ~150KB
        finding = {
            'finding_id': 'mem_001',
            'summary': large_text,
            'statistics': {'p_value': 0.01},
            'interpretation': large_text,
        }

        # Should complete without memory issues
        result = failure_detector.detect_failures(finding)
        assert isinstance(result, FailureDetectionResult)


# =============================================================================
# Realistic Scenario Tests
# =============================================================================


class TestRealisticScenarios:
    """Tests for realistic research scenarios."""

    def test_borderline_significant_finding(self, failure_detector, research_context):
        """Test handling of borderline significant finding."""
        finding = {
            'finding_id': 'border_001',
            'summary': 'Analysis shows a trend toward significance',
            'statistics': {
                'p_value': 0.06,  # Just above threshold
                'effect_size': 0.45,
                'sample_size': 80,
            },
            'interpretation': 'Results indicate a possible association that approaches statistical significance.',
        }

        result = failure_detector.detect_failures(finding, research_context)

        # Should not flag as over-interpretation since claims are appropriately hedged
        assert result.over_interpretation.score < 0.6

    def test_exploratory_analysis_finding(self, failure_detector, research_context):
        """Test handling of exploratory analysis finding."""
        finding = {
            'finding_id': 'explore_001',
            'summary': 'Exploratory analysis identified potential biomarkers',
            'statistics': {
                'p_value': 0.001,
                'effect_size': 0.8,
                'sample_size': 50,
                'multiple_testing_correction': 'none',  # Exploratory
            },
            'interpretation': 'These preliminary findings suggest candidates for follow-up validation studies.',
        }

        result = failure_detector.detect_failures(finding, research_context)

        # Hedged language should pass
        assert result.over_interpretation.score < 0.7

    def test_negative_result_finding(self, failure_detector, research_context):
        """Test handling of negative/null result."""
        finding = {
            'finding_id': 'null_001',
            'summary': 'No significant association was found between Gene X and cancer risk',
            'statistics': {
                'p_value': 0.45,
                'effect_size': 0.05,
                'sample_size': 500,
            },
            'interpretation': 'The data do not support an association between Gene X and cancer risk in this population.',
        }

        result = failure_detector.detect_failures(finding, research_context)

        # Negative results with appropriate language should pass
        assert result.over_interpretation.detected is False

    def test_replication_study_finding(self, failure_detector, research_context):
        """Test handling of replication study finding."""
        finding = {
            'finding_id': 'rep_001',
            'summary': 'Replication study confirms genetic factors associated with breast cancer susceptibility in European populations',
            'statistics': {
                'p_value': 0.0001,
                'effect_size': 0.75,
                'sample_size': 1000,
            },
            'interpretation': 'This replication study provides support for the previously reported genetic associations with cancer risk.',
        }

        result = failure_detector.detect_failures(finding, research_context)

        # Strong stats + moderate claims should have low over-interpretation
        assert result.over_interpretation.detected is False


# =============================================================================
# Edge Case Integration Tests
# =============================================================================


class TestEdgeCaseIntegration:
    """Integration tests for edge cases."""

    def test_missing_context(self, failure_detector, real_world_finding):
        """Test detection without context."""
        result = failure_detector.detect_failures(real_world_finding, context=None)

        # Should still work, just with limited rabbit hole detection
        assert isinstance(result, FailureDetectionResult)
        assert result.rabbit_hole.confidence < 0.5

    def test_partial_context(self, failure_detector, real_world_finding):
        """Test detection with partial context."""
        partial_context = {
            'research_question': 'Cancer genetics research',
            # Missing dataset_schema and hypothesis
        }

        result = failure_detector.detect_failures(real_world_finding, partial_context)
        assert isinstance(result, FailureDetectionResult)

    def test_conflicting_signals(self, failure_detector, cancer_research_question):
        """Test finding with conflicting quality signals."""
        finding = {
            'finding_id': 'conflict_001',
            'summary': 'Analysis of breast cancer genetics identifies key mutations',
            'statistics': {
                'p_value': 0.08,  # NOT significant
                'effect_size': 0.15,  # Small effect
                # Weak statistics to contrast with strong claims
            },
            'interpretation': 'This conclusively proves mutations cause cancer.',
            # Over-confident claims
        }
        context = {'research_question': cancer_research_question}

        result = failure_detector.detect_failures(finding, context)

        # Weak stats but strong claims = over-interpretation
        assert result.over_interpretation.score > 0.3
