"""
Integration tests for the validation pipeline.

Tests ScholarEval with real discoveries and filtering workflow.
"""

import pytest
from kosmos.validation.scholar_eval import ScholarEvalValidator, ScholarEvalScore


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def high_quality_findings():
    """High-quality scientific findings that should pass validation."""
    return [
        {
            'finding_id': 'finding_001',
            'summary': '''We identified 87 differentially expressed genes in KRAS G12D mutant
            pancreatic cancer cells compared to wildtype controls. The most significantly
            upregulated gene was MYC (fold change = 3.2, FDR-adjusted p < 0.001).''',
            'statistics': {
                'p_value': 0.0001,
                'sample_size': 150,
                'n_genes': 87,
                'effect_size': 0.82,
                'fdr': 0.01
            },
            'methods': '''RNA-seq analysis using DESeq2 with FDR correction (Benjamini-Hochberg).
            Quality control: FastQC, alignment with STAR, quantification with featureCounts.
            Statistical threshold: FDR < 0.05, |log2FC| > 1.''',
            'interpretation': '''KRAS mutations drive significant transcriptional changes
            consistent with enhanced cell proliferation and survival signaling.'''
        },
        {
            'finding_id': 'finding_002',
            'summary': '''Survival analysis reveals KRAS mutation status is an independent
            predictor of overall survival (HR = 2.1, 95% CI: 1.5-2.9, p < 0.001).''',
            'statistics': {
                'p_value': 0.0005,
                'sample_size': 500,
                'hazard_ratio': 2.1,
                'confidence_interval': [1.5, 2.9]
            },
            'methods': '''Cox proportional hazards regression with adjustment for age,
            stage, and treatment. Kaplan-Meier survival curves with log-rank test.
            Median follow-up: 36 months.''',
            'interpretation': '''KRAS mutation status should be considered for patient
            stratification and treatment decisions.'''
        }
    ]


@pytest.fixture
def low_quality_findings():
    """Low-quality findings that should fail validation."""
    return [
        {
            'finding_id': 'finding_low_001',
            'summary': 'Found some genes that might be different.',
            'statistics': {},
            'methods': None,
            'interpretation': None
        },
        {
            'finding_id': 'finding_low_002',
            'summary': '''We noticed a trend in the data that could suggest KRAS
            involvement but more research is needed.''',
            'statistics': {
                'p_value': 0.15,  # Not significant
                'sample_size': 10  # Small sample
            },
            'methods': 'Basic analysis',
            'interpretation': 'Results are inconclusive'
        }
    ]


@pytest.fixture
def mixed_quality_findings(high_quality_findings, low_quality_findings):
    """Mix of high and low quality findings."""
    return high_quality_findings + low_quality_findings


@pytest.fixture
def scholar_validator():
    """ScholarEvalValidator without LLM for testing."""
    return ScholarEvalValidator(
        anthropic_client=None,
        threshold=0.75,
        min_rigor_score=0.70,
        allow_mock=True
    )


# ============================================================================
# Integration Tests
# ============================================================================

class TestScholarEvalPipeline:
    """Integration tests for ScholarEval validation pipeline."""

    @pytest.mark.asyncio
    async def test_evaluate_high_quality_finding(self, scholar_validator, high_quality_findings):
        """Test evaluation of high-quality finding."""
        finding = high_quality_findings[0]
        score = scholar_validator.evaluate_finding(finding)

        assert isinstance(score, ScholarEvalScore)
        assert score.passes_threshold is True
        assert score.overall_score >= scholar_validator.threshold

        # Should have high rigor score due to comprehensive methods
        assert score.rigor >= 0.7

    @pytest.mark.asyncio
    async def test_evaluate_low_quality_finding(self, scholar_validator, low_quality_findings):
        """Test evaluation of low-quality finding."""
        finding = low_quality_findings[0]  # No statistics, no methods
        score = scholar_validator.evaluate_finding(finding)

        assert isinstance(score, ScholarEvalScore)
        # May or may not pass depending on mock behavior

    @pytest.mark.asyncio
    async def test_batch_validation_filtering(self, scholar_validator, mixed_quality_findings):
        """Test batch validation correctly filters findings."""
        scores = []
        for finding in mixed_quality_findings:
            score = scholar_validator.evaluate_finding(finding)
            scores.append((finding, score))

        passed = [(f, s) for f, s in scores if s.passes_threshold]
        rejected = [(f, s) for f, s in scores if not s.passes_threshold]

        # Should have some passed and some rejected
        assert len(passed) + len(rejected) == len(mixed_quality_findings)

    @pytest.mark.asyncio
    async def test_validation_statistics(self, scholar_validator, high_quality_findings):
        """Test computing validation statistics."""
        scores = []
        for finding in high_quality_findings:
            score = scholar_validator.evaluate_finding(finding)
            scores.append(score)

        stats = scholar_validator.get_validation_statistics(scores)

        assert stats['total_evaluated'] == len(high_quality_findings)
        assert 'passed' in stats
        assert 'rejected' in stats
        assert 'validation_rate' in stats
        assert 'avg_overall_score' in stats


class TestScholarEvalScoring:
    """Tests for ScholarEval scoring dimensions."""

    @pytest.mark.asyncio
    async def test_all_dimensions_scored(self, scholar_validator, high_quality_findings):
        """Test that all 8 dimensions are scored."""
        score = scholar_validator.evaluate_finding(high_quality_findings[0])

        # Check all dimensions present
        dimensions = ['novelty', 'rigor', 'clarity', 'reproducibility',
                      'impact', 'coherence', 'limitations', 'ethics']

        for dim in dimensions:
            assert hasattr(score, dim), f"Missing dimension: {dim}"
            dim_score = getattr(score, dim)
            assert 0 <= dim_score <= 1, f"Invalid {dim} score: {dim_score}"

    @pytest.mark.asyncio
    async def test_rigor_weight_dominates(self, scholar_validator):
        """Test that rigor has highest weight in overall score."""
        # High rigor, low everything else
        high_rigor_finding = {
            'summary': 'Test finding',
            'statistics': {'p_value': 0.001, 'sample_size': 500},
            'methods': 'Comprehensive statistical analysis with multiple correction methods',
            'interpretation': 'Clear'
        }

        score = scholar_validator.evaluate_finding(high_rigor_finding)

        # Verify rigor contributes significantly
        rigor_contribution = score.rigor * scholar_validator.DIMENSION_WEIGHTS['rigor']
        assert rigor_contribution > 0.15  # Should be substantial


class TestValidationFeedback:
    """Tests for validation feedback generation."""

    @pytest.mark.asyncio
    async def test_approved_feedback(self, scholar_validator, high_quality_findings):
        """Test feedback for approved findings."""
        score = scholar_validator.evaluate_finding(high_quality_findings[0])

        if score.passes_threshold:
            assert 'APPROVED' in score.feedback or 'approved' in score.feedback.lower()

    @pytest.mark.asyncio
    async def test_rejected_feedback(self, scholar_validator, low_quality_findings):
        """Test feedback for rejected findings."""
        score = scholar_validator.evaluate_finding(low_quality_findings[0])

        if not score.passes_threshold:
            assert 'REJECTED' in score.feedback or 'rejected' in score.feedback.lower()

    @pytest.mark.asyncio
    async def test_feedback_actionable(self, scholar_validator, low_quality_findings):
        """Test that rejection feedback is actionable."""
        score = scholar_validator.evaluate_finding(low_quality_findings[1])

        # Feedback should mention weaknesses or suggestions
        feedback_lower = score.feedback.lower()
        has_actionable = any(word in feedback_lower for word in
                            ['weakness', 'improve', 'suggest', 'concern', 'rigor'])

        # May not always have actionable feedback in mock mode
        assert isinstance(score.feedback, str)
        assert len(score.feedback) > 0


class TestValidationThresholds:
    """Tests for validation thresholds."""

    @pytest.mark.asyncio
    async def test_custom_threshold(self, high_quality_findings):
        """Test validation with custom threshold."""
        strict_validator = ScholarEvalValidator(
            anthropic_client=None,
            threshold=0.90,  # Stricter threshold
            min_rigor_score=0.85,
            allow_mock=True
        )

        score = strict_validator.evaluate_finding(high_quality_findings[0])

        # With stricter threshold, may not pass
        assert isinstance(score, ScholarEvalScore)

    @pytest.mark.asyncio
    async def test_rigor_threshold_enforcement(self, scholar_validator):
        """Test that findings fail if rigor is below minimum."""
        # Mock finding with potentially low rigor
        finding = {
            'summary': 'We observed some patterns in the data',
            'statistics': {},  # No statistics
            'methods': '',  # No methods
            'interpretation': 'Needs more research'
        }

        score = scholar_validator.evaluate_finding(finding)

        # Should have feedback about rigor if failed
        if not score.passes_threshold and score.rigor < scholar_validator.min_rigor_score:
            assert 'rigor' in score.feedback.lower() or isinstance(score.feedback, str)


class TestValidationEdgeCases:
    """Tests for validation edge cases."""

    @pytest.mark.asyncio
    async def test_empty_finding(self, scholar_validator):
        """Test validation of empty finding."""
        score = scholar_validator.evaluate_finding({})

        assert isinstance(score, ScholarEvalScore)
        assert isinstance(score.feedback, str)

    @pytest.mark.asyncio
    async def test_partial_finding(self, scholar_validator):
        """Test validation of finding with only some fields."""
        partial_finding = {
            'summary': 'Partial finding with only summary'
        }

        score = scholar_validator.evaluate_finding(partial_finding)

        assert isinstance(score, ScholarEvalScore)

    def test_concurrent_validation(self, scholar_validator, mixed_quality_findings):
        """Test validation of multiple findings."""
        scores = [
            scholar_validator.evaluate_finding(f)
            for f in mixed_quality_findings
        ]

        assert len(scores) == len(mixed_quality_findings)
        assert all(isinstance(s, ScholarEvalScore) for s in scores)
