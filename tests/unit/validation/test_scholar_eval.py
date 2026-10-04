"""
Unit tests for kosmos.validation.scholar_eval module.

Tests:
- ScholarEvalScore dataclass
- ScholarEvalValidator: 8-dimension scoring, validation thresholds
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch

from kosmos.validation.scholar_eval import (
    ScholarEvalScore,
    ScholarEvalValidator
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def high_quality_finding():
    """High quality finding that should pass validation."""
    return {
        'summary': 'We identified 42 differentially expressed genes associated with KRAS mutations',
        'statistics': {
            'test_type': 't_test',
            'statistic': 4.5,  # Strong test statistic for null model validation
            'p_value': 0.001,
            'sample_size': 150,
            'effect_size': 0.85,
            'degrees_of_freedom': 148,
            'fdr': 0.05
        },
        'methods': 'DESeq2 differential expression analysis with FDR correction',
        'interpretation': 'KRAS mutations are associated with significant transcriptional changes'
    }


@pytest.fixture
def low_quality_finding():
    """Low quality finding that should fail validation."""
    return {
        'summary': 'Found some genes',
        'statistics': {},
        'methods': None,
        'interpretation': None
    }


@pytest.fixture
def scholar_validator():
    """Create ScholarEvalValidator without LLM client."""
    return ScholarEvalValidator(anthropic_client=None, allow_mock=True)


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for evaluation."""
    response_content = json.dumps({
        'novelty': 0.85,
        'rigor': 0.90,
        'clarity': 0.80,
        'reproducibility': 0.85,
        'impact': 0.75,
        'coherence': 0.80,
        'limitations': 0.70,
        'ethics': 0.75,
        'reasoning': 'High quality finding with strong statistical support'
    })

    mock_response = Mock()
    mock_response.content = [Mock(text=response_content)]
    return mock_response


# ============================================================================
# ScholarEvalScore Tests
# ============================================================================

class TestScholarEvalScore:
    """Tests for ScholarEvalScore dataclass."""

    def test_basic_creation(self):
        """Test basic ScholarEvalScore creation."""
        score = ScholarEvalScore(
            novelty=0.8,
            rigor=0.85,
            clarity=0.75,
            reproducibility=0.80,
            impact=0.70,
            coherence=0.75,
            limitations=0.65,
            ethics=0.70,
            overall_score=0.78,
            passes_threshold=True,
            feedback='Good finding'
        )

        assert score.novelty == 0.8
        assert score.overall_score == 0.78
        assert score.passes_threshold is True

    def test_to_dict(self):
        """Test ScholarEvalScore to dictionary conversion."""
        score = ScholarEvalScore(
            novelty=0.8,
            rigor=0.85,
            clarity=0.75,
            reproducibility=0.80,
            impact=0.70,
            coherence=0.75,
            limitations=0.65,
            ethics=0.70,
            overall_score=0.78,
            passes_threshold=True,
            feedback='Good',
            reasoning='Test reasoning'
        )

        result = score.to_dict()

        assert result['novelty'] == 0.8
        assert result['overall_score'] == 0.78
        assert result['reasoning'] == 'Test reasoning'

    def test_from_dict(self):
        """Test creating ScholarEvalScore from dictionary."""
        data = {
            'novelty': 0.8,
            'rigor': 0.85,
            'clarity': 0.75,
            'reproducibility': 0.80,
            'impact': 0.70,
            'coherence': 0.75,
            'limitations': 0.65,
            'ethics': 0.70,
            'overall_score': 0.78,
            'passes_threshold': True,
            'feedback': 'Good'
        }

        score = ScholarEvalScore.from_dict(data)

        assert score.novelty == 0.8
        assert score.passes_threshold is True


# ============================================================================
# ScholarEvalValidator Initialization Tests
# ============================================================================

class TestScholarEvalValidatorInit:
    """Tests for ScholarEvalValidator initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        validator = ScholarEvalValidator()

        assert validator.client is None
        assert validator.threshold == 0.75
        assert validator.min_rigor_score == 0.70

    def test_custom_initialization(self):
        """Test custom initialization."""
        mock_client = Mock()
        validator = ScholarEvalValidator(
            anthropic_client=mock_client,
            threshold=0.80,
            min_rigor_score=0.75
        )

        assert validator.client == mock_client
        assert validator.threshold == 0.80
        assert validator.min_rigor_score == 0.75

    def test_dimension_weights_defined(self):
        """Test that dimension weights are defined and sum to 1."""
        validator = ScholarEvalValidator()

        assert len(validator.DIMENSION_WEIGHTS) == 8
        assert abs(sum(validator.DIMENSION_WEIGHTS.values()) - 1.0) < 0.001

    def test_rigor_highest_weight(self):
        """Test that rigor has the highest weight."""
        validator = ScholarEvalValidator()

        assert validator.DIMENSION_WEIGHTS['rigor'] == max(validator.DIMENSION_WEIGHTS.values())


# ============================================================================
# Mock Evaluation Tests
# ============================================================================

class TestMockEvaluation:
    """Tests for mock evaluation (no LLM)."""

    @pytest.mark.asyncio
    async def test_mock_evaluation_high_quality(self, scholar_validator, high_quality_finding):
        """Test mock evaluation of high quality finding."""
        score = scholar_validator.evaluate_finding(high_quality_finding)

        assert isinstance(score, ScholarEvalScore)
        # With statistics and methods, should pass
        assert score.passes_threshold is True

    @pytest.mark.asyncio
    async def test_mock_evaluation_low_quality(self, scholar_validator, low_quality_finding):
        """Test mock evaluation of low quality finding."""
        score = scholar_validator.evaluate_finding(low_quality_finding)

        assert isinstance(score, ScholarEvalScore)
        # May or may not pass depending on mock implementation

    @pytest.mark.asyncio
    async def test_mock_evaluation_scores_valid(self, scholar_validator, high_quality_finding):
        """Test that mock evaluation provides valid scores."""
        score = scholar_validator.evaluate_finding(high_quality_finding)

        # All scores should be between 0 and 1
        assert 0 <= score.novelty <= 1
        assert 0 <= score.rigor <= 1
        assert 0 <= score.clarity <= 1
        assert 0 <= score.reproducibility <= 1
        assert 0 <= score.impact <= 1
        assert 0 <= score.coherence <= 1
        assert 0 <= score.limitations <= 1
        assert 0 <= score.ethics <= 1
        assert 0 <= score.overall_score <= 1


# ============================================================================
# LLM Evaluation Tests
# ============================================================================

class TestLLMEvaluation:
    """Tests for LLM-based evaluation."""

    @pytest.mark.asyncio
    async def test_evaluation_with_llm(self, high_quality_finding, mock_llm_response):
        """Test evaluation with LLM client."""
        mock_client = Mock()
        mock_client.messages.create = AsyncMock(return_value=mock_llm_response)

        validator = ScholarEvalValidator(anthropic_client=mock_client, allow_mock=True)

        score = validator.evaluate_finding(high_quality_finding)

        assert isinstance(score, ScholarEvalScore)
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self, high_quality_finding):
        """Test fallback to mock evaluation on LLM failure."""
        mock_client = Mock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API Error"))

        validator = ScholarEvalValidator(anthropic_client=mock_client, allow_mock=True)

        score = validator.evaluate_finding(high_quality_finding)

        # Should still return a valid score (mock)
        assert isinstance(score, ScholarEvalScore)


# ============================================================================
# Score Calculation Tests
# ============================================================================

class TestScoreCalculation:
    """Tests for score calculation."""

    def test_calculate_overall_score(self, scholar_validator):
        """Test weighted overall score calculation."""
        scores = {
            'novelty': 0.8,
            'rigor': 0.9,
            'clarity': 0.7,
            'reproducibility': 0.8,
            'impact': 0.7,
            'coherence': 0.75,
            'limitations': 0.6,
            'ethics': 0.7
        }

        overall = scholar_validator._calculate_overall_score(scores)

        # Should be weighted average
        expected = sum(
            scores[dim] * scholar_validator.DIMENSION_WEIGHTS[dim]
            for dim in scores
        )
        assert abs(overall - expected) < 0.001

    def test_calculate_overall_score_uniform(self, scholar_validator):
        """Test overall score with uniform dimension scores."""
        uniform_score = 0.8
        scores = {dim: uniform_score for dim in scholar_validator.DIMENSION_WEIGHTS}

        overall = scholar_validator._calculate_overall_score(scores)

        assert abs(overall - uniform_score) < 0.001

    def test_calculate_overall_score_missing_dims(self, scholar_validator):
        """Test overall score with missing dimensions (uses default 0.5)."""
        scores = {'rigor': 0.9}  # Only one dimension

        overall = scholar_validator._calculate_overall_score(scores)

        # Should use 0.5 for missing dimensions
        assert overall > 0


# ============================================================================
# Prompt Building Tests
# ============================================================================

class TestPromptBuilding:
    """Tests for evaluation prompt construction."""

    def test_build_evaluation_prompt(self, scholar_validator, high_quality_finding):
        """Test building evaluation prompt."""
        prompt = scholar_validator._build_evaluation_prompt(high_quality_finding)

        assert 'Finding' in prompt
        assert 'Statistics' in prompt
        assert 'Methods' in prompt
        assert 'Novelty' in prompt
        assert 'Rigor' in prompt
        assert '0.0-1.0' in prompt

    def test_prompt_includes_finding_content(self, scholar_validator, high_quality_finding):
        """Test that prompt includes finding content."""
        prompt = scholar_validator._build_evaluation_prompt(high_quality_finding)

        assert '42 differentially expressed' in prompt
        assert 'DESeq2' in prompt

    def test_prompt_handles_missing_fields(self, scholar_validator):
        """Test prompt handles missing finding fields."""
        minimal_finding = {'summary': 'Test'}

        prompt = scholar_validator._build_evaluation_prompt(minimal_finding)

        assert 'Test' in prompt
        assert 'No summary' not in prompt  # Should use actual summary


# ============================================================================
# Response Parsing Tests
# ============================================================================

class TestResponseParsing:
    """Tests for LLM response parsing."""

    def test_parse_valid_response(self, scholar_validator):
        """Test parsing valid JSON response."""
        response = json.dumps({
            'novelty': 0.85,
            'rigor': 0.90,
            'clarity': 0.80,
            'reproducibility': 0.85,
            'impact': 0.75,
            'coherence': 0.80,
            'limitations': 0.70,
            'ethics': 0.75,
            'reasoning': 'Good finding'
        })

        result = scholar_validator._parse_llm_response(response)

        assert result['novelty'] == 0.85
        assert result['rigor'] == 0.90
        assert result['reasoning'] == 'Good finding'

    def test_parse_response_with_text(self, scholar_validator):
        """Test parsing response with surrounding text."""
        response = """Based on my analysis:
        {
            "novelty": 0.8,
            "rigor": 0.85,
            "clarity": 0.75,
            "reproducibility": 0.80,
            "impact": 0.70,
            "coherence": 0.75,
            "limitations": 0.65,
            "ethics": 0.70
        }
        This is a solid finding.
        """

        result = scholar_validator._parse_llm_response(response)

        assert result['novelty'] == 0.8

    def test_parse_invalid_json(self, scholar_validator):
        """Test parsing invalid JSON response."""
        response = "This is not valid JSON"

        result = scholar_validator._parse_llm_response(response)

        # Should return default scores
        assert all(result[dim] == 0.5 for dim in ['novelty', 'rigor', 'clarity'])

    def test_parse_score_clamping(self, scholar_validator):
        """Test that scores are clamped to [0, 1]."""
        response = json.dumps({
            'novelty': 1.5,  # Should clamp to 1
            'rigor': -0.5,  # Should clamp to 0
            'clarity': 0.75,
            'reproducibility': 0.80,
            'impact': 0.70,
            'coherence': 0.75,
            'limitations': 0.65,
            'ethics': 0.70
        })

        result = scholar_validator._parse_llm_response(response)

        assert result['novelty'] == 1.0
        assert result['rigor'] == 0.0

    def test_parse_missing_dimensions(self, scholar_validator):
        """Test parsing response with missing dimensions."""
        response = json.dumps({
            'novelty': 0.8,
            'rigor': 0.9
            # Missing other dimensions
        })

        result = scholar_validator._parse_llm_response(response)

        # Missing dimensions should be filled with 0.5
        assert result['novelty'] == 0.8
        assert result['clarity'] == 0.5


# ============================================================================
# Feedback Generation Tests
# ============================================================================

class TestFeedbackGeneration:
    """Tests for feedback generation."""

    def test_feedback_for_approved(self, scholar_validator, high_quality_finding):
        """Test feedback generation for approved finding."""
        scores = {
            'novelty': 0.85,
            'rigor': 0.90,
            'clarity': 0.80,
            'reproducibility': 0.85,
            'impact': 0.75,
            'coherence': 0.80,
            'limitations': 0.70,
            'ethics': 0.75
        }

        feedback = scholar_validator._generate_feedback(scores, True, high_quality_finding)

        assert 'APPROVED' in feedback
        assert 'Strengths' in feedback or 'overall' in feedback.lower()

    def test_feedback_for_rejected(self, scholar_validator, low_quality_finding):
        """Test feedback generation for rejected finding."""
        scores = {
            'novelty': 0.5,
            'rigor': 0.4,  # Low rigor
            'clarity': 0.5,
            'reproducibility': 0.4,
            'impact': 0.5,
            'coherence': 0.5,
            'limitations': 0.3,
            'ethics': 0.5
        }

        feedback = scholar_validator._generate_feedback(scores, False, low_quality_finding)

        assert 'REJECTED' in feedback
        assert 'Weaknesses' in feedback or 'rigor' in feedback.lower()

    def test_feedback_includes_critical_concerns(self, scholar_validator, low_quality_finding):
        """Test that feedback includes critical concerns for low rigor."""
        scores = {
            'novelty': 0.7,
            'rigor': 0.5,  # Below min_rigor_score
            'clarity': 0.7,
            'reproducibility': 0.7,
            'impact': 0.7,
            'coherence': 0.7,
            'limitations': 0.6,
            'ethics': 0.7
        }

        feedback = scholar_validator._generate_feedback(scores, False, low_quality_finding)

        assert 'CRITICAL' in feedback or 'rigor' in feedback.lower()


# ============================================================================
# Batch Evaluation Tests
# ============================================================================

class TestBatchEvaluation:
    """Tests for batch evaluation."""

    def test_batch_evaluate(self, scholar_validator, high_quality_finding, low_quality_finding):
        """Test batch evaluation of multiple findings."""
        findings = [high_quality_finding, low_quality_finding]

        # Note: batch_evaluate is sync, so we need to handle this differently
        # The actual implementation may need to be async
        # This is a placeholder test

    def test_get_validation_statistics(self, scholar_validator):
        """Test computing statistics over evaluations."""
        scores = [
            ScholarEvalScore(
                novelty=0.8, rigor=0.85, clarity=0.75, reproducibility=0.80,
                impact=0.70, coherence=0.75, limitations=0.65, ethics=0.70,
                overall_score=0.78, passes_threshold=True, feedback='Good'
            ),
            ScholarEvalScore(
                novelty=0.5, rigor=0.5, clarity=0.5, reproducibility=0.5,
                impact=0.5, coherence=0.5, limitations=0.5, ethics=0.5,
                overall_score=0.5, passes_threshold=False, feedback='Needs work'
            )
        ]

        stats = scholar_validator.get_validation_statistics(scores)

        assert stats['total_evaluated'] == 2
        assert stats['passed'] == 1
        assert stats['rejected'] == 1
        assert stats['validation_rate'] == 0.5

    def test_get_validation_statistics_empty(self, scholar_validator):
        """Test statistics for empty score list."""
        stats = scholar_validator.get_validation_statistics([])

        assert stats == {}


# ============================================================================
# Edge Cases
# ============================================================================

class TestScholarEvalEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_finding(self, scholar_validator):
        """Test evaluation of empty finding."""
        score = scholar_validator.evaluate_finding({})

        assert isinstance(score, ScholarEvalScore)

    @pytest.mark.asyncio
    async def test_finding_with_none_values(self, scholar_validator):
        """Test evaluation with None values."""
        finding = {
            'summary': 'Test',
            'statistics': None,
            'methods': None,
            'interpretation': None
        }

        score = scholar_validator.evaluate_finding(finding)

        assert isinstance(score, ScholarEvalScore)

    @pytest.mark.asyncio
    async def test_threshold_boundary(self):
        """Test evaluation at threshold boundary."""
        validator = ScholarEvalValidator(threshold=0.75, min_rigor_score=0.70, allow_mock=True)

        finding = {
            'summary': 'Borderline finding',
            'statistics': {'p_value': 0.05}
        }

        score = validator.evaluate_finding(finding)

        # Should be near threshold
        assert isinstance(score, ScholarEvalScore)

    @pytest.mark.asyncio
    async def test_very_long_summary(self, scholar_validator):
        """Test evaluation with very long summary."""
        finding = {
            'summary': 'Very long summary. ' * 500,
            'statistics': {'p_value': 0.01}
        }

        score = scholar_validator.evaluate_finding(finding)

        assert isinstance(score, ScholarEvalScore)
