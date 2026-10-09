"""
Unit tests for kosmos.orchestration.plan_reviewer module.

Tests:
- PlanReview dataclass
- PlanReviewerAgent: plan validation, scoring, structural requirements
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch

from kosmos.orchestration.plan_reviewer import (
    PlanReview,
    PlanReviewerAgent
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def valid_plan():
    """Valid plan that should pass review."""
    return {
        'cycle': 1,
        'tasks': [
            {'id': 1, 'type': 'data_analysis', 'description': 'Task 1', 'expected_output': 'Output 1'},
            {'id': 2, 'type': 'data_analysis', 'description': 'Task 2', 'expected_output': 'Output 2'},
            {'id': 3, 'type': 'data_analysis', 'description': 'Task 3', 'expected_output': 'Output 3'},
            {'id': 4, 'type': 'literature_review', 'description': 'Task 4', 'expected_output': 'Output 4'},
            {'id': 5, 'type': 'hypothesis_generation', 'description': 'Task 5', 'expected_output': 'Output 5'}
        ],
        'rationale': 'Test plan'
    }


@pytest.fixture
def invalid_plan_few_data_analysis():
    """Plan with too few data_analysis tasks."""
    return {
        'cycle': 1,
        'tasks': [
            {'id': 1, 'type': 'data_analysis', 'description': 'Task 1', 'expected_output': 'Output 1'},
            {'id': 2, 'type': 'data_analysis', 'description': 'Task 2', 'expected_output': 'Output 2'},
            {'id': 3, 'type': 'literature_review', 'description': 'Task 3', 'expected_output': 'Output 3'},
            {'id': 4, 'type': 'literature_review', 'description': 'Task 4', 'expected_output': 'Output 4'}
        ],
        'rationale': 'Test plan'
    }


@pytest.fixture
def invalid_plan_single_type():
    """Plan with only one task type."""
    return {
        'cycle': 1,
        'tasks': [
            {'id': 1, 'type': 'data_analysis', 'description': 'Task 1', 'expected_output': 'Output 1'},
            {'id': 2, 'type': 'data_analysis', 'description': 'Task 2', 'expected_output': 'Output 2'},
            {'id': 3, 'type': 'data_analysis', 'description': 'Task 3', 'expected_output': 'Output 3'},
            {'id': 4, 'type': 'data_analysis', 'description': 'Task 4', 'expected_output': 'Output 4'}
        ],
        'rationale': 'Test plan'
    }


@pytest.fixture
def sample_context():
    """Sample context for plan review."""
    return {
        'cycle': 1,
        'research_objective': 'Investigate KRAS mutations'
    }


@pytest.fixture
def plan_reviewer():
    """Create PlanReviewerAgent without LLM client."""
    return PlanReviewerAgent(anthropic_client=None)


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for plan review."""
    response_content = json.dumps({
        'scores': {
            'specificity': 8.0,
            'relevance': 8.5,
            'novelty': 7.5,
            'coverage': 8.0,
            'feasibility': 8.0
        },
        'feedback': 'Good plan overall',
        'required_changes': [],
        'suggestions': ['Consider adding more validation tasks']
    })

    mock_response = Mock()
    mock_response.content = [Mock(text=response_content)]
    return mock_response


# ============================================================================
# PlanReview Dataclass Tests
# ============================================================================

class TestPlanReview:
    """Tests for PlanReview dataclass."""

    def test_basic_creation(self):
        """Test basic PlanReview creation."""
        review = PlanReview(
            approved=True,
            scores={'specificity': 8.0, 'relevance': 7.5},
            average_score=7.75,
            min_score=7.5,
            feedback='Good plan',
            required_changes=[],
            suggestions=['Add more tasks']
        )

        assert review.approved is True
        assert review.average_score == 7.75
        assert len(review.suggestions) == 1

    def test_to_dict(self):
        """Test PlanReview to dictionary conversion."""
        review = PlanReview(
            approved=False,
            scores={'specificity': 5.0},
            average_score=5.0,
            min_score=5.0,
            feedback='Needs work',
            required_changes=['Add more detail'],
            suggestions=[]
        )

        result = review.to_dict()

        assert result['approved'] is False
        assert result['average_score'] == 5.0
        assert result['required_changes'] == ['Add more detail']


# ============================================================================
# PlanReviewerAgent Initialization Tests
# ============================================================================

class TestPlanReviewerAgentInit:
    """Tests for PlanReviewerAgent initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        reviewer = PlanReviewerAgent()

        assert reviewer.client is None
        assert reviewer.min_average_score == 7.0
        assert reviewer.min_dimension_score == 5.0

    def test_custom_initialization(self):
        """Test custom initialization."""
        mock_client = Mock()
        reviewer = PlanReviewerAgent(
            anthropic_client=mock_client,
            min_average_score=8.0,
            min_dimension_score=6.0
        )

        assert reviewer.client == mock_client
        assert reviewer.min_average_score == 8.0
        assert reviewer.min_dimension_score == 6.0

    def test_dimension_weights_defined(self):
        """Test that dimension weights are defined."""
        reviewer = PlanReviewerAgent()

        assert 'specificity' in reviewer.DIMENSION_WEIGHTS
        assert 'relevance' in reviewer.DIMENSION_WEIGHTS
        assert 'novelty' in reviewer.DIMENSION_WEIGHTS
        assert 'coverage' in reviewer.DIMENSION_WEIGHTS
        assert 'feasibility' in reviewer.DIMENSION_WEIGHTS


# ============================================================================
# Structural Requirements Tests
# ============================================================================

class TestStructuralRequirements:
    """Tests for structural requirement checking."""

    def test_valid_structure(self, plan_reviewer, valid_plan):
        """Test valid plan passes structural check."""
        result = plan_reviewer._meets_structural_requirements(valid_plan)

        assert result is True

    def test_too_few_data_analysis(self, plan_reviewer, invalid_plan_few_data_analysis):
        """Test plan with too few data_analysis tasks fails."""
        result = plan_reviewer._meets_structural_requirements(invalid_plan_few_data_analysis)

        assert result is False

    def test_single_task_type(self, plan_reviewer, invalid_plan_single_type):
        """Test plan with single task type fails."""
        result = plan_reviewer._meets_structural_requirements(invalid_plan_single_type)

        assert result is False

    def test_missing_description(self, plan_reviewer):
        """Test plan with missing task description fails."""
        plan = {
            'tasks': [
                {'id': 1, 'type': 'data_analysis', 'expected_output': 'Output'},
                {'id': 2, 'type': 'data_analysis', 'description': 'Task 2', 'expected_output': 'Output'},
                {'id': 3, 'type': 'data_analysis', 'description': 'Task 3', 'expected_output': 'Output'},
                {'id': 4, 'type': 'literature_review', 'description': 'Task 4', 'expected_output': 'Output'}
            ]
        }

        result = plan_reviewer._meets_structural_requirements(plan)

        assert result is False

    def test_missing_expected_output(self, plan_reviewer):
        """Test plan with missing expected_output fails."""
        plan = {
            'tasks': [
                {'id': 1, 'type': 'data_analysis', 'description': 'Task 1'},
                {'id': 2, 'type': 'data_analysis', 'description': 'Task 2', 'expected_output': 'Output'},
                {'id': 3, 'type': 'data_analysis', 'description': 'Task 3', 'expected_output': 'Output'},
                {'id': 4, 'type': 'literature_review', 'description': 'Task 4', 'expected_output': 'Output'}
            ]
        }

        result = plan_reviewer._meets_structural_requirements(plan)

        assert result is False

    def test_empty_tasks(self, plan_reviewer):
        """Test empty task list fails."""
        plan = {'tasks': []}

        result = plan_reviewer._meets_structural_requirements(plan)

        assert result is False


# ============================================================================
# Mock Review Tests
# ============================================================================

class TestMockReview:
    """Tests for mock review (no LLM)."""

    @pytest.mark.asyncio
    async def test_mock_review_valid_plan(self, plan_reviewer, valid_plan, sample_context):
        """Test mock review of valid plan."""
        review = await plan_reviewer.review_plan(valid_plan, sample_context)

        assert isinstance(review, PlanReview)
        assert review.approved is True
        assert review.average_score >= plan_reviewer.min_average_score

    @pytest.mark.asyncio
    async def test_mock_review_invalid_plan(self, plan_reviewer, invalid_plan_few_data_analysis, sample_context):
        """Test mock review of invalid plan."""
        review = await plan_reviewer.review_plan(invalid_plan_few_data_analysis, sample_context)

        assert review.approved is False
        assert len(review.required_changes) > 0

    @pytest.mark.asyncio
    async def test_mock_review_scores(self, plan_reviewer, valid_plan, sample_context):
        """Test mock review provides all scores."""
        review = await plan_reviewer.review_plan(valid_plan, sample_context)

        assert 'specificity' in review.scores
        assert 'relevance' in review.scores
        assert 'novelty' in review.scores
        assert 'coverage' in review.scores
        assert 'feasibility' in review.scores


# ============================================================================
# LLM Review Tests
# ============================================================================

class TestLLMReview:
    """Tests for LLM-based review."""

    @pytest.mark.asyncio
    async def test_review_with_llm(self, valid_plan, sample_context, mock_llm_response):
        """Test review with LLM client."""
        mock_client = Mock()
        mock_client.messages.create = AsyncMock(return_value=mock_llm_response)

        reviewer = PlanReviewerAgent(anthropic_client=mock_client)

        review = await reviewer.review_plan(valid_plan, sample_context)

        assert isinstance(review, PlanReview)
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self, valid_plan, sample_context):
        """Test fallback to mock review on LLM failure."""
        mock_client = Mock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API Error"))

        reviewer = PlanReviewerAgent(anthropic_client=mock_client)

        review = await reviewer.review_plan(valid_plan, sample_context)

        # Should still return a valid review (mock)
        assert isinstance(review, PlanReview)


# ============================================================================
# Prompt Building Tests
# ============================================================================

class TestPromptBuilding:
    """Tests for review prompt construction."""

    def test_build_review_prompt(self, plan_reviewer, valid_plan, sample_context):
        """Test building review prompt."""
        prompt = plan_reviewer._build_review_prompt(valid_plan, sample_context)

        assert 'Research Objective' in prompt
        assert 'Investigate KRAS' in prompt
        assert 'Specificity' in prompt
        assert 'Relevance' in prompt
        assert 'JSON' in prompt

    def test_prompt_includes_plan(self, plan_reviewer, valid_plan, sample_context):
        """Test that prompt includes plan JSON."""
        prompt = plan_reviewer._build_review_prompt(valid_plan, sample_context)

        # Plan should be JSON-formatted in prompt
        assert 'tasks' in prompt


# ============================================================================
# Response Parsing Tests
# ============================================================================

class TestResponseParsing:
    """Tests for LLM response parsing."""

    def test_parse_valid_response(self, plan_reviewer):
        """Test parsing valid JSON response."""
        response = json.dumps({
            'scores': {
                'specificity': 8.0,
                'relevance': 8.5,
                'novelty': 7.5,
                'coverage': 8.0,
                'feasibility': 8.0
            },
            'feedback': 'Good plan',
            'required_changes': [],
            'suggestions': []
        })

        result = plan_reviewer._parse_review_response(response)

        assert 'scores' in result
        assert result['scores']['specificity'] == 8.0
        assert result['feedback'] == 'Good plan'

    def test_parse_response_with_text(self, plan_reviewer):
        """Test parsing response with surrounding text."""
        response = """Here is my review:
        {
            "scores": {"specificity": 7.0, "relevance": 8.0, "novelty": 6.0, "coverage": 7.0, "feasibility": 8.0},
            "feedback": "Acceptable",
            "required_changes": [],
            "suggestions": []
        }
        Overall good plan.
        """

        result = plan_reviewer._parse_review_response(response)

        assert result['scores']['specificity'] == 7.0

    def test_parse_invalid_json(self, plan_reviewer):
        """Test parsing invalid JSON response."""
        response = "This is not valid JSON"

        result = plan_reviewer._parse_review_response(response)

        # Should return default scores
        assert 'scores' in result
        assert result['scores']['specificity'] == 5.0

    def test_parse_missing_scores(self, plan_reviewer):
        """Test parsing response with missing scores."""
        response = json.dumps({
            'feedback': 'Good plan',
            'required_changes': []
        })

        result = plan_reviewer._parse_review_response(response)

        # Should add default scores
        assert 'scores' in result
        assert 'specificity' in result['scores']

    def test_parse_score_clamping(self, plan_reviewer):
        """Test that scores are clamped to [0, 10]."""
        response = json.dumps({
            'scores': {
                'specificity': 15.0,  # Should clamp to 10
                'relevance': -5.0,    # Should clamp to 0
                'novelty': 7.0,
                'coverage': 8.0,
                'feasibility': 8.0
            },
            'feedback': 'Test'
        })

        result = plan_reviewer._parse_review_response(response)

        assert result['scores']['specificity'] == 10.0
        assert result['scores']['relevance'] == 0.0


# ============================================================================
# Approval Statistics Tests
# ============================================================================

class TestApprovalStatistics:
    """Tests for approval statistics computation."""

    def test_get_approval_statistics(self, plan_reviewer):
        """Test computing statistics over reviews."""
        reviews = [
            PlanReview(
                approved=True,
                scores={'specificity': 8.0, 'relevance': 8.0, 'novelty': 7.0, 'coverage': 7.5, 'feasibility': 8.0},
                average_score=7.7,
                min_score=7.0,
                feedback='Good',
                required_changes=[],
                suggestions=[]
            ),
            PlanReview(
                approved=False,
                scores={'specificity': 5.0, 'relevance': 5.0, 'novelty': 4.0, 'coverage': 5.0, 'feasibility': 5.0},
                average_score=4.8,
                min_score=4.0,
                feedback='Needs work',
                required_changes=['Add detail'],
                suggestions=[]
            )
        ]

        stats = plan_reviewer.get_approval_statistics(reviews)

        assert stats['total_reviewed'] == 2
        assert stats['approved'] == 1
        assert stats['rejected'] == 1
        assert stats['approval_rate'] == 0.5

    def test_get_approval_statistics_empty(self, plan_reviewer):
        """Test statistics for empty review list."""
        stats = plan_reviewer.get_approval_statistics([])

        assert stats == {}


# ============================================================================
# Edge Cases
# ============================================================================

class TestPlanReviewerEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_review_missing_context(self, plan_reviewer, valid_plan):
        """Test review with minimal context."""
        review = await plan_reviewer.review_plan(valid_plan, {})

        assert isinstance(review, PlanReview)

    @pytest.mark.asyncio
    async def test_review_large_plan(self, plan_reviewer, sample_context):
        """Test review of large plan."""
        large_plan = {
            'tasks': [
                {'id': i, 'type': 'data_analysis' if i % 3 == 0 else 'literature_review',
                 'description': f'Task {i}', 'expected_output': f'Output {i}'}
                for i in range(50)
            ]
        }

        review = await plan_reviewer.review_plan(large_plan, sample_context)

        assert isinstance(review, PlanReview)

    @pytest.mark.asyncio
    async def test_approval_threshold_boundary(self, sample_context):
        """Test approval at threshold boundary."""
        reviewer = PlanReviewerAgent(min_average_score=7.5)

        # Plan that should be just at threshold
        plan = {
            'tasks': [
                {'id': i, 'type': 'data_analysis' if i <= 3 else 'literature_review',
                 'description': f'Task {i}', 'expected_output': f'Output {i}'}
                for i in range(1, 6)
            ]
        }

        review = await reviewer.review_plan(plan, sample_context)

        # Review should be based on structural validity and mock scores
        assert isinstance(review, PlanReview)
