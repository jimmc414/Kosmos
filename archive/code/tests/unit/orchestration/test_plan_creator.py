"""
Unit tests for kosmos.orchestration.plan_creator module.

Tests:
- Task and ResearchPlan dataclasses
- PlanCreatorAgent: plan generation, exploration/exploitation balance
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch

from kosmos.orchestration.plan_creator import (
    Task,
    ResearchPlan,
    PlanCreatorAgent
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_context():
    """Sample context for plan creation."""
    return {
        'cycle': 5,
        'research_objective': 'Investigate KRAS mutations in cancer',
        'findings_count': 15,
        'recent_findings': [
            {'summary': 'Found 42 DEGs in KRAS mutant samples'},
            {'summary': 'Literature supports KRAS-PI3K pathway'}
        ],
        'unsupported_hypotheses': [
            {'hypothesis_id': 'hyp_001', 'statement': 'KRAS affects drug response'}
        ]
    }


@pytest.fixture
def plan_creator():
    """Create PlanCreatorAgent without LLM client."""
    return PlanCreatorAgent(anthropic_client=None)


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for plan generation."""
    response_content = json.dumps({
        'tasks': [
            {
                'id': 1,
                'type': 'data_analysis',
                'description': 'Analyze RNA-seq data',
                'expected_output': 'DEG list',
                'required_skills': ['deseq2'],
                'exploration': True,
                'priority': 1
            },
            {
                'id': 2,
                'type': 'literature_review',
                'description': 'Review KRAS papers',
                'expected_output': 'Literature summary',
                'required_skills': [],
                'exploration': False,
                'priority': 2
            }
        ],
        'rationale': 'Focus on KRAS pathway analysis'
    })

    mock_response = Mock()
    mock_response.content = [Mock(text=response_content)]
    return mock_response


# ============================================================================
# Task Dataclass Tests
# ============================================================================

class TestTask:
    """Tests for Task dataclass."""

    def test_basic_creation(self):
        """Test basic Task creation."""
        task = Task(
            task_id=1,
            task_type='data_analysis',
            description='Test task',
            expected_output='Results',
            required_skills=['pandas'],
            exploration=True
        )

        assert task.task_id == 1
        assert task.task_type == 'data_analysis'
        assert task.exploration is True
        assert task.priority == 1  # Default

    def test_to_dict(self):
        """Test Task to dictionary conversion."""
        task = Task(
            task_id=1,
            task_type='data_analysis',
            description='Test task',
            expected_output='Results',
            required_skills=['pandas', 'scipy'],
            exploration=True,
            target_hypotheses=['hyp_001'],
            priority=2
        )

        result = task.to_dict()

        assert result['id'] == 1
        assert result['type'] == 'data_analysis'
        assert result['exploration'] is True
        assert result['required_skills'] == ['pandas', 'scipy']
        assert result['target_hypotheses'] == ['hyp_001']

    def test_default_values(self):
        """Test Task default values."""
        task = Task(
            task_id=1,
            task_type='data_analysis',
            description='Test',
            expected_output='Output',
            required_skills=[],
            exploration=False
        )

        assert task.target_hypotheses is None
        assert task.priority == 1


# ============================================================================
# ResearchPlan Dataclass Tests
# ============================================================================

class TestResearchPlan:
    """Tests for ResearchPlan dataclass."""

    def test_basic_creation(self):
        """Test basic ResearchPlan creation."""
        tasks = [
            Task(1, 'data_analysis', 'Task 1', 'Output 1', [], True),
            Task(2, 'literature_review', 'Task 2', 'Output 2', [], False)
        ]

        plan = ResearchPlan(
            cycle=1,
            tasks=tasks,
            rationale='Test rationale',
            exploration_ratio=0.7
        )

        assert plan.cycle == 1
        assert len(plan.tasks) == 2
        assert plan.exploration_ratio == 0.7

    def test_to_dict(self):
        """Test ResearchPlan to dictionary conversion."""
        tasks = [
            Task(1, 'data_analysis', 'Task 1', 'Output 1', [], True)
        ]

        plan = ResearchPlan(
            cycle=5,
            tasks=tasks,
            rationale='Test rationale',
            exploration_ratio=0.5
        )

        result = plan.to_dict()

        assert result['cycle'] == 5
        assert len(result['tasks']) == 1
        assert result['tasks'][0]['id'] == 1
        assert result['rationale'] == 'Test rationale'
        assert result['exploration_ratio'] == 0.5


# ============================================================================
# PlanCreatorAgent Initialization Tests
# ============================================================================

class TestPlanCreatorAgentInit:
    """Tests for PlanCreatorAgent initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        agent = PlanCreatorAgent()

        assert agent.client is None
        assert agent.default_num_tasks == 10

    def test_custom_initialization(self):
        """Test custom initialization."""
        mock_client = Mock()
        agent = PlanCreatorAgent(
            anthropic_client=mock_client,
            model='claude-3-opus',
            default_num_tasks=5
        )

        assert agent.client == mock_client
        assert agent.model == 'claude-3-opus'
        assert agent.default_num_tasks == 5


# ============================================================================
# Exploration Ratio Tests
# ============================================================================

class TestExplorationRatio:
    """Tests for exploration/exploitation ratio."""

    def test_early_cycle_ratio(self, plan_creator):
        """Test exploration ratio for early cycles (1-7)."""
        for cycle in [1, 3, 5, 7]:
            ratio = plan_creator._get_exploration_ratio(cycle)
            assert ratio == 0.70

    def test_middle_cycle_ratio(self, plan_creator):
        """Test exploration ratio for middle cycles (8-14)."""
        for cycle in [8, 10, 14]:
            ratio = plan_creator._get_exploration_ratio(cycle)
            assert ratio == 0.50

    def test_late_cycle_ratio(self, plan_creator):
        """Test exploration ratio for late cycles (15-20)."""
        for cycle in [15, 18, 20]:
            ratio = plan_creator._get_exploration_ratio(cycle)
            assert ratio == 0.30


# ============================================================================
# Mock Plan Creation Tests
# ============================================================================

class TestMockPlanCreation:
    """Tests for mock plan creation (no LLM)."""

    @pytest.mark.asyncio
    async def test_create_mock_plan(self, plan_creator, sample_context):
        """Test creating mock plan without LLM."""
        plan = await plan_creator.create_plan(
            research_objective='Test objective',
            context=sample_context,
            num_tasks=10
        )

        assert isinstance(plan, ResearchPlan)
        assert len(plan.tasks) == 10
        assert plan.cycle == 5

    @pytest.mark.asyncio
    async def test_mock_plan_exploration_balance(self, plan_creator, sample_context):
        """Test mock plan respects exploration ratio."""
        plan = await plan_creator.create_plan(
            research_objective='Test objective',
            context=sample_context,
            num_tasks=10
        )

        # Count exploration tasks
        exploration_tasks = sum(1 for t in plan.tasks if t.exploration)

        # Should roughly match exploration ratio for cycle 5 (0.7)
        assert exploration_tasks >= 5

    @pytest.mark.asyncio
    async def test_mock_plan_different_cycles(self, plan_creator):
        """Test mock plan for different cycles."""
        for cycle in [1, 10, 20]:
            context = {'cycle': cycle}
            plan = await plan_creator.create_plan(
                research_objective='Test',
                context=context,
                num_tasks=10
            )

            assert plan.cycle == cycle
            expected_ratio = plan_creator._get_exploration_ratio(cycle)
            assert plan.exploration_ratio == expected_ratio


# ============================================================================
# LLM Plan Creation Tests
# ============================================================================

class TestLLMPlanCreation:
    """Tests for LLM-based plan creation."""

    @pytest.mark.asyncio
    async def test_create_plan_with_llm(self, sample_context, mock_llm_response):
        """Test creating plan with LLM client."""
        mock_client = Mock()
        mock_client.messages.create = AsyncMock(return_value=mock_llm_response)

        agent = PlanCreatorAgent(anthropic_client=mock_client)

        plan = await agent.create_plan(
            research_objective='Test objective',
            context=sample_context,
            num_tasks=10
        )

        assert isinstance(plan, ResearchPlan)
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self, sample_context):
        """Test fallback to mock plan on LLM failure."""
        mock_client = Mock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API Error"))

        agent = PlanCreatorAgent(anthropic_client=mock_client)

        plan = await agent.create_plan(
            research_objective='Test objective',
            context=sample_context,
            num_tasks=10
        )

        # Should still return a valid plan (mock)
        assert isinstance(plan, ResearchPlan)
        assert len(plan.tasks) == 10


# ============================================================================
# Prompt Building Tests
# ============================================================================

class TestPromptBuilding:
    """Tests for planning prompt construction."""

    def test_build_planning_prompt(self, plan_creator, sample_context):
        """Test building planning prompt."""
        prompt = plan_creator._build_planning_prompt(
            research_objective='Investigate KRAS',
            context=sample_context,
            num_tasks=10,
            exploration_ratio=0.7
        )

        assert 'Research Objective' in prompt
        assert 'Investigate KRAS' in prompt
        assert 'Cycle: 5' in prompt
        assert '70%' in prompt  # Exploration ratio
        assert 'JSON' in prompt

    def test_prompt_includes_recent_findings(self, plan_creator, sample_context):
        """Test that prompt includes recent findings."""
        prompt = plan_creator._build_planning_prompt(
            research_objective='Test',
            context=sample_context,
            num_tasks=10,
            exploration_ratio=0.5
        )

        assert 'Recent Findings' in prompt
        assert '42 DEGs' in prompt or 'finding' in prompt.lower()

    def test_prompt_includes_hypotheses(self, plan_creator, sample_context):
        """Test that prompt includes unsupported hypotheses."""
        prompt = plan_creator._build_planning_prompt(
            research_objective='Test',
            context=sample_context,
            num_tasks=10,
            exploration_ratio=0.5
        )

        assert 'Unsupported Hypotheses' in prompt


# ============================================================================
# Response Parsing Tests
# ============================================================================

class TestResponseParsing:
    """Tests for LLM response parsing."""

    def test_parse_valid_response(self, plan_creator):
        """Test parsing valid JSON response."""
        response = json.dumps({
            'tasks': [
                {
                    'type': 'data_analysis',
                    'description': 'Test task',
                    'expected_output': 'Output',
                    'required_skills': [],
                    'exploration': True,
                    'priority': 1
                }
            ],
            'rationale': 'Test rationale'
        })

        result = plan_creator._parse_plan_response(response)

        assert 'tasks' in result
        assert len(result['tasks']) == 1
        assert result['rationale'] == 'Test rationale'

    def test_parse_response_with_text(self, plan_creator):
        """Test parsing response with surrounding text."""
        response = """Here is the plan:
        {
            "tasks": [{"type": "data_analysis", "description": "Test"}],
            "rationale": "Because"
        }
        This is a good plan.
        """

        result = plan_creator._parse_plan_response(response)

        assert 'tasks' in result
        assert len(result['tasks']) == 1

    def test_parse_invalid_json(self, plan_creator):
        """Test parsing invalid JSON response."""
        response = "This is not valid JSON at all"

        result = plan_creator._parse_plan_response(response)

        assert result['tasks'] == []
        assert 'Failed to parse' in result['rationale']

    def test_parse_empty_response(self, plan_creator):
        """Test parsing empty response."""
        result = plan_creator._parse_plan_response("")

        assert result['tasks'] == []


# ============================================================================
# Plan Revision Tests
# ============================================================================

class TestPlanRevision:
    """Tests for plan revision."""

    @pytest.mark.asyncio
    async def test_revise_plan(self, plan_creator, sample_context):
        """Test revising plan based on feedback."""
        original_plan = ResearchPlan(
            cycle=5,
            tasks=[Task(1, 'data_analysis', 'Old task', 'Output', [], True)],
            rationale='Old rationale',
            exploration_ratio=0.7
        )

        review_feedback = {
            'feedback': 'Need more literature review tasks',
            'required_changes': ['Add 2 literature review tasks']
        }

        sample_context['research_objective'] = 'Test objective'

        revised_plan = await plan_creator.revise_plan(
            original_plan,
            review_feedback,
            sample_context
        )

        assert isinstance(revised_plan, ResearchPlan)
        # Should regenerate plan
        assert len(revised_plan.tasks) >= 1


# ============================================================================
# Edge Cases
# ============================================================================

class TestPlanCreatorEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_create_plan_empty_context(self, plan_creator):
        """Test creating plan with empty context."""
        plan = await plan_creator.create_plan(
            research_objective='Test',
            context={},
            num_tasks=5
        )

        assert isinstance(plan, ResearchPlan)
        assert len(plan.tasks) == 5
        assert plan.cycle == 1  # Default cycle

    @pytest.mark.asyncio
    async def test_create_plan_single_task(self, plan_creator, sample_context):
        """Test creating plan with single task."""
        plan = await plan_creator.create_plan(
            research_objective='Test',
            context=sample_context,
            num_tasks=1
        )

        assert len(plan.tasks) == 1

    def test_create_generic_task(self, plan_creator):
        """Test creating generic filler task."""
        task = plan_creator._create_generic_task(42)

        assert task.task_id == 42
        assert task.task_type == 'data_analysis'
        assert task.exploration is True

    @pytest.mark.asyncio
    async def test_plan_task_filling(self, sample_context, mock_llm_response):
        """Test that plan fills in missing tasks."""
        # Response with only 2 tasks
        mock_client = Mock()
        mock_client.messages.create = AsyncMock(return_value=mock_llm_response)

        agent = PlanCreatorAgent(anthropic_client=mock_client)

        plan = await agent.create_plan(
            research_objective='Test',
            context=sample_context,
            num_tasks=10  # Request 10
        )

        # Should fill to 10
        assert len(plan.tasks) == 10
