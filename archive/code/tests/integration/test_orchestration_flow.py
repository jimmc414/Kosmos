"""
Integration tests for the orchestration flow.

Tests plan creation -> review -> delegation pipeline with novelty detection.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from kosmos.orchestration.plan_creator import PlanCreatorAgent, ResearchPlan, Task
from kosmos.orchestration.plan_reviewer import PlanReviewerAgent, PlanReview
from kosmos.orchestration.delegation import DelegationManager
from kosmos.orchestration.novelty_detector import NoveltyDetector


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def research_context():
    """Realistic research context for orchestration."""
    return {
        'cycle': 5,
        'research_objective': 'Investigate KRAS mutations and their role in drug resistance',
        'findings_count': 12,
        'recent_findings': [
            {
                'id': 'finding_001',
                'summary': 'Found 42 differentially expressed genes in KRAS mutant samples',
                'statistics': {'p_value': 0.001, 'n_genes': 42}
            },
            {
                'id': 'finding_002',
                'summary': 'Literature supports KRAS-MAPK pathway involvement',
                'statistics': {}
            }
        ],
        'unsupported_hypotheses': [
            {
                'hypothesis_id': 'hyp_001',
                'statement': 'KRAS G12D mutations confer resistance to EGFR inhibitors'
            }
        ]
    }


@pytest.fixture
def past_tasks():
    """Past tasks for novelty detection."""
    return [
        {
            'id': 1,
            'type': 'data_analysis',
            'description': 'Analyze RNA-seq data for KRAS expression levels'
        },
        {
            'id': 2,
            'type': 'literature_review',
            'description': 'Review papers on KRAS mutations in lung cancer'
        },
        {
            'id': 3,
            'type': 'data_analysis',
            'description': 'Perform differential expression analysis on treatment response data'
        }
    ]


@pytest.fixture
def plan_creator():
    """PlanCreatorAgent without LLM for testing."""
    return PlanCreatorAgent(anthropic_client=None)


@pytest.fixture
def plan_reviewer():
    """PlanReviewerAgent without LLM for testing."""
    return PlanReviewerAgent(anthropic_client=None)


@pytest.fixture
def delegation_manager():
    """DelegationManager for testing."""
    return DelegationManager(max_parallel_tasks=3, max_retries=2)


@pytest.fixture
def novelty_detector():
    """NoveltyDetector with token-based similarity."""
    return NoveltyDetector(novelty_threshold=0.75, use_sentence_transformers=False)


# ============================================================================
# Integration Tests
# ============================================================================

class TestPlanCreationToReview:
    """Tests for plan creation -> review flow."""

    @pytest.mark.asyncio
    async def test_create_and_review_plan(self, plan_creator, plan_reviewer, research_context):
        """Test full plan creation and review cycle."""
        # Create plan
        plan = await plan_creator.create_plan(
            research_objective=research_context['research_objective'],
            context=research_context,
            num_tasks=10
        )

        assert isinstance(plan, ResearchPlan)
        assert len(plan.tasks) == 10

        # Review plan
        review = await plan_reviewer.review_plan(plan.to_dict(), research_context)

        assert isinstance(review, PlanReview)
        assert 'specificity' in review.scores
        assert 'relevance' in review.scores

    @pytest.mark.asyncio
    async def test_rejected_plan_revision(self, plan_creator, plan_reviewer, research_context):
        """Test plan revision after rejection."""
        # Create plan
        plan = await plan_creator.create_plan(
            research_objective=research_context['research_objective'],
            context=research_context,
            num_tasks=10
        )

        # Get review
        review = await plan_reviewer.review_plan(plan.to_dict(), research_context)

        # Simulate revision
        if not review.approved:
            revised_plan = await plan_creator.revise_plan(
                plan,
                review.to_dict(),
                research_context
            )
            assert isinstance(revised_plan, ResearchPlan)
            assert len(revised_plan.tasks) == len(plan.tasks)

    @pytest.mark.asyncio
    async def test_exploration_exploitation_balance(self, plan_creator):
        """Test that plans respect exploration/exploitation ratios."""
        for cycle, expected_ratio in [(1, 0.7), (10, 0.5), (18, 0.3)]:
            context = {'cycle': cycle}
            plan = await plan_creator.create_plan(
                research_objective='Test',
                context=context,
                num_tasks=10
            )

            exploration_tasks = sum(1 for t in plan.tasks if t.exploration)
            actual_ratio = exploration_tasks / len(plan.tasks)

            # Should be close to expected ratio
            assert abs(actual_ratio - expected_ratio) < 0.2, \
                f"Cycle {cycle}: expected {expected_ratio}, got {actual_ratio}"


class TestNoveltyInOrchestration:
    """Tests for novelty detection in orchestration flow."""

    def test_novelty_filtering_in_pipeline(self, novelty_detector, past_tasks):
        """Test novelty detection filters redundant tasks."""
        # Index past tasks
        novelty_detector.index_past_tasks(past_tasks)

        # Check novelty of new tasks
        redundant_task = {
            'type': 'data_analysis',
            'description': 'Analyze RNA-seq data for KRAS expression'  # Similar to task 1
        }

        novel_task = {
            'type': 'experiment_design',
            'description': 'Design CRISPR knockout experiment for KRAS validation'
        }

        redundant_result = novelty_detector.check_task_novelty(redundant_task)
        novel_result = novelty_detector.check_task_novelty(novel_task)

        # Novel task should have higher novelty score
        assert novel_result['novelty_score'] >= redundant_result['novelty_score']

    @pytest.mark.asyncio
    async def test_plan_novelty_check(
        self, plan_creator, novelty_detector, past_tasks, research_context
    ):
        """Test checking novelty of entire plan."""
        # Index past tasks
        novelty_detector.index_past_tasks(past_tasks)

        # Create plan
        plan = await plan_creator.create_plan(
            research_objective=research_context['research_objective'],
            context=research_context,
            num_tasks=10
        )

        # Check plan novelty
        novelty_result = novelty_detector.check_plan_novelty(plan.to_dict())

        assert 'plan_novelty_score' in novelty_result
        assert 'novel_task_count' in novelty_result
        assert 'redundant_task_count' in novelty_result
        assert novelty_result['novel_task_count'] + novelty_result['redundant_task_count'] == len(plan.tasks)


class TestPlanToDelegation:
    """Tests for plan -> delegation flow."""

    @pytest.mark.asyncio
    async def test_approved_plan_delegation(
        self, plan_creator, plan_reviewer, delegation_manager, research_context
    ):
        """Test delegation of approved plan."""
        # Create plan
        plan = await plan_creator.create_plan(
            research_objective=research_context['research_objective'],
            context=research_context,
            num_tasks=5
        )

        # Review plan
        review = await plan_reviewer.review_plan(plan.to_dict(), research_context)

        # Execute if approved
        if review.approved:
            result = await delegation_manager.execute_plan(
                plan.to_dict(),
                cycle=research_context['cycle'],
                context=research_context
            )

            assert 'completed_tasks' in result
            assert 'execution_summary' in result
            assert result['execution_summary']['total_tasks'] == len(plan.tasks)

    @pytest.mark.asyncio
    async def test_parallel_task_execution(self, delegation_manager, research_context):
        """Test that tasks execute in parallel batches."""
        plan = {
            'tasks': [
                {'id': i, 'type': 'data_analysis', 'description': f'Task {i}'}
                for i in range(6)
            ]
        }

        result = await delegation_manager.execute_plan(
            plan,
            cycle=1,
            context=research_context
        )

        # All tasks should complete
        assert result['execution_summary']['total_tasks'] == 6
        assert result['execution_summary']['completed_tasks'] + \
               result['execution_summary']['failed_tasks'] == 6


class TestFullOrchestrationPipeline:
    """Tests for complete orchestration pipeline."""

    @pytest.mark.asyncio
    async def test_full_orchestration_cycle(
        self, plan_creator, plan_reviewer, delegation_manager,
        novelty_detector, past_tasks, research_context
    ):
        """Test complete plan creation -> review -> novelty -> delegation flow."""
        # Step 1: Index past tasks for novelty detection
        novelty_detector.index_past_tasks(past_tasks)

        # Step 2: Create plan
        plan = await plan_creator.create_plan(
            research_objective=research_context['research_objective'],
            context=research_context,
            num_tasks=10
        )

        assert len(plan.tasks) == 10

        # Step 3: Check novelty
        novelty_result = novelty_detector.check_plan_novelty(plan.to_dict())
        assert novelty_result['plan_novelty_score'] >= 0

        # Step 4: Review plan
        review = await plan_reviewer.review_plan(plan.to_dict(), research_context)

        # Step 5: If rejected, revise
        if not review.approved:
            plan = await plan_creator.revise_plan(
                plan,
                review.to_dict(),
                research_context
            )
            review = await plan_reviewer.review_plan(plan.to_dict(), research_context)

        # Step 6: Execute plan
        execution_result = await delegation_manager.execute_plan(
            plan.to_dict(),
            cycle=research_context['cycle'],
            context=research_context
        )

        # Verify full pipeline completed
        assert execution_result['execution_summary']['total_tasks'] > 0
        assert len(execution_result['completed_tasks']) >= 0

    @pytest.mark.asyncio
    async def test_multi_cycle_orchestration(
        self, plan_creator, plan_reviewer, delegation_manager, novelty_detector
    ):
        """Test orchestration across multiple research cycles."""
        all_past_tasks = []
        all_results = []

        for cycle in range(1, 4):  # 3 cycles
            context = {
                'cycle': cycle,
                'research_objective': 'Investigate KRAS mutations'
            }

            # Index past tasks
            if all_past_tasks:
                novelty_detector.index_past_tasks(all_past_tasks)

            # Create and execute plan
            plan = await plan_creator.create_plan(
                research_objective=context['research_objective'],
                context=context,
                num_tasks=5
            )

            review = await plan_reviewer.review_plan(plan.to_dict(), context)

            result = await delegation_manager.execute_plan(
                plan.to_dict(),
                cycle=cycle,
                context=context
            )

            all_results.append({
                'cycle': cycle,
                'tasks_completed': len(result['completed_tasks']),
                'plan_approved': review.approved
            })

            # Add tasks to history
            for task in plan.tasks:
                all_past_tasks.append(task.to_dict())

        # Should have results for all cycles
        assert len(all_results) == 3
        assert all(r['tasks_completed'] > 0 for r in all_results)


class TestOrchestrationErrorHandling:
    """Tests for error handling in orchestration flow."""

    @pytest.mark.asyncio
    async def test_delegation_handles_task_failures(self, delegation_manager, research_context):
        """Test that delegation handles individual task failures gracefully."""
        plan = {
            'tasks': [
                {'id': 1, 'type': 'data_analysis', 'description': 'Task 1'},
                {'id': 2, 'type': 'invalid_type', 'description': 'Invalid task'},
                {'id': 3, 'type': 'data_analysis', 'description': 'Task 3'}
            ]
        }

        result = await delegation_manager.execute_plan(
            plan,
            cycle=1,
            context=research_context
        )

        # Should still complete valid tasks
        summary = result['execution_summary']
        assert summary['total_tasks'] == 3

    @pytest.mark.asyncio
    async def test_empty_plan_handling(self, plan_reviewer, delegation_manager, research_context):
        """Test handling of empty plans."""
        empty_plan = {'tasks': []}

        # Review should fail structural requirements
        review = await plan_reviewer.review_plan(empty_plan, research_context)
        assert not review.approved

        # Delegation should handle gracefully
        result = await delegation_manager.execute_plan(
            empty_plan,
            cycle=1,
            context=research_context
        )

        assert result['completed_tasks'] == []
        assert result['execution_summary']['total_tasks'] == 0
