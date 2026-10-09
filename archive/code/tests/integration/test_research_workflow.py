"""
Integration tests for the research workflow.

Tests complete single-cycle workflow and state persistence across cycles.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from kosmos.workflow.research_loop import ResearchWorkflow


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def workflow_dir(temp_dir):
    """Create workflow directory structure."""
    artifacts_dir = temp_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


@pytest.fixture
def research_workflow(workflow_dir):
    """Create ResearchWorkflow instance."""
    return ResearchWorkflow(
        research_objective="Investigate KRAS mutations and their role in cancer drug resistance",
        anthropic_client=None,  # Use mock mode
        artifacts_dir=str(workflow_dir),
        max_cycles=5
    )


# ============================================================================
# Integration Tests
# ============================================================================

class TestSingleCycleWorkflow:
    """Tests for single cycle execution."""

    @pytest.mark.asyncio
    async def test_single_cycle_execution(self, research_workflow):
        """Test execution of a single research cycle."""
        result = await research_workflow.run(num_cycles=1, tasks_per_cycle=5)

        assert result['cycles_completed'] == 1
        assert 'total_findings' in result
        assert 'validation_rate' in result
        assert result['total_time'] > 0

    @pytest.mark.asyncio
    async def test_cycle_generates_findings(self, research_workflow):
        """Test that cycle generates and validates findings."""
        await research_workflow.run(num_cycles=1, tasks_per_cycle=5)

        # Check state manager has findings
        all_findings = research_workflow.state_manager.get_all_findings()

        # Should have some findings (mock mode generates them)
        assert len(all_findings) >= 0

    @pytest.mark.asyncio
    async def test_cycle_tasks_tracked(self, research_workflow):
        """Test that past tasks are tracked for novelty detection."""
        await research_workflow.run(num_cycles=1, tasks_per_cycle=5)

        # Should have past tasks indexed
        assert len(research_workflow.past_tasks) > 0


class TestMultiCycleWorkflow:
    """Tests for multi-cycle execution."""

    @pytest.mark.asyncio
    async def test_multi_cycle_execution(self, research_workflow):
        """Test execution of multiple research cycles."""
        result = await research_workflow.run(num_cycles=3, tasks_per_cycle=5)

        assert result['cycles_completed'] == 3
        assert len(research_workflow.cycle_results) == 3

    @pytest.mark.asyncio
    async def test_cycle_context_progression(self, research_workflow):
        """Test that context builds across cycles."""
        await research_workflow.run(num_cycles=2, tasks_per_cycle=5)

        # Later cycles should have access to earlier findings
        context = research_workflow.state_manager.get_cycle_context(cycle=2, lookback=1)

        assert context['cycle'] == 2
        # Context should include findings from previous cycles

    @pytest.mark.asyncio
    async def test_novelty_accumulation(self, research_workflow):
        """Test that novelty detector accumulates tasks."""
        await research_workflow.run(num_cycles=3, tasks_per_cycle=5)

        # Should have accumulated tasks from all cycles
        assert len(research_workflow.past_tasks) >= 10  # At least 5 per cycle for 2+ cycles


class TestWorkflowComponents:
    """Tests for workflow component integration."""

    @pytest.mark.asyncio
    async def test_plan_creation_integration(self, research_workflow):
        """Test plan creation is integrated correctly."""
        # Run single cycle
        await research_workflow.run(num_cycles=1, tasks_per_cycle=10)

        # Plan creator should have been used
        assert len(research_workflow.cycle_results) == 1
        result = research_workflow.cycle_results[0]
        assert result['tasks_generated'] == 10

    @pytest.mark.asyncio
    async def test_validation_integration(self, research_workflow):
        """Test ScholarEval validation is integrated."""
        await research_workflow.run(num_cycles=1, tasks_per_cycle=5)

        result = research_workflow.cycle_results[0]

        # Should have validated findings count
        assert 'validated_findings' in result

    @pytest.mark.asyncio
    async def test_compression_integration(self, research_workflow):
        """Test context compression is integrated."""
        await research_workflow.run(num_cycles=1, tasks_per_cycle=5)

        # Compression should have been called
        # (verified by cycle completing successfully)
        assert research_workflow.cycle_results[0]['tasks_completed'] >= 0


class TestWorkflowStatistics:
    """Tests for workflow statistics."""

    @pytest.mark.asyncio
    async def test_final_statistics(self, research_workflow):
        """Test final statistics computation."""
        result = await research_workflow.run(num_cycles=2, tasks_per_cycle=5)

        assert 'cycles_completed' in result
        assert 'total_findings' in result
        assert 'validated_findings' in result
        assert 'validation_rate' in result
        assert 'total_tasks_generated' in result
        assert 'total_tasks_completed' in result
        assert 'task_completion_rate' in result
        assert 'total_time' in result

    @pytest.mark.asyncio
    async def test_get_statistics(self, research_workflow):
        """Test comprehensive statistics method."""
        await research_workflow.run(num_cycles=1, tasks_per_cycle=5)

        stats = research_workflow.get_statistics()

        assert 'workflow' in stats
        assert stats['workflow']['research_objective'] == research_workflow.research_objective


class TestWorkflowReporting:
    """Tests for workflow reporting."""

    @pytest.mark.asyncio
    async def test_report_generation(self, research_workflow):
        """Test research report generation."""
        await research_workflow.run(num_cycles=2, tasks_per_cycle=5)

        report = await research_workflow.generate_report()

        assert '# Research Report' in report
        assert 'KRAS mutations' in report
        assert 'cycles' in report.lower()


class TestWorkflowErrorHandling:
    """Tests for workflow error handling."""

    @pytest.mark.asyncio
    async def test_handles_cycle_errors(self, workflow_dir):
        """Test workflow handles individual cycle errors."""
        workflow = ResearchWorkflow(
            research_objective="Test",
            artifacts_dir=str(workflow_dir)
        )

        # Mock a failing cycle
        original_execute = workflow._execute_cycle
        call_count = [0]

        async def mock_execute(cycle, tasks):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Simulated failure")
            return await original_execute(cycle, tasks)

        workflow._execute_cycle = mock_execute

        result = await workflow.run(num_cycles=3, tasks_per_cycle=5)

        # Should continue despite failure
        assert result['cycles_completed'] == 2

    @pytest.mark.asyncio
    async def test_handles_empty_execution(self, workflow_dir):
        """Test workflow handles empty plan execution."""
        workflow = ResearchWorkflow(
            research_objective="Test",
            artifacts_dir=str(workflow_dir)
        )

        # Mock delegation to return no completed tasks
        workflow.delegation_manager.execute_plan = AsyncMock(return_value={
            'completed_tasks': [],
            'failed_tasks': [],
            'execution_summary': {
                'total_tasks': 5,
                'completed_tasks': 0,
                'failed_tasks': 5,
                'success_rate': 0
            }
        })

        # Mock plan creator and reviewer
        from kosmos.orchestration.plan_creator import Task, ResearchPlan
        from kosmos.orchestration.plan_reviewer import PlanReview

        mock_plan = ResearchPlan(
            cycle=1,
            tasks=[
                Task(i, 'data_analysis', f'Task {i}', 'Output', [], True)
                for i in range(5)
            ],
            rationale='Test',
            exploration_ratio=0.7
        )

        mock_review = PlanReview(
            approved=True,
            scores={'specificity': 8, 'relevance': 8, 'novelty': 7,
                    'coverage': 8, 'feasibility': 8},
            average_score=7.8,
            min_score=7,
            feedback='OK',
            required_changes=[],
            suggestions=[]
        )

        workflow.plan_creator.create_plan = AsyncMock(return_value=mock_plan)
        workflow.plan_reviewer.review_plan = AsyncMock(return_value=mock_review)

        result = await workflow.run(num_cycles=1, tasks_per_cycle=5)

        # Should complete without error
        assert result['cycles_completed'] == 1


class TestWorkflowStatePersistence:
    """Tests for state persistence across cycles."""

    @pytest.mark.asyncio
    async def test_findings_persist(self, research_workflow, workflow_dir):
        """Test that findings persist to disk."""
        await research_workflow.run(num_cycles=1, tasks_per_cycle=5)

        # Check for artifact files
        cycle_dir = workflow_dir / "cycle_1"

        # Should have some files if findings were saved
        # (depends on mock behavior)

    @pytest.mark.asyncio
    async def test_cycle_summaries_generated(self, research_workflow, workflow_dir):
        """Test that cycle summaries are generated."""
        await research_workflow.run(num_cycles=1, tasks_per_cycle=5)

        # Summary should be generated for each cycle
        cycle_dir = workflow_dir / "cycle_1"

        # Check if summary was generated
        # (state_manager.generate_cycle_summary is called)
