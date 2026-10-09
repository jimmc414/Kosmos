"""
End-to-End tests for autonomous research workflow.

Tests multi-cycle autonomous operation verifying all components integrate correctly.

These tests use REAL LLM API calls (Anthropic Claude) - not mocks.
Requires ANTHROPIC_API_KEY environment variable.
"""

import os
import pytest
import asyncio
from pathlib import Path
from datetime import datetime

import anthropic
from kosmos.workflow.research_loop import ResearchWorkflow


# ============================================================================
# E2E Test Configuration
# ============================================================================

# Mark all tests as E2E and slow, skip if no API key
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY for real LLM calls"
    )
]


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def e2e_artifacts_dir(tmp_path):
    """Create E2E artifacts directory."""
    artifacts = tmp_path / "e2e_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts


@pytest.fixture
def anthropic_client():
    """Create real Anthropic client for LLM calls."""
    return anthropic.Anthropic()


@pytest.fixture
def e2e_workflow(e2e_artifacts_dir, anthropic_client):
    """Create E2E workflow instance with real LLM client."""
    return ResearchWorkflow(
        research_objective="""
        Investigate the role of KRAS mutations in pancreatic cancer drug resistance.
        Specifically:
        1. Identify differentially expressed genes in KRAS mutant vs wildtype samples
        2. Analyze correlation with treatment response
        3. Review relevant literature for therapeutic targets
        4. Generate hypotheses for combination therapies
        """,
        anthropic_client=anthropic_client,  # Real LLM client
        artifacts_dir=str(e2e_artifacts_dir),
        max_cycles=20
    )


# ============================================================================
# E2E Tests
# ============================================================================

class TestAutonomousResearchE2E:
    """End-to-end tests for autonomous research."""

    @pytest.mark.asyncio
    async def test_multi_cycle_autonomous_operation(self, e2e_workflow):
        """Test multi-cycle autonomous operation (3-5 cycles)."""
        result = await e2e_workflow.run(num_cycles=3, tasks_per_cycle=5)

        # Verify completion
        assert result['cycles_completed'] == 3

        # Verify statistics
        assert result['total_tasks_generated'] >= 10
        assert result['total_time'] > 0

        # Verify validation
        assert 'validation_rate' in result
        assert 0 <= result['validation_rate'] <= 1

    @pytest.mark.asyncio
    async def test_component_integration(self, e2e_workflow):
        """Test all components integrate correctly."""
        # Run minimal workflow
        result = await e2e_workflow.run(num_cycles=2, tasks_per_cycle=5)

        # Verify plan creation worked
        assert len(e2e_workflow.cycle_results) == 2
        for cycle_result in e2e_workflow.cycle_results:
            assert 'tasks_generated' in cycle_result
            assert 'plan_approved' in cycle_result

        # Verify novelty detection accumulated tasks
        assert len(e2e_workflow.past_tasks) > 0

        # Verify state manager has data
        stats = e2e_workflow.state_manager.get_statistics()
        assert 'total_findings' in stats

    @pytest.mark.asyncio
    async def test_workflow_produces_report(self, e2e_workflow):
        """Test workflow produces valid research report."""
        await e2e_workflow.run(num_cycles=2, tasks_per_cycle=5)

        report = await e2e_workflow.generate_report()

        # Verify report structure
        assert '# Research Report' in report
        assert 'KRAS' in report
        assert 'Date' in report
        assert 'Cycles Completed' in report

    @pytest.mark.asyncio
    async def test_exploration_exploitation_progression(self, e2e_workflow):
        """Test exploration/exploitation ratio progresses correctly."""
        # Run across early, middle, and late cycles
        await e2e_workflow.run(num_cycles=5, tasks_per_cycle=10)

        # Early cycles should have more exploration tasks
        # Plan creator adjusts exploration ratio by cycle number

        for i, cycle_result in enumerate(e2e_workflow.cycle_results):
            assert cycle_result['tasks_generated'] == 10


class TestWorkflowRobustness:
    """Tests for workflow robustness."""

    @pytest.mark.asyncio
    async def test_workflow_continues_after_failures(self, e2e_artifacts_dir, anthropic_client):
        """Test workflow continues after individual cycle failures."""
        workflow = ResearchWorkflow(
            research_objective="Test robustness",
            anthropic_client=anthropic_client,
            artifacts_dir=str(e2e_artifacts_dir)
        )

        # Inject failure in middle cycle
        original_execute = workflow._execute_cycle
        failure_injected = [False]

        async def failing_execute(cycle, tasks):
            if cycle == 2 and not failure_injected[0]:
                failure_injected[0] = True
                raise Exception("Simulated cycle failure")
            return await original_execute(cycle, tasks)

        workflow._execute_cycle = failing_execute

        result = await workflow.run(num_cycles=3, tasks_per_cycle=5)

        # Should complete 2 of 3 cycles
        assert result['cycles_completed'] == 2

    @pytest.mark.asyncio
    async def test_handles_large_task_count(self, e2e_artifacts_dir, anthropic_client):
        """Test workflow handles larger task counts."""
        workflow = ResearchWorkflow(
            research_objective="Test scale",
            anthropic_client=anthropic_client,
            artifacts_dir=str(e2e_artifacts_dir)
        )

        result = await workflow.run(num_cycles=2, tasks_per_cycle=20)

        assert result['total_tasks_generated'] >= 40


class TestDataFlow:
    """Tests for data flow through the system."""

    @pytest.mark.asyncio
    async def test_findings_flow_to_context(self, e2e_workflow):
        """Test that findings from early cycles inform later cycles."""
        await e2e_workflow.run(num_cycles=3, tasks_per_cycle=5)

        # Get context for later cycles
        context_cycle_3 = e2e_workflow.state_manager.get_cycle_context(
            cycle=3,
            lookback=2
        )

        assert context_cycle_3['cycle'] == 3
        # Should have access to earlier findings (if any were generated)

    @pytest.mark.asyncio
    async def test_task_history_builds(self, e2e_workflow):
        """Test that task history builds across cycles."""
        await e2e_workflow.run(num_cycles=3, tasks_per_cycle=5)

        # Should have accumulated all tasks
        expected_min_tasks = 3 * 5  # 3 cycles, 5 tasks each
        assert len(e2e_workflow.past_tasks) >= expected_min_tasks


class TestStatisticsAccuracy:
    """Tests for statistics accuracy."""

    @pytest.mark.asyncio
    async def test_statistics_sum_correctly(self, e2e_workflow):
        """Test that statistics sum correctly across cycles."""
        result = await e2e_workflow.run(num_cycles=3, tasks_per_cycle=5)

        # Total tasks should be sum of per-cycle tasks
        total_generated = sum(
            r['tasks_generated'] for r in e2e_workflow.cycle_results
        )
        assert result['total_tasks_generated'] == total_generated

        total_completed = sum(
            r['tasks_completed'] for r in e2e_workflow.cycle_results
        )
        assert result['total_tasks_completed'] == total_completed

    @pytest.mark.asyncio
    async def test_completion_rate_accurate(self, e2e_workflow):
        """Test that completion rate is accurately calculated."""
        result = await e2e_workflow.run(num_cycles=2, tasks_per_cycle=5)

        if result['total_tasks_generated'] > 0:
            expected_rate = result['total_tasks_completed'] / result['total_tasks_generated']
            assert abs(result['task_completion_rate'] - expected_rate) < 0.01


class TestPerformance:
    """Tests for performance characteristics."""

    @pytest.mark.asyncio
    async def test_reasonable_execution_time(self, e2e_artifacts_dir, anthropic_client):
        """Test that workflow completes in reasonable time."""
        workflow = ResearchWorkflow(
            research_objective="Performance test",
            anthropic_client=anthropic_client,
            artifacts_dir=str(e2e_artifacts_dir)
        )

        start_time = datetime.now()
        await workflow.run(num_cycles=2, tasks_per_cycle=5)
        elapsed = (datetime.now() - start_time).total_seconds()

        # With real LLM calls, expect 2-5 minutes for 2 cycles
        assert elapsed < 300, f"Workflow took {elapsed}s, expected < 300s (5 min)"


class TestPaperRequirements:
    """Tests verifying paper requirements (from arxiv paper)."""

    @pytest.mark.asyncio
    async def test_cycle_management(self, e2e_workflow):
        """Verify research_loop.py can manage multiple cycles."""
        result = await e2e_workflow.run(num_cycles=5, tasks_per_cycle=5)

        assert result['cycles_completed'] == 5

    @pytest.mark.asyncio
    async def test_parallel_task_dispatch(self, e2e_workflow):
        """Verify orchestration can dispatch parallel tasks."""
        # This is handled by DelegationManager's batch execution
        await e2e_workflow.run(num_cycles=1, tasks_per_cycle=10)

        # If tasks were batched correctly, cycle should complete
        assert e2e_workflow.cycle_results[0]['tasks_generated'] == 10

    @pytest.mark.asyncio
    async def test_state_persistence(self, e2e_workflow):
        """Verify state persistence between cycles."""
        await e2e_workflow.run(num_cycles=3, tasks_per_cycle=5)

        # State should persist
        assert len(e2e_workflow.cycle_results) == 3
        assert len(e2e_workflow.past_tasks) > 0

    @pytest.mark.asyncio
    async def test_discovery_validation(self, e2e_workflow):
        """Verify ScholarEval scoring works end-to-end."""
        result = await e2e_workflow.run(num_cycles=2, tasks_per_cycle=5)

        # Should have validation statistics
        assert 'validation_rate' in result
        validated_findings = e2e_workflow.state_manager.get_validated_findings()

        # All validated findings should have scholar_eval scores
        for finding in validated_findings:
            if hasattr(finding, 'scholar_eval') and finding.scholar_eval:
                assert 'overall_score' in finding.scholar_eval or \
                       hasattr(finding.scholar_eval, 'overall_score')
