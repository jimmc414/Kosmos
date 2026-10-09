"""
Unit tests for kosmos.workflow.research_loop module.

Tests:
- ResearchWorkflow: initialization, cycle execution, statistics
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from kosmos.workflow.research_loop import ResearchWorkflow


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_components():
    """Create mock components for ResearchWorkflow."""
    mocks = {
        'context_compressor': Mock(),
        'state_manager': Mock(),
        'skill_loader': Mock(),
        'scholar_eval': Mock(),
        'plan_creator': Mock(),
        'plan_reviewer': Mock(),
        'delegation_manager': Mock(),
        'novelty_detector': Mock()
    }

    # Configure mock return values
    mocks['state_manager'].get_cycle_context.return_value = {
        'cycle': 1,
        'findings_count': 0,
        'recent_findings': [],
        'unsupported_hypotheses': [],
        'validated_discoveries': [],
        'statistics': {}
    }

    mocks['state_manager'].get_all_findings.return_value = []
    mocks['state_manager'].get_validated_findings.return_value = []
    mocks['state_manager'].get_statistics.return_value = {
        'total_findings': 0,
        'validated_findings': 0
    }

    mocks['skill_loader'].get_statistics.return_value = {
        'total_skills': 100
    }

    mocks['novelty_detector'].get_statistics.return_value = {
        'total_indexed_tasks': 0
    }

    return mocks


@pytest.fixture
def mock_plan():
    """Create mock research plan."""
    from kosmos.orchestration.plan_creator import Task, ResearchPlan

    tasks = [
        Task(
            task_id=1,
            task_type='data_analysis',
            description='Test task 1',
            expected_output='Output 1',
            required_skills=['pandas'],
            exploration=True
        )
    ]

    plan = ResearchPlan(
        cycle=1,
        tasks=tasks,
        rationale='Test plan',
        exploration_ratio=0.7
    )

    return plan


@pytest.fixture
def mock_review():
    """Create mock plan review."""
    from kosmos.orchestration.plan_reviewer import PlanReview

    return PlanReview(
        approved=True,
        scores={'specificity': 8.0, 'relevance': 8.0, 'novelty': 7.0,
                'coverage': 7.5, 'feasibility': 8.0},
        average_score=7.7,
        min_score=7.0,
        feedback='Good plan',
        required_changes=[],
        suggestions=[]
    )


@pytest.fixture
def mock_execution_result():
    """Create mock execution result."""
    return {
        'completed_tasks': [
            {
                'task_id': 1,
                'task_type': 'data_analysis',
                'status': 'completed',
                'finding': {
                    'summary': 'Found significant results',
                    'statistics': {'p_value': 0.01}
                }
            }
        ],
        'failed_tasks': [],
        'execution_summary': {
            'total_tasks': 1,
            'completed_tasks': 1,
            'failed_tasks': 0,
            'success_rate': 1.0
        }
    }


@pytest.fixture
def mock_scholar_eval_score():
    """Create mock ScholarEval score."""
    from kosmos.validation.scholar_eval import ScholarEvalScore

    return ScholarEvalScore(
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


# ============================================================================
# Initialization Tests
# ============================================================================

class TestResearchWorkflowInit:
    """Tests for ResearchWorkflow initialization."""

    def test_basic_initialization(self, temp_dir):
        """Test basic initialization without LLM client."""
        workflow = ResearchWorkflow(
            research_objective='Test objective',
            artifacts_dir=str(temp_dir / 'artifacts'),
            max_cycles=10
        )

        assert workflow.research_objective == 'Test objective'
        assert workflow.max_cycles == 10

    def test_components_initialized(self, temp_dir):
        """Test that all components are initialized."""
        workflow = ResearchWorkflow(
            research_objective='Test',
            artifacts_dir=str(temp_dir / 'artifacts')
        )

        assert workflow.context_compressor is not None
        assert workflow.state_manager is not None
        assert workflow.skill_loader is not None
        assert workflow.scholar_eval is not None
        assert workflow.plan_creator is not None
        assert workflow.plan_reviewer is not None
        assert workflow.delegation_manager is not None
        assert workflow.novelty_detector is not None

    def test_tracking_initialized(self, temp_dir):
        """Test that tracking attributes are initialized."""
        workflow = ResearchWorkflow(
            research_objective='Test',
            artifacts_dir=str(temp_dir / 'artifacts')
        )

        assert workflow.past_tasks == []
        assert workflow.cycle_results == []
        assert workflow.start_time is None


# ============================================================================
# Cycle Execution Tests
# ============================================================================

class TestCycleExecution:
    """Tests for cycle execution."""

    @pytest.mark.asyncio
    async def test_execute_cycle(
        self, temp_dir, mock_plan, mock_review,
        mock_execution_result, mock_scholar_eval_score
    ):
        """Test executing a single cycle."""
        workflow = ResearchWorkflow(
            research_objective='Test',
            artifacts_dir=str(temp_dir / 'artifacts')
        )

        # Mock components
        workflow.plan_creator.create_plan = AsyncMock(return_value=mock_plan)
        workflow.plan_reviewer.review_plan = AsyncMock(return_value=mock_review)
        workflow.delegation_manager.execute_plan = AsyncMock(return_value=mock_execution_result)
        workflow.scholar_eval.evaluate_finding = Mock(return_value=mock_scholar_eval_score)
        workflow.state_manager.save_finding_artifact = AsyncMock()
        workflow.state_manager.generate_cycle_summary = AsyncMock()
        workflow.novelty_detector.check_plan_novelty = Mock(return_value={
            'plan_novelty_score': 0.9,
            'novel_task_count': 1,
            'redundant_task_count': 0
        })
        workflow.context_compressor.compress_cycle_results = Mock(return_value=Mock(summary='Compressed'))

        result = await workflow._execute_cycle(1, 5)

        assert result['cycle'] == 1
        assert result['plan_approved'] is True

    @pytest.mark.asyncio
    async def test_execute_cycle_plan_rejected(
        self, temp_dir, mock_plan, mock_execution_result
    ):
        """Test cycle execution when plan is rejected."""
        from kosmos.orchestration.plan_reviewer import PlanReview

        rejected_review = PlanReview(
            approved=False,
            scores={'specificity': 5.0, 'relevance': 5.0, 'novelty': 4.0,
                    'coverage': 5.0, 'feasibility': 5.0},
            average_score=4.8,
            min_score=4.0,
            feedback='Needs work',
            required_changes=['Add more detail'],
            suggestions=[]
        )

        workflow = ResearchWorkflow(
            research_objective='Test',
            artifacts_dir=str(temp_dir / 'artifacts')
        )

        workflow.plan_creator.create_plan = AsyncMock(return_value=mock_plan)
        workflow.plan_creator.revise_plan = AsyncMock(return_value=mock_plan)
        workflow.plan_reviewer.review_plan = AsyncMock(return_value=rejected_review)
        workflow.state_manager.generate_cycle_summary = AsyncMock()

        result = await workflow._execute_cycle(1, 5)

        assert result['plan_approved'] is False
        # Execution should be skipped
        assert result['tasks_completed'] == 0


# ============================================================================
# Full Run Tests
# ============================================================================

class TestResearchWorkflowRun:
    """Tests for full workflow run."""

    @pytest.mark.asyncio
    async def test_run_multiple_cycles(
        self, temp_dir, mock_plan, mock_review,
        mock_execution_result, mock_scholar_eval_score
    ):
        """Test running multiple cycles."""
        workflow = ResearchWorkflow(
            research_objective='Test',
            artifacts_dir=str(temp_dir / 'artifacts')
        )

        # Mock all components
        workflow.plan_creator.create_plan = AsyncMock(return_value=mock_plan)
        workflow.plan_reviewer.review_plan = AsyncMock(return_value=mock_review)
        workflow.delegation_manager.execute_plan = AsyncMock(return_value=mock_execution_result)
        workflow.scholar_eval.evaluate_finding = Mock(return_value=mock_scholar_eval_score)
        workflow.state_manager.save_finding_artifact = AsyncMock()
        workflow.state_manager.generate_cycle_summary = AsyncMock()
        workflow.state_manager.get_all_findings = Mock(return_value=[])
        workflow.state_manager.get_validated_findings = Mock(return_value=[])
        workflow.novelty_detector.check_plan_novelty = Mock(return_value={
            'plan_novelty_score': 0.9,
            'novel_task_count': 1,
            'redundant_task_count': 0
        })
        workflow.context_compressor.compress_cycle_results = Mock(return_value=Mock(summary='Compressed'))

        result = await workflow.run(num_cycles=3, tasks_per_cycle=5)

        assert result['cycles_completed'] == 3
        assert 'total_findings' in result
        assert 'validation_rate' in result

    @pytest.mark.asyncio
    async def test_run_handles_cycle_failure(self, temp_dir, mock_plan, mock_review):
        """Test that run continues even if a cycle fails."""
        workflow = ResearchWorkflow(
            research_objective='Test',
            artifacts_dir=str(temp_dir / 'artifacts')
        )

        call_count = [0]

        async def mock_execute_cycle(cycle, num_tasks):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Simulated failure")
            return {
                'cycle': cycle,
                'tasks_generated': num_tasks,
                'tasks_completed': num_tasks,
                'validated_findings': 1,
                'plan_approved': True,
                'plan_score': 8.0
            }

        workflow._execute_cycle = mock_execute_cycle
        workflow.state_manager.get_all_findings = Mock(return_value=[])
        workflow.state_manager.get_validated_findings = Mock(return_value=[])

        result = await workflow.run(num_cycles=3)

        # Should complete 2 cycles (first and third, second failed)
        assert result['cycles_completed'] == 2


# ============================================================================
# Statistics Tests
# ============================================================================

class TestResearchWorkflowStatistics:
    """Tests for workflow statistics."""

    def test_compute_final_statistics(self, temp_dir):
        """Test computing final statistics."""
        workflow = ResearchWorkflow(
            research_objective='Test',
            artifacts_dir=str(temp_dir / 'artifacts')
        )

        workflow.start_time = datetime.now()
        workflow.cycle_results = [
            {'tasks_generated': 10, 'tasks_completed': 8},
            {'tasks_generated': 10, 'tasks_completed': 9}
        ]

        # Mock state manager
        workflow.state_manager.get_all_findings = Mock(return_value=[
            Mock(summary='Finding 1'),
            Mock(summary='Finding 2')
        ])
        workflow.state_manager.get_validated_findings = Mock(return_value=[
            Mock(summary='Finding 1')
        ])

        result = workflow._compute_final_statistics()

        assert result['cycles_completed'] == 2
        assert result['total_findings'] == 2
        assert result['validated_findings'] == 1
        assert result['validation_rate'] == 0.5
        assert result['total_tasks_generated'] == 20
        assert result['total_tasks_completed'] == 17

    def test_get_statistics(self, temp_dir):
        """Test get_statistics method."""
        workflow = ResearchWorkflow(
            research_objective='Test objective',
            artifacts_dir=str(temp_dir / 'artifacts'),
            max_cycles=10
        )

        workflow.cycle_results = [{'cycle': 1}]

        stats = workflow.get_statistics()

        assert 'workflow' in stats
        assert stats['workflow']['research_objective'] == 'Test objective'
        assert stats['workflow']['cycles_completed'] == 1


# ============================================================================
# Report Generation Tests
# ============================================================================

class TestReportGeneration:
    """Tests for report generation."""

    @pytest.mark.asyncio
    async def test_generate_report(self, temp_dir):
        """Test generating research report."""
        workflow = ResearchWorkflow(
            research_objective='Investigate KRAS mutations',
            artifacts_dir=str(temp_dir / 'artifacts')
        )

        workflow.cycle_results = [{'cycle': 1}, {'cycle': 2}]

        # Create mock findings
        mock_finding = Mock()
        mock_finding.summary = 'Found 42 significant genes'
        mock_finding.statistics = {'p_value': 0.001}
        mock_finding.notebook_path = '/path/to/notebook.ipynb'
        mock_finding.scholar_eval = {'overall_score': 0.85}

        workflow.state_manager.get_validated_findings = Mock(return_value=[mock_finding])

        report = await workflow.generate_report()

        assert '# Research Report' in report
        assert 'KRAS mutations' in report
        assert 'Finding 1' in report
        assert '42 significant genes' in report

    @pytest.mark.asyncio
    async def test_generate_report_empty(self, temp_dir):
        """Test generating report with no findings."""
        workflow = ResearchWorkflow(
            research_objective='Test',
            artifacts_dir=str(temp_dir / 'artifacts')
        )

        workflow.cycle_results = []
        workflow.state_manager.get_validated_findings = Mock(return_value=[])

        report = await workflow.generate_report()

        assert '# Research Report' in report
        assert '0 validated findings' in report


# ============================================================================
# Integration Points Tests
# ============================================================================

class TestIntegrationPoints:
    """Tests for integration between components."""

    @pytest.mark.asyncio
    async def test_novelty_detection_integration(self, temp_dir, mock_plan, mock_review, mock_execution_result, mock_scholar_eval_score):
        """Test novelty detection is called with past tasks."""
        workflow = ResearchWorkflow(
            research_objective='Test',
            artifacts_dir=str(temp_dir / 'artifacts')
        )

        # Add some past tasks
        workflow.past_tasks = [
            {'type': 'data_analysis', 'description': 'Previous task'}
        ]

        workflow.plan_creator.create_plan = AsyncMock(return_value=mock_plan)
        workflow.plan_reviewer.review_plan = AsyncMock(return_value=mock_review)
        workflow.delegation_manager.execute_plan = AsyncMock(return_value=mock_execution_result)
        workflow.scholar_eval.evaluate_finding = Mock(return_value=mock_scholar_eval_score)
        workflow.state_manager.save_finding_artifact = AsyncMock()
        workflow.state_manager.generate_cycle_summary = AsyncMock()
        workflow.novelty_detector.index_past_tasks = Mock()
        workflow.novelty_detector.check_plan_novelty = Mock(return_value={
            'plan_novelty_score': 0.8,
            'novel_task_count': 1,
            'redundant_task_count': 0
        })
        workflow.context_compressor.compress_cycle_results = Mock(return_value=Mock(summary='Compressed'))

        await workflow._execute_cycle(2, 5)

        # Novelty detector should be called
        workflow.novelty_detector.index_past_tasks.assert_called_once()

    @pytest.mark.asyncio
    async def test_compression_integration(self, temp_dir, mock_plan, mock_review, mock_execution_result, mock_scholar_eval_score):
        """Test context compression is called after cycle."""
        workflow = ResearchWorkflow(
            research_objective='Test',
            artifacts_dir=str(temp_dir / 'artifacts')
        )

        workflow.plan_creator.create_plan = AsyncMock(return_value=mock_plan)
        workflow.plan_reviewer.review_plan = AsyncMock(return_value=mock_review)
        workflow.delegation_manager.execute_plan = AsyncMock(return_value=mock_execution_result)
        workflow.scholar_eval.evaluate_finding = Mock(return_value=mock_scholar_eval_score)
        workflow.state_manager.save_finding_artifact = AsyncMock()
        workflow.state_manager.generate_cycle_summary = AsyncMock()
        workflow.novelty_detector.check_plan_novelty = Mock(return_value={
            'plan_novelty_score': 0.9,
            'novel_task_count': 1,
            'redundant_task_count': 0
        })
        workflow.context_compressor.compress_cycle_results = Mock(return_value=Mock(summary='Compressed'))

        await workflow._execute_cycle(1, 5)

        # Compression should be called
        workflow.context_compressor.compress_cycle_results.assert_called_once()


# ============================================================================
# Edge Cases
# ============================================================================

class TestResearchWorkflowEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_zero_cycles(self, temp_dir):
        """Test running with zero cycles."""
        workflow = ResearchWorkflow(
            research_objective='Test',
            artifacts_dir=str(temp_dir / 'artifacts')
        )

        workflow.state_manager.get_all_findings = Mock(return_value=[])
        workflow.state_manager.get_validated_findings = Mock(return_value=[])

        result = await workflow.run(num_cycles=0)

        assert result['cycles_completed'] == 0

    @pytest.mark.asyncio
    async def test_single_task_per_cycle(self, temp_dir, mock_plan, mock_review, mock_execution_result, mock_scholar_eval_score):
        """Test running with single task per cycle."""
        workflow = ResearchWorkflow(
            research_objective='Test',
            artifacts_dir=str(temp_dir / 'artifacts')
        )

        workflow.plan_creator.create_plan = AsyncMock(return_value=mock_plan)
        workflow.plan_reviewer.review_plan = AsyncMock(return_value=mock_review)
        workflow.delegation_manager.execute_plan = AsyncMock(return_value=mock_execution_result)
        workflow.scholar_eval.evaluate_finding = Mock(return_value=mock_scholar_eval_score)
        workflow.state_manager.save_finding_artifact = AsyncMock()
        workflow.state_manager.generate_cycle_summary = AsyncMock()
        workflow.state_manager.get_all_findings = Mock(return_value=[])
        workflow.state_manager.get_validated_findings = Mock(return_value=[])
        workflow.novelty_detector.check_plan_novelty = Mock(return_value={
            'plan_novelty_score': 0.9,
            'novel_task_count': 1,
            'redundant_task_count': 0
        })
        workflow.context_compressor.compress_cycle_results = Mock(return_value=Mock(summary='Compressed'))

        result = await workflow.run(num_cycles=1, tasks_per_cycle=1)

        assert result['cycles_completed'] == 1

    def test_empty_research_objective(self, temp_dir):
        """Test with empty research objective."""
        workflow = ResearchWorkflow(
            research_objective='',
            artifacts_dir=str(temp_dir / 'artifacts')
        )

        assert workflow.research_objective == ''
