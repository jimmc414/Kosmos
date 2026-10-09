"""
Unit tests for kosmos.orchestration.delegation module.

Tests:
- TaskResult and ExecutionSummary dataclasses
- DelegationManager: task batching, parallel execution, retry logic
- Agent wiring: RuntimeError when no agents provided
- Agent routing: mock agents receive correct calls
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from kosmos.orchestration.delegation import (
    TaskResult,
    ExecutionSummary,
    DelegationManager
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_tasks():
    """Sample task list for testing."""
    return [
        {
            'id': 1,
            'type': 'data_analysis',
            'description': 'Analyze gene expression data',
            'required_skills': ['pandas', 'scipy']
        },
        {
            'id': 2,
            'type': 'literature_review',
            'description': 'Review KRAS mutation papers',
            'required_skills': []
        },
        {
            'id': 3,
            'type': 'hypothesis_generation',
            'description': 'Generate new hypotheses',
            'required_skills': []
        },
        {
            'id': 4,
            'type': 'data_analysis',
            'description': 'Perform statistical tests',
            'required_skills': ['statsmodels']
        },
        {
            'id': 5,
            'type': 'data_analysis',
            'description': 'Create visualizations',
            'required_skills': ['matplotlib']
        }
    ]


@pytest.fixture
def sample_plan(sample_tasks):
    """Sample research plan for testing."""
    return {
        'cycle': 1,
        'tasks': sample_tasks,
        'rationale': 'Test plan'
    }


@pytest.fixture
def sample_context():
    """Sample context for task execution."""
    return {
        'cycle': 1,
        'research_objective': 'Study KRAS mutations',
        'recent_findings': []
    }


@pytest.fixture
def mock_agents():
    """Create mock agent instances for DelegationManager."""
    data_analyst = Mock()
    data_analyst.execute.return_value = {
        'success': True,
        'interpretation': {
            'summary': 'Mock data analysis result',
            'key_findings': ['finding1'],
            'overall_assessment': 'Good',
        }
    }

    literature_analyzer = Mock()
    literature_analyzer.execute.return_value = {
        'status': 'success',
        'insights': {
            'corpus_size': 5,
            'common_themes': ['theme1'],
            'research_gaps': ['gap1'],
        }
    }

    hypothesis_generator = Mock()
    mock_hyp = Mock()
    mock_hyp.statement = 'Test hypothesis statement'
    mock_hyp.testability_score = 0.8
    mock_response = Mock()
    mock_response.hypotheses = [mock_hyp]
    mock_response.generation_time_seconds = 1.5
    hypothesis_generator.generate_hypotheses.return_value = mock_response

    experiment_designer = Mock()
    mock_protocol = Mock()
    mock_protocol.name = 'Test Protocol'
    mock_design_response = Mock()
    mock_design_response.protocol = mock_protocol
    mock_design_response.rigor_score = 0.85
    mock_design_response.completeness_score = 0.9
    mock_design_response.feasibility_assessment = 'High'
    mock_design_response.design_time_seconds = 2.0
    experiment_designer.design_experiment.return_value = mock_design_response

    return {
        'data_analyst': data_analyst,
        'literature_analyzer': literature_analyzer,
        'hypothesis_generator': hypothesis_generator,
        'experiment_designer': experiment_designer,
    }


@pytest.fixture
def delegation_manager():
    """Create DelegationManager instance without agents."""
    return DelegationManager(
        max_parallel_tasks=3,
        max_retries=2,
        task_timeout=300
    )


@pytest.fixture
def delegation_manager_with_agents(mock_agents):
    """Create DelegationManager instance with mock agents."""
    return DelegationManager(
        max_parallel_tasks=3,
        max_retries=2,
        task_timeout=300,
        agents=mock_agents
    )


# ============================================================================
# TaskResult Tests
# ============================================================================

class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_basic_creation(self):
        """Test basic TaskResult creation."""
        result = TaskResult(
            task_id=1,
            task_type='data_analysis',
            status='completed'
        )

        assert result.task_id == 1
        assert result.task_type == 'data_analysis'
        assert result.status == 'completed'
        assert result.finding is None
        assert result.error is None

    def test_full_creation(self):
        """Test TaskResult with all fields."""
        result = TaskResult(
            task_id=1,
            task_type='data_analysis',
            status='completed',
            finding={'summary': 'Found results'},
            error=None,
            execution_time=10.5,
            retry_count=0
        )

        assert result.execution_time == 10.5
        assert result.finding['summary'] == 'Found results'

    def test_to_dict(self):
        """Test TaskResult to dictionary conversion."""
        result = TaskResult(
            task_id=1,
            task_type='data_analysis',
            status='completed',
            execution_time=5.0
        )

        result_dict = result.to_dict()

        assert result_dict['task_id'] == 1
        assert result_dict['status'] == 'completed'
        assert result_dict['execution_time'] == 5.0


# ============================================================================
# ExecutionSummary Tests
# ============================================================================

class TestExecutionSummary:
    """Tests for ExecutionSummary dataclass."""

    def test_basic_creation(self):
        """Test basic ExecutionSummary creation."""
        summary = ExecutionSummary(
            total_tasks=10,
            completed_tasks=8,
            failed_tasks=2,
            skipped_tasks=0,
            total_execution_time=120.5,
            success_rate=0.8
        )

        assert summary.total_tasks == 10
        assert summary.success_rate == 0.8

    def test_to_dict(self):
        """Test ExecutionSummary to dictionary conversion."""
        summary = ExecutionSummary(
            total_tasks=10,
            completed_tasks=8,
            failed_tasks=2,
            skipped_tasks=0,
            total_execution_time=120.5,
            success_rate=0.8
        )

        summary_dict = summary.to_dict()

        assert summary_dict['total_tasks'] == 10
        assert summary_dict['success_rate'] == 0.8


# ============================================================================
# DelegationManager Initialization Tests
# ============================================================================

class TestDelegationManagerInit:
    """Tests for DelegationManager initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        manager = DelegationManager()

        assert manager.max_parallel_tasks == 3
        assert manager.max_retries == 2
        assert manager.task_timeout == 300

    def test_custom_initialization(self):
        """Test custom initialization parameters."""
        manager = DelegationManager(
            max_parallel_tasks=5,
            max_retries=3,
            task_timeout=600
        )

        assert manager.max_parallel_tasks == 5
        assert manager.max_retries == 3
        assert manager.task_timeout == 600

    def test_agent_routing_defined(self):
        """Test that agent routing is defined."""
        manager = DelegationManager()

        assert 'data_analysis' in manager.AGENT_ROUTING
        assert 'literature_review' in manager.AGENT_ROUTING
        assert 'hypothesis_generation' in manager.AGENT_ROUTING

    def test_agents_stored(self, mock_agents):
        """Test that agents dict is stored."""
        manager = DelegationManager(agents=mock_agents)

        assert 'data_analyst' in manager.agents
        assert 'literature_analyzer' in manager.agents
        assert 'hypothesis_generator' in manager.agents
        assert 'experiment_designer' in manager.agents


# ============================================================================
# Task Batching Tests
# ============================================================================

class TestTaskBatching:
    """Tests for task batching functionality."""

    def test_create_batches_single(self, delegation_manager, sample_tasks):
        """Test batching with few tasks."""
        batches = delegation_manager._create_task_batches(sample_tasks[:2])

        assert len(batches) == 1
        assert len(batches[0]) == 2

    def test_create_batches_multiple(self, delegation_manager, sample_tasks):
        """Test batching creates multiple batches."""
        batches = delegation_manager._create_task_batches(sample_tasks)

        # 5 tasks, max 3 per batch = 2 batches
        assert len(batches) == 2
        assert len(batches[0]) == 3
        assert len(batches[1]) == 2

    def test_create_batches_empty(self, delegation_manager):
        """Test batching with empty task list."""
        batches = delegation_manager._create_task_batches([])

        assert batches == []

    def test_create_batches_exact_fit(self, delegation_manager):
        """Test batching when tasks fit exactly."""
        tasks = [{'id': i} for i in range(6)]
        batches = delegation_manager._create_task_batches(tasks)

        assert len(batches) == 2
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3


# ============================================================================
# Task Execution Tests — with agents
# ============================================================================

class TestTaskExecution:
    """Tests for task execution with mock agents."""

    @pytest.mark.asyncio
    async def test_execute_data_analysis(self, delegation_manager_with_agents, sample_context):
        """Test executing data analysis task with agent."""
        task = {
            'id': 1,
            'type': 'data_analysis',
            'description': 'Test analysis'
        }

        result = await delegation_manager_with_agents._execute_data_analysis(task, 1, sample_context)

        assert result['finding_id'] == 'cycle1_task1'
        assert 'statistics' in result
        assert result['evidence_type'] == 'data_analysis'

    @pytest.mark.asyncio
    async def test_execute_literature_review(self, delegation_manager_with_agents, sample_context):
        """Test executing literature review task with agent."""
        task = {
            'id': 2,
            'type': 'literature_review',
            'description': 'Review papers'
        }

        result = await delegation_manager_with_agents._execute_literature_review(task, 1, sample_context)

        assert result['finding_id'] == 'cycle1_task2'
        assert 'corpus_size' in result['statistics']
        assert result['evidence_type'] == 'literature_review'

    @pytest.mark.asyncio
    async def test_execute_hypothesis_generation(self, delegation_manager_with_agents, sample_context):
        """Test executing hypothesis generation task with agent."""
        task = {
            'id': 3,
            'type': 'hypothesis_generation',
            'description': 'Generate hypotheses'
        }

        result = await delegation_manager_with_agents._execute_hypothesis_generation(task, 1, sample_context)

        assert result['finding_id'] == 'cycle1_task3'
        assert 'hypotheses_generated' in result['statistics']

    @pytest.mark.asyncio
    async def test_execute_experiment_design(self, delegation_manager_with_agents, sample_context):
        """Test executing experiment design via generic handler."""
        task = {
            'id': 4,
            'type': 'experiment_design',
            'description': 'Design experiment'
        }

        result = await delegation_manager_with_agents._execute_generic_task(task, 1, sample_context)

        assert result['finding_id'] == 'cycle1_task4'
        assert result['evidence_type'] == 'experiment_design'

    @pytest.mark.asyncio
    async def test_execute_task_routing(self, delegation_manager_with_agents, sample_context):
        """Test task routing to correct executor."""
        task = {
            'id': 1,
            'type': 'data_analysis',
            'description': 'Test'
        }

        result = await delegation_manager_with_agents._execute_task(task, 1, sample_context)

        assert result['evidence_type'] == 'data_analysis'


# ============================================================================
# No-Agent RuntimeError Tests
# ============================================================================

class TestNoAgentErrors:
    """Tests that RuntimeError is raised when no agents are provided."""

    @pytest.mark.asyncio
    async def test_data_analysis_no_agent(self, delegation_manager, sample_context):
        """Test RuntimeError for data_analysis without agent."""
        task = {'id': 1, 'type': 'data_analysis', 'description': 'Test'}

        with pytest.raises(RuntimeError, match="No 'data_analyst' agent was provided"):
            await delegation_manager._execute_data_analysis(task, 1, sample_context)

    @pytest.mark.asyncio
    async def test_literature_review_no_agent(self, delegation_manager, sample_context):
        """Test RuntimeError for literature_review without agent."""
        task = {'id': 2, 'type': 'literature_review', 'description': 'Test'}

        with pytest.raises(RuntimeError, match="No 'literature_analyzer' agent was provided"):
            await delegation_manager._execute_literature_review(task, 1, sample_context)

    @pytest.mark.asyncio
    async def test_hypothesis_generation_no_agent(self, delegation_manager, sample_context):
        """Test RuntimeError for hypothesis_generation without agent."""
        task = {'id': 3, 'type': 'hypothesis_generation', 'description': 'Test'}

        with pytest.raises(RuntimeError, match="No 'hypothesis_generator' agent was provided"):
            await delegation_manager._execute_hypothesis_generation(task, 1, sample_context)

    @pytest.mark.asyncio
    async def test_generic_task_no_agents(self, delegation_manager, sample_context):
        """Test RuntimeError for unknown task type without any agents."""
        task = {'id': 4, 'type': 'unknown_type', 'description': 'Test'}

        with pytest.raises(RuntimeError, match="No agents were provided"):
            await delegation_manager._execute_generic_task(task, 1, sample_context)


# ============================================================================
# Retry Logic Tests
# ============================================================================

class TestRetryLogic:
    """Tests for retry logic."""

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self, delegation_manager_with_agents, sample_context):
        """Test successful execution without retry."""
        task = {
            'id': 1,
            'type': 'data_analysis',
            'description': 'Test'
        }

        result = await delegation_manager_with_agents._execute_task_with_retry(task, 1, sample_context)

        assert result.status == 'completed'
        assert result.retry_count == 0

    @pytest.mark.asyncio
    async def test_execute_with_retry_failure(self, delegation_manager, sample_context):
        """Test execution with retries after failure."""
        task = {
            'id': 1,
            'type': 'data_analysis',
            'description': 'Test'
        }

        # Mock _execute_task to fail
        with patch.object(delegation_manager, '_execute_task', side_effect=Exception("Test error")):
            result = await delegation_manager._execute_task_with_retry(task, 1, sample_context)

        assert result.status == 'failed'
        assert 'Test error' in result.error
        assert result.retry_count == delegation_manager.max_retries

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self, sample_context):
        """Test execution timeout handling."""
        manager = DelegationManager(task_timeout=0.1)  # Very short timeout

        task = {
            'id': 1,
            'type': 'data_analysis',
            'description': 'Test'
        }

        # Mock slow execution
        async def slow_execute(*args):
            await asyncio.sleep(1)
            return {}

        with patch.object(manager, '_execute_task', side_effect=slow_execute):
            result = await manager._execute_task_with_retry(task, 1, sample_context)

        assert result.status == 'failed'
        assert 'timeout' in result.error.lower()


# ============================================================================
# Batch Execution Tests
# ============================================================================

class TestBatchExecution:
    """Tests for batch execution."""

    @pytest.mark.asyncio
    async def test_execute_batch_success(self, delegation_manager_with_agents, sample_context):
        """Test successful batch execution."""
        batch = [
            {'id': 1, 'type': 'data_analysis', 'description': 'Task 1'},
            {'id': 2, 'type': 'data_analysis', 'description': 'Task 2'}
        ]

        results = await delegation_manager_with_agents._execute_batch(batch, 1, sample_context)

        assert len(results) == 2
        assert all(r.status == 'completed' for r in results)

    @pytest.mark.asyncio
    async def test_execute_batch_partial_failure(self, delegation_manager_with_agents, sample_context):
        """Test batch execution with partial failure."""
        batch = [
            {'id': 1, 'type': 'data_analysis', 'description': 'Task 1'},
            {'id': 2, 'type': 'data_analysis', 'description': 'Task 2'}
        ]

        # Make second task fail
        original_execute = delegation_manager_with_agents._execute_task_with_retry

        async def mock_execute(task, cycle, context):
            if task['id'] == 2:
                return TaskResult(
                    task_id=2,
                    task_type='data_analysis',
                    status='failed',
                    error='Mock failure'
                )
            return await original_execute(task, cycle, context)

        with patch.object(delegation_manager_with_agents, '_execute_task_with_retry', side_effect=mock_execute):
            results = await delegation_manager_with_agents._execute_batch(batch, 1, sample_context)

        assert len(results) == 2
        completed = [r for r in results if r.status == 'completed']
        failed = [r for r in results if r.status == 'failed']

        assert len(completed) == 1
        assert len(failed) == 1


# ============================================================================
# Plan Execution Tests
# ============================================================================

class TestPlanExecution:
    """Tests for plan execution."""

    @pytest.mark.asyncio
    async def test_execute_plan_success(self, delegation_manager_with_agents, sample_plan, sample_context):
        """Test successful plan execution."""
        result = await delegation_manager_with_agents.execute_plan(sample_plan, 1, sample_context)

        assert 'completed_tasks' in result
        assert 'failed_tasks' in result
        assert 'execution_summary' in result

        summary = result['execution_summary']
        assert summary['total_tasks'] == 5
        assert summary['success_rate'] > 0

    @pytest.mark.asyncio
    async def test_execute_plan_empty(self, delegation_manager, sample_context):
        """Test executing empty plan."""
        empty_plan = {'tasks': []}

        result = await delegation_manager.execute_plan(empty_plan, 1, sample_context)

        assert result['completed_tasks'] == []
        assert result['failed_tasks'] == []
        assert result['execution_summary']['total_tasks'] == 0

    @pytest.mark.asyncio
    async def test_execute_plan_tracks_time(self, delegation_manager_with_agents, sample_plan, sample_context):
        """Test that plan execution tracks time."""
        result = await delegation_manager_with_agents.execute_plan(sample_plan, 1, sample_context)

        summary = result['execution_summary']
        assert summary['total_execution_time'] >= 0


# ============================================================================
# Statistics Tests
# ============================================================================

class TestExecutionStatistics:
    """Tests for execution statistics."""

    def test_get_execution_statistics(self, delegation_manager):
        """Test getting execution statistics."""
        stats = delegation_manager.get_execution_statistics()

        assert stats['max_parallel_tasks'] == 3
        assert stats['max_retries'] == 2
        assert stats['task_timeout'] == 300
        assert 'agent_routing' in stats


# ============================================================================
# Edge Cases
# ============================================================================

class TestDelegationEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_missing_task_id(self, delegation_manager_with_agents, sample_context):
        """Test handling task without id."""
        task = {
            'type': 'data_analysis',
            'description': 'Task without ID'
        }

        result = await delegation_manager_with_agents._execute_task_with_retry(task, 1, sample_context)

        assert result.task_id == 0  # Default

    @pytest.mark.asyncio
    async def test_missing_task_type_no_agents(self, delegation_manager, sample_context):
        """Test handling task without type raises RuntimeError when no agents."""
        task = {
            'id': 1,
            'description': 'Task without type'
        }

        result = await delegation_manager._execute_task_with_retry(task, 1, sample_context)

        # Should fail since generic executor raises RuntimeError without agents
        assert result.status == 'failed'

    @pytest.mark.asyncio
    async def test_concurrent_plan_execution(self, delegation_manager_with_agents, sample_plan, sample_context):
        """Test that plan execution handles concurrency correctly."""
        # Execute same plan multiple times concurrently
        results = await asyncio.gather(
            delegation_manager_with_agents.execute_plan(sample_plan, 1, sample_context),
            delegation_manager_with_agents.execute_plan(sample_plan, 2, sample_context)
        )

        assert len(results) == 2
        assert all('completed_tasks' in r for r in results)
