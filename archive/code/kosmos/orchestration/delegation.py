"""
Delegation Manager for Kosmos.

Executes approved research plans by delegating tasks to specialized agents.

Key Features:
- Parallel execution (max 3 tasks concurrently)
- Task-type routing (data_analysis → DataAnalystAgent, etc.)
- Retry logic (max 2 attempts per failed task)
- Result aggregation and reporting

Performance Target: ~90% task completion rate
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from kosmos.core.providers.base import ProviderAPIError

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Container for task execution results."""
    task_id: int
    task_type: str
    status: str  # "completed", "failed", "skipped"
    finding: Optional[Dict] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    retry_count: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'status': self.status,
            'finding': self.finding,
            'error': self.error,
            'execution_time': self.execution_time,
            'retry_count': self.retry_count
        }


@dataclass
class ExecutionSummary:
    """Summary of plan execution."""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    skipped_tasks: int
    total_execution_time: float
    success_rate: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'skipped_tasks': self.skipped_tasks,
            'total_execution_time': self.total_execution_time,
            'success_rate': self.success_rate
        }


class DelegationManager:
    """
    Manages execution of research plans by delegating to specialized agents.

    Architecture:
    - Receives approved plan from PlanReviewerAgent
    - Routes tasks to appropriate agents based on task type
    - Executes tasks in parallel (configurable concurrency)
    - Handles errors with retry logic
    - Aggregates results for State Manager

    Design Philosophy:
    - Fail gracefully: Individual task failures don't crash the cycle
    - Parallel execution: Maximize throughput
    - Clear routing: Task type → Agent mapping
    """

    # Task type → Agent mapping
    AGENT_ROUTING = {
        'data_analysis': 'DataAnalystAgent',
        'literature_review': 'LiteratureAnalyzerAgent',
        'hypothesis_generation': 'HypothesisGeneratorAgent',
        'experiment_design': 'ExperimentDesignerAgent'
    }

    def __init__(
        self,
        max_parallel_tasks: int = 3,
        max_retries: int = 2,
        task_timeout: int = 300,  # 5 minutes
        agents: Optional[Dict] = None
    ):
        """
        Initialize Delegation Manager.

        Args:
            max_parallel_tasks: Maximum tasks to execute concurrently
            max_retries: Maximum retry attempts for failed tasks
            task_timeout: Task execution timeout in seconds
            agents: Dictionary of agent instances (optional)
        """
        self.max_parallel_tasks = max_parallel_tasks
        self.max_retries = max_retries
        self.task_timeout = task_timeout
        self.agents = agents or {}

        # Execution tracking
        self.task_retries: Dict[int, int] = {}

        logger.info(
            f"DelegationManager initialized "
            f"(max_parallel={max_parallel_tasks}, max_retries={max_retries})"
        )

    async def execute_plan(
        self,
        plan: Dict,
        cycle: int,
        context: Dict
    ) -> Dict:
        """
        Execute research plan.

        Args:
            plan: ResearchPlan dictionary with tasks
            cycle: Current research cycle
            context: Context from State Manager

        Returns:
            Dictionary with:
            - completed_tasks: List of TaskResult objects
            - failed_tasks: List of TaskResult objects
            - execution_summary: ExecutionSummary object
        """
        tasks = plan.get('tasks', [])

        logger.info(f"Executing plan with {len(tasks)} tasks for cycle {cycle}")

        # Reset retry counters
        self.task_retries = {task.get('id', i): 0 for i, task in enumerate(tasks, 1)}

        # Create batches for parallel execution
        batches = self._create_task_batches(tasks)

        # Execute batches
        completed_tasks = []
        failed_tasks = []
        start_time = datetime.now()

        for batch_idx, batch in enumerate(batches, 1):
            logger.info(f"Executing batch {batch_idx}/{len(batches)} ({len(batch)} tasks)")

            # Execute batch in parallel
            batch_results = await self._execute_batch(batch, cycle, context)

            # Classify results
            for result in batch_results:
                if result.status == 'completed':
                    completed_tasks.append(result)
                else:
                    failed_tasks.append(result)

        # Compute execution summary
        total_time = (datetime.now() - start_time).total_seconds()
        summary = ExecutionSummary(
            total_tasks=len(tasks),
            completed_tasks=len(completed_tasks),
            failed_tasks=len(failed_tasks),
            skipped_tasks=0,
            total_execution_time=total_time,
            success_rate=len(completed_tasks) / len(tasks) if tasks else 0
        )

        logger.info(
            f"Plan execution complete: {summary.completed_tasks}/{summary.total_tasks} "
            f"tasks completed ({summary.success_rate*100:.1f}%)"
        )

        return {
            'completed_tasks': [t.to_dict() for t in completed_tasks],
            'failed_tasks': [t.to_dict() for t in failed_tasks],
            'execution_summary': summary.to_dict()
        }

    def _create_task_batches(self, tasks: List[Dict]) -> List[List[Dict]]:
        """
        Create batches for parallel execution.

        Args:
            tasks: List of task dictionaries

        Returns:
            List of task batches
        """
        batches = []
        current_batch = []

        for task in tasks:
            current_batch.append(task)

            if len(current_batch) >= self.max_parallel_tasks:
                batches.append(current_batch)
                current_batch = []

        # Add remaining tasks
        if current_batch:
            batches.append(current_batch)

        return batches

    async def _execute_batch(
        self,
        batch: List[Dict],
        cycle: int,
        context: Dict
    ) -> List[TaskResult]:
        """
        Execute batch of tasks in parallel.

        Args:
            batch: List of task dictionaries
            cycle: Current cycle
            context: Context dictionary

        Returns:
            List of TaskResult objects
        """
        # Create coroutines for each task
        coroutines = [
            self._execute_task_with_retry(task, cycle, context)
            for task in batch
        ]

        # Execute in parallel with timeout
        try:
            results = await asyncio.gather(*coroutines, return_exceptions=True)

            # Convert exceptions to failed TaskResults
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    task = batch[i]
                    processed_results.append(
                        TaskResult(
                            task_id=task.get('id', 0),
                            task_type=task.get('type', 'unknown'),
                            status='failed',
                            error=str(result)
                        )
                    )
                else:
                    processed_results.append(result)

            return processed_results

        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            # Return failed results for all tasks
            return [
                TaskResult(
                    task_id=task.get('id', 0),
                    task_type=task.get('type', 'unknown'),
                    status='failed',
                    error=f"Batch execution error: {str(e)}"
                )
                for task in batch
            ]

    async def _execute_task_with_retry(
        self,
        task: Dict,
        cycle: int,
        context: Dict
    ) -> TaskResult:
        """
        Execute task with retry logic.

        Args:
            task: Task dictionary
            cycle: Current cycle
            context: Context dictionary

        Returns:
            TaskResult object
        """
        task_id = task.get('id', 0)
        task_type = task.get('type', 'unknown')

        # Try execution with retries
        last_error = None
        last_exception = None
        start_time = datetime.now()

        for attempt in range(self.max_retries + 1):
            try:
                # Execute task
                result = await asyncio.wait_for(
                    self._execute_task(task, cycle, context),
                    timeout=self.task_timeout
                )

                # Success!
                execution_time = (datetime.now() - start_time).total_seconds()
                return TaskResult(
                    task_id=task_id,
                    task_type=task_type,
                    status='completed',
                    finding=result,
                    execution_time=execution_time,
                    retry_count=attempt
                )

            except asyncio.TimeoutError:
                last_error = f"Task timeout after {self.task_timeout}s"
                last_exception = None
                logger.warning(f"Task {task_id} timeout (attempt {attempt+1})")

            except Exception as e:
                last_error = str(e)
                last_exception = e
                logger.warning(f"Task {task_id} failed (attempt {attempt+1}): {e}")

                # Check if error is non-recoverable (don't retry)
                if isinstance(e, ProviderAPIError) and not e.is_recoverable():
                    logger.info(f"Task {task_id} failed with non-recoverable error, not retrying")
                    break

            # Track retry
            self.task_retries[task_id] = attempt + 1

            # Small delay before retry (exponential backoff: 1s, 2s, 4s...)
            if attempt < self.max_retries:
                delay = min(2 ** attempt, 8)  # Cap at 8 seconds
                logger.debug(f"Waiting {delay}s before retry")
                await asyncio.sleep(delay)

        # All retries exhausted
        execution_time = (datetime.now() - start_time).total_seconds()
        return TaskResult(
            task_id=task_id,
            task_type=task_type,
            status='failed',
            error=last_error,
            execution_time=execution_time,
            retry_count=self.max_retries
        )

    async def _execute_task(
        self,
        task: Dict,
        cycle: int,
        context: Dict
    ) -> Dict:
        """
        Execute individual task by routing to appropriate agent.

        Args:
            task: Task dictionary
            cycle: Current cycle
            context: Context dictionary

        Returns:
            Finding dictionary
        """
        task_type = task.get('type', 'unknown')

        # Route to appropriate agent
        if task_type == 'data_analysis':
            return await self._execute_data_analysis(task, cycle, context)
        elif task_type == 'literature_review':
            return await self._execute_literature_review(task, cycle, context)
        elif task_type == 'hypothesis_generation':
            return await self._execute_hypothesis_generation(task, cycle, context)
        else:
            return await self._execute_generic_task(task, cycle, context)

    async def _execute_data_analysis(
        self,
        task: Dict,
        cycle: int,
        context: Dict
    ) -> Dict:
        """Execute data analysis task via DataAnalystAgent."""
        agent = self.agents.get('data_analyst')
        if not agent:
            raise RuntimeError(
                "No 'data_analyst' agent was provided to DelegationManager. "
                "Pass agents={'data_analyst': DataAnalystAgent(), ...} at init."
            )

        logger.info(f"Executing data analysis task: {task.get('description', '')[:50]}...")

        result = agent.execute({
            'action': 'interpret_results',
            'result': {'description': task.get('description', '')},
            'hypothesis': context.get('research_objective', ''),
            'lit_context': context.get('recent_findings', []),
        })

        interpretation = result.get('interpretation', {})
        return {
            'finding_id': f"cycle{cycle}_task{task.get('id', 0)}",
            'cycle': cycle,
            'task_id': task.get('id', 0),
            'summary': interpretation.get('summary', ''),
            'statistics': interpretation.get('key_findings', []),
            'methods': 'DataAnalystAgent.execute(interpret_results)',
            'interpretation': interpretation.get('overall_assessment', ''),
            'evidence_type': 'data_analysis',
            'metadata': {'libraries_used': task.get('required_skills', [])}
        }

    async def _execute_literature_review(
        self,
        task: Dict,
        cycle: int,
        context: Dict
    ) -> Dict:
        """Execute literature review task via LiteratureAnalyzerAgent."""
        agent = self.agents.get('literature_analyzer')
        if not agent:
            raise RuntimeError(
                "No 'literature_analyzer' agent was provided to DelegationManager. "
                "Pass agents={'literature_analyzer': LiteratureAnalyzerAgent(), ...} at init."
            )

        logger.info(f"Executing literature review: {task.get('description', '')[:50]}...")

        result = agent.execute({
            'task_type': 'analyze_corpus',
            'papers': context.get('recent_findings', []),
            'generate_insights': True,
        })

        insights = result.get('insights', {})
        return {
            'finding_id': f"cycle{cycle}_task{task.get('id', 0)}",
            'cycle': cycle,
            'task_id': task.get('id', 0),
            'summary': f"Literature review: {task.get('description', '')[:100]}",
            'statistics': {
                'corpus_size': insights.get('corpus_size', 0),
                'common_themes': insights.get('common_themes', []),
                'research_gaps': insights.get('research_gaps', []),
            },
            'methods': 'LiteratureAnalyzerAgent.execute(analyze_corpus)',
            'interpretation': '; '.join(insights.get('common_themes', [])),
            'evidence_type': 'literature_review'
        }

    async def _execute_hypothesis_generation(
        self,
        task: Dict,
        cycle: int,
        context: Dict
    ) -> Dict:
        """Execute hypothesis generation task via HypothesisGeneratorAgent."""
        agent = self.agents.get('hypothesis_generator')
        if not agent:
            raise RuntimeError(
                "No 'hypothesis_generator' agent was provided to DelegationManager. "
                "Pass agents={'hypothesis_generator': HypothesisGeneratorAgent(), ...} at init."
            )

        logger.info(f"Executing hypothesis generation: {task.get('description', '')[:50]}...")

        response = agent.generate_hypotheses(
            research_question=task.get('description', ''),
            store_in_db=False,
        )

        hypotheses = response.hypotheses if hasattr(response, 'hypotheses') else []
        return {
            'finding_id': f"cycle{cycle}_task{task.get('id', 0)}",
            'cycle': cycle,
            'task_id': task.get('id', 0),
            'summary': '; '.join(h.statement for h in hypotheses) if hypotheses else '',
            'statistics': {
                'hypotheses_generated': len(hypotheses),
                'testable_hypotheses': sum(
                    1 for h in hypotheses
                    if getattr(h, 'testability_score', 0) >= 0.5
                ),
                'generation_time_seconds': getattr(response, 'generation_time_seconds', None),
            },
            'methods': 'HypothesisGeneratorAgent.generate_hypotheses()',
            'interpretation': 'New testable hypotheses identified',
            'evidence_type': 'hypothesis_generation'
        }

    async def _execute_generic_task(
        self,
        task: Dict,
        cycle: int,
        context: Dict
    ) -> Dict:
        """Execute generic task -- routes experiment_design to its agent, else raises."""
        task_type = task.get('type', 'unknown')

        # Route experiment_design to ExperimentDesignerAgent
        if task_type == 'experiment_design':
            agent = self.agents.get('experiment_designer')
            if not agent:
                raise RuntimeError(
                    "No 'experiment_designer' agent was provided to DelegationManager. "
                    "Pass agents={'experiment_designer': ExperimentDesignerAgent(), ...} at init."
                )

            logger.info(f"Executing experiment design: {task.get('description', '')[:50]}...")

            response = agent.design_experiment(
                hypothesis=task.get('description', ''),
                store_in_db=False,
            )

            protocol = response.protocol if hasattr(response, 'protocol') else None
            return {
                'finding_id': f"cycle{cycle}_task{task.get('id', 0)}",
                'cycle': cycle,
                'task_id': task.get('id', 0),
                'summary': getattr(protocol, 'name', '') if protocol else '',
                'statistics': {
                    'rigor_score': getattr(response, 'rigor_score', None),
                    'completeness_score': getattr(response, 'completeness_score', None),
                    'feasibility': getattr(response, 'feasibility_assessment', None),
                    'design_time_seconds': getattr(response, 'design_time_seconds', None),
                },
                'methods': 'ExperimentDesignerAgent.design_experiment()',
                'interpretation': getattr(response, 'feasibility_assessment', ''),
                'evidence_type': 'experiment_design'
            }

        # No matching agent for unknown task types
        if not self.agents:
            raise RuntimeError(
                "No agents were provided to DelegationManager. "
                "Cannot execute task type: " + task_type
            )

        logger.warning(f"Unknown task type: {task_type}, no specific agent handler")
        raise RuntimeError(f"No agent handler for task type: {task_type}")

    def get_execution_statistics(self) -> Dict:
        """Get statistics about task execution."""
        return {
            'max_parallel_tasks': self.max_parallel_tasks,
            'max_retries': self.max_retries,
            'task_timeout': self.task_timeout,
            'available_agents': list(self.agents.keys()),
            'agent_routing': self.AGENT_ROUTING
        }
