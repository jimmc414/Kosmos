"""
Research Workflow Integration for Kosmos.

Orchestrates the complete autonomous research cycle integrating all 6 gaps.

This is the main entry point for running the Kosmos AI scientist system.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

# Gap imports
from kosmos.compression import ContextCompressor
from kosmos.world_model.artifacts import ArtifactStateManager
from kosmos.orchestration import (
    PlanCreatorAgent,
    PlanReviewerAgent,
    DelegationManager,
    NoveltyDetector
)
from kosmos.validation import ScholarEvalValidator
from kosmos.agents import SkillLoader

logger = logging.getLogger(__name__)


class ResearchWorkflow:
    """
    Complete autonomous research workflow.

    Integrates all 6 gap implementations:
    - Gap 0: Context compression
    - Gap 1: State management
    - Gap 2: Task generation & orchestration
    - Gap 3: Agent integration with skills
    - Gap 4: Python-first tooling
    - Gap 5: Discovery validation

    Example:
        workflow = ResearchWorkflow(
            research_objective="Investigate KRAS mutations in cancer",
            anthropic_api_key="your-key"
        )

        # Run 5 cycles
        results = await workflow.run(num_cycles=5)

        # Generate report
        report = await workflow.generate_report()
    """

    def __init__(
        self,
        research_objective: str,
        anthropic_client=None,
        artifacts_dir: str = "artifacts",
        world_model=None,
        max_cycles: int = 20,
        seed: Optional[int] = None,
        temperature: Optional[float] = None
    ):
        """
        Initialize Research Workflow.

        Args:
            research_objective: Main research goal
            anthropic_client: Anthropic client for LLM calls
            artifacts_dir: Directory for artifact storage
            world_model: Optional knowledge graph
            max_cycles: Maximum research cycles
            seed: Random seed for reproducibility (Issue #64)
            temperature: LLM temperature override (Issue #64)
        """
        self.research_objective = research_objective
        self.max_cycles = max_cycles
        self._seed = seed
        self._temperature = temperature

        # Apply seed if provided (Issue #64: Multi-Run Convergence)
        if seed is not None:
            from kosmos.safety.reproducibility import ReproducibilityManager
            self._reproducibility_manager = ReproducibilityManager(default_seed=seed)
            self._reproducibility_manager.set_seed(seed)
            logger.info(f"Set random seed: {seed}")
        else:
            self._reproducibility_manager = None

        # Initialize components
        logger.info("Initializing Kosmos Research Workflow...")

        # Gap 0: Context Compression
        self.context_compressor = ContextCompressor(anthropic_client)
        logger.info("✓ Gap 0: Context Compression initialized")

        # Gap 1: State Manager
        self.state_manager = ArtifactStateManager(
            artifacts_dir=artifacts_dir,
            world_model=world_model
        )
        logger.info("✓ Gap 1: State Manager initialized")

        # Gap 3: Skill Loader
        self.skill_loader = SkillLoader()
        logger.info("✓ Gap 3: Skill Loader initialized")

        # Gap 5: Discovery Validation
        self.scholar_eval = ScholarEvalValidator(anthropic_client)
        logger.info("✓ Gap 5: ScholarEval Validator initialized")

        # Gap 2: Orchestration Components
        self.plan_creator = PlanCreatorAgent(anthropic_client)
        self.plan_reviewer = PlanReviewerAgent(anthropic_client)
        self.delegation_manager = DelegationManager()
        self.novelty_detector = NoveltyDetector()
        logger.info("✓ Gap 2: Orchestration components initialized")

        # Tracking
        self.past_tasks = []
        self.cycle_results = []
        self.start_time = None

    async def run(
        self,
        num_cycles: int = 5,
        tasks_per_cycle: int = 10
    ) -> Dict:
        """
        Run autonomous research workflow.

        Args:
            num_cycles: Number of cycles to execute
            tasks_per_cycle: Tasks to generate per cycle

        Returns:
            Dictionary with:
            - cycles_completed: Number of cycles
            - total_findings: Total findings generated
            - validated_findings: Validated findings count
            - validation_rate: Percentage validated
            - total_time: Execution time in seconds
        """
        self.start_time = datetime.now()

        logger.info(
            f"\n{'='*70}\n"
            f"Starting Kosmos Research Workflow\n"
            f"Objective: {self.research_objective}\n"
            f"Cycles: {num_cycles}\n"
            f"Tasks per cycle: {tasks_per_cycle}\n"
            f"{'='*70}\n"
        )

        for cycle in range(1, num_cycles + 1):
            logger.info(f"\n--- Cycle {cycle}/{num_cycles} ---")

            try:
                cycle_result = await self._execute_cycle(cycle, tasks_per_cycle)
                self.cycle_results.append(cycle_result)

                logger.info(
                    f"Cycle {cycle} complete: "
                    f"{cycle_result.get('tasks_completed', 0)}/{tasks_per_cycle} tasks, "
                    f"{cycle_result.get('validated_findings', 0)} validated findings"
                )

            except Exception as e:
                logger.error(f"Cycle {cycle} failed: {e}")
                continue

        # Compute final statistics
        return self._compute_final_statistics()

    async def _execute_cycle(self, cycle: int, num_tasks: int) -> Dict:
        """Execute one research cycle."""

        # Step 1: Get context from State Manager
        context = self.state_manager.get_cycle_context(cycle, lookback=3)
        context['research_objective'] = self.research_objective

        logger.info(f"  Context: {context.get('findings_count', 0)} recent findings")

        # Step 2: Plan Creator generates tasks
        plan = self.plan_creator.create_plan(
            research_objective=self.research_objective,
            context=context,
            num_tasks=num_tasks
        )

        logger.info(f"  Generated plan with {len(plan.tasks)} tasks")

        # Step 3: Novelty Detector checks redundancy
        if self.past_tasks:
            self.novelty_detector.index_past_tasks(self.past_tasks)
            plan_novelty = self.novelty_detector.check_plan_novelty(plan.to_dict())
            logger.info(
                f"  Novelty: {plan_novelty['novel_task_count']}/{len(plan.tasks)} "
                f"novel tasks ({plan_novelty['plan_novelty_score']:.2f})"
            )

        # Step 4: Plan Reviewer validates quality
        review = self.plan_reviewer.review_plan(plan.to_dict(), context)

        logger.info(
            f"  Plan review: {'APPROVED' if review.approved else 'REJECTED'} "
            f"(score: {review.average_score:.1f}/10)"
        )

        # If rejected, attempt revision (once)
        if not review.approved:
            logger.info("  Revising plan based on feedback...")
            plan = self.plan_creator.revise_plan(plan, review.to_dict(), context)
            review = self.plan_reviewer.review_plan(plan.to_dict(), context)

            logger.info(
                f"  Revised plan: {'APPROVED' if review.approved else 'REJECTED'}"
            )

        # Step 5: Delegation Manager executes approved tasks
        completed_tasks = []
        if review.approved:
            execution_result = await self.delegation_manager.execute_plan(
                plan.to_dict(),
                cycle,
                context
            )

            completed_tasks = execution_result.get('completed_tasks', [])
            logger.info(
                f"  Execution: {len(completed_tasks)}/{num_tasks} tasks completed"
            )
        else:
            logger.warning("  Plan rejected after revision, skipping execution")

        # Step 6 & 7: Validate and save findings
        validated_count = 0
        for task_result in completed_tasks:
            finding = task_result.get('finding')
            if not finding:
                continue

            # ScholarEval validation
            eval_score = self.scholar_eval.evaluate_finding(finding)

            if eval_score.passes_threshold:
                # Save validated finding
                finding['scholar_eval'] = eval_score.to_dict()
                await self.state_manager.save_finding_artifact(
                    cycle,
                    task_result.get('task_id', 0),
                    finding
                )
                validated_count += 1
            else:
                logger.debug(
                    f"    Finding rejected: score={eval_score.overall_score:.2f}"
                )

        logger.info(f"  Validated: {validated_count}/{len(completed_tasks)} findings")

        # Step 8: Compress cycle results
        if completed_tasks:
            compressed_cycle = self.context_compressor.compress_cycle_results(
                cycle,
                completed_tasks
            )
            logger.info(
                f"  Compressed: {len(completed_tasks)} tasks → "
                f"{len(compressed_cycle.summary)} chars"
            )

        # Track tasks for novelty detection
        for task in plan.tasks:
            self.past_tasks.append(task.to_dict())

        # Generate cycle summary
        await self.state_manager.generate_cycle_summary(cycle)

        return {
            'cycle': cycle,
            'tasks_generated': len(plan.tasks),
            'tasks_completed': len(completed_tasks),
            'validated_findings': validated_count,
            'plan_approved': review.approved,
            'plan_score': review.average_score
        }

    def _compute_final_statistics(self) -> Dict:
        """Compute final statistics across all cycles."""
        total_time = (datetime.now() - self.start_time).total_seconds()

        all_findings = self.state_manager.get_all_findings()
        validated_findings = self.state_manager.get_validated_findings()

        total_tasks_completed = sum(
            r.get('tasks_completed', 0) for r in self.cycle_results
        )
        total_tasks_generated = sum(
            r.get('tasks_generated', 0) for r in self.cycle_results
        )

        results = {
            'cycles_completed': len(self.cycle_results),
            'total_findings': len(all_findings),
            'validated_findings': len(validated_findings),
            'validation_rate': (
                len(validated_findings) / len(all_findings)
                if all_findings else 0
            ),
            'total_tasks_generated': total_tasks_generated,
            'total_tasks_completed': total_tasks_completed,
            'task_completion_rate': (
                total_tasks_completed / total_tasks_generated
                if total_tasks_generated else 0
            ),
            'total_time': total_time,
            'research_objective': self.research_objective
        }

        logger.info(
            f"\n{'='*70}\n"
            f"Research Workflow Complete!\n"
            f"Cycles: {results['cycles_completed']}\n"
            f"Findings: {results['total_findings']} "
            f"({results['validated_findings']} validated, "
            f"{results['validation_rate']*100:.1f}%)\n"
            f"Tasks: {results['total_tasks_completed']}/{results['total_tasks_generated']} "
            f"({results['task_completion_rate']*100:.1f}% completion)\n"
            f"Time: {results['total_time']:.1f}s\n"
            f"{'='*70}\n"
        )

        return results

    async def generate_report(self) -> str:
        """
        Generate research report from findings.

        Returns:
            Markdown-formatted research report
        """
        validated_findings = self.state_manager.get_validated_findings()

        report = f"# Research Report\n\n"
        report += f"**Objective**: {self.research_objective}\n"
        report += f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n"
        report += f"**Cycles Completed**: {len(self.cycle_results)}\n\n"

        report += f"## Summary\n\n"
        report += f"This autonomous research system completed {len(self.cycle_results)} "
        report += f"research cycles, generating {len(validated_findings)} validated findings.\n\n"

        report += f"## Key Findings\n\n"
        for i, finding in enumerate(validated_findings[:10], 1):  # Top 10
            report += f"### Finding {i}\n\n"
            report += f"{finding.summary}\n\n"

            # Statistics
            if finding.statistics:
                report += "**Statistics**:\n"
                for key, value in finding.statistics.items():
                    if isinstance(value, float):
                        report += f"- {key}: {value:.4f}\n"
                    else:
                        report += f"- {key}: {value}\n"
                report += "\n"

            # Evidence with code provenance (Issue #62)
            if finding.code_provenance:
                prov = finding.code_provenance
                hyperlink = f"{prov['notebook_path']}#cell={prov['cell_index']}&line={prov['start_line']}"
                filename = prov['notebook_path'].split('/')[-1]
                report += f"**Code Citation**: [{filename}]({hyperlink})"
                if prov.get('start_line') and prov.get('end_line'):
                    report += f" (lines {prov['start_line']}-{prov['end_line']})"
                report += "\n\n"
            elif finding.notebook_path:
                report += f"**Evidence**: `{finding.notebook_path}`\n\n"

            # Quality score
            if finding.scholar_eval:
                overall = finding.scholar_eval.get('overall_score', 0)
                report += f"**Quality Score**: {overall:.2f}/1.0\n\n"

        return report

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        return {
            'workflow': {
                'research_objective': self.research_objective,
                'max_cycles': self.max_cycles,
                'cycles_completed': len(self.cycle_results)
            },
            'state_manager': self.state_manager.get_statistics(),
            'skill_loader': self.skill_loader.get_statistics(),
            'novelty_detector': self.novelty_detector.get_statistics()
        }
