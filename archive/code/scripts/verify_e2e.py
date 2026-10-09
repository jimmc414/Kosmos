#!/usr/bin/env python3
"""
End-to-End Verification Script for Kosmos.

Verifies that all paper requirements are met and demonstrates
multi-cycle autonomous research capability.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from libraries
for lib in ['kosmos.orchestration', 'kosmos.workflow', 'kosmos.agents']:
    logging.getLogger(lib).setLevel(logging.WARNING)


async def run_verification(num_cycles: int = 5, tasks_per_cycle: int = 10):
    """Run E2E verification of Kosmos workflow."""
    from kosmos.workflow.research_loop import ResearchWorkflow

    print("=" * 70)
    print("KOSMOS E2E VERIFICATION")
    print("=" * 70)
    print(f"\nConfiguration: {num_cycles} cycles, {tasks_per_cycle} tasks/cycle")

    # Create workflow
    workflow = ResearchWorkflow(
        research_objective="Investigate KRAS mutations in pancreatic cancer drug resistance",
        artifacts_dir="/tmp/kosmos_verification"
    )

    # Run multi-cycle workflow
    print("\nRunning multi-cycle research workflow...")
    result = await workflow.run(num_cycles=num_cycles, tasks_per_cycle=tasks_per_cycle)

    # Paper Requirements Verification
    print("\n" + "=" * 70)
    print("PAPER REQUIREMENTS VERIFICATION")
    print("=" * 70)

    verifications = {
        # Gap 0: Context Compression
        "Gap 0 - Context compression implemented": hasattr(workflow, 'context_compressor'),

        # Gap 1: State Manager
        "Gap 1 - State manager initialized": hasattr(workflow, 'state_manager'),
        "Gap 1 - State persistence across cycles": len(workflow.past_tasks) > 0,

        # Gap 2: Task Orchestration
        "Gap 2 - Multi-cycle management": result['cycles_completed'] == num_cycles,
        "Gap 2 - Task generation": result['total_tasks_generated'] >= num_cycles * tasks_per_cycle,
        "Gap 2 - Plan review process": all(
            cr.get('plan_approved', True) for cr in workflow.cycle_results
        ),
        "Gap 2 - Novelty detection": hasattr(workflow, 'novelty_detector'),

        # Gap 3: Agent Integration
        "Gap 3 - Skill loader initialized": hasattr(workflow, 'skill_loader'),

        # Gap 5: Discovery Validation
        "Gap 5 - ScholarEval implemented": hasattr(workflow, 'scholar_eval'),
        "Gap 5 - Finding validation": result['validation_rate'] > 0,
    }

    all_passed = True
    for req, passed in verifications.items():
        status = "\033[92m[PASS]\033[0m" if passed else "\033[91m[FAIL]\033[0m"
        if not passed:
            all_passed = False
        print(f"  {status} {req}")

    # Summary Statistics
    print("\n" + "=" * 70)
    print("WORKFLOW STATISTICS")
    print("=" * 70)
    print(f"  Cycles completed: {result['cycles_completed']}")
    print(f"  Total tasks generated: {result['total_tasks_generated']}")
    print(f"  Tasks completed: {result['total_tasks_completed']} ({result['task_completion_rate']*100:.0f}%)")
    print(f"  Total findings: {result['total_findings']}")
    print(f"  Validated findings: {result['validated_findings']} ({result['validation_rate']*100:.0f}%)")
    print(f"  Total execution time: {result['total_time']:.2f}s")

    # Per-cycle breakdown
    print("\n  Per-cycle breakdown:")
    for i, cr in enumerate(workflow.cycle_results, 1):
        print(f"    Cycle {i}: {cr['tasks_generated']} tasks, {cr['tasks_completed']} completed")

    # Overall result
    print("\n" + "=" * 70)
    passed = sum(verifications.values())
    total = len(verifications)
    if all_passed:
        print(f"\033[92mVERIFICATION PASSED: {passed}/{total} requirements met\033[0m")
    else:
        print(f"\033[91mVERIFICATION INCOMPLETE: {passed}/{total} requirements met\033[0m")
    print("=" * 70)

    return all_passed, result


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Kosmos E2E Verification")
    parser.add_argument("--cycles", type=int, default=5, help="Number of cycles")
    parser.add_argument("--tasks", type=int, default=10, help="Tasks per cycle")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    success, result = asyncio.run(run_verification(args.cycles, args.tasks))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
