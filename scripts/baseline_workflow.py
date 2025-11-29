#!/usr/bin/env python3
"""
Baseline Workflow Runner for Phase 3 Validation.

Runs a simplified multi-cycle research workflow using individual agents directly.
Part of the Kosmos validation roadmap - Phase 3.1 Baseline Measurement.

Note: Uses individual agents (HypothesisGeneratorAgent, ExperimentDesignerAgent,
DataAnalystAgent) directly rather than the message-based ResearchDirectorAgent
to ensure synchronous execution and metrics capture.
"""

import json
import time
from datetime import datetime
from pathlib import Path

# Force reload .env to override shell environment
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# Reset LLM client singleton to pick up new config
from kosmos.core.llm import get_client
get_client(reset=True)

from kosmos.agents.hypothesis_generator import HypothesisGeneratorAgent
from kosmos.agents.experiment_designer import ExperimentDesignerAgent
from kosmos.core.metrics import get_metrics


def run_baseline(num_cycles: int = 3):
    """
    Run baseline workflow using individual agents directly.

    Each cycle:
    1. Generate hypotheses
    2. Design experiments
    3. (Skip execution - no sandbox needed for baseline metrics)

    Args:
        num_cycles: Number of research cycles to execute

    Returns:
        Report dictionary with all metrics
    """
    # Initialize
    artifacts_dir = Path("./artifacts/baseline_run")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics = get_metrics()
    start_time = time.time()

    print("=" * 60)
    print("KOSMOS BASELINE WORKFLOW (Direct Agent Execution)")
    print("=" * 60)
    print(f"Research question: How does temperature affect enzyme activity?")
    print(f"Cycles: {num_cycles}")
    print(f"Artifacts: {artifacts_dir}")
    print("=" * 60)

    # Initialize agents
    # Disable literature search to avoid API hangs
    hypothesis_agent = HypothesisGeneratorAgent(config={"num_hypotheses": 2, "use_literature_context": False})
    experiment_agent = ExperimentDesignerAgent(config={})

    results = {
        "cycles_completed": 0,
        "hypotheses_generated": 0,
        "experiments_designed": 0,
        "cycle_times": [],
        "error": None
    }

    research_question = "How does temperature affect enzyme activity?"
    domain = "biology"

    try:
        for cycle in range(1, num_cycles + 1):
            cycle_start = time.time()
            print(f"\n{'='*60}")
            print(f"CYCLE {cycle}/{num_cycles}")
            print(f"{'='*60}")

            # Step 1: Generate hypotheses
            print(f"\n[{cycle}.1] Generating hypotheses...")
            step_start = time.time()

            response = hypothesis_agent.generate_hypotheses(
                research_question=research_question,
                domain=domain,
                num_hypotheses=2,
                store_in_db=False  # Don't persist for baseline
            )

            step_time = time.time() - step_start

            # Extract hypotheses from response
            hypotheses = response.hypotheses if hasattr(response, 'hypotheses') else []
            num_hyps = len(hypotheses)
            results["hypotheses_generated"] += num_hyps

            print(f"  Generated: {num_hyps} hypotheses")
            print(f"  Time: {step_time:.1f}s")

            if hypotheses:
                for i, hyp in enumerate(hypotheses[:2], 1):
                    statement = hyp.statement[:80] if hasattr(hyp, 'statement') else str(hyp)[:80]
                    print(f"  [{i}] {statement}...")

            # Step 2: Design experiment for first hypothesis
            if hypotheses:
                print(f"\n[{cycle}.2] Designing experiment...")
                step_start = time.time()

                # Use the first hypothesis directly (it's already a Hypothesis object)
                hyp = hypotheses[0]

                try:
                    design_response = experiment_agent.design_experiment(
                        hypothesis=hyp,
                        store_in_db=False  # Don't persist for baseline
                    )

                    step_time = time.time() - step_start

                    if design_response and hasattr(design_response, 'protocol'):
                        protocol = design_response.protocol
                        results["experiments_designed"] += 1
                        protocol_name = protocol.name if hasattr(protocol, 'name') else "Protocol"
                        print(f"  Designed: {protocol_name[:60]}...")
                        print(f"  Time: {step_time:.1f}s")
                    else:
                        print(f"  Design returned but no protocol")
                        print(f"  Time: {step_time:.1f}s")

                except Exception as e:
                    step_time = time.time() - step_start
                    print(f"  Failed to design experiment: {e}")
                    print(f"  Time: {step_time:.1f}s")

            # Record cycle time
            cycle_time = time.time() - cycle_start
            results["cycle_times"].append(cycle_time)
            results["cycles_completed"] = cycle

            print(f"\n  Cycle {cycle} complete in {cycle_time:.1f}s")

    except Exception as e:
        print(f"\n[ERROR] Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        results["error"] = str(e)

    total_time = time.time() - start_time

    # Collect metrics from provider
    try:
        provider = get_client()
        if hasattr(provider, 'get_usage_stats'):
            usage = provider.get_usage_stats()
        else:
            usage = {}
    except Exception:
        usage = {}

    # Build report
    report = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "config": {
            "research_question": research_question,
            "domain": domain,
            "num_cycles": num_cycles,
            "llm_provider": "litellm/ollama/qwen3-kosmos-fast"
        },
        "workflow_results": results,
        "provider_usage": usage,
        "timing": {
            "total_seconds": total_time,
            "total_minutes": total_time / 60,
            "avg_per_cycle": total_time / max(results["cycles_completed"], 1),
            "cycle_times": results["cycle_times"]
        }
    }

    # Save report
    report_path = artifacts_dir / "baseline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE WORKFLOW RESULTS")
    print("=" * 60)

    if results.get("error"):
        print(f"[ERROR] {results['error']}")

    print(f"Cycles completed: {results['cycles_completed']}/{num_cycles}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    if results["cycles_completed"] > 0:
        avg_time = total_time / results["cycles_completed"]
        print(f"Avg time/cycle: {avg_time:.1f}s")

    print(f"\nResearch Metrics:")
    print(f"  Hypotheses generated: {results['hypotheses_generated']}")
    print(f"  Experiments designed: {results['experiments_designed']}")

    if usage:
        print(f"\nProvider Usage:")
        print(f"  Total requests: {usage.get('total_requests', 0)}")
        print(f"  Total tokens: {usage.get('total_tokens', 0)}")
        print(f"  Total cost: ${usage.get('total_cost_usd', 0):.4f}")

    print(f"\nReport saved to: {report_path}")
    print("=" * 60)

    return report


if __name__ == "__main__":
    import sys

    # Allow command-line override of cycles
    num_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    run_baseline(num_cycles=num_cycles)
