"""
Run command for Kosmos CLI.

Executes autonomous research with live progress visualization.
"""

import sys
import time
import logging
from typing import Optional
from datetime import datetime
from pathlib import Path

import typer
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.layout import Layout
from rich.text import Text

from kosmos.cli.utils import (
    console,
    print_success,
    print_error,
    print_info,
    get_icon,
    format_timestamp,
    create_status_text,
)
from kosmos.cli.interactive import run_interactive_mode
from kosmos.cli.views.results_viewer import ResultsViewer

logger = logging.getLogger(__name__)


def run_research(
    question: Optional[str] = typer.Argument(None, help="Research question to investigate"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Research domain (biology, neuroscience, materials, etc.)"),
    max_iterations: int = typer.Option(10, "--max-iterations", "-i", help="Maximum number of research iterations"),
    budget: Optional[float] = typer.Option(None, "--budget", "-b", help="Budget limit in USD"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable caching"),
    interactive: bool = typer.Option(False, "--interactive", help="Use interactive mode"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to file (JSON or Markdown)"),
):
    """
    Run autonomous research on a scientific question.

    Examples:

        # Interactive mode (recommended for first time)
        kosmos run --interactive

        # Direct command
        kosmos run "What metabolic pathways differ between cancer and normal cells?" --domain biology

        # With budget limit
        kosmos run "How do perovskites optimize efficiency?" --domain materials --budget 50

        # Save results
        kosmos run "Question" --output results.json
    """
    # Use interactive mode if requested or no question provided
    if interactive or not question:
        config = run_interactive_mode()

        if not config:
            console.print("[warning]Research cancelled.[/warning]")
            raise typer.Exit(0)

        # Extract config
        question = config["question"]
        domain = config["domain"]
        max_iterations = config["max_iterations"]
        budget = config.get("budget_usd")
        no_cache = not config.get("enable_cache", True)

    # Validate inputs
    if not question:
        print_error("No research question provided. Use --interactive or provide a question.")
        raise typer.Exit(1)

    # Show starting message
    console.print()
    console.print(
        Panel(
            f"[cyan]Starting autonomous research...[/cyan]\n\n"
            f"**Question:** {question}\n"
            f"**Domain:** {domain or 'auto-detect'}\n"
            f"**Max Iterations:** {max_iterations}\n"
            f"**Budget:** ${budget} USD" if budget else "**Budget:** No limit",
            title=f"[bright_blue]{get_icon('rocket')} Kosmos Research[/bright_blue]",
            border_style="bright_blue",
        )
    )
    console.print()

    # Initialize research
    try:
        from kosmos.agents.research_director import ResearchDirectorAgent
        from kosmos.config import get_config

        # Get configuration
        config_obj = get_config()

        # Override with CLI parameters
        if domain:
            config_obj.research.enabled_domains = [domain]
        config_obj.research.max_iterations = max_iterations
        if budget:
            config_obj.research.budget_usd = budget
        config_obj.claude.enable_cache = not no_cache

        # Create flattened config dict for agents
        # Agents expect flat keys, not nested KosmosConfig structure
        flat_config = {
            # Research settings
            "max_iterations": config_obj.research.max_iterations,
            "enabled_domains": config_obj.research.enabled_domains,
            "enabled_experiment_types": config_obj.research.enabled_experiment_types,
            "min_novelty_score": config_obj.research.min_novelty_score,
            "enable_autonomous_iteration": config_obj.research.enable_autonomous_iteration,
            "budget_usd": config_obj.research.budget_usd,

            # Performance/concurrent operations settings
            "enable_concurrent_operations": config_obj.performance.enable_concurrent_operations,
            "max_parallel_hypotheses": config_obj.performance.max_parallel_hypotheses,
            "max_concurrent_experiments": config_obj.performance.max_concurrent_experiments,
            "max_concurrent_llm_calls": config_obj.performance.max_concurrent_llm_calls,
            "llm_rate_limit_per_minute": config_obj.performance.llm_rate_limit_per_minute,

            # LLM provider settings
            "llm_provider": config_obj.llm_provider,
            "enable_cache": config_obj.claude.enable_cache,
        }

        # Create research director
        director = ResearchDirectorAgent(
            research_question=question,
            domain=domain,
            config=flat_config
        )

        # Run research with live progress
        results = run_with_progress(director, question, max_iterations)

        # Display results
        viewer = ResultsViewer()
        viewer.display_research_overview(results)
        viewer.display_hypotheses_table(results.get("hypotheses", []))
        viewer.display_experiments_table(results.get("experiments", []))

        if "metrics" in results:
            viewer.display_metrics_summary(results["metrics"])

        # Export if requested
        if output:
            if output.suffix == ".json":
                viewer.export_to_json(results, output)
            elif output.suffix in [".md", ".markdown"]:
                viewer.export_to_markdown(results, output)
            else:
                print_error(f"Unsupported output format: {output.suffix}")

        print_success("Research completed successfully!", title="Complete")

    except KeyboardInterrupt:
        console.print("\n[warning]Research interrupted by user[/warning]")
        raise typer.Exit(130)

    except Exception as e:
        print_error(f"Research failed: {str(e)}", title="Error")
        if "--debug" in sys.argv:
            raise
        raise typer.Exit(1)


def run_with_progress(director, question: str, max_iterations: int) -> dict:
    """
    Run research with live progress display.

    Args:
        director: ResearchDirectorAgent instance
        question: Research question
        max_iterations: Maximum iterations

    Returns:
        Research results dictionary
    """
    # Create progress bars
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    # Create tasks for each phase
    hypothesis_task = progress.add_task("[cyan]Generating hypotheses...", total=100)
    experiment_task = progress.add_task("[yellow]Designing experiments...", total=100)
    execution_task = progress.add_task("[green]Executing experiments...", total=100)
    analysis_task = progress.add_task("[magenta]Analyzing results...", total=100)
    iteration_task = progress.add_task("[bright_blue]Research progress...", total=max_iterations)

    # Create current hypothesis table
    def create_status_table():
        table = Table(title="Current Status", box=None, show_header=True)
        table.add_column("Phase", style="cyan")
        table.add_column("Status", style="white")

        # Get current state from director
        state = getattr(director.workflow, "current_state", "INITIALIZING")
        iteration = getattr(director.research_plan, "iteration_count", 0)

        table.add_row("Workflow State", create_status_text(state))
        table.add_row("Iteration", f"{iteration}/{max_iterations}")
        table.add_row("Started", format_timestamp(datetime.utcnow()))

        return table

    # Run with live display
    with Live(progress, console=console, refresh_per_second=4):
        try:
            # Start research
            director.execute({"action": "start_research"})

            # Research loop - execute until convergence or max iterations
            iteration = 0
            while iteration < max_iterations:
                # Get current status
                status = director.get_research_status()

                # Update iteration progress
                progress.update(iteration_task, completed=iteration + 1)

                # Update phase-specific progress based on workflow state
                # WorkflowState enum uses lowercase values
                workflow_state = status.get("workflow_state", "initializing")

                if workflow_state == "generating_hypotheses":
                    progress.update(hypothesis_task, completed=50)
                elif workflow_state == "designing_experiments":
                    progress.update(hypothesis_task, completed=100)
                    progress.update(experiment_task, completed=50)
                elif workflow_state == "executing":
                    progress.update(experiment_task, completed=100)
                    progress.update(execution_task, completed=50)
                elif workflow_state == "analyzing":
                    progress.update(execution_task, completed=100)
                    progress.update(analysis_task, completed=50)
                elif workflow_state in ["refining", "converged"]:
                    progress.update(analysis_task, completed=100)

                # Check for convergence
                if status.get("has_converged", False):
                    progress.update(iteration_task, completed=max_iterations)
                    break

                # Execute next research step
                director.execute({"action": "step"})

                # Update iteration counter
                iteration = status.get("iteration", iteration)

                # Small delay to allow UI updates
                time.sleep(0.05)

            # Mark all tasks as complete
            progress.update(hypothesis_task, completed=100)
            progress.update(experiment_task, completed=100)
            progress.update(execution_task, completed=100)
            progress.update(analysis_task, completed=100)

            # Get final research status
            final_status = director.get_research_status()

            # Build results from actual research
            # Fetch actual hypothesis and experiment objects from database
            from kosmos.db import get_session
            from kosmos.db.operations import get_hypothesis, get_experiment

            hypotheses_data = []
            experiments_data = []

            try:
                with get_session() as session:
                    # Fetch hypotheses from database using IDs
                    for h_id in director.research_plan.hypothesis_pool:
                        hypothesis = get_hypothesis(session, h_id)
                        if hypothesis:
                            hypotheses_data.append(hypothesis.to_dict() if hasattr(hypothesis, 'to_dict') else str(hypothesis))

                    # Fetch experiments from database using IDs
                    for e_id in director.research_plan.completed_experiments:
                        experiment = get_experiment(session, e_id)
                        if experiment:
                            experiments_data.append(experiment.to_dict() if hasattr(experiment, 'to_dict') else str(experiment))
            except Exception as e:
                logger.warning(f"Could not fetch all objects from database: {e}")
                # Fallback: use IDs as strings
                hypotheses_data = list(director.research_plan.hypothesis_pool)
                experiments_data = list(director.research_plan.completed_experiments)

            results = {
                "id": f"research_{int(time.time())}",
                "question": question,
                "domain": final_status.get("domain", "auto"),
                "state": final_status.get("workflow_state", "COMPLETED"),
                "current_iteration": final_status.get("iteration", 0),
                "max_iterations": max_iterations,
                "has_converged": final_status.get("has_converged", False),
                "convergence_reason": final_status.get("convergence_reason"),
                "hypotheses": hypotheses_data,
                "experiments": experiments_data,
                "metrics": {
                    "api_calls": getattr(director.llm_client, 'total_requests', 0),
                    "cache_hits": getattr(director.llm_client, 'cache_hits', 0),
                    "cache_misses": getattr(director.llm_client, 'cache_misses', 0),
                    "hypotheses_generated": final_status.get("hypothesis_pool_size", 0),
                    "hypotheses_tested": final_status.get("hypotheses_tested", 0),
                    "hypotheses_supported": final_status.get("hypotheses_supported", 0),
                    "hypotheses_rejected": final_status.get("hypotheses_rejected", 0),
                    "experiments_executed": final_status.get("experiments_completed", 0),
                },
            }

            return results

        except Exception as e:
            console.print(f"\n[error]Error during research: {str(e)}[/error]")
            raise


if __name__ == "__main__":
    # Allow standalone testing
    typer.run(run_research)
