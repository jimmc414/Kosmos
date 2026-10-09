"""
Status command for Kosmos CLI.

Shows current status of a research run with live updates.
"""

import time
from typing import Optional
from datetime import datetime

import typer
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn
from rich.layout import Layout

from kosmos.cli.utils import (
    console,
    print_error,
    print_info,
    get_icon,
    create_table,
    create_status_text,
    create_domain_text,
    create_metric_text,
    format_timestamp,
    format_duration,
    validate_run_id,
    get_db_session,
)
from kosmos.cli.views.results_viewer import ResultsViewer


def show_status(
    run_id: Optional[str] = typer.Argument(None, help="Research run ID to check (omit for latest)"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch mode - refresh every 5 seconds"),
    show_details: bool = typer.Option(False, "--details", "-d", help="Show detailed information"),
):
    """
    Show status of a research run.

    Examples:

        # Show latest research status
        kosmos status

        # Show specific run
        kosmos status research_12345

        # Watch mode (live updates)
        kosmos status --watch

        # Detailed view
        kosmos status research_12345 --details
    """
    try:
        # Get research data
        research_data = get_research_data(run_id)

        if not research_data:
            print_error(f"Research run not found: {run_id or 'latest'}")
            raise typer.Exit(1)

        # Display status
        if watch:
            display_status_live(research_data, show_details)
        else:
            display_status_once(research_data, show_details)

    except KeyboardInterrupt:
        console.print("\n[warning]Status display cancelled[/warning]")
        raise typer.Exit(130)

    except typer.Exit:
        # An intentional exit (with its own code) is not a failure to report
        raise

    except Exception as e:
        print_error(f"Failed to get status: {str(e)}")
        raise typer.Exit(1)


def get_research_data(run_id: Optional[str] = None) -> Optional[dict]:
    """
    Get research data from database.

    Args:
        run_id: Optional run ID (None for latest)

    Returns:
        Research data dictionary or None
    """
    try:
        with get_db_session() as session:
            from kosmos.db.models import ResearchRun

            # Get specific run or latest
            if run_id:
                run = session.query(ResearchRun).filter_by(id=run_id).first()
            else:
                run = session.query(ResearchRun).order_by(ResearchRun.created_at.desc()).first()

            if not run:
                return None

            # Convert to dict (simplified - in real implementation would include relations)
            return {
                "id": run.id,
                "question": run.research_question,
                "domain": run.domain or "general",
                "state": run.state,
                "current_iteration": run.current_iteration,
                "max_iterations": run.max_iterations,
                "created_at": run.created_at,
                "updated_at": run.updated_at,
                "hypotheses": [],  # Would query related hypotheses
                "experiments": [],  # Would query related experiments
                "metrics": {
                    "api_calls": run.total_api_calls or 0,
                    "cache_hits": run.cache_hits or 0,
                    "cache_misses": run.cache_misses or 0,
                    "total_cost_usd": run.total_cost or 0.0,
                    "hypotheses_generated": 0,  # Would count
                    "experiments_executed": 0,  # Would count
                },
            }

    except Exception as e:
        console.print(f"[error]Database error: {str(e)}[/error]")
        return None


def display_status_once(research_data: dict, show_details: bool = False):
    """Display status once."""
    viewer = ResultsViewer()

    # Show overview
    viewer.display_research_overview(research_data)

    # Show progress
    display_progress_bar(research_data)

    # Show workflow state
    display_workflow_state(research_data)

    # Show metrics
    if "metrics" in research_data:
        viewer.display_metrics_summary(research_data["metrics"])

    # Show details if requested
    if show_details:
        viewer.display_hypotheses_table(research_data.get("hypotheses", []))
        viewer.display_experiments_table(research_data.get("experiments", []))


def display_status_live(research_data: dict, show_details: bool = False):
    """Display status with live updates."""
    console.print("[info]Live status mode - refreshing every 5 seconds (Ctrl+C to exit)[/info]")
    console.print()

    try:
        while True:
            # Clear and redisplay
            console.clear()
            display_status_once(research_data, show_details)

            # Wait before refresh
            time.sleep(5)

            # Reload data
            research_data = get_research_data(research_data["id"])
            if not research_data:
                print_error("Research run no longer exists")
                break

            # Check if completed
            if research_data["state"] in ["COMPLETED", "FAILED", "STOPPED"]:
                console.print(f"\n[success]Research {research_data['state']}[/success]")
                break

    except KeyboardInterrupt:
        console.print("\n[warning]Live status stopped[/warning]")


def display_progress_bar(research_data: dict):
    """Display progress bar for research."""
    iteration = research_data.get("current_iteration", 0)
    max_iterations = research_data.get("max_iterations", 10)
    progress_pct = (iteration / max_iterations * 100) if max_iterations > 0 else 0

    # Create progress bar
    progress = Progress(BarColumn(), console=console, expand=True)
    task = progress.add_task("Research Progress", total=max_iterations, completed=iteration)

    console.print(
        Panel(
            progress,
            title=f"[cyan]Progress: {iteration}/{max_iterations} ({progress_pct:.1f}%)[/cyan]",
            border_style="cyan",
        )
    )
    console.print()


def display_workflow_state(research_data: dict):
    """Display workflow state information."""
    state = research_data.get("state", "UNKNOWN")
    created_at = research_data.get("created_at")
    updated_at = research_data.get("updated_at")

    # Calculate duration
    if created_at and updated_at:
        duration = (updated_at - created_at).total_seconds()
    else:
        duration = 0

    # State table
    table = create_table(
        title=f"{get_icon('info')} Workflow Information",
        columns=["Property", "Value"],
        show_lines=True,
    )

    table.add_row("Current State", create_status_text(state))
    table.add_row("Started", format_timestamp(created_at, relative=False) if created_at else "[muted]Unknown[/muted]")
    table.add_row("Last Updated", format_timestamp(updated_at) if updated_at else "[muted]Unknown[/muted]")
    table.add_row("Duration", format_duration(duration))

    console.print(table)
    console.print()


if __name__ == "__main__":
    typer.run(show_status)
