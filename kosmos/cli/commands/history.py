"""
History command for Kosmos CLI.

Browse past research runs and view detailed history.
"""

from typing import Optional
from datetime import datetime, timedelta, timezone

import typer
from rich.prompt import Prompt

from kosmos.cli.utils import (
    console,
    print_error,
    print_info,
    get_icon,
    create_table,
    create_status_text,
    create_domain_text,
    format_timestamp,
    get_db_session,
)
from kosmos.cli.views.results_viewer import ResultsViewer, view_research_results


def show_history(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of runs to show"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Filter by domain"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    days: Optional[int] = typer.Option(None, "--days", help="Show runs from last N days"),
    show_details: bool = typer.Option(False, "--details", help="Show detailed view"),
):
    """
    Browse research history.

    Examples:

        # Show last 10 runs
        kosmos history

        # Show last 20 runs
        kosmos history --limit 20

        # Filter by domain
        kosmos history --domain biology

        # Show runs from last 7 days
        kosmos history --days 7

        # Detailed view
        kosmos history --details
    """
    try:
        # Get research runs
        runs = get_research_runs(limit, domain, status, days)

        if not runs:
            print_info("No research runs found.")
            raise typer.Exit(0)

        # Display history
        if show_details:
            display_detailed_history(runs)
        else:
            display_history_table(runs)

        # Offer to view specific run
        if not show_details and len(runs) > 0:
            console.print()
            view_run = Prompt.ask(
                "[cyan]View details for a run?[/cyan] (Enter run # or 'n' to skip)",
                default="n"
            )

            if view_run.lower() != 'n':
                try:
                    run_idx = int(view_run) - 1
                    if 0 <= run_idx < len(runs):
                        console.print()
                        view_run_details(runs[run_idx])
                except ValueError:
                    print_error("Invalid run number")

    except KeyboardInterrupt:
        console.print("\n[warning]History display cancelled[/warning]")
        raise typer.Exit(130)

    except typer.Exit:
        # An intentional exit (with its own code) is not a failure to report
        raise

    except Exception as e:
        print_error(f"Failed to get history: {str(e)}")
        raise typer.Exit(1)


def get_research_runs(
    limit: int = 10,
    domain: Optional[str] = None,
    status: Optional[str] = None,
    days: Optional[int] = None,
) -> list:
    """
    Get research runs from database.

    Args:
        limit: Maximum number of runs to return
        domain: Optional domain filter
        status: Optional status filter
        days: Optional day range filter

    Returns:
        List of research run dictionaries
    """
    try:
        with get_db_session() as session:
            from kosmos.db.models import ResearchRun

            query = session.query(ResearchRun)

            # Apply filters
            if domain:
                query = query.filter(ResearchRun.domain == domain)

            if status:
                query = query.filter(ResearchRun.state == status)

            if days:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
                query = query.filter(ResearchRun.created_at >= cutoff_date)

            # Order by most recent and limit
            query = query.order_by(ResearchRun.created_at.desc()).limit(limit)

            # Convert to dictionaries
            runs = []
            for run in query.all():
                runs.append({
                    "id": run.id,
                    "question": run.research_question,
                    "domain": run.domain or "general",
                    "state": run.state,
                    "current_iteration": run.current_iteration,
                    "max_iterations": run.max_iterations,
                    "created_at": run.created_at,
                    "updated_at": run.updated_at,
                })

            return runs

    except Exception as e:
        console.print(f"[error]Database error: {str(e)}[/error]")
        return []


def display_history_table(runs: list):
    """Display history as a table."""
    console.print()
    console.print(f"[h2]{get_icon('book')} Research History[/h2]", justify="center")
    console.print()

    table = create_table(
        title=f"Showing {len(runs)} runs",
        columns=["#", "Run ID", "Question", "Domain", "Status", "Progress", "Created"],
        show_lines=False,
    )

    for i, run in enumerate(runs, 1):
        run_id = run["id"][:12] + "..." if len(run["id"]) > 15 else run["id"]
        question = run["question"][:40] + "..." if len(run["question"]) > 43 else run["question"]
        progress = f"{run['current_iteration']}/{run['max_iterations']}"

        table.add_row(
            str(i),
            run_id,
            question,
            create_domain_text(run["domain"]),
            create_status_text(run["state"]),
            progress,
            format_timestamp(run["created_at"]),
        )

    console.print(table)
    console.print()


def display_detailed_history(runs: list):
    """Display detailed history for each run."""
    console.print()
    console.print(f"[h2]{get_icon('book')} Detailed Research History[/h2]", justify="center")
    console.print()

    for i, run in enumerate(runs, 1):
        console.print(f"[h3]{i}. Run: {run['id']}[/h3]")
        console.print()

        # Create detail table
        table = create_table(
            title="",
            columns=["Property", "Value"],
            show_lines=True,
        )

        table.add_row("Question", run["question"])
        table.add_row("Domain", create_domain_text(run["domain"]))
        table.add_row("Status", create_status_text(run["state"]))
        table.add_row("Progress", f"{run['current_iteration']}/{run['max_iterations']}")
        table.add_row("Created", format_timestamp(run["created_at"], relative=False))
        table.add_row("Updated", format_timestamp(run["updated_at"], relative=False))

        console.print(table)
        console.print()


def view_run_details(run: dict):
    """View detailed information for a specific run."""
    # Get full data including hypotheses and experiments
    with get_db_session() as session:
        from kosmos.db.models import ResearchRun

        full_run = session.query(ResearchRun).filter_by(id=run["id"]).first()

        if not full_run:
            print_error("Run not found")
            return

        # Build full research data (would include relations in real implementation)
        research_data = {
            "id": full_run.id,
            "question": full_run.research_question,
            "domain": full_run.domain or "general",
            "state": full_run.state,
            "current_iteration": full_run.current_iteration,
            "max_iterations": full_run.max_iterations,
            "created_at": full_run.created_at,
            "updated_at": full_run.updated_at,
            "hypotheses": [],  # Would load from DB
            "experiments": [],  # Would load from DB
            "metrics": {
                "api_calls": full_run.total_api_calls or 0,
                "cache_hits": full_run.cache_hits or 0,
                "cache_misses": full_run.cache_misses or 0,
                "total_cost_usd": full_run.total_cost or 0.0,
            },
        }

    # Display using results viewer
    view_research_results(research_data)


if __name__ == "__main__":
    typer.run(show_history)
