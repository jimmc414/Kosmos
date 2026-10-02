"""
Results viewer for Kosmos CLI.

Provides beautiful visualization of research results, hypotheses, experiments,
and analysis using Rich library components.
"""

import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.text import Text
from rich.columns import Columns

from kosmos.cli.utils import (
    console,
    create_table,
    format_timestamp,
    format_duration,
    format_currency,
    truncate_text,
    create_status_text,
    create_domain_text,
    create_metric_text,
    get_icon,
)
from kosmos.cli.themes import get_domain_color, get_state_color, get_box_style


class ResultsViewer:
    """Viewer for displaying research results in various formats."""

    def __init__(self, console_instance: Optional[Console] = None):
        """
        Initialize results viewer.

        Args:
            console_instance: Optional Rich Console instance
        """
        self.console = console_instance or console

    def display_research_overview(self, research_data: Dict[str, Any]):
        """
        Display overview of a research run.

        Args:
            research_data: Research run data with metadata
        """
        run_id = research_data.get("id", "Unknown")
        question = research_data.get("question", "Unknown")
        domain = research_data.get("domain") or "general"  # key is present with None when no --domain
        state = research_data.get("state", "Unknown")
        iteration = research_data.get("current_iteration", 0)
        max_iterations = research_data.get("max_iterations", 10)

        # Create overview panel
        overview_text = [
            f"**Run ID:** {run_id}",
            f"**Domain:** {domain.title()}",
            f"**State:** {state}",
            f"**Progress:** Iteration {iteration}/{max_iterations} ({iteration/max_iterations*100:.1f}%)",
            "",
            f"**Question:** {question}",
        ]

        self.console.print()
        self.console.print(
            Panel(
                "\n".join(overview_text),
                title=f"[h2]{get_icon('flask')} Research Overview[/h2]",
                border_style=get_domain_color(domain),
                box=get_box_style("default"),
            )
        )
        self.console.print()

    def display_hypotheses_table(self, hypotheses: List[Dict[str, Any]]):
        """
        Display table of hypotheses.

        Args:
            hypotheses: List of hypothesis dictionaries
        """
        if not hypotheses:
            self.console.print("[muted]No hypotheses yet.[/muted]")
            return

        table = create_table(
            title=f"{get_icon('magnifying_glass')} Hypotheses",
            columns=["#", "Claim", "Novelty", "Priority", "Status"],
            show_lines=False,
        )

        for i, hyp in enumerate(hypotheses, 1):
            claim = truncate_text(hyp.get("claim", "Unknown"), 50)
            novelty = hyp.get("novelty_score", 0.0)
            priority = hyp.get("priority_score", 0.0)
            status = hyp.get("status", "pending")

            table.add_row(
                str(i),
                claim,
                create_metric_text(novelty, format_type="number"),
                create_metric_text(priority, format_type="number"),
                create_status_text(status),
            )

        self.console.print(table)
        self.console.print()

    def display_hypothesis_tree(self, hypotheses: List[Dict[str, Any]]):
        """
        Display hypothesis evolution as a tree.

        Args:
            hypotheses: List of hypothesis dictionaries with parent relationships
        """
        if not hypotheses:
            self.console.print("[muted]No hypothesis tree available.[/muted]")
            return

        # Build tree structure
        tree = Tree(
            f"[h2]{get_icon('brain')} Hypothesis Evolution[/h2]",
            guide_style="bright_black"
        )

        # Group by parent
        root_hypotheses = [h for h in hypotheses if not h.get("parent_id")]
        children_map = {}

        for hyp in hypotheses:
            parent_id = hyp.get("parent_id")
            if parent_id:
                if parent_id not in children_map:
                    children_map[parent_id] = []
                children_map[parent_id].append(hyp)

        def add_hypothesis_node(parent_node, hypothesis):
            """Recursively add hypothesis nodes."""
            claim = truncate_text(hypothesis.get("claim", "Unknown"), 60)
            novelty = hypothesis.get("novelty_score", 0.0)
            status = hypothesis.get("status", "pending")

            node_label = (
                f"{claim}\n"
                f"[muted]Novelty: {novelty:.2f} | Status: {status}[/muted]"
            )

            node = parent_node.add(node_label)

            # Add children
            hyp_id = hypothesis.get("id")
            if hyp_id in children_map:
                for child in children_map[hyp_id]:
                    add_hypothesis_node(node, child)

        # Add root hypotheses
        for hyp in root_hypotheses:
            add_hypothesis_node(tree, hyp)

        self.console.print(tree)
        self.console.print()

    def display_experiments_table(self, experiments: List[Dict[str, Any]]):
        """
        Display table of experiments.

        Args:
            experiments: List of experiment dictionaries
        """
        if not experiments:
            self.console.print("[muted]No experiments yet.[/muted]")
            return

        table = create_table(
            title=f"{get_icon('flask')} Experiments",
            columns=["#", "Type", "Status", "Duration", "Timestamp"],
            show_lines=False,
        )

        for i, exp in enumerate(experiments, 1):
            exp_type = exp.get("type", "Unknown")
            status = exp.get("status", "pending")
            duration = exp.get("duration_seconds", 0)
            timestamp = exp.get("created_at")

            # Parse timestamp if string
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    timestamp = None

            table.add_row(
                str(i),
                exp_type,
                create_status_text(status),
                format_duration(duration),
                format_timestamp(timestamp) if timestamp else "[muted]Unknown[/muted]",
            )

        self.console.print(table)
        self.console.print()

    def display_experiment_details(self, experiment: Dict[str, Any]):
        """
        Display detailed view of a single experiment.

        Args:
            experiment: Experiment dictionary with full details
        """
        exp_id = experiment.get("id", "Unknown")
        exp_type = experiment.get("type", "Unknown")
        status = experiment.get("status", "Unknown")

        # Header panel
        header = [
            f"**Experiment ID:** {exp_id}",
            f"**Type:** {exp_type}",
            f"**Status:** {status}",
        ]

        self.console.print()
        self.console.print(
            Panel(
                "\n".join(header),
                title=f"[cyan]{get_icon('flask')} Experiment Details[/cyan]",
                border_style="cyan",
            )
        )

        # Parameters
        if "parameters" in experiment:
            self.console.print("\n[h3]Parameters:[/h3]")
            params_json = json.dumps(experiment["parameters"], indent=2)
            syntax = Syntax(params_json, "json", theme="monokai", line_numbers=False)
            self.console.print(syntax)

        # Results
        if "results" in experiment:
            self.console.print("\n[h3]Results:[/h3]")
            results_json = json.dumps(experiment["results"], indent=2)
            syntax = Syntax(results_json, "json", theme="monokai", line_numbers=False)
            self.console.print(syntax)

        # Code (if available)
        if "code" in experiment:
            self.console.print("\n[h3]Generated Code:[/h3]")
            syntax = Syntax(
                experiment["code"],
                "python",
                theme="monokai",
                line_numbers=True,
            )
            self.console.print(syntax)

        self.console.print()

    def display_metrics_summary(self, metrics: Dict[str, Any]):
        """
        Display research metrics summary.

        Args:
            metrics: Metrics dictionary
        """
        # API metrics
        api_table = create_table(
            title=f"{get_icon('info')} API Usage",
            columns=["Metric", "Value"],
            show_lines=True,
        )

        api_calls = metrics.get("api_calls", 0)
        cache_hits = metrics.get("cache_hits", 0)
        cache_misses = metrics.get("cache_misses", 0)
        total_cache = cache_hits + cache_misses
        hit_rate = (cache_hits / total_cache * 100) if total_cache > 0 else 0

        api_table.add_row("Total API Calls", str(api_calls))
        api_table.add_row("Cache Hits", f"{cache_hits} ({hit_rate:.1f}%)")
        api_table.add_row("Cache Misses", str(cache_misses))

        if "total_cost_usd" in metrics:
            api_table.add_row("Total Cost", format_currency(metrics["total_cost_usd"]))

        self.console.print(api_table)
        self.console.print()

        # Research metrics
        research_table = create_table(
            title=f"{get_icon('brain')} Research Progress",
            columns=["Metric", "Value"],
            show_lines=True,
        )

        research_table.add_row("Hypotheses Generated", str(metrics.get("hypotheses_generated", 0)))
        research_table.add_row("Experiments Executed", str(metrics.get("experiments_executed", 0)))
        research_table.add_row("Successful Experiments", str(metrics.get("successful_experiments", 0)))
        research_table.add_row("Failed Experiments", str(metrics.get("failed_experiments", 0)))

        self.console.print(research_table)
        self.console.print()

    def export_to_json(self, data: Dict[str, Any], output_path: Path):
        """
        Export results to JSON file.

        Args:
            data: Data to export
            output_path: Output file path
        """
        try:
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            self.console.print(f"[success]Exported to {output_path}[/success]")
        except Exception as e:
            self.console.print(f"[error]Export failed: {str(e)}[/error]")

    def export_to_markdown(self, data: Dict[str, Any], output_path: Path):
        """
        Export results to Markdown file.

        Args:
            data: Data to export
            output_path: Output file path
        """
        try:
            lines = [
                f"# Research Results: {data.get('question', 'Unknown')}",
                "",
                f"**Run ID:** {data.get('id', 'Unknown')}",
                f"**Domain:** {data.get('domain', 'Unknown')}",
                f"**Status:** {data.get('state', 'Unknown')}",
                "",
                "## Hypotheses",
                "",
            ]

            for i, hyp in enumerate(data.get("hypotheses", []), 1):
                lines.extend([
                    f"### {i}. {hyp.get('claim', 'Unknown')}",
                    f"",
                    f"- **Novelty:** {hyp.get('novelty_score', 0):.2f}",
                    f"- **Priority:** {hyp.get('priority_score', 0):.2f}",
                    f"- **Status:** {hyp.get('status', 'Unknown')}",
                    "",
                ])

            lines.extend([
                "## Experiments",
                "",
            ])

            for i, exp in enumerate(data.get("experiments", []), 1):
                lines.extend([
                    f"### {i}. {exp.get('type', 'Unknown')}",
                    f"",
                    f"- **Status:** {exp.get('status', 'Unknown')}",
                    f"- **Duration:** {format_duration(exp.get('duration_seconds', 0))}",
                    "",
                ])

            with open(output_path, "w") as f:
                f.write("\n".join(lines))

            self.console.print(f"[success]Exported to {output_path}[/success]")
        except Exception as e:
            self.console.print(f"[error]Export failed: {str(e)}[/error]")


# Convenience functions
def view_research_results(research_data: Dict[str, Any]):
    """Display complete research results."""
    viewer = ResultsViewer()

    viewer.display_research_overview(research_data)
    viewer.display_hypotheses_table(research_data.get("hypotheses", []))
    viewer.display_experiments_table(research_data.get("experiments", []))

    if "metrics" in research_data:
        viewer.display_metrics_summary(research_data["metrics"])


def view_hypothesis_evolution(hypotheses: List[Dict[str, Any]]):
    """Display hypothesis evolution tree."""
    viewer = ResultsViewer()
    viewer.display_hypothesis_tree(hypotheses)


def view_experiment_details(experiment: Dict[str, Any]):
    """Display detailed experiment view."""
    viewer = ResultsViewer()
    viewer.display_experiment_details(experiment)
