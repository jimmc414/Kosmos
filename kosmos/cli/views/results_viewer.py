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
from rich.markup import escape
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
                _score_text(novelty),
                _score_text(priority),
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

    def display_results_table(self, results: List[Dict[str, Any]]):
        """
        Display one row per experiment result: execution, data source, test and verdict.

        Args:
            results: Result dictionaries from build_run_results
        """
        if not results:
            self.console.print("[muted]No results yet.[/muted]")
            return

        table = create_table(
            title=f"{get_icon('success')} Results",
            columns=["Result", "Hyp", "Exec", "Data", "Test", "Stat", "p", "Verdict", "Validation"],
            show_lines=False,
        )

        for res in results:
            failed = res.get("execution_success") is not True
            validation = res.get("validation_status") or "-"
            if res.get("validation_reason"):
                validation = f"{validation} ({res['validation_reason']})"
            if failed and res.get("error_message"):
                validation = f"{validation}: {_one_line(res['error_message'])[:60]}"

            row = [
                (res.get("result_id") or "")[:8],
                (res.get("hypothesis_id") or "-")[:8],
                "FAIL" if failed else "OK",
                res.get("data_source") or "none",
                res.get("test_type") or "-",
                _num(res.get("statistic"), "{:.4g}"),
                _num(res.get("p_value"), "{:.3g}"),
                _verdict(res.get("supports_hypothesis")),
                validation,
            ]
            table.add_row(*(escape(str(cell)) for cell in row), style="red" if failed else None)

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

        api_table.add_row("Total API Calls", str(metrics.get("api_calls", 0)))
        if "cache_hits" in metrics or "cache_misses" in metrics:
            cache_hits = metrics.get("cache_hits", 0)
            cache_misses = metrics.get("cache_misses", 0)
            total_cache = cache_hits + cache_misses
            hit_rate = (cache_hits / total_cache * 100) if total_cache > 0 else 0
            api_table.add_row("Cache Hits", f"{cache_hits} ({hit_rate:.1f}%)")
            api_table.add_row("Cache Misses", str(cache_misses))

        api_table.add_row("Total Cost", _cost(metrics.get("total_cost_usd")))
        api_table.add_row(
            "Tokens (in / out)",
            f"{metrics.get('input_tokens', 0)} / {metrics.get('output_tokens', 0)}",
        )
        if "cost_per_validated_finding" in metrics:
            api_table.add_row(
                "Cost per Validated Finding", _cost(metrics.get("cost_per_validated_finding"))
            )

        self.console.print(api_table)
        self.console.print()

        # Research metrics
        research_table = create_table(
            title=f"{get_icon('brain')} Research Progress",
            columns=["Metric", "Value"],
            show_lines=True,
        )

        research_table.add_row("Hypotheses Generated", str(metrics.get("hypotheses_generated", 0)))
        research_table.add_row("Hypotheses Untestable", str(metrics.get("hypotheses_untestable", 0)))
        research_table.add_row("Experiments Attempted", str(metrics.get("experiments_attempted", 0)))
        research_table.add_row("Experiments Succeeded", str(metrics.get("experiments_succeeded", 0)))
        research_table.add_row("Experiments Failed", str(metrics.get("experiments_failed", 0)))
        research_table.add_row(
            "Results (file / synthetic)",
            f"{metrics.get('results_from_file', 0)} / {metrics.get('results_synthetic', 0)}",
        )
        research_table.add_row("Findings Validated", str(metrics.get("findings_validated", 0)))
        research_table.add_row("Findings Rejected", str(metrics.get("findings_rejected", 0)))

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
                    f"- **Novelty:** {_num(hyp.get('novelty_score'), '{:.2f}')}",
                    f"- **Priority:** {_num(hyp.get('priority_score'), '{:.2f}')}",
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

            lines.extend([
                "## Results",
                "",
                "| Result | Hyp | Exec | Data | Test | Stat | p | Verdict | Validation | Cost |",
                "|---|---|---|---|---|---|---|---|---|---|",
            ])
            for res in data.get("results", []):
                failed = res.get("execution_success") is not True
                validation = res.get("validation_status") or "-"
                if res.get("validation_reason"):
                    validation = f"{validation} ({res['validation_reason']})"
                if failed and res.get("error_message"):
                    validation = f"{validation}: {_one_line(res['error_message'])[:60]}"
                lines.append(
                    f"| {(res.get('result_id') or '')[:8]} | {(res.get('hypothesis_id') or '-')[:8]} "
                    f"| {'FAIL' if failed else 'OK'} | {res.get('data_source') or 'none'} "
                    f"| {res.get('test_type') or '-'} | {_num(res.get('statistic'), '{:.4g}')} "
                    f"| {_num(res.get('p_value'), '{:.3g}')} | {_verdict(res.get('supports_hypothesis'))} "
                    f"| {validation.replace('|', '/')} | {_cost(res.get('cost_usd'))} |"
                )

            metrics = data.get("metrics", {})
            if metrics:
                lines.extend([
                    "",
                    "## Metrics",
                    "",
                    f"- **API calls:** {metrics.get('api_calls', 0)}",
                    f"- **Total cost:** {_cost(metrics.get('total_cost_usd'))}",
                    f"- **Tokens (in / out):** {metrics.get('input_tokens', 0)} / {metrics.get('output_tokens', 0)}",
                    f"- **Experiments succeeded / failed:** "
                    f"{metrics.get('experiments_succeeded', 0)} / {metrics.get('experiments_failed', 0)}",
                    f"- **Findings validated / rejected:** "
                    f"{metrics.get('findings_validated', 0)} / {metrics.get('findings_rejected', 0)}",
                    f"- **Cost per validated finding:** {_cost(metrics.get('cost_per_validated_finding'))}",
                    f"- **Hypotheses untestable:** {metrics.get('hypotheses_untestable', 0)}",
                ])

            with open(output_path, "w") as f:
                f.write("\n".join(lines))

            self.console.print(f"[success]Exported to {output_path}[/success]")
        except Exception as e:
            self.console.print(f"[error]Export failed: {str(e)}[/error]")


def _num(value: Any, fmt: str) -> str:
    """Format a number, or '-' when it is missing."""
    if value is None or isinstance(value, bool):
        return "-"
    try:
        return fmt.format(value)
    except (TypeError, ValueError):
        return str(value)


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _score_text(value: Any) -> Text:
    if value is None:
        return Text("-", style="muted")
    return create_metric_text(value, format_type="number")


def _cost(value: Any) -> str:
    """Cost in USD with four decimals (LLM runs often cost cents), or '-' when unknown."""
    return _num(value, "${:.4f}")


def _verdict(supports: Optional[bool]) -> str:
    if supports is True:
        return "supports"
    if supports is False:
        return "refutes"
    return "inconclusive"


# Convenience functions
def view_research_results(research_data: Dict[str, Any]):
    """Display complete research results."""
    viewer = ResultsViewer()

    viewer.display_research_overview(research_data)
    viewer.display_hypotheses_table(research_data.get("hypotheses", []))
    viewer.display_experiments_table(research_data.get("experiments", []))
    viewer.display_results_table(research_data.get("results", []))

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
