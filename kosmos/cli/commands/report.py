"""
`kosmos report`: render a finished run as Markdown.

Reads the run's result rows from the database (get_results_for_run) and writes
the validated and rejected findings with their provenance, the plan section 7
metrics and the failed experiments with their error messages. The structure
follows the library loop's generate_report. No LLM is called.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.markup import escape

from kosmos.cli.commands.run_results import _number, _result_row, result_metrics
from kosmos.cli.utils import console, print_error
from kosmos.db import get_session
from kosmos.db.operations import get_research_session, get_results_for_run

logger = logging.getLogger(__name__)


def _fmt(value: Any) -> str:
    """A table cell: n/a for None, four significant digits for floats."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _cost(value: Optional[float]) -> str:
    return "unknown" if value is None else f"${value:.4f}"


def _validation_reason(status: Optional[str], detail: Dict[str, Any]) -> str:
    """One line on why the gate decided what it did, built from validation_detail."""
    parts: List[str] = []
    if detail.get("reason"):
        parts.append(str(detail["reason"]))
    if detail.get("recompute_error"):
        parts.append(f"recomputation failed: {detail['recompute_error']}")
    elif "recomputed_match" in detail:
        parts.append(
            "recomputed statistic matches" if detail["recomputed_match"]
            else "recomputed statistic does not match"
        )
    null = detail.get("null_model") or {}
    if null:
        verdict = "passes" if null.get("passes_null_test") else "fails"
        line = f"permutation p = {_fmt(null.get('permutation_p_value'))} ({verdict} the null test"
        if null.get("persists_in_noise"):
            line += ", persists in noise"
        parts.append(line + ")")
    scholar = detail.get("scholar_eval") or {}
    if scholar:
        verdict = "passes" if scholar.get("passes_threshold") else "below threshold"
        parts.append(f"ScholarEval {_fmt(scholar.get('overall_score'))} ({verdict})")
    return "; ".join(parts) if parts else (status or "no validation recorded")


def _finding_section(index: int, r, row: Dict[str, Any]) -> str:
    detail = r.validation_detail or {}
    prov = r.provenance or {}
    out = f"### Finding {index}: `{r.id}`\n\n"
    if r.interpretation:
        out += f"{r.interpretation}\n\n"
    out += f"- Hypothesis: {_fmt(row['hypothesis_id'])}\n"
    out += f"- Test: {_fmt(row['test_type'])}\n"
    out += f"- Statistic: {_fmt(row['statistic'])}\n"
    out += f"- p-value: {_fmt(row['p_value'])}\n"
    out += f"- Effect size: {_fmt(row['effect_size'])}\n"
    out += f"- n: {_fmt(row['n'])}\n"
    out += f"- Data source: {_fmt(row['data_source'])}\n"
    out += f"- Supports hypothesis: {_fmt(row['supports_hypothesis'])}\n"
    out += f"- Validation: {r.validation_status} ({_validation_reason(r.validation_status, detail)})\n\n"
    out += "**Provenance**:\n"
    out += f"- git_sha: {_fmt(prov.get('git_sha'))}\n"
    out += f"- data_sha256: {_fmt(prov.get('data_sha256'))}\n"
    out += f"- seed: {_fmt(prov.get('seed', r.random_seed))}\n"
    out += f"- code: {_fmt(prov.get('code_path'))}\n\n"
    return out


def render_report(run_id: str, results: List[Any], question: Optional[str] = None) -> str:
    """Markdown report of a run from its result rows (oldest first)."""
    rows = [_result_row(r) for r in results]
    known_costs = [c for c in (_number(r.cost_usd) for r in results) if c is not None]
    total_cost = sum(known_costs) if known_costs else None
    metrics = result_metrics(rows, total_cost)

    validated = [(r, row) for r, row in zip(results, rows, strict=True) if r.validation_status == "validated"]
    rejected = [(r, row) for r, row in zip(results, rows, strict=True) if r.validation_status == "rejected"]
    failed = [r for r in results if r.execution_success is not True]
    unvalidated = [
        (r, row) for r, row in zip(results, rows, strict=True)
        if r.execution_success is True and r.validation_status not in ("validated", "rejected")
    ]

    report = "# Research Report\n\n"
    report += f"- **Run**: {run_id}\n"
    report += f"- **Question**: {question or 'unknown'}\n"
    report += f"- **Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n\n"

    report += "## Summary\n\n"
    report += (
        f"This run attempted {metrics['experiments_attempted']} experiments, "
        f"{metrics['experiments_succeeded']} of which executed, producing "
        f"{metrics['findings_validated']} validated and {len(rejected)} rejected findings.\n\n"
    )

    report += "## Metrics\n\n"
    report += "| Metric | Value |\n|---|---|\n"
    for label, value in (
        ("Experiments attempted", metrics["experiments_attempted"]),
        ("Experiments succeeded", metrics["experiments_succeeded"]),
        ("Experiments failed", metrics["experiments_failed"]),
        ("Results from file", metrics["results_from_file"]),
        ("Results synthetic", metrics["results_synthetic"]),
        ("Findings validated", metrics["findings_validated"]),
        ("Findings rejected", metrics["findings_rejected"]),
        ("LLM cost attributed to results", _cost(total_cost)),
        ("Cost per validated finding", _cost(metrics["cost_per_validated_finding"])),
    ):
        report += f"| {label} | {value} |\n"
    report += "\n"

    report += "## Validated Findings\n\n"
    if not validated:
        report += "None.\n\n"
    for i, (r, row) in enumerate(validated, 1):
        report += _finding_section(i, r, row)

    report += "## Rejected Findings\n\n"
    if not rejected:
        report += "None.\n\n"
    for i, (r, row) in enumerate(rejected, 1):
        report += _finding_section(i, r, row)

    if unvalidated:
        report += "## Unvalidated Results\n\n"
        for r, row in unvalidated:
            reason = (r.validation_detail or {}).get("reason") or r.validation_status or "not analyzed"
            report += (
                f"- `{r.id}`: {reason} (data source {_fmt(row['data_source'])}, "
                f"p-value {_fmt(row['p_value'])})\n"
            )
        report += "\n"

    report += "## Failed Experiments\n\n"
    if not failed:
        report += "None.\n"
    for r in failed:
        status = f", {r.validation_status}" if r.validation_status == "rejected_unsafe" else ""
        report += (
            f"- `{r.id}` (experiment {r.experiment_id}{status}): "
            f"{r.error_message or 'no error message recorded'}\n"
        )
    return report


def _default_artifacts_dir() -> Path:
    """The director's layout (D-08): config research.artifacts_dir, else ./artifacts/runs."""
    try:
        from kosmos.config import get_config
        configured = get_config().research.artifacts_dir
    except Exception as e:
        logger.debug("Config not available for the artifacts dir: %s", e)
        configured = None
    return Path(configured or Path.cwd() / "artifacts" / "runs")


def generate_report(
    run_id: str = typer.Option(..., "--run-id", help="Run id printed by `kosmos run`"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Report path (default: <artifacts_dir>/<run_id>/report.md)",
    ),
):
    """
    Write a Markdown report of a finished research run.

    Examples:

        kosmos report --run-id run_0123456789ab

        kosmos report --run-id run_0123456789ab --output report.md
    """
    try:
        with get_session() as session:
            results = get_results_for_run(session, run_id)
            research = get_research_session(session, run_id)
            question = research.research_question if research else None
            markdown = render_report(run_id, results, question) if results else None
    except Exception as e:
        print_error(f"Could not read run {run_id} from the database: {e}")
        raise typer.Exit(1) from None

    if markdown is None:
        print_error(
            f"No results recorded for run id '{run_id}'. "
            "Check the id printed at the end of `kosmos run`, or list runs with `kosmos history`."
        )
        raise typer.Exit(1)

    path = output or _default_artifacts_dir() / run_id / "report.md"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    except OSError as e:
        print_error(f"Could not write the report to {path}: {e}")
        raise typer.Exit(1) from None

    console.print(f"[success]Report written:[/success] {escape(str(path))}", soft_wrap=True)
