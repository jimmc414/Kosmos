"""Tests for the end-of-run results display."""

import io

from rich.console import Console

from kosmos.cli.views.results_viewer import ResultsViewer


def test_overview_accepts_missing_domain():
    """`kosmos run` without --domain reports domain None; the overview must still render."""
    out = io.StringIO()
    viewer = ResultsViewer(console_instance=Console(file=out, width=120))

    viewer.display_research_overview({
        "id": "research_1", "question": "Q", "domain": None,
        "state": "converged", "current_iteration": 1, "max_iterations": 1,
    })

    assert "General" in out.getvalue()
