"""
Integration tests for Kosmos CLI.

Tests all CLI commands with various options and scenarios.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner
from pathlib import Path
import tempfile
import json

from kosmos.cli.main import app
from kosmos.cli.interactive import run_interactive_mode
from kosmos.cli.views.results_viewer import ResultsViewer


@pytest.fixture
def cli_runner():
    """Create CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_config():
    """Mock configuration."""
    config = Mock()
    config.claude.model = "claude-3-5-sonnet-20241022"
    config.claude.is_cli_mode = False
    config.claude.max_tokens = 4096
    config.claude.temperature = 1.0
    config.claude.enable_cache = True
    config.research.max_iterations = 10
    config.research.enabled_domains = ["biology", "neuroscience"]
    config.research.experiment_types = ["computational", "data_analysis"]
    config.research.enabled_experiment_types = ["computational", "data_analysis"]
    config.research.budget_usd = None
    config.database.url = "sqlite:///test.db"
    return config


@pytest.fixture
def no_db_init():
    """Skip the main callback's database setup when get_config is a Mock."""
    with patch("kosmos.db.init_from_config") as init:
        yield init


@pytest.fixture
def mock_cache_manager():
    """Mock cache manager."""
    manager = Mock()
    manager.get_stats.return_value = {
        "claude": {"hits": 100, "misses": 50, "size": 150, "storage_size_mb": 10},
        "experiment": {"hits": 50, "misses": 25, "size": 75, "storage_size_mb": 5},
        "embedding": {"hits": 200, "misses": 100, "size": 300, "storage_size_mb": 15},
        "general": {"hits": 75, "misses": 25, "size": 100, "storage_size_mb": 3},
    }
    manager.health_check.return_value = {
        "claude": {"healthy": True, "details": "OK"},
        "experiment": {"healthy": True, "details": "OK"},
        "embedding": {"healthy": True, "details": "OK"},
        "general": {"healthy": True, "details": "OK"},
    }
    manager.cleanup_expired.return_value = {
        "claude": 5,
        "experiment": 3,
        "embedding": 10,
        "general": 2,
    }
    return manager


class TestCLIBasicCommands:
    """Test basic CLI commands."""

    def test_cli_help(self, cli_runner):
        """Test CLI help command."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Kosmos AI Scientist" in result.stdout
        assert "Commands" in result.stdout

    def test_version_command(self, cli_runner):
        """Test version command."""
        result = cli_runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "Kosmos AI Scientist" in result.stdout
        assert "v0.2.0" in result.stdout

    @patch("kosmos.config.get_config")
    def test_info_command(self, mock_get_config, cli_runner, mock_config, no_db_init):
        """Test info command."""
        mock_get_config.return_value = mock_config

        result = cli_runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "System Information" in result.stdout
        assert "Configuration" in result.stdout

    def test_doctor_command(self, cli_runner, tmp_path):
        """Doctor passes on the hermetic test environment (litellm + temp database)."""
        with patch("kosmos.cli.utils.get_cache_dir", return_value=tmp_path), \
             patch("kosmos.core.providers.claude_code.ClaudeCodeProvider.cli_version",
                   return_value="2.0.0 (Claude Code)"):
            result = cli_runner.invoke(app, ["doctor"])
        assert result.exit_code == 0, result.output
        assert "Running Diagnostics" in result.stdout
        assert "Diagnostic Results" in result.stdout
        assert "All checks passed" in result.stdout


class TestConfigCommand:
    """Test config command."""

    @patch("kosmos.config.get_config")
    def test_config_show(self, mock_get_config, cli_runner, mock_config, no_db_init):
        """Test config show."""
        mock_get_config.return_value = mock_config

        result = cli_runner.invoke(app, ["config", "--show"])
        assert result.exit_code == 0
        assert "Current Configuration" in result.stdout
        assert "Claude API Configuration" in result.stdout
        assert "claude-3-5-sonnet-20241022" in result.stdout
        assert "computational, data_analysis" in result.stdout

    def test_config_path(self, cli_runner):
        """Test config path."""
        result = cli_runner.invoke(app, ["config", "--path"])
        assert result.exit_code == 0
        assert "Configuration File Locations" in result.stdout

    @patch("kosmos.config.get_config")
    def test_config_validate(self, mock_get_config, cli_runner, mock_config, no_db_init):
        """Test config validate."""
        mock_get_config.return_value = mock_config

        result = cli_runner.invoke(app, ["config", "--validate"])
        # May fail due to database not existing, but should run
        assert "Validating Configuration" in result.stdout


class TestCacheCommand:
    """Test cache command."""

    @patch("kosmos.core.cache_manager.get_cache_manager")
    def test_cache_stats(self, mock_get_manager, cli_runner, mock_cache_manager):
        """Test cache stats."""
        mock_get_manager.return_value = mock_cache_manager

        result = cli_runner.invoke(app, ["cache", "--stats"])
        assert result.exit_code == 0
        assert "Cache Statistics" in result.stdout
        assert "Overall Cache Performance" in result.stdout

    @patch("kosmos.core.cache_manager.get_cache_manager")
    def test_cache_health(self, mock_get_manager, cli_runner, mock_cache_manager):
        """Test cache health check."""
        mock_get_manager.return_value = mock_cache_manager

        result = cli_runner.invoke(app, ["cache", "--health"])
        assert result.exit_code == 0
        assert "Running Health Check" in result.stdout
        assert "Health Check Results" in result.stdout

    @patch("kosmos.core.cache_manager.get_cache_manager")
    def test_cache_optimize(self, mock_get_manager, cli_runner, mock_cache_manager):
        """Test cache optimize."""
        mock_get_manager.return_value = mock_cache_manager

        result = cli_runner.invoke(app, ["cache", "--optimize"])
        assert result.exit_code == 0
        assert "Optimizing Caches" in result.stdout
        assert "Optimization Results" in result.stdout


class TestRunCommand:
    """Test run command."""

    def test_run_help(self, cli_runner):
        """Test run command help."""
        result = cli_runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run autonomous research" in result.stdout

    def test_run_without_question_shows_error(self, cli_runner):
        """Test run without question."""
        # Provide 'n' to the interactive prompt to cancel
        result = cli_runner.invoke(app, ["run"], input="n\n")
        # Should either exit with 0 (cancelled) or 1 (error)
        assert result.exit_code in [0, 1]


class TestStatusCommand:
    """Test status command."""

    def test_status_help(self, cli_runner):
        """Test status command help."""
        result = cli_runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0
        assert "Show status of a research run" in result.stdout

    @patch("kosmos.cli.commands.status.get_research_data")
    def test_status_not_found(self, mock_get_data, cli_runner):
        """Test status for non-existent run."""
        mock_get_data.return_value = None

        result = cli_runner.invoke(app, ["status", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.stdout


class TestHistoryCommand:
    """Test history command."""

    def test_history_help(self, cli_runner):
        """Test history command help."""
        result = cli_runner.invoke(app, ["history", "--help"])
        assert result.exit_code == 0
        assert "Browse research history" in result.stdout

    @patch("kosmos.cli.commands.history.get_research_runs")
    def test_history_empty(self, mock_get_runs, cli_runner):
        """Test history with no runs."""
        mock_get_runs.return_value = []

        result = cli_runner.invoke(app, ["history"])
        assert result.exit_code == 0
        assert "No research runs found" in result.stdout

    @patch("kosmos.cli.commands.history.get_research_runs")
    def test_history_with_runs(self, mock_get_runs, cli_runner):
        """Test history with runs."""
        from datetime import datetime

        mock_get_runs.return_value = [
            {
                "id": "run_123",
                "question": "Test question?",
                "domain": "biology",
                "state": "COMPLETED",
                "current_iteration": 5,
                "max_iterations": 10,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
        ]

        result = cli_runner.invoke(app, ["history"], input="n\n")
        assert result.exit_code == 0
        assert "Research History" in result.stdout


class TestResultsViewer:
    """Test results viewer functionality."""

    def test_results_viewer_init(self):
        """Test results viewer initialization."""
        viewer = ResultsViewer()
        assert viewer.console is not None

    def test_display_research_overview(self, capsys):
        """Test research overview display."""
        viewer = ResultsViewer()

        research_data = {
            "id": "test_run",
            "question": "Test question?",
            "domain": "biology",
            "state": "COMPLETED",
            "current_iteration": 5,
            "max_iterations": 10,
        }

        # Should not raise exception
        viewer.display_research_overview(research_data)

    def test_display_hypotheses_table_empty(self, capsys):
        """Test hypotheses table with no data."""
        viewer = ResultsViewer()
        viewer.display_hypotheses_table([])

        # Should display message about no hypotheses

    def test_display_hypotheses_table_with_data(self, capsys):
        """Test hypotheses table with data."""
        viewer = ResultsViewer()

        hypotheses = [
            {
                "claim": "Test hypothesis",
                "novelty_score": 0.85,
                "priority_score": 0.90,
                "status": "pending",
            }
        ]

        # Should not raise exception
        viewer.display_hypotheses_table(hypotheses)

    def test_export_to_json(self):
        """Test JSON export."""
        viewer = ResultsViewer()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            data = {"test": "data"}
            viewer.export_to_json(data, output_path)

            # Verify file was created and contains correct data
            assert output_path.exists()
            with open(output_path) as f:
                loaded = json.load(f)
            assert loaded == data

        finally:
            output_path.unlink()

    def test_export_to_markdown(self):
        """Test Markdown export."""
        viewer = ResultsViewer()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            output_path = Path(f.name)

        try:
            data = {
                "question": "Test question?",
                "id": "test_run",
                "domain": "biology",
                "state": "COMPLETED",
                "hypotheses": [],
                "experiments": [],
            }

            viewer.export_to_markdown(data, output_path)

            # Verify file was created
            assert output_path.exists()
            content = output_path.read_text()
            assert "# Research Results" in content

        finally:
            output_path.unlink()


class TestInteractiveMode:
    """Test interactive mode."""

    @patch("kosmos.cli.interactive.Prompt.ask")
    @patch("kosmos.cli.interactive.Confirm.ask")
    @patch("kosmos.cli.interactive.IntPrompt.ask")
    def test_interactive_mode_complete_flow(
        self, mock_int_prompt, mock_confirm, mock_prompt
    ):
        """Test complete interactive flow."""
        # Mock user inputs
        mock_prompt.side_effect = [
            "6",  # domain selection (general)
            "What are the patterns in complex systems?",  # question
        ]

        mock_confirm.side_effect = [
            True,  # confirm question
            False,  # enable budget
            True,  # enable cache
            True,  # auto model selection
            False,  # parallel execution
            True,  # confirm and start
        ]

        mock_int_prompt.side_effect = [
            10,  # max iterations
        ]

        result = run_interactive_mode()

        assert result is not None
        assert result["domain"] == "general"
        assert "question" in result
        assert result["max_iterations"] == 10

    @patch("kosmos.cli.interactive.Prompt.ask")
    @patch("kosmos.cli.interactive.Confirm.ask")
    def test_interactive_mode_cancel(self, mock_confirm, mock_prompt):
        """Test interactive mode cancellation."""
        mock_prompt.side_effect = ["6"]  # domain selection
        mock_confirm.return_value = False  # cancel at question confirmation

        # Should handle cancellation gracefully
        # (This test would need more setup to fully test)


@pytest.mark.parametrize("command,args", [
    ("version", []),
    ("info", []),
    ("config", ["--show"]),
    ("cache", ["--stats"]),
])
def test_command_does_not_crash(cli_runner, command, args):
    """Test that commands don't crash."""
    with patch("kosmos.config.get_config"):
        with patch("kosmos.core.cache_manager.get_cache_manager"):
            result = cli_runner.invoke(app, [command] + args)
            # Should either succeed or fail gracefully, not crash
            assert result.exit_code in [0, 1]


def test_cli_keyboard_interrupt(cli_runner):
    """Test CLI handles keyboard interrupt gracefully."""
    # This is difficult to test directly, but we can verify the code structure
    # supports it by checking that KeyboardInterrupt is caught in commands
    from kosmos.cli.commands import run, status, history, cache, config as config_cmd

    # Verify KeyboardInterrupt handlers exist
    import inspect

    for module in [run, status, history, cache, config_cmd]:
        source = inspect.getsource(module)
        assert "KeyboardInterrupt" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
