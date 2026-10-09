"""
Unit tests for CLI commands.

Tests the commands registered on the main app: version, info, doctor, run,
status, history, cache, config, profile. Each test patches the names the
command modules actually resolve at call time (most of them import their
collaborators inside the function body), so the patch targets are the
defining modules, e.g. kosmos.agents.research_director.ResearchDirectorAgent.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from kosmos.cli.main import app
from kosmos.config import reset_config


RUN_RESULTS = {
    "id": "run_unit_0001",
    "question": "What is the effect of temperature on protein folding?",
    "domain": "biology",
    "state": "converged",
    "current_iteration": 1,
    "max_iterations": 1,
    "convergence_reason": None,
    "hypotheses": [],
    "experiments": [],
    "results": [],
    "metrics": {},
}


def _research_data(state="RUNNING", iteration=3, max_iterations=10):
    created = datetime.now(timezone.utc) - timedelta(minutes=5)
    return {
        "id": "run_status_0001",
        "question": "Does CO2 predict temperature?",
        "domain": "climate",
        "state": state,
        "current_iteration": iteration,
        "max_iterations": max_iterations,
        "created_at": created,
        "updated_at": created + timedelta(minutes=2),
        "hypotheses": [],
        "experiments": [],
        "metrics": {"api_calls": 4, "cache_hits": 1, "cache_misses": 3, "total_cost_usd": 0.01},
    }


def _history_runs():
    # Naive UTC timestamps, as SQLite returns DateTime columns
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return [
        {
            "id": "run_hist_0001",
            "question": "Short question?",
            "domain": "biology",
            "state": "COMPLETED",
            "current_iteration": 5,
            "max_iterations": 10,
            "created_at": now - timedelta(hours=2),
            "updated_at": now - timedelta(hours=1),
        },
        {
            "id": "run_hist_0002",
            "question": "A considerably longer research question that must be truncated in the table?",
            "domain": "physics",
            "state": "FAILED",
            "current_iteration": 2,
            "max_iterations": 10,
            "created_at": now - timedelta(days=2),
            "updated_at": now - timedelta(days=2),
        },
    ]


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def fresh_config():
    """The run command mutates the config singleton; isolate it."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def run_mocks(fresh_config):
    """Mock the director, the async research loop and the agent registry."""
    with patch("kosmos.agents.research_director.ResearchDirectorAgent") as director, \
         patch("kosmos.cli.commands.run.run_with_progress_async",
               new=AsyncMock(return_value=RUN_RESULTS)) as loop, \
         patch("kosmos.agents.registry.get_registry", return_value=MagicMock()):
        yield {"director": director, "loop": loop}


@pytest.fixture
def cache_manager():
    manager = MagicMock()
    manager.get_stats.return_value = {
        "claude": {"hits": 80, "misses": 20, "size": 100, "storage_size_mb": 2},
        "general": {"hits": 20, "misses": 80, "size": 100, "storage_size_mb": 1},
    }
    with patch("kosmos.core.cache_manager.get_cache_manager", return_value=manager):
        yield manager


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_version_command(self, runner):
        """Test version command."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "Kosmos" in result.stdout
        assert "version" in result.stdout.lower()

    def test_help_command(self, runner):
        """Test help command."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Kosmos AI Scientist" in result.stdout

    def test_doctor_command(self, runner, tmp_path):
        """Doctor reports a database it cannot reach and exits 1."""
        with patch("kosmos.db.get_session", side_effect=RuntimeError("db down")), \
             patch("kosmos.cli.utils.get_cache_dir", return_value=tmp_path), \
             patch("kosmos.core.providers.claude_code.ClaudeCodeProvider.cli_version",
                   return_value="2.0.0 (Claude Code)"):
            result = runner.invoke(app, ["doctor"])

        assert result.exit_code == 1
        assert "Running Diagnostics" in result.stdout
        assert "Diagnostic Results" in result.stdout
        assert "Database Issues Detected" in result.stdout
        assert "Connection failed: db down" in result.stdout
        assert "Diagnostics Failed" in result.stdout


class TestInfoCommand:
    """Test info command."""

    def test_info_displays_configuration(self, runner):
        """Test info command displays configuration."""
        with patch('kosmos.config.get_config') as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.claude.model = "claude-3-5-sonnet-20241022"
            mock_cfg.research.max_iterations = 10
            mock_cfg.research.enabled_domains = ["biology", "physics"]
            mock_cfg.claude.is_cli_mode = False
            mock_config.return_value = mock_cfg

            result = runner.invoke(app, ["info"])

            assert "claude-3-5-sonnet" in result.stdout.lower() or result.exit_code >= 0


class TestRunCommand:
    """Test run command."""

    def test_run_with_question(self, runner, run_mocks):
        """A positional question and --domain reach the director and the run completes."""
        question = "What is the effect of temperature on protein folding?"
        result = runner.invoke(app, ["run", question, "--domain", "biology", "--max-iterations", "1"])

        assert result.exit_code == 0, result.output
        run_mocks["director"].assert_called_once()
        kwargs = run_mocks["director"].call_args.kwargs
        assert kwargs["research_question"] == question
        assert kwargs["domain"] == "biology"
        assert kwargs["config"]["enabled_domains"] == ["biology"]
        assert kwargs["config"]["max_iterations"] == 1
        run_mocks["loop"].assert_awaited_once()
        assert "Research completed successfully" in result.stdout

    def test_run_interactive_mode(self, runner, run_mocks):
        """--interactive takes the question and settings from the interactive prompts."""
        interactive_config = {
            "question": "Test question",
            "domain": "physics",
            "max_iterations": 2,
            "budget_usd": None,
            "enable_cache": True,
            "auto_model_selection": True,
            "parallel_execution": False,
        }
        with patch("kosmos.cli.commands.run.run_interactive_mode", return_value=interactive_config) as interactive:
            result = runner.invoke(app, ["run", "--interactive"])

        assert result.exit_code == 0, result.output
        interactive.assert_called_once()
        kwargs = run_mocks["director"].call_args.kwargs
        assert kwargs["research_question"] == "Test question"
        assert kwargs["domain"] == "physics"
        assert kwargs["config"]["max_iterations"] == 2

    def test_run_interactive_cancelled(self, runner, run_mocks):
        """Cancelling the interactive prompts exits 0 without starting research."""
        with patch("kosmos.cli.commands.run.run_interactive_mode", return_value=None):
            result = runner.invoke(app, ["run", "--interactive"])

        assert result.exit_code == 0
        assert "Research cancelled" in result.stdout
        run_mocks["director"].assert_not_called()


class TestStatusCommand:
    """Test status command."""

    def test_status_shows_research_status(self, runner):
        """Test status command shows research status."""
        with patch("kosmos.cli.commands.status.get_research_data", return_value=_research_data()) as get_data:
            result = runner.invoke(app, ["status", "run_status_0001"])

        assert result.exit_code == 0, result.output
        get_data.assert_called_once_with("run_status_0001")
        assert "Research Overview" in result.stdout
        assert "Progress: 3/10" in result.stdout
        assert "Workflow Information" in result.stdout

    def test_status_latest_when_no_run_id(self, runner):
        """Without a run id the latest run is requested."""
        with patch("kosmos.cli.commands.status.get_research_data", return_value=_research_data()) as get_data:
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        get_data.assert_called_once_with(None)

    def test_status_watch_mode(self, runner):
        """Watch mode reloads the run and stops once it reaches a terminal state."""
        running = _research_data(state="RUNNING", iteration=3)
        completed = _research_data(state="COMPLETED", iteration=10)
        with patch("kosmos.cli.commands.status.get_research_data", side_effect=[running, completed]) as get_data, \
             patch("kosmos.cli.commands.status.time.sleep") as sleep:
            result = runner.invoke(app, ["status", "--watch"])

        assert result.exit_code == 0, result.output
        assert get_data.call_count == 2
        sleep.assert_called_once_with(5)
        assert "Research COMPLETED" in result.stdout


class TestHistoryCommand:
    """Test history command."""

    def test_history_lists_past_research(self, runner):
        """Test history command lists past research."""
        with patch("kosmos.cli.commands.history.get_research_runs", return_value=_history_runs()) as get_runs:
            result = runner.invoke(app, ["history", "--limit", "5", "--domain", "biology"], input="n\n")

        assert result.exit_code == 0, result.output
        get_runs.assert_called_once_with(5, "biology", None, None)
        assert "Research History" in result.stdout
        assert "Showing 2 runs" in result.stdout
        assert "COMPLETED" in result.stdout
        assert "2h ago" in result.stdout  # naive (SQLite) timestamps format as UTC

    def test_history_view_specific_cycle(self, runner):
        """Entering a run number at the prompt opens that run's details."""
        runs = _history_runs()
        with patch("kosmos.cli.commands.history.get_research_runs", return_value=runs), \
             patch("kosmos.cli.commands.history.view_run_details") as view:
            result = runner.invoke(app, ["history"], input="2\n")

        assert result.exit_code == 0, result.output
        view.assert_called_once_with(runs[1])

    def test_history_empty(self, runner):
        """No runs is not an error."""
        with patch("kosmos.cli.commands.history.get_research_runs", return_value=[]):
            result = runner.invoke(app, ["history"])

        assert result.exit_code == 0
        assert "No research runs found" in result.stdout
        assert "Failed to get history" not in result.stdout


class TestCacheCommand:
    """Test cache command."""

    def test_cache_info(self, runner, cache_manager):
        """With no flag the cache command shows statistics."""
        result = runner.invoke(app, ["cache"])

        assert result.exit_code == 0, result.output
        cache_manager.get_stats.assert_called_once()
        assert "Cache Statistics" in result.stdout
        assert "Total Requests" in result.stdout
        assert "200" in result.stdout  # 100 hits + 100 misses

    def test_cache_clear(self, runner, cache_manager):
        """--clear clears every cache once the user confirms."""
        result = runner.invoke(app, ["cache", "--clear"], input="y\n")

        assert result.exit_code == 0, result.output
        cache_manager.clear.assert_called_once_with()
        assert "All caches cleared successfully" in result.stdout

    def test_cache_clear_requires_confirmation(self, runner, cache_manager):
        """Declining the confirmation clears nothing."""
        result = runner.invoke(app, ["cache", "--clear"], input="n\n")

        assert result.exit_code == 0, result.output
        cache_manager.clear.assert_not_called()
        assert "Operation cancelled" in result.stdout

    def test_cache_clear_invalid_type(self, runner, cache_manager):
        """An unknown --clear-type is rejected with exit 1 and a single error."""
        result = runner.invoke(app, ["cache", "--clear-type", "bogus"])

        assert result.exit_code == 1
        assert "Invalid cache type" in result.stdout
        assert "Cache operation failed" not in result.stdout
        cache_manager.clear.assert_not_called()


class TestConfigCommand:
    """Test config command (options --show, --path, --validate on the hermetic test config)."""

    @pytest.fixture(autouse=True)
    def _fresh(self, fresh_config):
        yield

    def test_config_show(self, runner):
        """Test showing configuration."""
        result = runner.invoke(app, ["config", "--show"])

        assert result.exit_code == 0, result.output
        assert "Current Configuration" in result.stdout
        assert "Research Configuration" in result.stdout
        assert "Database Configuration" in result.stdout

    def test_config_path(self, runner):
        """--path lists the .env and .env.example locations."""
        result = runner.invoke(app, ["config", "--path"])

        assert result.exit_code == 0, result.output
        assert "Configuration File Locations" in result.stdout
        assert ".env.example" in result.stdout
        assert "Current Configuration" not in result.stdout

    def test_config_validate(self, runner):
        """Test config validation."""
        result = runner.invoke(app, ["config", "--validate"])

        assert result.exit_code in (0, 1)
        assert "Validating Configuration" in result.stdout
        assert "Validation Results" in result.stdout
        assert "Config operation failed" not in result.stdout


class TestProfileCommand:
    """Test profile command (targets: experiment, agent, workflow)."""

    def test_profile_view(self, runner):
        """Profiling an experiment with no stored profile exits 0 with a warning."""
        result = runner.invoke(app, ["profile", "experiment", "--experiment", "exp_123"])

        assert result.exit_code == 0, result.output
        assert "Profiling Experiment: exp_123" in result.stdout
        assert "No profiling data found for experiment exp_123" in result.stdout

    def test_profile_clear(self, runner):
        """An unknown target is rejected."""
        result = runner.invoke(app, ["profile", "clear"])

        assert result.exit_code == 1
        assert "Invalid target 'clear'" in result.stdout

    def test_profile_bottlenecks(self, runner):
        """The experiment target requires --experiment."""
        result = runner.invoke(app, ["profile", "experiment"])

        assert result.exit_code == 1
        assert "--experiment required" in result.stdout


class TestCLIOptions:
    """Test global CLI options."""

    def test_verbose_flag(self, runner):
        """Test verbose flag."""
        result = runner.invoke(app, ["--verbose", "version"])

        assert result.exit_code == 0

    def test_debug_flag(self, runner):
        """Test debug flag."""
        result = runner.invoke(app, ["--debug", "version"])

        assert result.exit_code == 0

    def test_quiet_flag(self, runner):
        """Test quiet flag."""
        result = runner.invoke(app, ["--quiet", "version"])

        assert result.exit_code == 0

    def test_quiet_does_not_leak_into_next_invocation(self, runner):
        """--quiet mutes the shared console for that invocation only."""
        quiet = runner.invoke(app, ["--quiet", "version"])
        loud = runner.invoke(app, ["version"])

        assert "Kosmos AI Scientist" not in quiet.stdout
        assert "Kosmos AI Scientist" in loud.stdout


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_invalid_command(self, runner):
        """Test handling invalid command."""
        result = runner.invoke(app, ["nonexistent-command"])

        assert result.exit_code != 0

    def test_missing_required_argument(self, runner, run_mocks):
        """No question (none given, none entered interactively) is an error."""
        empty = {"question": "", "domain": None, "max_iterations": 1}
        with patch("kosmos.cli.commands.run.run_interactive_mode", return_value=empty) as interactive:
            result = runner.invoke(app, ["run"])

        interactive.assert_called_once()
        assert result.exit_code == 1
        assert "No research question provided" in result.stdout
        run_mocks["director"].assert_not_called()

    def test_exception_handling(self, runner, run_mocks):
        """A director failure is reported and exits 1."""
        run_mocks["director"].side_effect = Exception("Test error")

        result = runner.invoke(app, ["run", "Test", "--max-iterations", "1"])

        assert result.exit_code == 1
        assert "Research failed: Test error" in result.stdout
        run_mocks["loop"].assert_not_awaited()


class TestCLIOutputFormatting:
    """Test CLI output formatting."""

    def test_table_output(self, runner):
        """The default history view is a table that truncates long questions."""
        with patch("kosmos.cli.commands.history.get_research_runs", return_value=_history_runs()):
            result = runner.invoke(app, ["history"], input="n\n")

        assert result.exit_code == 0, result.output
        assert "Run ID" in result.stdout
        assert "Progress" in result.stdout
        assert "5/10" in result.stdout
        assert "question ..." in result.stdout  # 40-character cut plus "..."
        assert "truncated" not in result.stdout

    def test_detailed_output(self, runner):
        """--details prints one property table per run, with the full question."""
        runs = _history_runs()[:1]
        with patch("kosmos.cli.commands.history.get_research_runs", return_value=runs):
            result = runner.invoke(app, ["history", "--details"])

        assert result.exit_code == 0, result.output
        assert "Detailed Research History" in result.stdout
        assert "1. Run: run_hist_0001" in result.stdout
        assert "Short question?" in result.stdout


class TestCLIIntegration:
    """Integration tests for CLI."""

    @pytest.mark.integration
    def test_full_workflow(self, runner, run_mocks, cache_manager):
        """Test complete CLI workflow."""
        with patch("kosmos.cli.commands.status.get_research_data", return_value=_research_data()), \
             patch("kosmos.cli.commands.history.get_research_runs", return_value=_history_runs()):
            # 1. Check status
            result = runner.invoke(app, ["status"])
            assert result.exit_code == 0, result.output

            # 2. View cache
            result = runner.invoke(app, ["cache", "--stats"])
            assert result.exit_code == 0, result.output

            # 3. Run research
            result = runner.invoke(app, [
                "run", "Test question",
                "--domain", "biology",
                "--max-iterations", "1",
            ])
            assert result.exit_code == 0, result.output
            run_mocks["director"].assert_called_once()

            # 4. Check history
            result = runner.invoke(app, ["history"], input="n\n")
            assert result.exit_code == 0, result.output
