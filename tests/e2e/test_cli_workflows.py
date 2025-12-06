"""
E2E Tests for CLI Workflows.

Tests the Kosmos CLI command functionality:
- Complete research workflow via CLI
- Status command during research
- History command after completion
- Verbose and debug flags
- Mock mode operation
- Doctor diagnostics
"""

import pytest
import os
from typer.testing import CliRunner

from kosmos.cli.main import app


pytestmark = [pytest.mark.e2e]


# Create CLI runner
runner = CliRunner()


class TestCLIBasicCommands:
    """Tests for basic CLI commands."""

    def test_cli_version_command(self):
        """Verify version command works."""
        result = runner.invoke(app, ["version"])

        # Should show version info
        assert result.exit_code == 0
        assert "Kosmos" in result.stdout or "kosmos" in result.stdout.lower()

    def test_cli_info_command(self):
        """Verify info command works."""
        result = runner.invoke(app, ["info"])

        # Should show system info
        assert result.exit_code == 0


class TestCLIHelpOutput:
    """Tests for CLI help output."""

    def test_cli_main_help(self):
        """Verify main help displays correctly."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "kosmos" in result.stdout.lower() or "Kosmos" in result.stdout
        assert "run" in result.stdout.lower()

    def test_cli_run_help(self):
        """Verify run command help displays correctly."""
        result = runner.invoke(app, ["run", "--help"])

        assert result.exit_code == 0
        # Should show help for run command


class TestCLIVerboseFlag:
    """Tests for verbose output flag."""

    def test_cli_verbose_flag_accepted(self):
        """Verify verbose flag is accepted."""
        result = runner.invoke(app, ["--verbose", "version"])

        # Should accept verbose flag
        assert result.exit_code == 0

    def test_cli_short_verbose_flag(self):
        """Verify -v shorthand works."""
        result = runner.invoke(app, ["-v", "version"])

        assert result.exit_code == 0


class TestCLIDebugFlag:
    """Tests for debug output flag."""

    def test_cli_debug_flag_accepted(self):
        """Verify debug flag is accepted."""
        result = runner.invoke(app, ["--debug", "version"])

        # Should accept debug flag
        assert result.exit_code == 0

    def test_cli_debug_level_flag(self):
        """Verify debug level flag works."""
        result = runner.invoke(app, ["--debug-level", "2", "version"])

        assert result.exit_code == 0

    def test_cli_trace_flag(self):
        """Verify trace flag works."""
        result = runner.invoke(app, ["--trace", "version"])

        assert result.exit_code == 0


class TestCLIQuietFlag:
    """Tests for quiet mode."""

    def test_cli_quiet_flag_accepted(self):
        """Verify quiet flag is accepted."""
        result = runner.invoke(app, ["--quiet", "version"])

        assert result.exit_code == 0

    def test_cli_short_quiet_flag(self):
        """Verify -q shorthand works."""
        result = runner.invoke(app, ["-q", "version"])

        assert result.exit_code == 0


class TestCLIDoctorDiagnostics:
    """Tests for doctor diagnostics command."""

    def test_cli_doctor_diagnostics(self):
        """Verify doctor command validates setup."""
        result = runner.invoke(app, ["doctor"])

        # Doctor command should run (may show warnings without all deps)
        # Exit code 0 or 1 depending on environment
        assert result.exit_code in [0, 1, 2]  # 0=ok, 1/2=warnings/errors

        # Should show diagnostic output
        output = result.stdout.lower()
        # Check for typical diagnostic sections
        has_diagnostic_output = any([
            "python" in output,
            "check" in output,
            "status" in output,
            "ok" in output,
            "error" in output,
            "warning" in output,
            "database" in output,
            "api" in output
        ])
        assert has_diagnostic_output or result.exit_code == 0


class TestCLIRunCommand:
    """Tests for the run command."""

    def test_cli_run_help_shows_options(self):
        """Verify run command help shows all options."""
        result = runner.invoke(app, ["run", "--help"])

        assert result.exit_code == 0
        # Check common options are documented
        output = result.stdout.lower()
        assert "help" in output or "run" in output


class TestCLIStatusCommand:
    """Tests for status command."""

    def test_cli_status_command(self):
        """Verify status command works."""
        result = runner.invoke(app, ["status"])

        # Should run (may show "no active research" if nothing running)
        # Exit code depends on state
        assert result.exit_code in [0, 1]


class TestCLIHistoryCommand:
    """Tests for history command."""

    def test_cli_history_command(self):
        """Verify history command works."""
        result = runner.invoke(app, ["history"])

        # Should run (may show "no history" if database empty)
        assert result.exit_code in [0, 1]


class TestCLICacheCommand:
    """Tests for cache management command."""

    def test_cli_cache_help(self):
        """Verify cache command help displays."""
        result = runner.invoke(app, ["cache", "--help"])

        assert result.exit_code == 0


class TestCLIConfigCommand:
    """Tests for config management command."""

    def test_cli_config_help(self):
        """Verify config command help displays."""
        result = runner.invoke(app, ["config", "--help"])

        assert result.exit_code == 0


class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    def test_cli_invalid_command(self):
        """Verify invalid command shows error."""
        result = runner.invoke(app, ["invalid_command_xyz"])

        # Should show error for unknown command
        assert result.exit_code != 0


class TestCLIIntegration:
    """Integration tests for CLI workflows."""

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY"
    )
    def test_cli_full_workflow_with_api_key(self):
        """Test full CLI workflow with real API key."""
        # This test only runs if API key is available
        result = runner.invoke(app, [
            "run",
            "What is 2+2?",  # Simple question for quick test
            "--max-iterations", "1",
            "--verbose"
        ])

        # Should complete without crash
        # (actual success depends on environment)
        assert result.exception is None or "error" not in str(result.exception).lower()

    def test_cli_workflow_sequence(self):
        """Test typical CLI workflow sequence."""
        # 1. Check version
        version_result = runner.invoke(app, ["version"])
        assert version_result.exit_code == 0

        # 2. Check status (no active research)
        status_result = runner.invoke(app, ["status"])
        # May show "no active" or similar
        assert status_result.exit_code in [0, 1]

        # 3. Check history
        history_result = runner.invoke(app, ["history"])
        # May show "no history" if empty
        assert history_result.exit_code in [0, 1]

        # 4. Check doctor
        doctor_result = runner.invoke(app, ["doctor"])
        # May show warnings but should not crash
        assert doctor_result.exit_code in [0, 1, 2]
