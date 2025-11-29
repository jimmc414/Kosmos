"""
Kosmos CLI - Main entry point.

Beautiful command-line interface for the Kosmos AI Scientist using Typer and Rich.
"""

import sys
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.traceback import install as install_rich_traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from kosmos.cli.utils import (
    console,
    print_error,
    print_info,
    print_success,
    get_icon,
)
from kosmos.cli.themes import KOSMOS_THEME


# Install rich traceback handler for beautiful error messages
install_rich_traceback(show_locals=False, width=120, console=console)


# Create main Typer app
app = typer.Typer(
    name="kosmos",
    help="Kosmos AI Scientist - Autonomous scientific research powered by Claude",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


# Configure logging
def setup_logging(verbose: bool = False, debug: bool = False, trace: bool = False, debug_level: int = 0):
    """Configure logging for CLI.

    Args:
        verbose: Enable verbose (INFO level) output
        debug: Enable debug mode (DEBUG level)
        trace: Enable trace-level logging (maximum verbosity, enables all debug flags)
        debug_level: Debug verbosity level (0-3)
    """
    from kosmos.cli.utils import get_log_dir

    log_dir = get_log_dir()
    log_file = log_dir / "kosmos.log"

    # Determine log level
    if trace or debug_level >= 2:
        level = logging.DEBUG
    elif debug or debug_level >= 1:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    # If trace is enabled, also enable all debug toggles in config
    if trace:
        try:
            from kosmos.config import get_config
            config = get_config()
            config.logging.log_llm_calls = True
            config.logging.log_agent_messages = True
            config.logging.log_workflow_transitions = True
            config.logging.stage_tracking_enabled = True
            config.logging.debug_level = 3
        except Exception:
            pass  # Config may not be available yet

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() if (debug or trace or debug_level > 0) else logging.NullHandler(),
        ]
    )


# Global options for all commands
@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
    trace: bool = typer.Option(False, "--trace", help="Enable trace-level logging (maximum verbosity)"),
    debug_level: int = typer.Option(0, "--debug-level", "-dl", help="Debug level 0-3 (0=off, 1=critical path, 2=full trace, 3=data dumps)"),
    debug_modules: Optional[str] = typer.Option(None, "--debug-modules", help="Comma-separated modules to debug"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress non-essential output"),
):
    """
    Kosmos AI Scientist - Autonomous scientific research powered by Claude.

    A fully autonomous AI scientist that can:

    • Generate and test hypotheses across multiple domains
    • Design and execute computational experiments
    • Analyze results and synthesize insights
    • Learn iteratively from outcomes
    • Produce publication-quality research

    For detailed help on any command, run:

        kosmos COMMAND --help
    """
    # Store global options in context
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug
    ctx.obj["trace"] = trace
    ctx.obj["debug_level"] = debug_level
    ctx.obj["debug_modules"] = debug_modules
    ctx.obj["quiet"] = quiet

    # Configure logging
    setup_logging(verbose=verbose, debug=debug, trace=trace, debug_level=debug_level)

    # Set debug_modules in config if provided
    if debug_modules:
        try:
            from kosmos.config import get_config
            config = get_config()
            config.logging.debug_modules = [m.strip() for m in debug_modules.split(",")]
        except Exception:
            pass  # Config may not be available yet

    # Initialize database
    try:
        from kosmos.db import init_from_config
        init_from_config()
    except Exception as e:
        # Always show database initialization errors to the user
        error_msg = f"Database initialization failed: {e}"
        logger.error(error_msg, exc_info=debug)

        # Show user-friendly message
        if not quiet:
            print_error(
                "Failed to initialize database. This may be due to:\n"
                "  • Missing .env file (will be auto-created from .env.example)\n"
                "  • Missing ANTHROPIC_API_KEY environment variable\n"
                "  • Database connection issues\n\n"
                f"Error: {str(e)}\n\n"
                "Run with --debug for full error details.",
                title="Database Initialization Error"
            )

        # Don't exit here - let commands handle the error if they need database
        # Some commands (like --help, version) don't need database

    # Suppress console output if quiet mode
    if quiet:
        console.quiet = True


@app.command()
def version():
    """Show Kosmos version and system information."""
    from kosmos import __version__
    import platform
    import anthropic

    info_lines = [
        f"**Kosmos AI Scientist** v{__version__}",
        "",
        "**System Information:**",
        f"  • Python: {platform.python_version()}",
        f"  • Platform: {platform.system()} {platform.release()}",
        f"  • Anthropic SDK: {anthropic.__version__}",
        "",
        "Built with Claude Sonnet 4.5",
    ]

    console.print(
        Panel(
            "\n".join(info_lines),
            title=f"[bright_blue]{get_icon('rocket')} Kosmos Version[/bright_blue]",
            border_style="bright_blue",
        )
    )


@app.command()
def info():
    """Show system status and configuration."""
    from kosmos.cli.utils import get_config_value, format_size, get_cache_dir
    from kosmos.config import get_config
    from kosmos.cli.utils import create_table
    import os

    console.print()
    console.print(f"[h2]{get_icon('info')} System Information[/h2]", justify="center")
    console.print()

    # Get configuration
    try:
        config = get_config()

        # Configuration table
        config_table = create_table(
            title="Configuration",
            columns=["Setting", "Value"],
            show_lines=True,
        )

        if config.claude:
            config_table.add_row("Claude Model", config.claude.model)
        elif config.openai:
            config_table.add_row("OpenAI Model", config.openai.model)
        config_table.add_row("Max Iterations", str(config.research.max_iterations))
        if config.claude:
            config_table.add_row("API Mode", "CLI" if config.claude.is_cli_mode else "API")
        config_table.add_row(
            "Domains",
            ", ".join(config.research.enabled_domains) if config.research.enabled_domains else "All"
        )

        console.print(config_table)
        console.print()

        # Cache information
        cache_dir = get_cache_dir()
        if cache_dir.exists():
            cache_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())

            cache_table = create_table(
                title="Cache Status",
                columns=["Item", "Value"],
                show_lines=True,
            )

            cache_table.add_row("Cache Directory", str(cache_dir))
            cache_table.add_row("Cache Size", format_size(cache_size))

            console.print(cache_table)
            console.print()

        # API key status
        api_key_status = "✓ Configured" if os.getenv("ANTHROPIC_API_KEY") else "✗ Not configured"
        api_color = "success" if os.getenv("ANTHROPIC_API_KEY") else "error"

        console.print(f"[{api_color}]API Key: {api_key_status}[/{api_color}]")
        console.print()

    except Exception as e:
        print_error(f"Failed to load configuration: {str(e)}")
        raise typer.Exit(1)


@app.command()
def doctor():
    """Run diagnostic checks on the Kosmos installation."""
    from kosmos.cli.utils import create_table
    import importlib
    import os

    console.print()
    console.print(f"[h2]{get_icon('flask')} Running Diagnostics[/h2]", justify="center")
    console.print()

    checks = []

    # Check Python version
    import sys
    python_ok = sys.version_info >= (3, 9)
    checks.append(("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}", python_ok))

    # Check required packages
    required_packages = [
        "anthropic",
        "typer",
        "rich",
        "pydantic",
        "sqlalchemy",
        "numpy",
        "pandas",
        "scipy",
    ]

    for package in required_packages:
        try:
            importlib.import_module(package)
            checks.append((f"Package: {package}", "Installed", True))
        except ImportError:
            checks.append((f"Package: {package}", "Missing", False))

    # Check API key based on provider
    llm_provider = os.getenv("LLM_PROVIDER", "anthropic")
    if llm_provider == "openai":
        api_key_ok = bool(os.getenv("OPENAI_API_KEY"))
        checks.append(("OpenAI API Key", "Configured" if api_key_ok else "Not set", api_key_ok))
    else:
        api_key_ok = bool(os.getenv("ANTHROPIC_API_KEY"))
        checks.append(("Anthropic API Key", "Configured" if api_key_ok else "Not set", api_key_ok))

    # Check cache directory
    from kosmos.cli.utils import get_cache_dir
    cache_dir = get_cache_dir()
    cache_ok = cache_dir.exists() and os.access(cache_dir, os.W_OK)
    checks.append(("Cache Directory", str(cache_dir), cache_ok))

    # Check database connection and schema
    db_issues = []
    try:
        from kosmos.db import get_session
        from kosmos.config import get_config
        from kosmos.utils.setup import validate_database_schema

        # Test connection
        with get_session() as session:
            pass

        # Validate schema completeness
        config = get_config()
        validation = validate_database_schema(config.database.normalized_url)

        if validation["complete"]:
            checks.append(("Database Connection", "Connected", True))
            checks.append(("Database Schema", "Complete", True))
        else:
            checks.append(("Database Connection", "Connected", True))
            checks.append(("Database Schema", "Incomplete", False))

            # Track issues for detailed reporting
            if validation["missing_tables"]:
                db_issues.append(f"Missing tables: {', '.join(validation['missing_tables'])}")
            if validation["missing_indexes"]:
                db_issues.append(f"Missing indexes: {', '.join(validation['missing_indexes'])}")
            if validation["errors"]:
                db_issues.extend(validation["errors"])

    except Exception as e:
        checks.append(("Database Connection", f"Error: {str(e)}", False))
        checks.append(("Database Schema", "Unknown", False))
        db_issues.append(f"Connection failed: {e}")

    # Display results
    table = create_table(
        title="Diagnostic Results",
        columns=["Check", "Status", "Result"],
        show_lines=True,
    )

    for check, status, ok in checks:
        result = "[success]✓ PASS[/success]" if ok else "[error]✗ FAIL[/error]"
        table.add_row(check, status, result)

    console.print(table)
    console.print()

    # Show database issues if any
    if db_issues:
        console.print("[error]Database Issues Detected:[/error]")
        for issue in db_issues:
            console.print(f"  • {issue}")
        console.print()
        console.print("[warning]To fix database issues, try:[/warning]")
        console.print("  1. Ensure .env file exists (will be auto-created)")
        if llm_provider == "openai":
            console.print("  2. Set OPENAI_API_KEY environment variable")
        else:
            console.print("  2. Set ANTHROPIC_API_KEY environment variable")
        console.print("  3. Run: [code]alembic upgrade head[/code]")
        console.print("  4. Or reinstall: [code]make install[/code]")
        console.print()

    # Overall status
    all_ok = all(check[2] for check in checks)
    if all_ok:
        print_success("All checks passed! Kosmos is ready to use.", title="Diagnostics Complete")
    else:
        print_error(
            "Some checks failed. Please resolve the issues above before using Kosmos.",
            title="Diagnostics Failed"
        )
        raise typer.Exit(1)


# Import and register command modules
# These will be implemented in separate files
def register_commands():
    """Register all CLI command groups."""
    # Import command modules when they're implemented
    try:
        from kosmos.cli.commands import run, status, history, cache, config as config_cmd, profile, graph

        # Register commands
        app.command(name="run")(run.run_research)
        app.command(name="status")(status.show_status)
        app.command(name="history")(history.show_history)
        app.command(name="cache")(cache.manage_cache)
        app.command(name="config")(config_cmd.manage_config)
        app.command(name="profile")(profile.profile_command)
        app.command(name="graph")(graph.manage_graph)

    except ImportError as e:
        # Commands not yet implemented - silently skip
        logging.debug(f"Command import failed: {e}")
        pass


# Register commands
register_commands()


def cli_entrypoint():
    """Main entry point for CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[warning]Operation cancelled by user[/warning]")
        sys.exit(130)
    except Exception as e:
        if "--debug" in sys.argv:
            # Re-raise for full traceback in debug mode
            raise
        else:
            print_error(f"An error occurred: {str(e)}", title="Error")
            console.print("\n[muted]Run with --debug for full traceback[/muted]")
            sys.exit(1)


if __name__ == "__main__":
    cli_entrypoint()
