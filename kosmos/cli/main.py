"""
Kosmos CLI - Main entry point.

Beautiful command-line interface for the Kosmos AI Scientist using Typer and Rich.
"""

import sys
from pathlib import Path
from typing import Optional
import logging

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
def setup_logging(verbose: bool = False, debug: bool = False):
    """Configure logging for CLI."""
    from kosmos.cli.utils import get_log_dir

    log_dir = get_log_dir()
    log_file = log_dir / "kosmos.log"

    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() if debug else logging.NullHandler(),
        ]
    )


# Global options for all commands
@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
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
    ctx.obj["quiet"] = quiet

    # Configure logging
    setup_logging(verbose=verbose, debug=debug)

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

        config_table.add_row("Claude Model", config.claude.model)
        config_table.add_row("Max Iterations", str(config.research.max_iterations))
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

    # Check API key
    api_key_ok = bool(os.getenv("ANTHROPIC_API_KEY"))
    checks.append(("Anthropic API Key", "Configured" if api_key_ok else "Not set", api_key_ok))

    # Check cache directory
    from kosmos.cli.utils import get_cache_dir
    cache_dir = get_cache_dir()
    cache_ok = cache_dir.exists() and os.access(cache_dir, os.W_OK)
    checks.append(("Cache Directory", str(cache_dir), cache_ok))

    # Check database
    try:
        from kosmos.db import get_session
        with get_session() as session:
            # Just open and close to verify connection
            pass
        checks.append(("Database", "Connected", True))
    except Exception:
        checks.append(("Database", "Error", False))

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
