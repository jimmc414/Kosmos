"""
Config command for Kosmos CLI.

View and manage Kosmos configuration.
"""

from typing import Optional
from pathlib import Path

import typer
from rich.syntax import Syntax
from rich.panel import Panel

from kosmos.cli.utils import (
    console,
    print_success,
    print_error,
    print_info,
    get_icon,
    create_table,
)


def manage_config(
    show: bool = typer.Option(False, "--show", "-s", help="Show current configuration"),
    edit: bool = typer.Option(False, "--edit", "-e", help="Open config file in editor"),
    validate: bool = typer.Option(False, "--validate", "-v", help="Validate configuration"),
    reset: bool = typer.Option(False, "--reset", help="Reset to default configuration"),
    path: bool = typer.Option(False, "--path", "-p", help="Show config file path"),
):
    """
    View and manage configuration.

    Examples:

        # Show current config
        kosmos config --show

        # Validate config
        kosmos config --validate

        # Show config file path
        kosmos config --path

        # Edit config
        kosmos config --edit
    """
    try:
        # Default to showing config if no options
        if not (show or edit or validate or reset or path):
            show = True

        # Show config file path
        if path:
            show_config_path()

        # Show configuration
        if show:
            display_config()

        # Validate
        if validate:
            validate_config()

        # Edit
        if edit:
            edit_config()

        # Reset
        if reset:
            reset_config()

    except KeyboardInterrupt:
        console.print("\n[warning]Config operation cancelled[/warning]")
        raise typer.Exit(130)

    except Exception as e:
        print_error(f"Config operation failed: {str(e)}")
        raise typer.Exit(1)


def show_config_path():
    """Show configuration file path."""
    import os

    console.print()
    console.print(f"[h3]{get_icon('path')} Configuration File Locations[/h3]")
    console.print()

    table = create_table(
        title="",
        columns=["File", "Path", "Exists"],
        show_lines=True,
    )

    # .env file
    env_path = Path(".env")
    table.add_row(
        ".env",
        str(env_path.absolute()),
        "[success]✓[/success]" if env_path.exists() else "[error]✗[/error]"
    )

    # Example .env
    env_example = Path(".env.example")
    table.add_row(
        ".env.example",
        str(env_example.absolute()),
        "[success]✓[/success]" if env_example.exists() else "[error]✗[/error]"
    )

    console.print(table)
    console.print()


def display_config():
    """Display current configuration."""
    console.print()
    console.print(f"[h2]{get_icon('info')} Current Configuration[/h2]", justify="center")
    console.print()

    try:
        from kosmos.config import get_config

        config = get_config()

        # Claude configuration
        if config.claude:
            claude_table = create_table(
                title="Claude API Configuration",
                columns=["Setting", "Value"],
                show_lines=True,
            )

            claude_table.add_row("Model", config.claude.model)
            claude_table.add_row("API Mode", "CLI" if config.claude.is_cli_mode else "API")
            claude_table.add_row("Max Tokens", str(config.claude.max_tokens))
            claude_table.add_row("Temperature", str(config.claude.temperature))
            claude_table.add_row("Cache Enabled", str(config.claude.enable_cache))

            console.print(claude_table)
            console.print()
        elif config.openai:
            openai_table = create_table(
                title="OpenAI API Configuration",
                columns=["Setting", "Value"],
                show_lines=True,
            )

            openai_table.add_row("Model", config.openai.model)
            openai_table.add_row("Max Tokens", str(config.openai.max_tokens))
            openai_table.add_row("Temperature", str(config.openai.temperature))

            console.print(openai_table)
            console.print()

        # Research configuration
        research_table = create_table(
            title="Research Configuration",
            columns=["Setting", "Value"],
            show_lines=True,
        )

        research_table.add_row("Max Iterations", str(config.research.max_iterations))
        research_table.add_row(
            "Enabled Domains",
            ", ".join(config.research.enabled_domains) if config.research.enabled_domains else "All"
        )
        research_table.add_row(
            "Experiment Types",
            ", ".join(config.research.enabled_experiment_types)
        )
        research_table.add_row(
            "Budget (USD)",
            f"${config.research.budget_usd}" if config.research.budget_usd else "No limit"
        )

        console.print(research_table)
        console.print()

        # Database configuration
        db_table = create_table(
            title="Database Configuration",
            columns=["Setting", "Value"],
            show_lines=True,
        )

        db_table.add_row("Database URL", config.database.url)

        console.print(db_table)
        console.print()

    except Exception as e:
        print_error(f"Failed to load configuration: {str(e)}")
        raise typer.Exit(1)


def validate_config():
    """Validate configuration."""
    console.print()
    console.print(f"[h2]{get_icon('flask')} Validating Configuration[/h2]", justify="center")
    console.print()

    try:
        from kosmos.config import get_config
        import os

        config = get_config()

        # Validation checks
        checks = []

        # Check API key based on provider
        llm_provider = os.getenv("LLM_PROVIDER", "anthropic")
        if llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            checks.append((
                "OpenAI API Key",
                "Configured" if api_key else "Missing",
                bool(api_key)
            ))
        else:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            checks.append((
                "Anthropic API Key",
                "Configured" if api_key else "Missing",
                bool(api_key)
            ))

        # Check model
        if config.claude:
            checks.append((
                "Claude Model",
                config.claude.model,
                config.claude.model in ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]
            ))
        elif config.openai:
            checks.append((
                "OpenAI Model",
                config.openai.model,
                True  # Any valid model is acceptable
            ))

        # Check domains
        valid_domains = {"biology", "neuroscience", "materials", "physics", "chemistry", "general"}
        domains_valid = all(d in valid_domains for d in config.research.enabled_domains) if config.research.enabled_domains else True
        checks.append((
            "Enabled Domains",
            ", ".join(config.research.enabled_domains) if config.research.enabled_domains else "All",
            domains_valid
        ))

        # Check database
        db_exists = Path(config.database.url.replace("sqlite:///", "")).exists()
        checks.append((
            "Database",
            config.database.url,
            db_exists
        ))

        # Display results
        table = create_table(
            title="Validation Results",
            columns=["Check", "Value", "Status"],
            show_lines=True,
        )

        for check, value, passed in checks:
            status = "[success]✓ Valid[/success]" if passed else "[error]✗ Invalid[/error]"
            table.add_row(check, value, status)

        console.print(table)
        console.print()

        # Overall result
        all_valid = all(check[2] for check in checks)
        if all_valid:
            print_success("Configuration is valid!", title="Validation Complete")
        else:
            print_error("Configuration has issues. Please review and fix.", title="Validation Failed")
            raise typer.Exit(1)

    except Exception as e:
        print_error(f"Validation failed: {str(e)}")
        raise typer.Exit(1)


def edit_config():
    """Open config file in default editor."""
    import os
    import subprocess

    env_path = Path(".env")

    if not env_path.exists():
        print_info("No .env file found. Creating from .env.example...")

        env_example = Path(".env.example")
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_path)
        else:
            # Create minimal .env
            env_path.write_text("# Kosmos Configuration\nANTHROPIC_API_KEY=your_api_key_here\n")

    # Open in editor
    editor = os.environ.get("EDITOR", "nano")

    try:
        subprocess.run([editor, str(env_path)])
        print_success(f"Configuration file edited: {env_path}")
    except Exception as e:
        print_error(f"Failed to open editor: {str(e)}")


def reset_config():
    """Reset configuration to defaults."""
    console.print()

    from rich.prompt import Confirm

    if not Confirm.ask("[warning]Reset configuration to defaults? This will overwrite your .env file.[/warning]"):
        console.print("[info]Reset cancelled[/info]")
        return

    env_path = Path(".env")
    env_example = Path(".env.example")

    if env_example.exists():
        import shutil
        shutil.copy(env_example, env_path)
        print_success("Configuration reset to defaults", title="Reset Complete")
    else:
        print_error(".env.example not found. Cannot reset.")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(manage_config)
