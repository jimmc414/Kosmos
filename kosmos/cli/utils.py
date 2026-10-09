"""
Shared utilities for Kosmos CLI.

Provides common formatting, database helpers, and utility functions
used across all CLI commands.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Any, Dict, List
from contextlib import contextmanager

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markup import escape

from kosmos.cli.themes import (
    KOSMOS_THEME,
    get_domain_color,
    get_state_color,
    get_metric_color,
    get_box_style,
    get_icon,
    get_status_icon,
)


# Global console instance with theme
console = Console(theme=KOSMOS_THEME)


def format_timestamp(dt: datetime, relative: bool = True) -> str:
    """
    Format timestamp for display.

    Args:
        dt: Datetime to format
        relative: If True, show relative time (e.g., "2 hours ago")

    Returns:
        Formatted timestamp string
    """
    if relative:
        if dt.tzinfo is None:
            # DateTime columns without timezone=True come back naive (UTC by convention)
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        diff = now - dt

        if diff.total_seconds() < 60:
            return "just now"
        elif diff.total_seconds() < 3600:
            minutes = int(diff.total_seconds() / 60)
            return f"{minutes}m ago"
        elif diff.total_seconds() < 86400:
            hours = int(diff.total_seconds() / 3600)
            return f"{hours}h ago"
        elif diff.total_seconds() < 604800:
            days = int(diff.total_seconds() / 86400)
            return f"{days}d ago"
        else:
            return dt.strftime("%Y-%m-%d")
    else:
        return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def format_size(bytes_size: int) -> str:
    """Format byte size to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage value."""
    return f"{value:.{decimals}f}%"


def format_currency(value: float, currency: str = "USD") -> str:
    """Format currency value."""
    return f"${value:.2f}" if currency == "USD" else f"{value:.2f} {currency}"


def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """Truncate text to max length with suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def print_success(message: str, title: str = "Success"):
    """Print success message in styled panel."""
    console.print(
        Panel(
            f"[success]{get_icon('success')} {message}[/success]",
            title=f"[success]{title}[/success]",
            box=get_box_style("default"),
            border_style="success",
        )
    )


def print_error(message: str, title: str = "Error"):
    """Print error message in styled panel."""
    console.print(
        Panel(
            f"[error]{get_icon('error')} {message}[/error]",
            title=f"[error]{title}[/error]",
            box=get_box_style("default"),
            border_style="error",
        )
    )


def print_warning(message: str, title: str = "Warning"):
    """Print warning message in styled panel."""
    console.print(
        Panel(
            f"[warning]{get_icon('warning')} {message}[/warning]",
            title=f"[warning]{title}[/warning]",
            box=get_box_style("default"),
            border_style="warning",
        )
    )


def print_info(message: str, title: str = "Info"):
    """Print info message in styled panel."""
    console.print(
        Panel(
            f"[info]{get_icon('info')} {message}[/info]",
            title=f"[info]{title}[/info]",
            box=get_box_style("default"),
            border_style="info",
        )
    )


def create_status_text(status: str, add_icon: bool = True) -> Text:
    """Create colored status text with optional icon."""
    color = get_state_color(status)
    icon = get_status_icon(status) if add_icon else ""
    text = f"{icon} {status}" if icon else status
    return Text(text, style=color)


def create_domain_text(domain: str) -> Text:
    """Create colored domain text."""
    color = get_domain_color(domain)
    return Text(domain, style=color)


def create_metric_text(
    value: float,
    format_type: str = "percentage",
    thresholds: Optional[Dict] = None
) -> Text:
    """
    Create colored metric text based on value.

    Args:
        value: Metric value
        format_type: How to format ('percentage', 'number', 'currency')
        thresholds: Custom thresholds for coloring

    Returns:
        Styled Text object
    """
    color = get_metric_color(value, thresholds)

    if format_type == "percentage":
        formatted = format_percentage(value * 100)
    elif format_type == "currency":
        formatted = format_currency(value)
    else:
        formatted = f"{value:.2f}"

    return Text(formatted, style=color)


def create_table(
    title: str,
    columns: List[str],
    rows: Optional[List[List[Any]]] = None,
    caption: Optional[str] = None,
    show_header: bool = True,
    show_lines: bool = False,
) -> Table:
    """
    Create a styled Rich table.

    Args:
        title: Table title
        columns: List of column names
        rows: Optional list of row data
        caption: Optional table caption
        show_header: Whether to show header row
        show_lines: Whether to show lines between rows

    Returns:
        Configured Rich Table
    """
    table = Table(
        title=title,
        caption=caption,
        show_header=show_header,
        show_lines=show_lines,
        box=get_box_style("default"),
        title_style="h2",
        caption_style="table.caption",
    )

    # Add columns
    for col in columns:
        table.add_column(col, style="white")

    # Add rows if provided
    if rows:
        for row in rows:
            # Convert all values to strings, handling Text objects
            str_row = []
            for val in row:
                if isinstance(val, Text):
                    str_row.append(val)
                else:
                    str_row.append(str(val))
            table.add_row(*str_row)

    return table


@contextmanager
def get_db_session():
    """
    Context manager for database sessions.

    Yields:
        SQLAlchemy session
    """
    from kosmos.db import get_session

    with get_session() as session:
        yield session


def validate_run_id(run_id: str) -> bool:
    """
    Validate research run ID format.

    Args:
        run_id: Run ID to validate

    Returns:
        True if valid format
    """
    # Run IDs should be UUIDs or similar format
    return len(run_id) >= 8 and run_id.replace("-", "").replace("_", "").isalnum()


def get_config_value(key: str, default: Any = None) -> Any:
    """
    Get configuration value.

    Args:
        key: Config key (dot notation supported, e.g., "claude.model")
        default: Default value if not found

    Returns:
        Configuration value
    """
    from kosmos.config import get_config

    config = get_config()

    # Handle dot notation
    parts = key.split(".")
    value = config
    for part in parts:
        if hasattr(value, part):
            value = getattr(value, part)
        else:
            return default

    return value


def format_hypothesis_summary(hypothesis: Dict) -> str:
    """Format hypothesis for display."""
    claim = hypothesis.get("claim", "Unknown")
    novelty = hypothesis.get("novelty_score", 0.0)
    priority = hypothesis.get("priority_score", 0.0)

    return (
        f"{truncate_text(claim, 60)}\n"
        f"  Novelty: {novelty:.2f} | Priority: {priority:.2f}"
    )


def format_experiment_summary(experiment: Dict) -> str:
    """Format experiment for display."""
    exp_type = experiment.get("type", "Unknown")
    status = experiment.get("status", "Unknown")
    duration = experiment.get("duration_seconds", 0)

    return (
        f"{exp_type} [{status}]\n"
        f"  Duration: {format_duration(duration)}"
    )


def confirm_action(message: str, default: bool = False) -> bool:
    """
    Prompt user for confirmation.

    Args:
        message: Confirmation message
        default: Default value if user just presses Enter

    Returns:
        True if user confirms
    """
    from rich.prompt import Confirm
    return Confirm.ask(message, default=default, console=console)


def get_cache_dir() -> Path:
    """Get cache directory path."""
    cache_dir = Path.home() / ".kosmos_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def get_log_dir() -> Path:
    """Get logs directory path."""
    log_dir = Path.home() / ".kosmos" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir
