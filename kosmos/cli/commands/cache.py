"""
Cache command for Kosmos CLI.

Manage caching system - view stats, clear cache, health check.
"""

from typing import Optional

import typer
from rich.panel import Panel
from rich.progress import Progress, BarColumn
from rich.text import Text

from kosmos.cli.utils import (
    console,
    print_success,
    print_error,
    print_info,
    get_icon,
    create_table,
    create_metric_text,
    format_size,
    format_currency,
    confirm_action,
)


def manage_cache(
    stats: bool = typer.Option(False, "--stats", "-s", help="Show cache statistics"),
    clear: bool = typer.Option(False, "--clear", "-c", help="Clear all caches"),
    clear_type: Optional[str] = typer.Option(None, "--clear-type", help="Clear specific cache type (claude, experiment, embedding, general)"),
    health: bool = typer.Option(False, "--health", "-h", help="Run health check"),
    optimize: bool = typer.Option(False, "--optimize", "-o", help="Optimize caches (cleanup expired)"),
):
    """
    Manage caching system.

    Examples:

        # Show cache statistics
        kosmos cache --stats

        # Clear all caches
        kosmos cache --clear

        # Clear specific cache
        kosmos cache --clear-type claude

        # Run health check
        kosmos cache --health

        # Optimize caches
        kosmos cache --optimize
    """
    try:
        from kosmos.core.cache_manager import get_cache_manager

        cache_manager = get_cache_manager()

        # Default to showing stats if no options
        if not (stats or clear or clear_type or health or optimize):
            stats = True

        # Show statistics
        if stats:
            display_cache_stats(cache_manager)

        # Health check
        if health:
            display_health_check(cache_manager)

        # Optimize
        if optimize:
            optimize_caches(cache_manager)

        # Clear caches
        if clear:
            clear_all_caches(cache_manager)

        if clear_type:
            clear_specific_cache(cache_manager, clear_type)

    except KeyboardInterrupt:
        console.print("\n[warning]Cache operation cancelled[/warning]")
        raise typer.Exit(130)

    except Exception as e:
        print_error(f"Cache operation failed: {str(e)}")
        raise typer.Exit(1)


def display_cache_stats(cache_manager):
    """Display cache statistics."""
    console.print()
    console.print(f"[h2]{get_icon('info')} Cache Statistics[/h2]", justify="center")
    console.print()

    # Get stats from cache manager
    stats = cache_manager.get_stats()

    # Overall statistics table
    overall_table = create_table(
        title="Overall Cache Performance",
        columns=["Metric", "Value"],
        show_lines=True,
    )

    total_hits = sum(s.get("hits", 0) for s in stats.values())
    total_misses = sum(s.get("misses", 0) for s in stats.values())
    total_requests = total_hits + total_misses
    hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0

    overall_table.add_row("Total Requests", str(total_requests))
    overall_table.add_row("Cache Hits", str(total_hits))
    overall_table.add_row("Cache Misses", str(total_misses))
    overall_table.add_row(
        "Hit Rate",
        create_metric_text(hit_rate / 100, format_type="percentage")
    )

    console.print(overall_table)
    console.print()

    # Per-cache statistics
    cache_table = create_table(
        title="Cache Details",
        columns=["Cache Type", "Hits", "Misses", "Hit Rate", "Size", "Storage"],
        show_lines=False,
    )

    for cache_type, cache_stats in stats.items():
        hits = cache_stats.get("hits", 0)
        misses = cache_stats.get("misses", 0)
        total = hits + misses
        rate = (hits / total * 100) if total > 0 else 0
        size = cache_stats.get("size", 0)
        storage = cache_stats.get("storage_size_mb", 0)

        cache_table.add_row(
            cache_type.title(),
            str(hits),
            str(misses),
            f"{rate:.1f}%",
            str(size),
            format_size(storage * 1024 * 1024),
        )

    console.print(cache_table)
    console.print()

    # Cost savings estimate
    estimate_cost_savings(total_hits)


def estimate_cost_savings(cache_hits: int):
    """Estimate cost savings from cache hits."""
    # Rough estimate: ~$0.003 per hit saved (average Sonnet request)
    avg_cost_per_request = 0.003
    estimated_savings = cache_hits * avg_cost_per_request

    savings_panel = Panel(
        f"[success]Estimated Cost Savings: {format_currency(estimated_savings)}[/success]\n"
        f"[muted]Based on {cache_hits} cache hits × ${avg_cost_per_request} avg cost[/muted]",
        title=f"[green]{get_icon('sparkle')} Cost Savings[/green]",
        border_style="green",
    )

    console.print(savings_panel)
    console.print()


def display_health_check(cache_manager):
    """Run and display cache health check."""
    console.print()
    console.print(f"[h2]{get_icon('flask')} Running Health Check[/h2]", justify="center")
    console.print()

    # Run health check
    health_results = cache_manager.health_check()

    # Display results
    table = create_table(
        title="Health Check Results",
        columns=["Cache", "Status", "Details"],
        show_lines=True,
    )

    for cache_type, result in health_results.items():
        status = "[success]✓ Healthy[/success]" if result["healthy"] else "[error]✗ Unhealthy[/error]"
        details = result.get("details", "OK")

        table.add_row(cache_type.title(), status, details)

    console.print(table)
    console.print()

    # Overall status
    all_healthy = all(r["healthy"] for r in health_results.values())
    if all_healthy:
        print_success("All caches are healthy!", title="Health Check Complete")
    else:
        print_error("Some caches have issues. Consider running --optimize", title="Health Check Failed")


def optimize_caches(cache_manager):
    """Optimize all caches."""
    console.print()
    console.print(f"[h2]{get_icon('rocket')} Optimizing Caches[/h2]", justify="center")
    console.print()

    # Run optimization
    with console.status("[cyan]Cleaning up expired entries...[/cyan]"):
        result = cache_manager.cleanup_expired()

    # Display results
    table = create_table(
        title="Optimization Results",
        columns=["Cache", "Entries Removed"],
        show_lines=True,
    )

    total_removed = 0
    for cache_type, removed in result.items():
        table.add_row(cache_type.title(), str(removed))
        total_removed += removed

    console.print(table)
    console.print()

    print_success(f"Removed {total_removed} expired entries", title="Optimization Complete")


def clear_all_caches(cache_manager):
    """Clear all caches with confirmation."""
    console.print()

    if not confirm_action("Are you sure you want to clear ALL caches? This cannot be undone."):
        console.print("[warning]Operation cancelled[/warning]")
        return

    with console.status("[yellow]Clearing all caches...[/yellow]"):
        cache_manager.clear()

    print_success("All caches cleared successfully", title="Caches Cleared")


def clear_specific_cache(cache_manager, cache_type: str):
    """Clear a specific cache type."""
    console.print()

    # Validate cache type - must match CacheType enum values
    valid_types = ["claude", "experiment", "embedding", "general", "literature"]
    if cache_type.lower() not in valid_types:
        print_error(f"Invalid cache type. Must be one of: {', '.join(valid_types)}")
        raise typer.Exit(1)

    if not confirm_action(f"Are you sure you want to clear the {cache_type} cache?"):
        console.print("[warning]Operation cancelled[/warning]")
        return

    with console.status(f"[yellow]Clearing {cache_type} cache...[/yellow]"):
        from kosmos.core.cache_manager import CacheType

        # Handle both uppercase and lowercase cache types
        try:
            cache_type_enum = CacheType[cache_type.upper()]
        except KeyError:
            # Try the value directly if uppercase doesn't work
            cache_type_enum = CacheType(cache_type.lower())

        cache_manager.clear(cache_type_enum)

    print_success(f"{cache_type.title()} cache cleared successfully", title="Cache Cleared")


if __name__ == "__main__":
    typer.run(manage_cache)
