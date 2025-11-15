"""
Graph command for Kosmos CLI.

Manage knowledge graphs - view stats, export/import, reset.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel
from rich.text import Text

from kosmos.cli.utils import (
    console,
    print_success,
    print_error,
    print_info,
    get_icon,
    create_table,
    format_size,
    confirm_action,
)


def manage_graph(
    stats: bool = typer.Option(False, "--stats", "-s", help="Show knowledge graph statistics"),
    info: bool = typer.Option(False, "--info", "-i", help="Show knowledge graph statistics (alias for --stats)"),
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Export graph to JSON file"),
    import_file: Optional[str] = typer.Option(None, "--import", help="Import graph from JSON file"),
    clear: bool = typer.Option(False, "--clear", "-c", help="Clear graph before import (use with --import)"),
    reset: bool = typer.Option(False, "--reset", "-r", help="Clear all graph data (DANGEROUS)"),
):
    """
    Manage knowledge graphs.

    Examples:

        # Show graph statistics
        kosmos graph
        kosmos graph --stats

        # Export graph to file
        kosmos graph --export backup.json

        # Import graph from file
        kosmos graph --import backup.json

        # Clear and import
        kosmos graph --import backup.json --clear

        # Reset (clear all data)
        kosmos graph --reset
    """
    try:
        from kosmos.world_model import get_world_model

        wm = get_world_model()

        # Default to showing stats if no options
        if not (stats or info or export or import_file or reset):
            stats = True

        # Show statistics
        if stats or info:
            display_graph_stats(wm)

        # Export graph
        if export:
            export_graph(wm, export)

        # Import graph
        if import_file:
            import_graph(wm, import_file, clear)

        # Reset graph
        if reset:
            reset_graph(wm)

    except KeyboardInterrupt:
        console.print("\n[warning]Graph operation cancelled[/warning]")
        raise typer.Exit(130)

    except Exception as e:
        print_error(f"Graph operation failed: {str(e)}")
        raise typer.Exit(1)


def display_graph_stats(wm):
    """Display knowledge graph statistics."""
    console.print()
    console.print(f"[h2]{get_icon('graph')} Knowledge Graph Statistics[/h2]", justify="center")
    console.print()

    # Get stats from world model
    stats = wm.get_statistics()

    # Overall statistics table
    overall_table = create_table(
        title="Graph Overview",
        columns=["Metric", "Value"],
        show_lines=True,
    )

    overall_table.add_row("Entities", f"{stats.get('entity_count', 0):,}")
    overall_table.add_row("Relationships", f"{stats.get('relationship_count', 0):,}")
    overall_table.add_row("Annotations", f"{stats.get('annotation_count', 0):,}")

    console.print(overall_table)
    console.print()

    # Entity types breakdown
    entity_types = stats.get('entity_types', {})
    if entity_types:
        types_table = create_table(
            title="Entity Types",
            columns=["Type", "Count"],
            show_lines=False,
        )

        for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
            types_table.add_row(entity_type, f"{count:,}")

        console.print(types_table)
        console.print()

    # Relationship types breakdown
    relationship_types = stats.get('relationship_types', {})
    if relationship_types:
        rel_table = create_table(
            title="Relationship Types",
            columns=["Type", "Count"],
            show_lines=False,
        )

        for rel_type, count in sorted(relationship_types.items(), key=lambda x: x[1], reverse=True):
            rel_table.add_row(rel_type, f"{count:,}")

        console.print(rel_table)
        console.print()

    # Additional info
    if stats.get('entity_count', 0) == 0:
        console.print("[muted]No entities in graph. Run research queries to build knowledge.[/muted]")
        console.print()


def export_graph(wm, filepath: str):
    """Export knowledge graph to JSON file."""
    console.print()
    console.print(f"[h2]{get_icon('save')} Exporting Knowledge Graph[/h2]", justify="center")
    console.print()

    path = Path(filepath)

    # Create parent directory if needed
    if path.parent != Path('.'):
        path.parent.mkdir(parents=True, exist_ok=True)

    # Get stats before export
    stats = wm.get_statistics()
    entity_count = stats.get('entity_count', 0)
    relationship_count = stats.get('relationship_count', 0)

    # Export
    with console.status(f"[cyan]Exporting {entity_count:,} entities and {relationship_count:,} relationships...[/cyan]"):
        wm.export_graph(str(path))

    # Get file size
    file_size = path.stat().st_size if path.exists() else 0

    # Success message
    success_panel = Panel(
        f"[success]✓ Exported successfully[/success]\n\n"
        f"  Entities: {entity_count:,}\n"
        f"  Relationships: {relationship_count:,}\n"
        f"  File: {path}\n"
        f"  Size: {format_size(file_size)}",
        title=f"[green]{get_icon('check')} Export Complete[/green]",
        border_style="green",
    )

    console.print(success_panel)
    console.print()

    print_info(
        f"Use 'kosmos graph --import {filepath}' to restore this graph",
        title="Next Steps"
    )


def import_graph(wm, filepath: str, clear: bool = False):
    """Import knowledge graph from JSON file."""
    console.print()
    console.print(f"[h2]{get_icon('upload')} Importing Knowledge Graph[/h2]", justify="center")
    console.print()

    path = Path(filepath)

    # Check file exists
    if not path.exists():
        print_error(f"File not found: {filepath}")
        raise typer.Exit(1)

    # Check file is readable
    if not path.is_file():
        print_error(f"Not a file: {filepath}")
        raise typer.Exit(1)

    # Warn if clearing
    if clear:
        if not confirm_action(
            "⚠️  Clear existing graph before import? This will DELETE all current data."
        ):
            console.print("[warning]Import cancelled[/warning]")
            return

    # Get current stats
    current_stats = wm.get_statistics()
    current_entities = current_stats.get('entity_count', 0)

    # Import
    mode_str = "Clearing and importing" if clear else "Importing"
    with console.status(f"[cyan]{mode_str} from {path.name}...[/cyan]"):
        wm.import_graph(str(path), clear=clear)

    # Get new stats
    new_stats = wm.get_statistics()
    new_entities = new_stats.get('entity_count', 0)
    new_relationships = new_stats.get('relationship_count', 0)

    # Calculate delta
    if clear:
        added = new_entities
        delta_text = f"Imported {added:,} entities"
    else:
        added = new_entities - current_entities
        delta_text = f"Added {added:,} entities" if added > 0 else "No new entities"

    # Success message
    success_panel = Panel(
        f"[success]✓ Import successful[/success]\n\n"
        f"  {delta_text}\n"
        f"  Total Entities: {new_entities:,}\n"
        f"  Total Relationships: {new_relationships:,}\n"
        f"  Mode: {'Replace' if clear else 'Append'}",
        title=f"[green]{get_icon('check')} Import Complete[/green]",
        border_style="green",
    )

    console.print(success_panel)
    console.print()


def reset_graph(wm):
    """Clear all knowledge graph data."""
    console.print()
    console.print(f"[h2]{get_icon('warning')} Reset Knowledge Graph[/h2]", justify="center")
    console.print()

    # Get current stats
    stats = wm.get_statistics()
    entity_count = stats.get('entity_count', 0)
    relationship_count = stats.get('relationship_count', 0)

    if entity_count == 0:
        console.print("[muted]Graph is already empty.[/muted]")
        console.print()
        return

    # Confirmation
    console.print(f"[warning]This will DELETE:[/warning]")
    console.print(f"  • {entity_count:,} entities")
    console.print(f"  • {relationship_count:,} relationships")
    console.print()

    if not confirm_action(
        "⚠️  Are you SURE you want to delete ALL graph data? This CANNOT be undone."
    ):
        console.print("[warning]Reset cancelled[/warning]")
        return

    # Double confirmation for safety
    if not confirm_action(
        "⚠️  FINAL WARNING: Delete all graph data permanently?"
    ):
        console.print("[warning]Reset cancelled[/warning]")
        return

    # Reset
    with console.status("[yellow]Deleting all graph data...[/yellow]"):
        wm.reset()

    # Success
    print_success(
        f"Deleted {entity_count:,} entities and {relationship_count:,} relationships",
        title="Reset Complete"
    )

    console.print()
    print_info(
        "Graph is now empty. Run research queries to rebuild knowledge.",
        title="Next Steps"
    )


if __name__ == "__main__":
    typer.run(manage_graph)
