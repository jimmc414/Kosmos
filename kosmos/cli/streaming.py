"""
CLI Streaming Display.

Rich-based real-time display for streaming events during research workflows.
Shows LLM token streaming, progress updates, and stage tracking.
"""

import logging
from typing import List, Optional

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.text import Text
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    Console = None
    Live = None
    Panel = None
    Progress = None
    Text = None
    Table = None

from kosmos.core.event_bus import get_event_bus, EventSubscription
from kosmos.core.events import (
    StreamingEvent,
    EventType,
    WorkflowEvent,
    CycleEvent,
    LLMEvent,
    StageEvent,
    TaskEvent,
)

logger = logging.getLogger(__name__)


class StreamingDisplay:
    """
    Rich-based streaming display for CLI.

    Provides real-time visualization of:
    - LLM token streaming
    - Workflow progress
    - Cycle and task status
    - Stage tracking

    Example:
        display = StreamingDisplay(console, "research_abc123")
        display.start()

        # ... run workflow ...

        display.stop()
    """

    def __init__(
        self,
        console: Optional["Console"] = None,
        process_id: Optional[str] = None,
        show_tokens: bool = True,
        show_stages: bool = True,
        max_token_display: int = 500
    ):
        """
        Initialize streaming display.

        Args:
            console: Rich Console instance (created if not provided)
            process_id: Optional process ID to filter events
            show_tokens: Whether to display streaming LLM tokens
            show_stages: Whether to display stage events
            max_token_display: Maximum tokens to keep in display buffer
        """
        if not HAS_RICH:
            raise ImportError(
                "Rich library required for StreamingDisplay. "
                "Install with: pip install rich"
            )

        self.console = console or Console()
        self.process_id = process_id
        self.show_tokens = show_tokens
        self.show_stages = show_stages
        self.max_token_display = max_token_display

        # Display state
        self._current_tokens: List[str] = []
        self._current_stage: Optional[str] = None
        self._current_cycle: int = 0
        self._max_cycles: int = 0
        self._findings_count: int = 0
        self._subscribed = False

        # Event tracking
        self._llm_calls_count: int = 0
        self._tasks_completed: int = 0
        self._stages_completed: int = 0

    def start(self) -> None:
        """
        Start listening to events and displaying.

        Subscribes to the global event bus for the configured process.
        """
        if self._subscribed:
            return

        event_bus = get_event_bus()

        # Subscribe to all relevant event types
        event_types = [
            EventType.WORKFLOW_STARTED,
            EventType.WORKFLOW_PROGRESS,
            EventType.WORKFLOW_COMPLETED,
            EventType.CYCLE_STARTED,
            EventType.CYCLE_COMPLETED,
            EventType.STAGE_STARTED,
            EventType.STAGE_COMPLETED,
            EventType.LLM_TOKEN,
            EventType.LLM_CALL_STARTED,
            EventType.LLM_CALL_COMPLETED,
            EventType.TASK_STARTED,
            EventType.TASK_COMPLETED,
        ]

        process_ids = [self.process_id] if self.process_id else None
        event_bus.subscribe(self._handle_event, event_types, process_ids)
        self._subscribed = True

        logger.debug(f"StreamingDisplay started for process: {self.process_id}")

    def stop(self) -> None:
        """
        Stop listening to events.

        Unsubscribes from the event bus.
        """
        if not self._subscribed:
            return

        get_event_bus().unsubscribe(self._handle_event)
        self._subscribed = False

        logger.debug("StreamingDisplay stopped")

    def _handle_event(self, event: StreamingEvent) -> None:
        """
        Handle incoming streaming event.

        Routes event to appropriate handler based on type.

        Args:
            event: The streaming event to handle
        """
        if not hasattr(event, 'type'):
            return

        handlers = {
            EventType.WORKFLOW_STARTED: self._on_workflow_started,
            EventType.WORKFLOW_PROGRESS: self._on_workflow_progress,
            EventType.WORKFLOW_COMPLETED: self._on_workflow_completed,
            EventType.CYCLE_STARTED: self._on_cycle_started,
            EventType.CYCLE_COMPLETED: self._on_cycle_completed,
            EventType.STAGE_STARTED: self._on_stage_started,
            EventType.STAGE_COMPLETED: self._on_stage_completed,
            EventType.LLM_TOKEN: self._on_llm_token,
            EventType.LLM_CALL_STARTED: self._on_llm_call_started,
            EventType.LLM_CALL_COMPLETED: self._on_llm_call_completed,
            EventType.TASK_STARTED: self._on_task_started,
            EventType.TASK_COMPLETED: self._on_task_completed,
        }

        handler = handlers.get(event.type)
        if handler:
            try:
                handler(event)
            except Exception as e:
                logger.debug(f"Error handling event {event.type}: {e}")

    def _on_workflow_started(self, event: WorkflowEvent) -> None:
        """Handle workflow started event."""
        self._max_cycles = event.max_cycles
        self._current_cycle = 0

        self.console.print()
        self.console.rule("[bold blue]Research Workflow Started")
        if event.research_question:
            self.console.print(
                f"[dim]Objective:[/dim] {event.research_question[:100]}"
            )
        self.console.print(f"[dim]Cycles:[/dim] {event.max_cycles}")
        self.console.print()

    def _on_workflow_progress(self, event: WorkflowEvent) -> None:
        """Handle workflow progress event."""
        self._current_cycle = event.cycle
        self._findings_count = event.findings_count

    def _on_workflow_completed(self, event: WorkflowEvent) -> None:
        """Handle workflow completed event."""
        self.console.print()
        self.console.rule("[bold green]Research Workflow Completed")
        self.console.print(
            f"[dim]Findings:[/dim] {event.findings_count} "
            f"([green]{event.validated_count}[/green] validated)"
        )
        self.console.print()

    def _on_cycle_started(self, event: CycleEvent) -> None:
        """Handle cycle started event."""
        self._current_cycle = event.cycle
        self.console.print(
            f"\n[bold cyan]Cycle {event.cycle}/{event.max_cycles}[/bold cyan]"
        )

    def _on_cycle_completed(self, event: CycleEvent) -> None:
        """Handle cycle completed event."""
        duration_str = f" ({event.duration_ms}ms)" if event.duration_ms else ""
        self.console.print(
            f"  [green]Completed:[/green] "
            f"{event.completed_tasks}/{event.tasks_count} tasks, "
            f"{event.findings_count} findings{duration_str}"
        )

    def _on_stage_started(self, event: StageEvent) -> None:
        """Handle stage started event."""
        if not self.show_stages:
            return

        self._current_stage = event.stage
        stage_name = event.stage.replace("_", " ").title()
        self.console.print(f"  [dim]> {stage_name}...[/dim]", end="")

    def _on_stage_completed(self, event: StageEvent) -> None:
        """Handle stage completed event."""
        if not self.show_stages:
            return

        self._stages_completed += 1
        duration_str = f" ({event.duration_ms}ms)" if event.duration_ms else ""
        self.console.print(f" [green]done[/green]{duration_str}")
        self._current_stage = None

    def _on_llm_token(self, event: LLMEvent) -> None:
        """Handle LLM token streaming event."""
        if not self.show_tokens or not event.token:
            return

        # Buffer tokens
        self._current_tokens.append(event.token)

        # Trim buffer if too long
        if len(self._current_tokens) > self.max_token_display:
            self._current_tokens = self._current_tokens[-self.max_token_display:]

        # Print token without newline for streaming effect
        self.console.print(event.token, end="", highlight=False)

    def _on_llm_call_started(self, event: LLMEvent) -> None:
        """Handle LLM call started event."""
        self._llm_calls_count += 1
        self._current_tokens = []

        if self.show_tokens:
            self.console.print(f"\n    [dim]LLM ({event.model}):[/dim] ", end="")

    def _on_llm_call_completed(self, event: LLMEvent) -> None:
        """Handle LLM call completed event."""
        if self.show_tokens:
            duration_str = f" ({event.duration_ms}ms)" if event.duration_ms else ""
            self.console.print(f" [dim]{duration_str}[/dim]")
            self._current_tokens = []

    def _on_task_started(self, event: TaskEvent) -> None:
        """Handle task started event."""
        if event.description:
            desc = event.description[:60] + "..." if len(event.description) > 60 else event.description
            self.console.print(f"    [dim]Task:[/dim] {desc}")

    def _on_task_completed(self, event: TaskEvent) -> None:
        """Handle task completed event."""
        self._tasks_completed += 1
        status = "[green]done[/green]" if event.success else "[red]failed[/red]"
        duration_str = f" ({event.duration_ms}ms)" if event.duration_ms else ""
        self.console.print(f"      {status}{duration_str}")

    def get_stats(self) -> dict:
        """
        Get display statistics.

        Returns:
            Dictionary with event counts
        """
        return {
            "llm_calls": self._llm_calls_count,
            "tasks_completed": self._tasks_completed,
            "stages_completed": self._stages_completed,
            "current_cycle": self._current_cycle,
            "max_cycles": self._max_cycles,
            "findings_count": self._findings_count,
        }


class SimpleStreamingDisplay:
    """
    Simplified streaming display without Rich dependency.

    Falls back to basic print statements for environments without Rich.
    """

    def __init__(self, process_id: Optional[str] = None):
        """Initialize simple display."""
        self.process_id = process_id
        self._subscribed = False

    def start(self) -> None:
        """Start listening to events."""
        if self._subscribed:
            return

        event_bus = get_event_bus()
        process_ids = [self.process_id] if self.process_id else None
        event_bus.subscribe(self._handle_event, process_ids=process_ids)
        self._subscribed = True

    def stop(self) -> None:
        """Stop listening to events."""
        if not self._subscribed:
            return

        get_event_bus().unsubscribe(self._handle_event)
        self._subscribed = False

    def _handle_event(self, event: StreamingEvent) -> None:
        """Handle events with simple print."""
        if hasattr(event, 'type'):
            if event.type == EventType.LLM_TOKEN and hasattr(event, 'token'):
                print(event.token, end="", flush=True)
            elif event.type == EventType.WORKFLOW_STARTED:
                print(f"\n=== Workflow Started ===")
            elif event.type == EventType.WORKFLOW_COMPLETED:
                print(f"\n=== Workflow Completed ===")
            elif event.type == EventType.CYCLE_STARTED:
                if hasattr(event, 'cycle'):
                    print(f"\n--- Cycle {event.cycle} ---")
            elif event.type == EventType.CYCLE_COMPLETED:
                if hasattr(event, 'cycle'):
                    print(f"--- Cycle {event.cycle} Complete ---")


def create_streaming_display(
    console: Optional["Console"] = None,
    process_id: Optional[str] = None,
    **kwargs
) -> StreamingDisplay:
    """
    Factory function to create appropriate streaming display.

    Args:
        console: Rich Console instance
        process_id: Process ID to filter
        **kwargs: Additional arguments for StreamingDisplay

    Returns:
        StreamingDisplay or SimpleStreamingDisplay instance
    """
    if HAS_RICH:
        return StreamingDisplay(console, process_id, **kwargs)
    else:
        return SimpleStreamingDisplay(process_id)
