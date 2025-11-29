"""
Real-time stage tracking for debug observability.

Provides context managers and utilities for tracking multi-step research processes.
"""

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Literal, List

logger = logging.getLogger(__name__)


@dataclass
class StageEvent:
    """Represents a single stage event for tracking."""
    timestamp: str
    process_id: str
    stage: str
    status: Literal["started", "completed", "failed", "skipped"]
    duration_ms: Optional[int] = None
    iteration: int = 0
    substage: Optional[str] = None
    parent_stage: Optional[str] = None
    output_summary: Optional[str] = None
    error: Optional[Dict[str, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), default=str)


class StageTracker:
    """
    Context manager for tracking stage execution with timing.

    Usage:
        tracker = StageTracker(process_id="research_123")
        with tracker.track("GENERATE_HYPOTHESIS", hypothesis_count=5):
            # do work
            pass
    """

    def __init__(
        self,
        process_id: str,
        output_file: Optional[str] = None,
        emit_to_stdout: bool = False,
        enabled: bool = True
    ):
        self.process_id = process_id
        self.output_file = output_file or "logs/stages.jsonl"
        self.emit_to_stdout = emit_to_stdout
        self.enabled = enabled
        self._stage_stack: List[str] = []
        self.current_iteration = 0
        self._events: List[StageEvent] = []

        # Ensure output directory exists
        if self.enabled and self.output_file:
            Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)

    def set_iteration(self, iteration: int):
        """Update current iteration number."""
        self.current_iteration = iteration

    @contextmanager
    def track(self, stage: str, **metadata):
        """
        Track a stage with timing and status.

        Args:
            stage: Stage name (e.g., "GENERATE_HYPOTHESIS")
            **metadata: Additional metadata to include in event
        """
        if not self.enabled:
            yield None
            return

        start = time.time()
        event = StageEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            process_id=self.process_id,
            stage=stage,
            status="started",
            iteration=self.current_iteration,
            parent_stage=self._stage_stack[-1] if self._stage_stack else None,
            metadata=metadata
        )

        self._emit(event)
        self._stage_stack.append(stage)

        try:
            yield event
            event.status = "completed"
            event.duration_ms = int((time.time() - start) * 1000)
        except Exception as e:
            event.status = "failed"
            event.duration_ms = int((time.time() - start) * 1000)
            event.error = {
                "type": type(e).__name__,
                "message": str(e)[:500]
            }
            raise
        finally:
            self._stage_stack.pop()
            self._emit(event)
            self._events.append(event)

    def log_substage(self, substage: str, parent_stage: str, **metadata):
        """Log a substage event without context manager."""
        if not self.enabled:
            return

        event = StageEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            process_id=self.process_id,
            stage=parent_stage,
            substage=substage,
            status="completed",
            iteration=self.current_iteration,
            metadata=metadata
        )
        self._emit(event)

    def _emit(self, event: StageEvent):
        """Emit stage event to configured outputs."""
        event_json = event.to_json()

        if self.emit_to_stdout:
            print(f"[STAGE] {event_json}")

        if self.output_file:
            try:
                with open(self.output_file, "a") as f:
                    f.write(event_json + "\n")
            except Exception as e:
                logger.warning(f"Failed to write stage event: {e}")

    def get_events(self) -> List[StageEvent]:
        """Get all recorded events."""
        return self._events.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of tracked stages."""
        completed = [e for e in self._events if e.status == "completed"]
        failed = [e for e in self._events if e.status == "failed"]

        total_duration = sum(e.duration_ms or 0 for e in completed)

        return {
            "process_id": self.process_id,
            "total_stages": len(self._events),
            "completed": len(completed),
            "failed": len(failed),
            "total_duration_ms": total_duration,
            "iterations": self.current_iteration
        }


# Singleton instance
_tracker: Optional[StageTracker] = None


def get_stage_tracker(process_id: Optional[str] = None) -> StageTracker:
    """Get or create stage tracker singleton."""
    global _tracker

    if _tracker is None or (process_id and _tracker.process_id != process_id):
        # Check config for settings
        try:
            from kosmos.config import get_config
            config = get_config()
            _tracker = StageTracker(
                process_id=process_id or f"research_{int(time.time())}",
                output_file=config.logging.stage_tracking_file,
                enabled=config.logging.stage_tracking_enabled
            )
        except Exception:
            # Fallback if config not available
            _tracker = StageTracker(
                process_id=process_id or f"research_{int(time.time())}",
                enabled=False
            )

    return _tracker


def reset_stage_tracker():
    """Reset the stage tracker singleton (useful for testing)."""
    global _tracker
    _tracker = None
