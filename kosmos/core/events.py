"""
Streaming Event Types for Kosmos.

Defines typed events for real-time visibility into long-running operations.
Events are published via EventBus and consumed by CLI, API endpoints, and loggers.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union


class EventType(str, Enum):
    """All event types in the streaming system."""

    # Workflow lifecycle events
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_PROGRESS = "workflow.progress"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"

    # Research cycle events
    CYCLE_STARTED = "cycle.started"
    CYCLE_COMPLETED = "cycle.completed"
    CYCLE_FAILED = "cycle.failed"

    # Task events
    TASK_GENERATED = "task.generated"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"

    # LLM call events
    LLM_CALL_STARTED = "llm.call_started"
    LLM_TOKEN = "llm.token"
    LLM_CALL_COMPLETED = "llm.call_completed"
    LLM_CALL_FAILED = "llm.call_failed"

    # Code execution events
    CODE_VALIDATING = "execution.validating"
    CODE_EXECUTING = "execution.executing"
    CODE_OUTPUT = "execution.output"
    CODE_COMPLETED = "execution.completed"
    CODE_FAILED = "execution.failed"

    # Stage tracking events (from StageTracker)
    STAGE_STARTED = "stage.started"
    STAGE_COMPLETED = "stage.completed"
    STAGE_FAILED = "stage.failed"


def _default_timestamp() -> str:
    """Generate ISO timestamp with Z suffix."""
    return datetime.utcnow().isoformat() + "Z"


@dataclass
class BaseEvent:
    """
    Base class for all streaming events.

    All events share common fields for identification and correlation.
    """
    type: EventType
    timestamp: str = field(default_factory=_default_timestamp)
    process_id: Optional[str] = None
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        data = asdict(self)
        data['type'] = self.type.value
        return data

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseEvent":
        """Create event from dictionary."""
        data = data.copy()
        if isinstance(data.get('type'), str):
            data['type'] = EventType(data['type'])
        return cls(**data)


@dataclass
class WorkflowEvent(BaseEvent):
    """
    Workflow lifecycle events.

    Emitted at workflow start, progress updates, and completion.
    """
    research_question: str = ""
    domain: Optional[str] = None
    state: str = ""
    cycle: int = 0
    max_cycles: int = 0
    progress_percent: float = 0.0
    findings_count: int = 0
    validated_count: int = 0

    def __post_init__(self):
        if not isinstance(self.type, EventType):
            self.type = EventType(self.type)


@dataclass
class CycleEvent(BaseEvent):
    """
    Research cycle events.

    Emitted at cycle start and completion within a workflow.
    """
    cycle: int = 0
    max_cycles: int = 0
    tasks_count: int = 0
    completed_tasks: int = 0
    findings_count: int = 0
    duration_ms: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.type, EventType):
            self.type = EventType(self.type)


@dataclass
class TaskEvent(BaseEvent):
    """
    Task generation and execution events.

    Emitted when tasks are created, started, and completed.
    """
    task_id: str = ""
    task_type: str = ""
    description: str = ""
    priority: float = 0.0
    cycle: int = 0
    result_summary: Optional[str] = None
    success: Optional[bool] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.type, EventType):
            self.type = EventType(self.type)


@dataclass
class LLMEvent(BaseEvent):
    """
    LLM call events with optional token streaming.

    Emitted at call start, for each token (streaming), and at completion.
    """
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    token: Optional[str] = None  # For streaming individual tokens
    content: Optional[str] = None  # Full response for completion
    cost_usd: Optional[float] = None
    duration_ms: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.type, EventType):
            self.type = EventType(self.type)


@dataclass
class ExecutionEvent(BaseEvent):
    """
    Code execution events.

    Emitted during code validation, execution, and output streaming.
    """
    code_hash: str = ""  # Short hash for identification
    language: str = "python"
    output_line: Optional[str] = None  # For streaming stdout/stderr
    output_source: Literal["stdout", "stderr"] = "stdout"
    exit_code: Optional[int] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.type, EventType):
            self.type = EventType(self.type)


@dataclass
class StageEvent(BaseEvent):
    """
    Stage tracking events.

    Bridges with existing StageTracker infrastructure.
    """
    stage: str = ""
    substage: Optional[str] = None
    parent_stage: Optional[str] = None
    status: Literal["started", "completed", "failed", "skipped"] = "started"
    iteration: int = 0
    duration_ms: Optional[int] = None
    output_summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.type, EventType):
            self.type = EventType(self.type)


# Union type for all streaming events
StreamingEvent = Union[
    BaseEvent,
    WorkflowEvent,
    CycleEvent,
    TaskEvent,
    LLMEvent,
    ExecutionEvent,
    StageEvent
]


# Event type to class mapping for deserialization
EVENT_CLASS_MAP: Dict[EventType, type] = {
    EventType.WORKFLOW_STARTED: WorkflowEvent,
    EventType.WORKFLOW_PROGRESS: WorkflowEvent,
    EventType.WORKFLOW_COMPLETED: WorkflowEvent,
    EventType.WORKFLOW_FAILED: WorkflowEvent,
    EventType.CYCLE_STARTED: CycleEvent,
    EventType.CYCLE_COMPLETED: CycleEvent,
    EventType.CYCLE_FAILED: CycleEvent,
    EventType.TASK_GENERATED: TaskEvent,
    EventType.TASK_STARTED: TaskEvent,
    EventType.TASK_COMPLETED: TaskEvent,
    EventType.TASK_FAILED: TaskEvent,
    EventType.LLM_CALL_STARTED: LLMEvent,
    EventType.LLM_TOKEN: LLMEvent,
    EventType.LLM_CALL_COMPLETED: LLMEvent,
    EventType.LLM_CALL_FAILED: LLMEvent,
    EventType.CODE_VALIDATING: ExecutionEvent,
    EventType.CODE_EXECUTING: ExecutionEvent,
    EventType.CODE_OUTPUT: ExecutionEvent,
    EventType.CODE_COMPLETED: ExecutionEvent,
    EventType.CODE_FAILED: ExecutionEvent,
    EventType.STAGE_STARTED: StageEvent,
    EventType.STAGE_COMPLETED: StageEvent,
    EventType.STAGE_FAILED: StageEvent,
}


def parse_event(data: Dict[str, Any]) -> StreamingEvent:
    """
    Parse a dictionary into the appropriate event type.

    Args:
        data: Dictionary with event data including 'type' field

    Returns:
        Typed event instance

    Raises:
        ValueError: If event type is unknown
    """
    event_type_str = data.get('type')
    if not event_type_str:
        raise ValueError("Event data missing 'type' field")

    try:
        event_type = EventType(event_type_str)
    except ValueError:
        raise ValueError(f"Unknown event type: {event_type_str}")

    event_class = EVENT_CLASS_MAP.get(event_type, BaseEvent)

    # Filter data to only include fields the class accepts
    import inspect
    valid_fields = {f.name for f in event_class.__dataclass_fields__.values()}
    filtered_data = {k: v for k, v in data.items() if k in valid_fields}

    return event_class(**filtered_data)
