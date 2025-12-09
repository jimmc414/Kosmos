"""
Unit tests for streaming events module.
"""

import json
import pytest
from datetime import datetime

from kosmos.core.events import (
    EventType,
    BaseEvent,
    WorkflowEvent,
    CycleEvent,
    TaskEvent,
    LLMEvent,
    ExecutionEvent,
    StageEvent,
    parse_event,
    EVENT_CLASS_MAP,
)


class TestEventType:
    """Tests for EventType enum."""

    def test_event_type_values(self):
        """Event types have expected string values."""
        assert EventType.WORKFLOW_STARTED.value == "workflow.started"
        assert EventType.LLM_TOKEN.value == "llm.token"
        assert EventType.STAGE_COMPLETED.value == "stage.completed"

    def test_event_type_from_string(self):
        """Event types can be created from string values."""
        assert EventType("workflow.started") == EventType.WORKFLOW_STARTED
        assert EventType("llm.token") == EventType.LLM_TOKEN

    def test_invalid_event_type_raises(self):
        """Invalid event type strings raise ValueError."""
        with pytest.raises(ValueError):
            EventType("invalid.event")

    def test_all_event_types_in_class_map(self):
        """All event types have a corresponding class mapping."""
        for event_type in EventType:
            assert event_type in EVENT_CLASS_MAP


class TestBaseEvent:
    """Tests for BaseEvent class."""

    def test_base_event_creation(self):
        """BaseEvent can be created with required fields."""
        event = BaseEvent(type=EventType.WORKFLOW_STARTED)
        assert event.type == EventType.WORKFLOW_STARTED
        assert event.timestamp is not None
        assert event.process_id is None
        assert event.correlation_id is None

    def test_base_event_with_all_fields(self):
        """BaseEvent can be created with all fields."""
        event = BaseEvent(
            type=EventType.WORKFLOW_STARTED,
            timestamp="2025-12-09T00:00:00Z",
            process_id="proc_123",
            correlation_id="corr_456"
        )
        assert event.process_id == "proc_123"
        assert event.correlation_id == "corr_456"

    def test_to_dict(self):
        """Event can be converted to dictionary."""
        event = BaseEvent(
            type=EventType.WORKFLOW_STARTED,
            process_id="test_proc"
        )
        data = event.to_dict()
        assert data['type'] == "workflow.started"
        assert data['process_id'] == "test_proc"
        assert 'timestamp' in data

    def test_to_json(self):
        """Event can be converted to JSON."""
        event = BaseEvent(
            type=EventType.WORKFLOW_STARTED,
            process_id="test_proc"
        )
        json_str = event.to_json()
        data = json.loads(json_str)
        assert data['type'] == "workflow.started"
        assert data['process_id'] == "test_proc"

    def test_from_dict(self):
        """Event can be created from dictionary."""
        data = {
            'type': 'workflow.started',
            'process_id': 'test_proc',
            'timestamp': '2025-12-09T00:00:00Z'
        }
        event = BaseEvent.from_dict(data)
        assert event.type == EventType.WORKFLOW_STARTED
        assert event.process_id == "test_proc"


class TestWorkflowEvent:
    """Tests for WorkflowEvent class."""

    def test_workflow_event_creation(self):
        """WorkflowEvent can be created."""
        event = WorkflowEvent(
            type=EventType.WORKFLOW_STARTED,
            research_question="Test question",
            domain="biology"
        )
        assert event.type == EventType.WORKFLOW_STARTED
        assert event.research_question == "Test question"
        assert event.domain == "biology"

    def test_workflow_event_defaults(self):
        """WorkflowEvent has correct defaults."""
        event = WorkflowEvent(type=EventType.WORKFLOW_STARTED)
        assert event.research_question == ""
        assert event.domain is None
        assert event.state == ""
        assert event.cycle == 0
        assert event.max_cycles == 0
        assert event.progress_percent == 0.0
        assert event.findings_count == 0
        assert event.validated_count == 0

    def test_workflow_event_progress(self):
        """WorkflowEvent tracks progress."""
        event = WorkflowEvent(
            type=EventType.WORKFLOW_PROGRESS,
            cycle=3,
            max_cycles=10,
            progress_percent=30.0,
            findings_count=5,
            validated_count=2
        )
        assert event.cycle == 3
        assert event.max_cycles == 10
        assert event.progress_percent == 30.0
        assert event.findings_count == 5
        assert event.validated_count == 2

    def test_workflow_event_to_dict(self):
        """WorkflowEvent serializes correctly."""
        event = WorkflowEvent(
            type=EventType.WORKFLOW_COMPLETED,
            research_question="Test",
            findings_count=10
        )
        data = event.to_dict()
        assert data['type'] == "workflow.completed"
        assert data['research_question'] == "Test"
        assert data['findings_count'] == 10


class TestCycleEvent:
    """Tests for CycleEvent class."""

    def test_cycle_event_creation(self):
        """CycleEvent can be created."""
        event = CycleEvent(
            type=EventType.CYCLE_STARTED,
            cycle=1,
            max_cycles=5
        )
        assert event.type == EventType.CYCLE_STARTED
        assert event.cycle == 1
        assert event.max_cycles == 5

    def test_cycle_event_completion(self):
        """CycleEvent tracks completion data."""
        event = CycleEvent(
            type=EventType.CYCLE_COMPLETED,
            cycle=2,
            tasks_count=10,
            completed_tasks=8,
            findings_count=3,
            duration_ms=5000
        )
        assert event.tasks_count == 10
        assert event.completed_tasks == 8
        assert event.findings_count == 3
        assert event.duration_ms == 5000


class TestTaskEvent:
    """Tests for TaskEvent class."""

    def test_task_event_creation(self):
        """TaskEvent can be created."""
        event = TaskEvent(
            type=EventType.TASK_STARTED,
            task_id="task_001",
            task_type="hypothesis_generation",
            description="Generate hypotheses"
        )
        assert event.type == EventType.TASK_STARTED
        assert event.task_id == "task_001"
        assert event.task_type == "hypothesis_generation"

    def test_task_event_completion(self):
        """TaskEvent tracks completion."""
        event = TaskEvent(
            type=EventType.TASK_COMPLETED,
            task_id="task_001",
            success=True,
            result_summary="Generated 3 hypotheses",
            duration_ms=2500
        )
        assert event.success is True
        assert event.result_summary == "Generated 3 hypotheses"
        assert event.duration_ms == 2500

    def test_task_event_failure(self):
        """TaskEvent tracks failure."""
        event = TaskEvent(
            type=EventType.TASK_FAILED,
            task_id="task_001",
            success=False,
            error="Connection timeout"
        )
        assert event.success is False
        assert event.error == "Connection timeout"


class TestLLMEvent:
    """Tests for LLMEvent class."""

    def test_llm_call_started(self):
        """LLMEvent tracks call start."""
        event = LLMEvent(
            type=EventType.LLM_CALL_STARTED,
            model="claude-3-5-sonnet",
            provider="anthropic"
        )
        assert event.type == EventType.LLM_CALL_STARTED
        assert event.model == "claude-3-5-sonnet"
        assert event.provider == "anthropic"

    def test_llm_token_event(self):
        """LLMEvent tracks streaming tokens."""
        event = LLMEvent(
            type=EventType.LLM_TOKEN,
            model="claude-3-5-sonnet",
            token="Hello"
        )
        assert event.type == EventType.LLM_TOKEN
        assert event.token == "Hello"

    def test_llm_call_completed(self):
        """LLMEvent tracks call completion."""
        event = LLMEvent(
            type=EventType.LLM_CALL_COMPLETED,
            model="claude-3-5-sonnet",
            provider="anthropic",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01,
            duration_ms=3000
        )
        assert event.prompt_tokens == 100
        assert event.completion_tokens == 50
        assert event.total_tokens == 150
        assert event.cost_usd == 0.01
        assert event.duration_ms == 3000


class TestExecutionEvent:
    """Tests for ExecutionEvent class."""

    def test_execution_started(self):
        """ExecutionEvent tracks execution start."""
        event = ExecutionEvent(
            type=EventType.CODE_EXECUTING,
            code_hash="abc123",
            language="python"
        )
        assert event.type == EventType.CODE_EXECUTING
        assert event.code_hash == "abc123"
        assert event.language == "python"

    def test_execution_output(self):
        """ExecutionEvent tracks output streaming."""
        event = ExecutionEvent(
            type=EventType.CODE_OUTPUT,
            code_hash="abc123",
            output_line="Processing data...",
            output_source="stdout"
        )
        assert event.type == EventType.CODE_OUTPUT
        assert event.output_line == "Processing data..."
        assert event.output_source == "stdout"

    def test_execution_completed(self):
        """ExecutionEvent tracks completion."""
        event = ExecutionEvent(
            type=EventType.CODE_COMPLETED,
            code_hash="abc123",
            exit_code=0,
            duration_ms=5000
        )
        assert event.exit_code == 0
        assert event.duration_ms == 5000

    def test_execution_failed(self):
        """ExecutionEvent tracks failure."""
        event = ExecutionEvent(
            type=EventType.CODE_FAILED,
            code_hash="abc123",
            exit_code=1,
            error="SyntaxError: invalid syntax"
        )
        assert event.exit_code == 1
        assert event.error == "SyntaxError: invalid syntax"


class TestStageEvent:
    """Tests for StageEvent class."""

    def test_stage_started(self):
        """StageEvent tracks stage start."""
        event = StageEvent(
            type=EventType.STAGE_STARTED,
            stage="GENERATE_HYPOTHESIS",
            iteration=1
        )
        assert event.type == EventType.STAGE_STARTED
        assert event.stage == "GENERATE_HYPOTHESIS"
        assert event.iteration == 1

    def test_stage_with_substage(self):
        """StageEvent supports substages."""
        event = StageEvent(
            type=EventType.STAGE_STARTED,
            stage="EXECUTE_EXPERIMENT",
            substage="code_generation",
            parent_stage="MAIN_LOOP"
        )
        assert event.substage == "code_generation"
        assert event.parent_stage == "MAIN_LOOP"

    def test_stage_completed(self):
        """StageEvent tracks completion."""
        event = StageEvent(
            type=EventType.STAGE_COMPLETED,
            stage="GENERATE_HYPOTHESIS",
            status="completed",
            duration_ms=1500,
            output_summary="Generated 5 hypotheses"
        )
        assert event.status == "completed"
        assert event.duration_ms == 1500
        assert event.output_summary == "Generated 5 hypotheses"

    def test_stage_with_metadata(self):
        """StageEvent supports metadata."""
        event = StageEvent(
            type=EventType.STAGE_COMPLETED,
            stage="VALIDATE_FINDINGS",
            metadata={"validated": 3, "rejected": 1}
        )
        assert event.metadata == {"validated": 3, "rejected": 1}


class TestParseEvent:
    """Tests for parse_event function."""

    def test_parse_workflow_event(self):
        """Parse WorkflowEvent from dict."""
        data = {
            'type': 'workflow.started',
            'research_question': 'Test question',
            'process_id': 'proc_123'
        }
        event = parse_event(data)
        assert isinstance(event, WorkflowEvent)
        assert event.research_question == 'Test question'

    def test_parse_llm_event(self):
        """Parse LLMEvent from dict."""
        data = {
            'type': 'llm.token',
            'model': 'claude-3-5-sonnet',
            'token': 'Hello'
        }
        event = parse_event(data)
        assert isinstance(event, LLMEvent)
        assert event.token == 'Hello'

    def test_parse_stage_event(self):
        """Parse StageEvent from dict."""
        data = {
            'type': 'stage.completed',
            'stage': 'TEST_STAGE',
            'duration_ms': 1000
        }
        event = parse_event(data)
        assert isinstance(event, StageEvent)
        assert event.stage == 'TEST_STAGE'

    def test_parse_missing_type_raises(self):
        """Parse raises on missing type."""
        with pytest.raises(ValueError, match="missing 'type'"):
            parse_event({})

    def test_parse_invalid_type_raises(self):
        """Parse raises on invalid type."""
        with pytest.raises(ValueError, match="Unknown event type"):
            parse_event({'type': 'invalid.type'})

    def test_parse_ignores_extra_fields(self):
        """Parse ignores extra fields not in dataclass."""
        data = {
            'type': 'workflow.started',
            'research_question': 'Test',
            'extra_field': 'ignored'
        }
        event = parse_event(data)
        assert isinstance(event, WorkflowEvent)
        assert not hasattr(event, 'extra_field')


class TestEventTimestamp:
    """Tests for automatic timestamp generation."""

    def test_timestamp_auto_generated(self):
        """Timestamp is auto-generated if not provided."""
        event = BaseEvent(type=EventType.WORKFLOW_STARTED)
        assert event.timestamp is not None
        assert event.timestamp.endswith('Z')

    def test_timestamp_can_be_overridden(self):
        """Timestamp can be explicitly set."""
        ts = "2025-01-01T00:00:00Z"
        event = BaseEvent(type=EventType.WORKFLOW_STARTED, timestamp=ts)
        assert event.timestamp == ts

    def test_timestamp_format_is_iso(self):
        """Timestamp is in ISO format."""
        event = BaseEvent(type=EventType.WORKFLOW_STARTED)
        # Should parse without error
        ts = event.timestamp.rstrip('Z')
        datetime.fromisoformat(ts)
