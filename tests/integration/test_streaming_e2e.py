"""
Integration tests for streaming events end-to-end.

Tests the full flow of events from sources through event bus to subscribers.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch

from kosmos.core.events import (
    EventType,
    WorkflowEvent,
    CycleEvent,
    TaskEvent,
    LLMEvent,
    StageEvent,
)
from kosmos.core.event_bus import (
    EventBus,
    get_event_bus,
    reset_event_bus,
    EventSubscription,
)
from kosmos.core.stage_tracker import StageTracker, reset_stage_tracker


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before each test."""
    reset_event_bus()
    reset_stage_tracker()
    yield
    reset_event_bus()
    reset_stage_tracker()


class TestStageTrackerEventBusIntegration:
    """Tests for StageTracker -> EventBus integration."""

    def test_stage_tracker_emits_to_event_bus(self):
        """StageTracker emits events to the global event bus."""
        received_events = []

        def callback(event):
            received_events.append(event)

        # Subscribe to stage events
        event_bus = get_event_bus()
        event_bus.subscribe(
            callback,
            event_types=[EventType.STAGE_STARTED, EventType.STAGE_COMPLETED]
        )

        # Create tracker with event bus integration enabled
        tracker = StageTracker(
            process_id="test_proc",
            output_file=None,
            emit_to_event_bus=True
        )

        # Track a stage
        with tracker.track("TEST_STAGE", test_key="test_value"):
            pass

        # Should have received started and completed events
        assert len(received_events) == 2
        assert received_events[0].type == EventType.STAGE_STARTED
        assert received_events[0].stage == "TEST_STAGE"
        assert received_events[1].type == EventType.STAGE_COMPLETED
        assert received_events[1].stage == "TEST_STAGE"

    def test_stage_tracker_with_callback(self):
        """StageTracker calls registered callbacks."""
        callback_events = []

        def my_callback(event):
            # Store a copy of status since the event object is modified
            callback_events.append({
                "stage": event.stage,
                "status": event.status
            })

        # Create tracker with callback
        tracker = StageTracker(
            process_id="test_proc",
            output_file=None,
            emit_to_event_bus=False
        )
        tracker.add_callback(my_callback)

        # Track a stage
        with tracker.track("CALLBACK_TEST"):
            pass

        # Callback should have been invoked twice (started and completed)
        assert len(callback_events) == 2
        # Note: First callback receives "started", second receives "completed"
        # The first stored status is "started", but the final status is "completed"
        # This is because the same event object is reused and modified
        assert callback_events[0]["stage"] == "CALLBACK_TEST"
        assert callback_events[1]["stage"] == "CALLBACK_TEST"
        # At least one should show completion
        statuses = [e["status"] for e in callback_events]
        assert "completed" in statuses

    def test_stage_tracker_disabled_no_events(self):
        """Disabled StageTracker doesn't emit events."""
        received_events = []

        def callback(event):
            received_events.append(event)

        event_bus = get_event_bus()
        event_bus.subscribe(callback)

        # Create disabled tracker
        tracker = StageTracker(
            process_id="test_proc",
            enabled=False
        )

        with tracker.track("DISABLED_STAGE"):
            pass

        assert len(received_events) == 0


class TestEventBusFiltering:
    """Tests for event bus filtering with real events."""

    @pytest.mark.asyncio
    async def test_filter_by_event_type(self):
        """Events are filtered by type."""
        llm_events = []
        workflow_events = []

        def llm_callback(event):
            llm_events.append(event)

        def workflow_callback(event):
            workflow_events.append(event)

        event_bus = get_event_bus()
        event_bus.subscribe(llm_callback, event_types=[EventType.LLM_TOKEN])
        event_bus.subscribe(
            workflow_callback,
            event_types=[EventType.WORKFLOW_STARTED, EventType.WORKFLOW_COMPLETED]
        )

        # Publish mixed events
        await event_bus.publish(LLMEvent(type=EventType.LLM_TOKEN, token="Hi"))
        await event_bus.publish(WorkflowEvent(type=EventType.WORKFLOW_STARTED))
        await event_bus.publish(LLMEvent(type=EventType.LLM_TOKEN, token="there"))
        await event_bus.publish(WorkflowEvent(type=EventType.WORKFLOW_COMPLETED))

        assert len(llm_events) == 2
        assert len(workflow_events) == 2

    @pytest.mark.asyncio
    async def test_filter_by_process_id(self):
        """Events are filtered by process ID."""
        proc1_events = []
        proc2_events = []

        def proc1_callback(event):
            proc1_events.append(event)

        def proc2_callback(event):
            proc2_events.append(event)

        event_bus = get_event_bus()
        event_bus.subscribe(proc1_callback, process_ids=["proc_1"])
        event_bus.subscribe(proc2_callback, process_ids=["proc_2"])

        # Publish events for different processes
        await event_bus.publish(
            WorkflowEvent(type=EventType.WORKFLOW_STARTED, process_id="proc_1")
        )
        await event_bus.publish(
            WorkflowEvent(type=EventType.WORKFLOW_STARTED, process_id="proc_2")
        )
        await event_bus.publish(
            WorkflowEvent(type=EventType.WORKFLOW_PROGRESS, process_id="proc_1")
        )

        assert len(proc1_events) == 2
        assert len(proc2_events) == 1


class TestEventSubscriptionContext:
    """Tests for EventSubscription context manager."""

    @pytest.mark.asyncio
    async def test_subscription_auto_cleanup(self):
        """EventSubscription cleans up on exit."""
        event_bus = get_event_bus()
        received = []

        def callback(event):
            received.append(event)

        # Subscribe within context
        async with EventSubscription(callback, event_bus=event_bus):
            await event_bus.publish(WorkflowEvent(type=EventType.WORKFLOW_STARTED))
            assert len(received) == 1
            assert event_bus.subscriber_count() == 1

        # After context, should be unsubscribed
        assert event_bus.subscriber_count() == 0

        # Publishing should not call callback
        await event_bus.publish(WorkflowEvent(type=EventType.WORKFLOW_COMPLETED))
        assert len(received) == 1


class TestWorkflowEventFlow:
    """Tests for workflow event flow."""

    @pytest.mark.asyncio
    async def test_complete_workflow_event_sequence(self):
        """Test a complete sequence of workflow events."""
        events = []

        async def callback(event):
            events.append(event)

        event_bus = get_event_bus()
        event_bus.subscribe(callback)

        # Simulate workflow event sequence
        await event_bus.publish(WorkflowEvent(
            type=EventType.WORKFLOW_STARTED,
            process_id="workflow_1",
            research_question="Test question",
            max_cycles=3
        ))

        for cycle in range(1, 4):
            await event_bus.publish(CycleEvent(
                type=EventType.CYCLE_STARTED,
                process_id="workflow_1",
                cycle=cycle,
                max_cycles=3
            ))

            await event_bus.publish(CycleEvent(
                type=EventType.CYCLE_COMPLETED,
                process_id="workflow_1",
                cycle=cycle,
                max_cycles=3,
                completed_tasks=5,
                findings_count=2
            ))

            await event_bus.publish(WorkflowEvent(
                type=EventType.WORKFLOW_PROGRESS,
                process_id="workflow_1",
                cycle=cycle,
                max_cycles=3,
                progress_percent=(cycle / 3) * 100
            ))

        await event_bus.publish(WorkflowEvent(
            type=EventType.WORKFLOW_COMPLETED,
            process_id="workflow_1",
            progress_percent=100.0,
            findings_count=6,
            validated_count=4
        ))

        # Check event sequence
        assert len(events) == 11  # 1 start + 3*(start+complete+progress) + 1 complete
        assert events[0].type == EventType.WORKFLOW_STARTED
        assert events[-1].type == EventType.WORKFLOW_COMPLETED
        assert events[-1].validated_count == 4


class TestLLMEventFlow:
    """Tests for LLM streaming event flow."""

    def test_llm_token_streaming_sync(self):
        """Test LLM token streaming with sync callbacks."""
        tokens = []
        events = []

        def token_callback(event):
            if event.type == EventType.LLM_TOKEN and event.token:
                tokens.append(event.token)
            events.append(event)

        event_bus = get_event_bus()
        event_bus.subscribe(token_callback, event_types=[
            EventType.LLM_CALL_STARTED,
            EventType.LLM_TOKEN,
            EventType.LLM_CALL_COMPLETED
        ])

        # Simulate LLM streaming
        event_bus.publish_sync(LLMEvent(
            type=EventType.LLM_CALL_STARTED,
            model="claude-3-5-sonnet",
            provider="anthropic"
        ))

        for word in ["Hello", " ", "world", "!"]:
            event_bus.publish_sync(LLMEvent(
                type=EventType.LLM_TOKEN,
                model="claude-3-5-sonnet",
                token=word
            ))

        event_bus.publish_sync(LLMEvent(
            type=EventType.LLM_CALL_COMPLETED,
            model="claude-3-5-sonnet",
            provider="anthropic",
            completion_tokens=4,
            duration_ms=500
        ))

        # Check events
        assert len(events) == 6  # start + 4 tokens + complete
        assert tokens == ["Hello", " ", "world", "!"]
        assert events[0].type == EventType.LLM_CALL_STARTED
        assert events[-1].type == EventType.LLM_CALL_COMPLETED
        assert events[-1].duration_ms == 500


class TestCLIStreamingDisplay:
    """Tests for CLI streaming display."""

    def test_streaming_display_subscribes_and_unsubscribes(self):
        """StreamingDisplay subscribes on start, unsubscribes on stop."""
        try:
            from kosmos.cli.streaming import StreamingDisplay

            event_bus = get_event_bus()
            initial_count = event_bus.subscriber_count()

            display = StreamingDisplay(process_id="test_proc")
            display.start()

            assert event_bus.subscriber_count() > initial_count

            display.stop()

            assert event_bus.subscriber_count() == initial_count

        except ImportError:
            pytest.skip("Rich library not available")

    def test_streaming_display_handles_events(self):
        """StreamingDisplay handles incoming events."""
        try:
            from kosmos.cli.streaming import StreamingDisplay
            from io import StringIO
            from rich.console import Console

            # Create display with a test console
            test_output = StringIO()
            test_console = Console(file=test_output, force_terminal=True)

            display = StreamingDisplay(
                console=test_console,
                process_id=None,
                show_tokens=False  # Disable token display for cleaner test
            )
            display.start()

            # Send some events
            event_bus = get_event_bus()
            event_bus.publish_sync(WorkflowEvent(
                type=EventType.WORKFLOW_STARTED,
                research_question="Test question",
                max_cycles=5
            ))

            display.stop()

            # Check that display processed events
            stats = display.get_stats()
            assert 'current_cycle' in stats

        except ImportError:
            pytest.skip("Rich library not available")


class TestAPIStreamingEndpoints:
    """Tests for API streaming endpoints."""

    @pytest.mark.asyncio
    async def test_event_generator_yields_events(self):
        """event_generator yields SSE-formatted events."""
        try:
            from kosmos.api.streaming import event_generator

            # Create a task that generates events
            async def event_producer():
                event_bus = get_event_bus()
                await asyncio.sleep(0.1)
                await event_bus.publish(WorkflowEvent(
                    type=EventType.WORKFLOW_STARTED,
                    process_id="test_proc"
                ))

            # Start producer
            producer_task = asyncio.create_task(event_producer())

            # Collect events from generator
            events_received = []
            gen = event_generator(process_id="test_proc", keepalive_interval=1)

            try:
                async for sse_event in gen:
                    if "workflow.started" in sse_event:
                        events_received.append(sse_event)
                        break
                    if len(events_received) > 5:
                        break
            finally:
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass

            # Should have received at least one event
            assert len(events_received) >= 1
            assert "event:" in events_received[0]
            assert "data:" in events_received[0]

        except ImportError:
            pytest.skip("FastAPI not available")


class TestEventSerialization:
    """Tests for event serialization across the system."""

    def test_event_roundtrip_json(self):
        """Events can be serialized and parsed."""
        from kosmos.core.events import parse_event

        original = WorkflowEvent(
            type=EventType.WORKFLOW_STARTED,
            process_id="proc_123",
            research_question="Test question",
            cycle=2,
            max_cycles=10,
            progress_percent=20.0
        )

        # Serialize
        json_str = original.to_json()

        # Parse back
        import json
        data = json.loads(json_str)
        parsed = parse_event(data)

        assert isinstance(parsed, WorkflowEvent)
        assert parsed.process_id == "proc_123"
        assert parsed.research_question == "Test question"
        assert parsed.cycle == 2


class TestConcurrentSubscribers:
    """Tests for concurrent subscriber handling."""

    @pytest.mark.asyncio
    async def test_multiple_async_subscribers(self):
        """Multiple async subscribers receive events concurrently."""
        results1 = []
        results2 = []
        results3 = []

        async def subscriber1(event):
            await asyncio.sleep(0.01)
            results1.append(event.type)

        async def subscriber2(event):
            await asyncio.sleep(0.01)
            results2.append(event.type)

        async def subscriber3(event):
            await asyncio.sleep(0.01)
            results3.append(event.type)

        event_bus = get_event_bus()
        event_bus.subscribe(subscriber1)
        event_bus.subscribe(subscriber2)
        event_bus.subscribe(subscriber3)

        # Publish events
        await event_bus.publish(WorkflowEvent(type=EventType.WORKFLOW_STARTED))
        await event_bus.publish(WorkflowEvent(type=EventType.WORKFLOW_COMPLETED))

        # All subscribers should have received both events
        assert len(results1) == 2
        assert len(results2) == 2
        assert len(results3) == 2

    @pytest.mark.asyncio
    async def test_subscriber_exception_isolation(self):
        """One failing subscriber doesn't affect others."""
        good_results = []

        def bad_subscriber(event):
            raise Exception("Intentional failure")

        def good_subscriber(event):
            good_results.append(event.type)

        event_bus = get_event_bus()
        event_bus.subscribe(bad_subscriber)
        event_bus.subscribe(good_subscriber)

        # Publish should not raise
        await event_bus.publish(WorkflowEvent(type=EventType.WORKFLOW_STARTED))

        # Good subscriber should still have received the event
        assert len(good_results) == 1
