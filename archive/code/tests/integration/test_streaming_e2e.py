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
