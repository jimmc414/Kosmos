"""
Unit tests for EventBus module.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock

from kosmos.core.events import (
    EventType,
    BaseEvent,
    WorkflowEvent,
    LLMEvent,
    TaskEvent,
)
from kosmos.core.event_bus import (
    EventBus,
    get_event_bus,
    reset_event_bus,
    EventSubscription,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset event bus singleton before each test."""
    reset_event_bus()
    yield
    reset_event_bus()


@pytest.fixture
def event_bus():
    """Create a fresh EventBus for testing."""
    return EventBus()


class TestEventBusBasics:
    """Tests for basic EventBus functionality."""

    def test_event_bus_creation(self, event_bus):
        """EventBus can be created."""
        assert event_bus is not None
        assert event_bus.enabled is True

    def test_subscribe_callback(self, event_bus):
        """Can subscribe a callback."""
        callback = Mock()
        event_bus.subscribe(callback)
        assert event_bus.subscriber_count() == 1

    def test_unsubscribe_callback(self, event_bus):
        """Can unsubscribe a callback."""
        callback = Mock()
        event_bus.subscribe(callback)
        event_bus.unsubscribe(callback)
        assert event_bus.subscriber_count() == 0

    def test_unsubscribe_all(self, event_bus):
        """Can unsubscribe all callbacks."""
        callback1 = Mock()
        callback2 = Mock()
        event_bus.subscribe(callback1)
        event_bus.subscribe(callback2)
        event_bus.unsubscribe_all()
        assert event_bus.subscriber_count() == 0

    def test_enable_disable(self, event_bus):
        """Can enable and disable event bus."""
        assert event_bus.enabled is True
        event_bus.disable()
        assert event_bus.enabled is False
        event_bus.enable()
        assert event_bus.enabled is True


class TestPublishSync:
    """Tests for synchronous event publishing."""

    def test_publish_sync_calls_subscriber(self, event_bus):
        """publish_sync calls subscribed callback."""
        callback = Mock()
        event_bus.subscribe(callback)

        event = WorkflowEvent(type=EventType.WORKFLOW_STARTED)
        event_bus.publish_sync(event)

        callback.assert_called_once_with(event)

    def test_publish_sync_calls_multiple_subscribers(self, event_bus):
        """publish_sync calls all subscribed callbacks."""
        callback1 = Mock()
        callback2 = Mock()
        event_bus.subscribe(callback1)
        event_bus.subscribe(callback2)

        event = WorkflowEvent(type=EventType.WORKFLOW_STARTED)
        event_bus.publish_sync(event)

        callback1.assert_called_once_with(event)
        callback2.assert_called_once_with(event)

    def test_publish_sync_disabled_no_call(self, event_bus):
        """publish_sync does nothing when disabled."""
        callback = Mock()
        event_bus.subscribe(callback)
        event_bus.disable()

        event = WorkflowEvent(type=EventType.WORKFLOW_STARTED)
        event_bus.publish_sync(event)

        callback.assert_not_called()

    def test_publish_sync_exception_continues(self, event_bus):
        """publish_sync continues after callback exception."""
        callback1 = Mock(side_effect=Exception("error"))
        callback2 = Mock()
        event_bus.subscribe(callback1)
        event_bus.subscribe(callback2)

        event = WorkflowEvent(type=EventType.WORKFLOW_STARTED)
        event_bus.publish_sync(event)  # Should not raise

        callback2.assert_called_once()


class TestPublishAsync:
    """Tests for asynchronous event publishing."""

    @pytest.mark.asyncio
    async def test_publish_async_calls_subscriber(self, event_bus):
        """publish calls subscribed callback."""
        callback = Mock()
        event_bus.subscribe(callback)

        event = WorkflowEvent(type=EventType.WORKFLOW_STARTED)
        await event_bus.publish(event)

        callback.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_publish_async_awaits_async_callback(self, event_bus):
        """publish awaits async callbacks."""
        callback = AsyncMock()
        event_bus.subscribe(callback)

        event = WorkflowEvent(type=EventType.WORKFLOW_STARTED)
        await event_bus.publish(event)

        callback.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_publish_async_disabled_no_call(self, event_bus):
        """publish does nothing when disabled."""
        callback = AsyncMock()
        event_bus.subscribe(callback)
        event_bus.disable()

        event = WorkflowEvent(type=EventType.WORKFLOW_STARTED)
        await event_bus.publish(event)

        callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_publish_async_exception_continues(self, event_bus):
        """publish continues after callback exception."""
        callback1 = AsyncMock(side_effect=Exception("error"))
        callback2 = AsyncMock()
        event_bus.subscribe(callback1)
        event_bus.subscribe(callback2)

        event = WorkflowEvent(type=EventType.WORKFLOW_STARTED)
        await event_bus.publish(event)  # Should not raise

        callback2.assert_awaited_once()


class TestEventTypeFiltering:
    """Tests for event type filtering."""

    def test_subscribe_to_specific_type(self, event_bus):
        """Subscriber receives only specified event types."""
        callback = Mock()
        event_bus.subscribe(callback, event_types=[EventType.LLM_TOKEN])

        # Send LLM_TOKEN - should be received
        event1 = LLMEvent(type=EventType.LLM_TOKEN, token="Hi")
        event_bus.publish_sync(event1)
        assert callback.call_count == 1

        # Send WORKFLOW_STARTED - should NOT be received
        event2 = WorkflowEvent(type=EventType.WORKFLOW_STARTED)
        event_bus.publish_sync(event2)
        assert callback.call_count == 1  # Still 1

    def test_subscribe_to_multiple_types(self, event_bus):
        """Subscriber receives multiple specified event types."""
        callback = Mock()
        event_bus.subscribe(
            callback,
            event_types=[EventType.LLM_TOKEN, EventType.LLM_CALL_COMPLETED]
        )

        event1 = LLMEvent(type=EventType.LLM_TOKEN, token="Hi")
        event_bus.publish_sync(event1)

        event2 = LLMEvent(type=EventType.LLM_CALL_COMPLETED)
        event_bus.publish_sync(event2)

        event3 = LLMEvent(type=EventType.LLM_CALL_STARTED)
        event_bus.publish_sync(event3)

        assert callback.call_count == 2

    def test_global_subscriber_receives_all(self, event_bus):
        """Subscriber without type filter receives all events."""
        callback = Mock()
        event_bus.subscribe(callback)  # No type filter

        event1 = WorkflowEvent(type=EventType.WORKFLOW_STARTED)
        event2 = LLMEvent(type=EventType.LLM_TOKEN, token="Hi")
        event3 = TaskEvent(type=EventType.TASK_STARTED)

        event_bus.publish_sync(event1)
        event_bus.publish_sync(event2)
        event_bus.publish_sync(event3)

        assert callback.call_count == 3

    def test_subscriber_count_by_type(self, event_bus):
        """subscriber_count respects type filter."""
        callback1 = Mock()
        callback2 = Mock()

        event_bus.subscribe(callback1, event_types=[EventType.LLM_TOKEN])
        event_bus.subscribe(callback2)  # Global

        assert event_bus.subscriber_count(EventType.LLM_TOKEN) == 1
        assert event_bus.subscriber_count(EventType.WORKFLOW_STARTED) == 0
        assert event_bus.subscriber_count() == 2  # Total unique


class TestProcessIdFiltering:
    """Tests for process ID filtering."""

    def test_filter_by_process_id(self, event_bus):
        """Subscriber receives only events from specified process."""
        callback = Mock()
        event_bus.subscribe(callback, process_ids=["proc_123"])

        event1 = WorkflowEvent(
            type=EventType.WORKFLOW_STARTED,
            process_id="proc_123"
        )
        event_bus.publish_sync(event1)
        assert callback.call_count == 1

        event2 = WorkflowEvent(
            type=EventType.WORKFLOW_STARTED,
            process_id="proc_456"
        )
        event_bus.publish_sync(event2)
        assert callback.call_count == 1  # Still 1

    def test_filter_by_multiple_process_ids(self, event_bus):
        """Subscriber receives events from multiple specified processes."""
        callback = Mock()
        event_bus.subscribe(callback, process_ids=["proc_123", "proc_456"])

        event1 = WorkflowEvent(
            type=EventType.WORKFLOW_STARTED,
            process_id="proc_123"
        )
        event2 = WorkflowEvent(
            type=EventType.WORKFLOW_STARTED,
            process_id="proc_456"
        )
        event3 = WorkflowEvent(
            type=EventType.WORKFLOW_STARTED,
            process_id="proc_789"
        )

        event_bus.publish_sync(event1)
        event_bus.publish_sync(event2)
        event_bus.publish_sync(event3)

        assert callback.call_count == 2

    def test_no_process_filter_receives_all(self, event_bus):
        """Subscriber without process filter receives all events."""
        callback = Mock()
        event_bus.subscribe(callback)  # No process filter

        event1 = WorkflowEvent(
            type=EventType.WORKFLOW_STARTED,
            process_id="proc_123"
        )
        event2 = WorkflowEvent(
            type=EventType.WORKFLOW_STARTED,
            process_id="proc_456"
        )

        event_bus.publish_sync(event1)
        event_bus.publish_sync(event2)

        assert callback.call_count == 2

    def test_combined_type_and_process_filter(self, event_bus):
        """Filters can be combined for type and process."""
        callback = Mock()
        event_bus.subscribe(
            callback,
            event_types=[EventType.LLM_TOKEN],
            process_ids=["proc_123"]
        )

        # Match both - should receive
        event1 = LLMEvent(
            type=EventType.LLM_TOKEN,
            process_id="proc_123",
            token="Hi"
        )
        event_bus.publish_sync(event1)
        assert callback.call_count == 1

        # Match type but not process - should NOT receive
        event2 = LLMEvent(
            type=EventType.LLM_TOKEN,
            process_id="proc_456",
            token="Hi"
        )
        event_bus.publish_sync(event2)
        assert callback.call_count == 1

        # Match process but not type - should NOT receive
        event3 = WorkflowEvent(
            type=EventType.WORKFLOW_STARTED,
            process_id="proc_123"
        )
        event_bus.publish_sync(event3)
        assert callback.call_count == 1


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_event_bus_returns_same_instance(self):
        """get_event_bus returns singleton."""
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2

    def test_reset_event_bus_creates_new_instance(self):
        """reset_event_bus creates new singleton."""
        bus1 = get_event_bus()
        callback = Mock()
        bus1.subscribe(callback)

        reset_event_bus()

        bus2 = get_event_bus()
        assert bus2.subscriber_count() == 0


class TestEventSubscription:
    """Tests for EventSubscription context manager."""

    def test_subscription_context_manager(self, event_bus):
        """EventSubscription subscribes on enter, unsubscribes on exit."""
        callback = Mock()

        with EventSubscription(callback, event_bus=event_bus):
            assert event_bus.subscriber_count() == 1
            event = WorkflowEvent(type=EventType.WORKFLOW_STARTED)
            event_bus.publish_sync(event)
            callback.assert_called_once()

        assert event_bus.subscriber_count() == 0

    def test_subscription_with_filters(self, event_bus):
        """EventSubscription respects filters."""
        callback = Mock()

        with EventSubscription(
            callback,
            event_types=[EventType.LLM_TOKEN],
            process_ids=["proc_123"],
            event_bus=event_bus
        ):
            # Should receive
            event1 = LLMEvent(
                type=EventType.LLM_TOKEN,
                process_id="proc_123",
                token="Hi"
            )
            event_bus.publish_sync(event1)
            assert callback.call_count == 1

            # Should NOT receive
            event2 = LLMEvent(
                type=EventType.LLM_TOKEN,
                process_id="proc_456",
                token="Hi"
            )
            event_bus.publish_sync(event2)
            assert callback.call_count == 1

    @pytest.mark.asyncio
    async def test_subscription_async_context(self, event_bus):
        """EventSubscription works as async context manager."""
        callback = Mock()

        async with EventSubscription(callback, event_bus=event_bus):
            assert event_bus.subscriber_count() == 1

        assert event_bus.subscriber_count() == 0

    def test_subscription_uses_global_bus(self):
        """EventSubscription uses global bus by default."""
        callback = Mock()
        global_bus = get_event_bus()

        with EventSubscription(callback):
            assert global_bus.subscriber_count() == 1

        assert global_bus.subscriber_count() == 0


class TestDuplicateSubscriptions:
    """Tests for duplicate subscription handling."""

    def test_same_callback_subscribed_once(self, event_bus):
        """Same callback is only added once per type."""
        callback = Mock()
        event_bus.subscribe(callback)
        event_bus.subscribe(callback)

        assert event_bus.subscriber_count() == 1

        event = WorkflowEvent(type=EventType.WORKFLOW_STARTED)
        event_bus.publish_sync(event)
        callback.assert_called_once()

    def test_same_callback_different_types(self, event_bus):
        """Same callback can subscribe to multiple types."""
        callback = Mock()
        event_bus.subscribe(callback, event_types=[EventType.LLM_TOKEN])
        event_bus.subscribe(callback, event_types=[EventType.WORKFLOW_STARTED])

        # Callback is in two type lists
        event1 = LLMEvent(type=EventType.LLM_TOKEN, token="Hi")
        event_bus.publish_sync(event1)
        assert callback.call_count == 1

        event2 = WorkflowEvent(type=EventType.WORKFLOW_STARTED)
        event_bus.publish_sync(event2)
        assert callback.call_count == 2


class TestEventWithoutProcessId:
    """Tests for events without process_id field."""

    def test_event_without_process_id_passes_no_filter(self, event_bus):
        """Event without process_id passes when no filter set."""
        callback = Mock()
        event_bus.subscribe(callback)  # No process filter

        event = BaseEvent(type=EventType.WORKFLOW_STARTED)
        event_bus.publish_sync(event)

        callback.assert_called_once()

    def test_event_with_none_process_id(self, event_bus):
        """Event with process_id=None handled correctly."""
        callback = Mock()
        event_bus.subscribe(callback, process_ids=["proc_123"])

        event = WorkflowEvent(
            type=EventType.WORKFLOW_STARTED,
            process_id=None
        )
        event_bus.publish_sync(event)

        callback.assert_not_called()
