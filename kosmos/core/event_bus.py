"""
Event Bus for Streaming Events.

Central pub/sub system for routing streaming events to multiple subscribers.
Supports both sync and async callbacks with optional filtering.
"""

import asyncio
import logging
from typing import Callable, Dict, List, Optional, Set, Union
from weakref import WeakSet

from kosmos.core.events import (
    BaseEvent,
    EventType,
    StreamingEvent,
)

logger = logging.getLogger(__name__)

# Callback type definitions
SyncCallback = Callable[[StreamingEvent], None]
AsyncCallback = Callable[[StreamingEvent], "asyncio.coroutine"]
Callback = Union[SyncCallback, AsyncCallback]


class EventBus:
    """
    Central event bus for routing streaming events.

    Features:
    - Type-safe event routing
    - Sync and async subscriber support
    - Filtering by event type and process ID
    - Thread-safe operations

    Example:
        bus = EventBus()

        def on_token(event):
            print(event.token)

        bus.subscribe(on_token, event_types=[EventType.LLM_TOKEN])
        bus.publish_sync(LLMEvent(type=EventType.LLM_TOKEN, token="Hello"))
    """

    def __init__(self):
        """Initialize the event bus."""
        # Map: event_type -> list of callbacks (None key = all events)
        self._subscribers: Dict[Optional[EventType], List[Callback]] = {}
        # Map: callback -> set of process_ids to filter (None = all)
        self._process_filters: Dict[Callback, Optional[Set[str]]] = {}
        self._enabled = True
        self._lock = asyncio.Lock()

    def subscribe(
        self,
        callback: Callback,
        event_types: Optional[List[EventType]] = None,
        process_ids: Optional[List[str]] = None
    ) -> None:
        """
        Subscribe to events.

        Args:
            callback: Function to call when event occurs.
                     Can be sync or async.
            event_types: Optional list of event types to filter.
                        None means subscribe to all events.
            process_ids: Optional list of process IDs to filter.
                        None means receive events from all processes.

        Example:
            # Subscribe to all events
            bus.subscribe(my_callback)

            # Subscribe to specific event types
            bus.subscribe(my_callback, event_types=[EventType.LLM_TOKEN])

            # Subscribe to events from specific process
            bus.subscribe(my_callback, process_ids=["research_123"])
        """
        if event_types:
            for event_type in event_types:
                if event_type not in self._subscribers:
                    self._subscribers[event_type] = []
                if callback not in self._subscribers[event_type]:
                    self._subscribers[event_type].append(callback)
        else:
            # Subscribe to all events
            if None not in self._subscribers:
                self._subscribers[None] = []
            if callback not in self._subscribers[None]:
                self._subscribers[None].append(callback)

        # Store process filter
        if process_ids:
            self._process_filters[callback] = set(process_ids)
        else:
            self._process_filters[callback] = None

    def unsubscribe(self, callback: Callback) -> None:
        """
        Remove a subscriber.

        Args:
            callback: The callback to remove
        """
        for subscribers in self._subscribers.values():
            if callback in subscribers:
                subscribers.remove(callback)

        self._process_filters.pop(callback, None)

    def unsubscribe_all(self) -> None:
        """Remove all subscribers."""
        self._subscribers.clear()
        self._process_filters.clear()

    async def publish(self, event: StreamingEvent) -> None:
        """
        Publish an event asynchronously.

        Notifies all matching subscribers. Async callbacks are awaited,
        sync callbacks are called directly.

        Args:
            event: The event to publish
        """
        if not self._enabled:
            return

        callbacks = self._get_callbacks(event)

        for callback in callbacks:
            # Check process filter
            if not self._passes_process_filter(callback, event):
                continue

            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Event subscriber error: {e}", exc_info=True)

    def publish_sync(self, event: StreamingEvent) -> None:
        """
        Publish an event synchronously.

        For use in non-async contexts. Async callbacks are scheduled
        on the event loop if available.

        Args:
            event: The event to publish
        """
        if not self._enabled:
            return

        callbacks = self._get_callbacks(event)

        for callback in callbacks:
            # Check process filter
            if not self._passes_process_filter(callback, event):
                continue

            try:
                if asyncio.iscoroutinefunction(callback):
                    # Try to schedule on event loop
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(callback(event))
                    except RuntimeError:
                        # No running event loop, skip async callback
                        logger.debug(
                            f"Skipping async callback {callback.__name__} - "
                            "no event loop"
                        )
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Event subscriber error: {e}", exc_info=True)

    def _get_callbacks(self, event: StreamingEvent) -> List[Callback]:
        """Get all callbacks for an event type."""
        callbacks = []

        # Global subscribers (subscribed to all events)
        callbacks.extend(self._subscribers.get(None, []))

        # Type-specific subscribers
        if hasattr(event, 'type'):
            callbacks.extend(self._subscribers.get(event.type, []))

        return callbacks

    def _passes_process_filter(
        self,
        callback: Callback,
        event: StreamingEvent
    ) -> bool:
        """Check if event passes callback's process filter."""
        filter_set = self._process_filters.get(callback)

        if filter_set is None:
            # No filter, accept all
            return True

        if not hasattr(event, 'process_id') or event.process_id is None:
            # Event has no process_id, accept if filter allows None
            return None in filter_set

        return event.process_id in filter_set

    def enable(self) -> None:
        """Enable event publishing."""
        self._enabled = True

    def disable(self) -> None:
        """Disable event publishing (events are dropped)."""
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Check if event bus is enabled."""
        return self._enabled

    def subscriber_count(
        self,
        event_type: Optional[EventType] = None
    ) -> int:
        """
        Get number of subscribers.

        Args:
            event_type: Optional event type to count subscribers for.
                       None returns total unique subscribers.

        Returns:
            Number of subscribers
        """
        if event_type is not None:
            return len(self._subscribers.get(event_type, []))

        # Count unique subscribers
        all_callbacks = set()
        for callbacks in self._subscribers.values():
            all_callbacks.update(callbacks)
        return len(all_callbacks)


# Global singleton instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """
    Get or create the global event bus singleton.

    Returns:
        The global EventBus instance
    """
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def reset_event_bus() -> None:
    """
    Reset the event bus singleton.

    Useful for testing to ensure clean state.
    """
    global _event_bus
    if _event_bus is not None:
        _event_bus.unsubscribe_all()
    _event_bus = None


class EventSubscription:
    """
    Context manager for event subscriptions.

    Automatically unsubscribes when exiting context.

    Example:
        with EventSubscription(callback, [EventType.LLM_TOKEN]) as sub:
            # Receive events
            pass
        # Automatically unsubscribed
    """

    def __init__(
        self,
        callback: Callback,
        event_types: Optional[List[EventType]] = None,
        process_ids: Optional[List[str]] = None,
        event_bus: Optional[EventBus] = None
    ):
        self.callback = callback
        self.event_types = event_types
        self.process_ids = process_ids
        self.event_bus = event_bus or get_event_bus()

    def __enter__(self) -> "EventSubscription":
        self.event_bus.subscribe(
            self.callback,
            self.event_types,
            self.process_ids
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.event_bus.unsubscribe(self.callback)
        return False

    async def __aenter__(self) -> "EventSubscription":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)
