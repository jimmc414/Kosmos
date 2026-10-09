"""
Server-Sent Events (SSE) Streaming Endpoint.

Provides real-time event streaming for long-running research workflows.
Clients can subscribe to events and receive updates via SSE.
"""

import asyncio
import json
import logging
from dataclasses import asdict
from typing import AsyncGenerator, List, Optional

try:
    from fastapi import APIRouter, Query
    from fastapi.responses import StreamingResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    APIRouter = None
    Query = None
    StreamingResponse = None

from kosmos.core.event_bus import get_event_bus
from kosmos.core.events import StreamingEvent, EventType

logger = logging.getLogger(__name__)

# Create router only if FastAPI is available
if HAS_FASTAPI:
    router = APIRouter(prefix="/stream", tags=["streaming"])
else:
    router = None


async def event_generator(
    process_id: Optional[str] = None,
    event_types: Optional[List[str]] = None,
    keepalive_interval: int = 30
) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events from the event bus.

    Args:
        process_id: Optional filter for specific process ID
        event_types: Optional list of event type strings to filter
        keepalive_interval: Seconds between keepalive messages

    Yields:
        SSE-formatted event strings
    """
    queue: asyncio.Queue[StreamingEvent] = asyncio.Queue()

    async def queue_callback(event: StreamingEvent):
        """Queue callback for async event handling."""
        await queue.put(event)

    # Subscribe to events
    event_bus = get_event_bus()

    # Parse event type filters
    filter_types = None
    if event_types:
        try:
            filter_types = [EventType(t) for t in event_types]
        except ValueError as e:
            logger.warning(f"Invalid event type filter: {e}")

    # Parse process ID filter
    filter_pids = [process_id] if process_id else None

    event_bus.subscribe(queue_callback, filter_types, filter_pids)
    logger.debug(
        f"SSE client subscribed: process_id={process_id}, "
        f"event_types={event_types}"
    )

    try:
        while True:
            try:
                # Wait for event with timeout for keepalive
                event = await asyncio.wait_for(
                    queue.get(),
                    timeout=float(keepalive_interval)
                )

                # Format as SSE
                data = json.dumps(asdict(event), default=str)
                event_type = event.type.value if hasattr(event, 'type') else "message"
                yield f"event: {event_type}\ndata: {data}\n\n"

            except asyncio.TimeoutError:
                # Send keepalive comment
                yield f": keepalive\n\n"

    except asyncio.CancelledError:
        logger.debug("SSE stream cancelled")
    finally:
        event_bus.unsubscribe(queue_callback)
        logger.debug("SSE client unsubscribed")


if HAS_FASTAPI:
    @router.get("/events")
    async def stream_events(
        process_id: Optional[str] = Query(
            None,
            description="Filter events by process ID"
        ),
        types: Optional[str] = Query(
            None,
            description="Comma-separated list of event types to filter"
        ),
        keepalive: int = Query(
            30,
            ge=5,
            le=120,
            description="Keepalive interval in seconds"
        )
    ):
        """
        Stream events via Server-Sent Events (SSE).

        This endpoint provides real-time event streaming for monitoring
        research workflows, LLM calls, and code execution.

        **Event Types**:
        - `workflow.started`, `workflow.progress`, `workflow.completed`
        - `cycle.started`, `cycle.completed`
        - `task.started`, `task.completed`, `task.failed`
        - `llm.call_started`, `llm.token`, `llm.call_completed`
        - `execution.executing`, `execution.output`, `execution.completed`
        - `stage.started`, `stage.completed`

        **Example Usage**:
        ```bash
        # Stream all events
        curl -N http://localhost:8000/stream/events

        # Filter by process ID
        curl -N http://localhost:8000/stream/events?process_id=research_abc123

        # Filter by event types
        curl -N http://localhost:8000/stream/events?types=llm.token,workflow.progress
        ```

        **SSE Format**:
        ```
        event: workflow.started
        data: {"type": "workflow.started", "process_id": "research_abc", ...}

        event: llm.token
        data: {"type": "llm.token", "token": "Hello", ...}
        ```
        """
        event_types = types.split(",") if types else None

        return StreamingResponse(
            event_generator(process_id, event_types, keepalive),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Access-Control-Allow-Origin": "*",
            }
        )

    @router.get("/health")
    async def streaming_health():
        """
        Check streaming endpoint health.

        Returns event bus status and subscriber count.
        """
        event_bus = get_event_bus()
        return {
            "status": "healthy",
            "enabled": event_bus.enabled,
            "subscriber_count": event_bus.subscriber_count()
        }
