"""
WebSocket Streaming Endpoint.

Provides bidirectional real-time event streaming for research workflows.
Clients can subscribe to events and send commands via WebSocket.
"""

import asyncio
import json
import logging
from dataclasses import asdict
from typing import List, Optional, Set

try:
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    APIRouter = None
    WebSocket = None
    WebSocketDisconnect = None
    Query = None

from kosmos.core.event_bus import get_event_bus
from kosmos.core.events import StreamingEvent, EventType

logger = logging.getLogger(__name__)

# Create router only if FastAPI is available
if HAS_FASTAPI:
    router = APIRouter(prefix="/ws", tags=["websocket"])
else:
    router = None


class ConnectionManager:
    """
    Manage WebSocket connections.

    Tracks active connections and handles broadcast operations.
    """

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        """
        Accept and track a WebSocket connection.

        Args:
            websocket: The WebSocket connection to accept
        """
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.debug(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        """
        Remove a WebSocket connection.

        Args:
            websocket: The WebSocket connection to remove
        """
        self.active_connections.discard(websocket)
        logger.debug(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_event(
        self,
        websocket: WebSocket,
        event: StreamingEvent
    ) -> bool:
        """
        Send an event to a specific connection.

        Args:
            websocket: Target WebSocket connection
            event: Event to send

        Returns:
            True if successful, False if connection is closed
        """
        try:
            data = json.dumps(asdict(event), default=str)
            await websocket.send_text(data)
            return True
        except Exception as e:
            logger.debug(f"Failed to send to WebSocket: {e}")
            self.disconnect(websocket)
            return False

    async def broadcast(self, event: StreamingEvent) -> None:
        """
        Broadcast an event to all connections.

        Args:
            event: Event to broadcast
        """
        data = json.dumps(asdict(event), default=str)
        disconnected = []

        for connection in list(self.active_connections):
            try:
                await connection.send_text(data)
            except Exception:
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    @property
    def connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)


# Global connection manager
manager = ConnectionManager()


if HAS_FASTAPI:
    @router.websocket("/events")
    async def websocket_events(
        websocket: WebSocket,
        process_id: Optional[str] = Query(
            None,
            description="Filter events by process ID"
        ),
        types: Optional[str] = Query(
            None,
            description="Comma-separated list of event types"
        )
    ):
        """
        WebSocket endpoint for bidirectional event streaming.

        **Connecting**:
        ```javascript
        const ws = new WebSocket('ws://localhost:8000/ws/events?process_id=research_abc');

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('Event:', data.type, data);
        };
        ```

        **Client Commands**:
        Clients can send JSON messages to control their subscription:

        ```json
        // Update event type filter
        {"action": "subscribe", "event_types": ["llm.token", "workflow.progress"]}

        // Clear filters (receive all events)
        {"action": "subscribe", "event_types": null}

        // Ping to keep connection alive
        {"action": "ping"}
        ```

        **Server Responses**:
        ```json
        // Event message
        {"type": "llm.token", "process_id": "research_abc", "token": "Hello", ...}

        // Pong response
        {"action": "pong", "timestamp": "2025-12-09T..."}

        // Subscription confirmation
        {"action": "subscribed", "event_types": ["llm.token"], "process_id": "research_abc"}
        ```
        """
        await manager.connect(websocket)

        # Parse initial filters
        event_types: Optional[List[EventType]] = None
        if types:
            try:
                event_types = [EventType(t) for t in types.split(",")]
            except ValueError as e:
                logger.warning(f"Invalid event type: {e}")

        filter_pids = [process_id] if process_id else None

        # Event queue for this connection
        queue: asyncio.Queue[StreamingEvent] = asyncio.Queue()

        async def queue_callback(event: StreamingEvent):
            """Queue events for this WebSocket."""
            await queue.put(event)

        # Subscribe to event bus
        event_bus = get_event_bus()
        event_bus.subscribe(queue_callback, event_types, filter_pids)

        # Send subscription confirmation
        await websocket.send_json({
            "action": "subscribed",
            "event_types": [t.value for t in event_types] if event_types else None,
            "process_id": process_id
        })

        async def send_events():
            """Task to send queued events to WebSocket."""
            while True:
                try:
                    event = await queue.get()
                    data = json.dumps(asdict(event), default=str)
                    await websocket.send_text(data)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.debug(f"Error sending event: {e}")
                    break

        async def receive_messages():
            """Task to receive and process client messages."""
            nonlocal event_types, filter_pids

            while True:
                try:
                    data = await websocket.receive_text()
                    msg = json.loads(data)
                    action = msg.get("action", "")

                    if action == "ping":
                        # Respond to ping
                        from datetime import datetime
                        await websocket.send_json({
                            "action": "pong",
                            "timestamp": datetime.utcnow().isoformat() + "Z"
                        })

                    elif action == "subscribe":
                        # Update subscription filters
                        event_bus.unsubscribe(queue_callback)

                        # Parse new filters
                        new_types = msg.get("event_types")
                        if new_types:
                            try:
                                event_types = [EventType(t) for t in new_types]
                            except ValueError as e:
                                await websocket.send_json({
                                    "action": "error",
                                    "message": f"Invalid event type: {e}"
                                })
                                continue
                        else:
                            event_types = None

                        new_pid = msg.get("process_id")
                        if new_pid is not None:
                            filter_pids = [new_pid] if new_pid else None

                        # Re-subscribe with new filters
                        event_bus.subscribe(queue_callback, event_types, filter_pids)

                        # Confirm subscription update
                        await websocket.send_json({
                            "action": "subscribed",
                            "event_types": [t.value for t in event_types] if event_types else None,
                            "process_id": filter_pids[0] if filter_pids else None
                        })

                except asyncio.CancelledError:
                    break
                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "action": "error",
                        "message": "Invalid JSON"
                    })
                except Exception as e:
                    logger.debug(f"Error receiving message: {e}")
                    break

        # Run send and receive concurrently
        send_task = asyncio.create_task(send_events())
        receive_task = asyncio.create_task(receive_messages())

        try:
            # Wait for either task to complete (usually receive on disconnect)
            done, pending = await asyncio.wait(
                [send_task, receive_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except WebSocketDisconnect:
            pass
        finally:
            manager.disconnect(websocket)
            event_bus.unsubscribe(queue_callback)
            logger.debug("WebSocket connection closed")

    @router.get("/connections")
    async def get_connections():
        """
        Get WebSocket connection statistics.

        Returns the number of active WebSocket connections.
        """
        return {
            "active_connections": manager.connection_count,
            "event_bus_subscribers": get_event_bus().subscriber_count()
        }
