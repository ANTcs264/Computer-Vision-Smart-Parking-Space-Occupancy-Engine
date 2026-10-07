"""
Event Handlers Module
Pluggable subscribers for processing, persisting, and forwarding domain events.
"""

from abc import ABC, abstractmethod
import json
import logging
from pathlib import Path
from queue import Queue
from typing import List

from .schemas import EventType, ParkingEvent

logger = logging.getLogger("cv_engine.events")


class BaseEventHandler(ABC):
    """Abstract base class for all event handlers / subscribers."""

    @abstractmethod
    def handle(self, event: ParkingEvent) -> None:
        """Processes a single domain event."""
        pass


class LoggingEventHandler(BaseEventHandler):
    """Logs domain events using structured Python logging."""

    def handle(self, event: ParkingEvent) -> None:
        severity = logging.INFO
        if event.event_type in (EventType.PARKING_FULL, EventType.CAMERA_OFFLINE):
            severity = logging.WARNING
        elif event.event_type == EventType.LOW_DETECTION_CONFIDENCE:
            severity = logging.WARNING

        details = []
        if event.slot_id:
            details.append(f"Slot: {event.slot_id}")
        if event.vehicle_id is not None:
            details.append(f"Vehicle: #{event.vehicle_id}")
        if event.confidence < 1.0:
            details.append(f"Conf: {event.confidence:.2f}")

        detail_str = f" ({', '.join(details)})" if details else ""
        logger.log(
            severity,
            f"📢 EVENT [{event.event_type.value}]{detail_str} [Cam: {event.camera_id}]"
        )


class FileAuditEventHandler(BaseEventHandler):
    """
    Appends events to a persistent JSON-Lines audit log file.
    Ensures zero data loss even if server restarts.
    """

    def __init__(self, log_path: Path | str = "logs/parking_events.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def handle(self, event: ParkingEvent) -> None:
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(event.to_json() + "\n")
        except Exception as e:
            logger.error(f"Failed to append event to audit file {self.log_path}: {e}")


class MemoryQueueHandler(BaseEventHandler):
    """
    Buffers events in an in-memory queue.
    Useful for testing, WebSocket streaming, and asynchronous batch workers.
    """

    def __init__(self, maxsize: int = 1000):
        self.queue: Queue[ParkingEvent] = Queue(maxsize=maxsize)

    def handle(self, event: ParkingEvent) -> None:
        if not self.queue.full():
            self.queue.put(event)

    def pop_all(self) -> List[ParkingEvent]:
        """Drains and returns all queued events."""
        events: List[ParkingEvent] = []
        while not self.queue.empty():
            events.append(self.queue.get())
        return events
