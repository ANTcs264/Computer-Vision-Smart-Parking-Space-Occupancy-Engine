"""
Events Package Interface
Exports EventType, ParkingEvent, EventDispatcher, and standard event handlers.
"""

from .schemas import EventType, ParkingEvent
from .dispatcher import EventDispatcher
from .handlers import (
    BaseEventHandler,
    LoggingEventHandler,
    FileAuditEventHandler,
    MemoryQueueHandler
)

__all__ = [
    "EventType",
    "ParkingEvent",
    "EventDispatcher",
    "BaseEventHandler",
    "LoggingEventHandler",
    "FileAuditEventHandler",
    "MemoryQueueHandler"
]
