"""
Event Dispatcher Module
Coordinates domain event generation, handler registration, and decoupled broadcasting.
"""

import logging
from typing import List, Optional
from .schemas import EventType, ParkingEvent
from .handlers import BaseEventHandler, LoggingEventHandler

logger = logging.getLogger("cv_engine.events")


class EventDispatcher:
    """
    Central Event Hub.
    Decouples computer vision inference loops from persistence and networking.
    """

    def __init__(self, camera_id: str = "CAM01"):
        self.camera_id = camera_id
        self._handlers: List[BaseEventHandler] = []
        self._total_dispatched = 0

    @property
    def total_dispatched(self) -> int:
        return self._total_dispatched

    def register_handler(self, handler: BaseEventHandler) -> None:
        """Registers a new event handler."""
        if handler not in self._handlers:
            self._handlers.append(handler)

    def unregister_handler(self, handler: BaseEventHandler) -> None:
        """Removes an event handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    def emit(self, event: ParkingEvent) -> None:
        """
        Broadcasts an event to all registered handlers.
        Catches handler exceptions to prevent CV loop disruptions.
        """
        self._total_dispatched += 1
        for handler in self._handlers:
            try:
                handler.handle(event)
            except Exception as e:
                logger.error(f"Error executing handler {handler.__class__.__name__}: {e}", exc_info=True)

    # --- Convenience Factory Dispatchers ---

    def emit_slot_occupied(
        self,
        slot_id: str,
        vehicle_id: Optional[int] = None,
        confidence: float = 1.0,
        metadata: Optional[dict] = None
    ) -> ParkingEvent:
        event = ParkingEvent(
            event_type=EventType.SLOT_OCCUPIED,
            camera_id=self.camera_id,
            slot_id=slot_id,
            vehicle_id=vehicle_id,
            confidence=confidence,
            metadata=metadata or {}
        )
        self.emit(event)
        return event

    def emit_slot_available(
        self,
        slot_id: str,
        dwell_duration_sec: float = 0.0,
        metadata: Optional[dict] = None
    ) -> ParkingEvent:
        meta = metadata or {}
        meta["dwell_duration_sec"] = round(dwell_duration_sec, 2)
        event = ParkingEvent(
            event_type=EventType.SLOT_AVAILABLE,
            camera_id=self.camera_id,
            slot_id=slot_id,
            confidence=1.0,
            metadata=meta
        )
        self.emit(event)
        return event

    def emit_vehicle_entered(
        self,
        vehicle_id: int,
        tripwire_name: str = "GATE",
        metadata: Optional[dict] = None
    ) -> ParkingEvent:
        meta = metadata or {}
        meta["tripwire"] = tripwire_name
        event = ParkingEvent(
            event_type=EventType.VEHICLE_ENTERED,
            camera_id=self.camera_id,
            vehicle_id=vehicle_id,
            metadata=meta
        )
        self.emit(event)
        return event

    def emit_vehicle_exited(
        self,
        vehicle_id: int,
        tripwire_name: str = "GATE",
        metadata: Optional[dict] = None
    ) -> ParkingEvent:
        meta = metadata or {}
        meta["tripwire"] = tripwire_name
        event = ParkingEvent(
            event_type=EventType.VEHICLE_EXITED,
            camera_id=self.camera_id,
            vehicle_id=vehicle_id,
            metadata=meta
        )
        self.emit(event)
        return event

    def emit_parking_full(self, total_slots: int, metadata: Optional[dict] = None) -> ParkingEvent:
        meta = metadata or {}
        meta["total_slots"] = total_slots
        meta["occupancy_rate"] = 100.0
        event = ParkingEvent(
            event_type=EventType.PARKING_FULL,
            camera_id=self.camera_id,
            metadata=meta
        )
        self.emit(event)
        return event

    def emit_parking_warning(self, occupancy_rate: float, metadata: Optional[dict] = None) -> ParkingEvent:
        meta = metadata or {}
        meta["occupancy_rate"] = round(occupancy_rate, 1)
        event = ParkingEvent(
            event_type=EventType.PARKING_WARNING,
            camera_id=self.camera_id,
            metadata=meta
        )
        self.emit(event)
        return event

    def emit_camera_offline(self, last_frame_sec: float, metadata: Optional[dict] = None) -> ParkingEvent:
        meta = metadata or {}
        meta["seconds_since_last_frame"] = round(last_frame_sec, 2)
        event = ParkingEvent(
            event_type=EventType.CAMERA_OFFLINE,
            camera_id=self.camera_id,
            metadata=meta
        )
        self.emit(event)
        return event

    def emit_low_confidence(self, avg_confidence: float, metadata: Optional[dict] = None) -> ParkingEvent:
        meta = metadata or {}
        meta["avg_confidence"] = round(avg_confidence, 3)
        event = ParkingEvent(
            event_type=EventType.LOW_DETECTION_CONFIDENCE,
            camera_id=self.camera_id,
            confidence=avg_confidence,
            metadata=meta
        )
        self.emit(event)
        return event
