"""
Database Package Interface
Exports models, session, engine, and CRUD utilities.
"""

from .session import Base, engine, SessionLocal, get_db
from .models import (
    User,
    Camera,
    ParkingZone,
    ParkingSlot,
    Vehicle,
    ParkingEvent,
    OccupancyHistory,
    ModelPrediction,
    SystemAlert
)
from .crud import (
    get_or_create_camera,
    get_or_create_zone,
    sync_slots_from_json,
    record_parking_event,
    record_occupancy_snapshot,
    create_system_alert,
    get_parking_summary
)

__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "get_db",
    "User",
    "Camera",
    "ParkingZone",
    "ParkingSlot",
    "Vehicle",
    "ParkingEvent",
    "OccupancyHistory",
    "ModelPrediction",
    "SystemAlert",
    "get_or_create_camera",
    "get_or_create_zone",
    "sync_slots_from_json",
    "record_parking_event",
    "record_occupancy_snapshot",
    "create_system_alert",
    "get_parking_summary"
]
