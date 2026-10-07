"""
Database CRUD (Create, Read, Update, Delete) Operations Module
Provides high-performance repository functions for event logging, slot state synchronization,
occupancy snapshots, and analytical aggregation.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from .models import (
    Camera,
    ParkingZone,
    ParkingSlot,
    Vehicle,
    ParkingEvent,
    OccupancyHistory,
    ModelPrediction,
    SystemAlert
)

logger = logging.getLogger("database.crud")


def get_or_create_camera(
    db: Session,
    camera_code: str = "CAM01",
    name: str = "Main Parking Lot Camera",
    stream_url: str = "0",
    resolution: str = "1280x720"
) -> Camera:
    """Finds or creates camera sensor record."""
    camera = db.query(Camera).filter(Camera.camera_code == camera_code).first()
    if not camera:
        camera = Camera(
            camera_code=camera_code,
            name=name,
            stream_url=stream_url,
            resolution=resolution
        )
        db.add(camera)
        db.commit()
        db.refresh(camera)
        logger.info(f"Registered new camera in database: {camera_code}")
    return camera


def get_or_create_zone(
    db: Session,
    camera: Camera,
    zone_code: str = "ZONE_A",
    name: str = "Main Parking Zone"
) -> ParkingZone:
    """Finds or creates parking zone record."""
    zone = db.query(ParkingZone).filter(
        ParkingZone.camera_id == camera.id,
        ParkingZone.zone_code == zone_code
    ).first()
    if not zone:
        zone = ParkingZone(
            camera_id=camera.id,
            zone_code=zone_code,
            name=name
        )
        db.add(zone)
        db.commit()
        db.refresh(zone)
        logger.info(f"Registered new parking zone in database: {zone_code}")
    return zone


def sync_slots_from_json(db: Session, json_path: Path | str) -> List[ParkingSlot]:
    """
    Parses parking_slots.json and synchronizes all slots into the database.
    Inserts missing slots and updates coordinates for existing slots.
    """
    path = Path(json_path)
    if not path.is_file():
        logger.warning(f"Slots file not found for DB sync: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    camera_code = data.get("camera_id", "CAM01")
    zone_code = data.get("zone_id", "ZONE_A")

    camera = get_or_create_camera(db, camera_code=camera_code)
    zone = get_or_create_zone(db, camera=camera, zone_code=zone_code)

    synced_slots = []
    for item in data.get("slots", []):
        slot_code = item["slot_id"]
        poly = item["polygon"]
        status = item.get("status", "AVAILABLE")

        slot = db.query(ParkingSlot).filter(
            ParkingSlot.camera_id == camera.id,
            ParkingSlot.slot_code == slot_code
        ).first()

        if slot:
            slot.polygon_coordinates = poly
            slot.zone_id = zone.id
        else:
            slot = ParkingSlot(
                camera_id=camera.id,
                zone_id=zone.id,
                slot_code=slot_code,
                polygon_coordinates=poly,
                current_status=status
            )
            db.add(slot)

        synced_slots.append(slot)

    db.commit()
    logger.info(f"Synchronized {len(synced_slots)} parking slots into database.")
    return synced_slots


def record_parking_event(
    db: Session,
    camera_code: str,
    event_type: str,
    slot_code: Optional[str] = None,
    track_id: Optional[int] = None,
    confidence: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None
) -> ParkingEvent:
    """
    Inserts a validated domain event and updates parking slot state atomically.
    """
    camera = get_or_create_camera(db, camera_code=camera_code)

    slot = None
    if slot_code:
        slot = db.query(ParkingSlot).filter(
            ParkingSlot.camera_id == camera.id,
            ParkingSlot.slot_code == slot_code
        ).first()

    vehicle = None
    if track_id is not None:
        vehicle = db.query(Vehicle).filter(Vehicle.track_id == track_id).first()
        if not vehicle:
            v_type = metadata.get("class", "car") if metadata else "car"
            vehicle = Vehicle(track_id=track_id, vehicle_type=v_type)
            db.add(vehicle)
            db.flush()

    event = ParkingEvent(
        camera_id=camera.id,
        slot_id=slot.id if slot else None,
        vehicle_id=vehicle.id if vehicle else None,
        event_type=event_type,
        confidence=confidence,
        extra_metadata=metadata or {}
    )
    db.add(event)

    # Update slot real-time status if occupancy event
    if slot:
        now = datetime.now(timezone.utc)
        if event_type == "SLOT_OCCUPIED":
            slot.current_status = "OCCUPIED"
            slot.assigned_vehicle_id = track_id
            slot.last_status_change = now
        elif event_type == "SLOT_AVAILABLE":
            slot.current_status = "AVAILABLE"
            slot.assigned_vehicle_id = None
            slot.last_status_change = now

    db.commit()
    db.refresh(event)
    return event


def record_occupancy_snapshot(
    db: Session,
    slot_id: str,
    camera_id: str,
    is_occupied: bool
) -> OccupancyHistory:
    """Records a periodic time-series occupancy state."""
    snapshot = OccupancyHistory(
        slot_id=slot_id,
        camera_id=camera_id,
        is_occupied=is_occupied
    )
    db.add(snapshot)
    db.commit()
    return snapshot


def create_system_alert(
    db: Session,
    camera_code: str,
    alert_type: str,
    severity: str,
    message: str
) -> SystemAlert:
    """Inserts a system alert record."""
    camera = get_or_create_camera(db, camera_code=camera_code)
    alert = SystemAlert(
        camera_id=camera.id,
        alert_type=alert_type,
        severity=severity,
        message=message
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


def get_parking_summary(db: Session, camera_code: str = "CAM01") -> Dict[str, Any]:
    """Returns real-time capacity and occupancy metrics."""
    camera = db.query(Camera).filter(Camera.camera_code == camera_code).first()
    if not camera:
        return {"total": 0, "occupied": 0, "available": 0, "occupancy_rate": 0.0, "slots": []}

    slots = db.query(ParkingSlot).filter(ParkingSlot.camera_id == camera.id).all()
    total = len(slots)
    occupied = sum(1 for s in slots if s.current_status == "OCCUPIED")
    available = sum(1 for s in slots if s.current_status == "AVAILABLE")
    rate = (occupied / total * 100.0) if total > 0 else 0.0

    return {
        "camera_code": camera_code,
        "total_slots": total,
        "occupied_slots": occupied,
        "available_slots": available,
        "occupancy_rate": round(rate, 1),
        "slots": [
            {
                "slot_code": s.slot_code,
                "status": s.current_status,
                "assigned_vehicle_id": s.assigned_vehicle_id,
                "last_changed": s.last_status_change.isoformat() if s.last_status_change else None
            }
            for s in slots
        ]
    }
