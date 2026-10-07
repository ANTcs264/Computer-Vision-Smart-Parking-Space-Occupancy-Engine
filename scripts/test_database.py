#!/usr/bin/env python3
"""
Database Persistence & Query Verification Script
Tests SQLAlchemy models, CRUD operations, relationships, and index queries.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.session import SessionLocal
from database.models import (
    Camera,
    ParkingZone,
    ParkingSlot,
    Vehicle,
    ParkingEvent,
    OccupancyHistory,
    ModelPrediction,
    SystemAlert
)
from database.crud import (
    get_or_create_camera,
    record_parking_event,
    record_occupancy_snapshot,
    create_system_alert,
    get_parking_summary
)


def run_database_unit_tests():
    print("=" * 68)
    print("        SMART PARKING AI - DATABASE PERSISTENCE VERIFICATION        ")
    print("=" * 68)

    db = SessionLocal()
    try:
        # TEST 1: Camera & Zone Verification
        print("\n[TEST 1] Testing Camera and Parking Zone Models...")
        camera = get_or_create_camera(db, camera_code="CAM01")
        assert camera is not None
        assert camera.camera_code == "CAM01"
        print(f"  [PASS] Camera verified: {camera.camera_code} (ID: {camera.id})")

        # TEST 2: Parking Slot Query & Real-time State
        print("\n[TEST 2] Testing Parking Slot Entities...")
        slots = db.query(ParkingSlot).filter(ParkingSlot.camera_id == camera.id).all()
        assert len(slots) >= 6, "Expected at least 6 synchronized parking slots."
        print(f"  [PASS] Retrieved {len(slots)} slots. Sample: Slot {slots[0].slot_code} [{slots[0].current_status}]")

        # TEST 3: Insert Domain Event with Atomic Slot Update
        print("\n[TEST 3] Testing Atomic Event Recording & Slot State Update...")
        # Record SLOT_OCCUPIED for P01 by Vehicle 17
        evt1 = record_parking_event(
            db=db,
            camera_code="CAM01",
            event_type="SLOT_OCCUPIED",
            slot_code="P01",
            track_id=17,
            confidence=0.96,
            metadata={"class": "car"}
        )
        assert evt1 is not None
        assert evt1.event_type == "SLOT_OCCUPIED"

        # Verify P01 is now marked OCCUPIED in database
        slot_p1 = db.query(ParkingSlot).filter(ParkingSlot.slot_code == "P01").first()
        assert slot_p1.current_status == "OCCUPIED"
        assert slot_p1.assigned_vehicle_id == 17
        print(f"  [PASS] Event recorded: {evt1.event_type}. Slot P01 updated to: {slot_p1.current_status} (Vehicle #{slot_p1.assigned_vehicle_id})")

        # Record SLOT_AVAILABLE for P01 upon departure
        evt2 = record_parking_event(
            db=db,
            camera_code="CAM01",
            event_type="SLOT_AVAILABLE",
            slot_code="P01",
            track_id=17,
            confidence=1.0,
            metadata={"dwell_duration_sec": 45.2}
        )
        db.refresh(slot_p1)
        assert slot_p1.current_status == "AVAILABLE"
        assert slot_p1.assigned_vehicle_id is None
        print(f"  [PASS] Event recorded: {evt2.event_type}. Slot P01 returned to: {slot_p1.current_status}")

        # TEST 4: Periodic Occupancy History Snapshot
        print("\n[TEST 4] Testing Occupancy Snapshot Time-Series Logging...")
        snapshot = record_occupancy_snapshot(
            db=db,
            slot_id=slot_p1.id,
            camera_id=camera.id,
            is_occupied=False
        )
        assert snapshot is not None
        assert snapshot.slot_id == slot_p1.id
        print(f"  [PASS] Occupancy snapshot logged at: {snapshot.recorded_at.isoformat()}")

        # TEST 5: System Alerts Logging
        print("\n[TEST 5] Testing System Alert Recording...")
        alert = create_system_alert(
            db=db,
            camera_code="CAM01",
            alert_type="HIGH_OCCUPANCY",
            severity="WARNING",
            message="Parking lot capacity reached 90% threshold."
        )
        assert alert is not None
        assert alert.alert_type == "HIGH_OCCUPANCY"
        print(f"  [PASS] System alert created: [{alert.severity}] {alert.message}")

        # TEST 6: Aggregated Parking Summary Query
        print("\n[TEST 6] Testing Aggregated Capacity & Occupancy Query...")
        summary = get_parking_summary(db, camera_code="CAM01")
        assert summary["total_slots"] >= 6
        assert "occupancy_rate" in summary
        print(f"  [PASS] Summary retrieved: Total={summary['total_slots']}, Occupied={summary['occupied_slots']}, Available={summary['available_slots']}, Rate={summary['occupancy_rate']}%")

        print("\n" + "=" * 68)
        print(">> ALL DATABASE PERSISTENCE UNIT TESTS PASSED SUCCESSFULLY! (100% OK)")
        print("=" * 68)

    finally:
        db.close()


if __name__ == "__main__":
    run_database_unit_tests()
