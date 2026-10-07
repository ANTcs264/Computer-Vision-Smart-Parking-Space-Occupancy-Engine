#!/usr/bin/env python3
"""
Parking Event Engine Verification Script
Validates event schemas, dispatcher pub/sub routing, and JSON-L audit file persistence.
"""

import json
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv_engine.events import (
    EventType,
    ParkingEvent,
    EventDispatcher,
    LoggingEventHandler,
    FileAuditEventHandler,
    MemoryQueueHandler
)


def run_events_unit_tests():
    print("=" * 68)
    print("        SMART PARKING AI - EVENT ENGINE VERIFICATION        ")
    print("=" * 68)

    test_log_file = PROJECT_ROOT / "logs" / "test_events.jsonl"
    if test_log_file.is_file():
        test_log_file.unlink()

    # 1. Initialize Dispatcher and Handlers
    dispatcher = EventDispatcher(camera_id="CAM01")
    mem_handler = MemoryQueueHandler(maxsize=100)
    file_handler = FileAuditEventHandler(log_path=test_log_file)
    log_handler = LoggingEventHandler()

    dispatcher.register_handler(mem_handler)
    dispatcher.register_handler(file_handler)
    dispatcher.register_handler(log_handler)

    # TEST 1: Schema Construction & Serialization
    print("\n[TEST 1] Testing ParkingEvent Schema & JSON Serialization...")
    evt = ParkingEvent(
        event_type=EventType.SLOT_OCCUPIED,
        camera_id="CAM01",
        slot_id="P01",
        vehicle_id=17,
        confidence=0.94,
        metadata={"dwell_expected_min": 45}
    )
    raw_dict = evt.to_dict()
    assert raw_dict["event_type"] == "SLOT_OCCUPIED"
    assert raw_dict["vehicle_id"] == 17
    assert raw_dict["slot_id"] == "P01"
    assert raw_dict["confidence"] == 0.94

    json_str = evt.to_json()
    reconstructed = ParkingEvent.from_dict(json.loads(json_str))
    assert reconstructed.event_id == evt.event_id
    assert reconstructed.event_type == EventType.SLOT_OCCUPIED
    print("  [PASS] Event schema and bidirectional JSON serialization verified.")

    # TEST 2: Dissemination of All 8 Domain Event Types
    print("\n[TEST 2] Testing Dispatching All 8 Core Domain Events...")
    e1 = dispatcher.emit_vehicle_entered(vehicle_id=42, tripwire_name="NORTH GATE")
    e2 = dispatcher.emit_vehicle_exited(vehicle_id=42, tripwire_name="NORTH GATE")
    e3 = dispatcher.emit_slot_occupied(slot_id="P03", vehicle_id=42, confidence=0.96)
    e4 = dispatcher.emit_slot_available(slot_id="P03", dwell_duration_sec=142.5)
    e5 = dispatcher.emit_parking_warning(occupancy_rate=88.5)
    e6 = dispatcher.emit_parking_full(total_slots=50)
    e7 = dispatcher.emit_camera_offline(last_frame_sec=3.5)
    e8 = dispatcher.emit_low_confidence(avg_confidence=0.32)

    assert dispatcher.total_dispatched == 8
    print(f"  [PASS] Dispatched {dispatcher.total_dispatched} events through EventDispatcher.")

    # TEST 3: Memory Queue Delivery
    print("\n[TEST 3] Testing In-Memory Queue Delivery...")
    queued = mem_handler.pop_all()
    assert len(queued) == 8
    types = [e.event_type for e in queued]
    assert EventType.VEHICLE_ENTERED in types
    assert EventType.PARKING_FULL in types
    assert EventType.SLOT_AVAILABLE in types
    assert queued[3].metadata["dwell_duration_sec"] == 142.5
    print(f"  [PASS] All {len(queued)} events captured in MemoryQueueHandler.")

    # TEST 4: File Audit Log Persistence
    print("\n[TEST 4] Testing JSON-Lines Audit File Persistence...")
    assert test_log_file.is_file(), "Audit file must exist on disk."
    with open(test_log_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 8
    first_record = json.loads(lines[0])
    assert first_record["event_type"] == "VEHICLE_ENTERED"
    assert first_record["vehicle_id"] == 42
    print(f"  [PASS] Audit log contains {len(lines)} JSON records: {test_log_file}")

    print("\n" + "=" * 68)
    print(">> ALL EVENT ENGINE UNIT TESTS PASSED SUCCESSFULLY! (100% OK)")
    print("=" * 68)


if __name__ == "__main__":
    run_events_unit_tests()
