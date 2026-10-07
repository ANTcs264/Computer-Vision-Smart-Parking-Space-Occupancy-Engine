#!/usr/bin/env python3
"""
Vehicle Tracking & Directional Counting Verification Script
Validates TrackedVehicle trajectory management, 2D line segment intersection,
and directional entry/exit tripwire counting.
"""

import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv_engine.tracking import TrackedVehicle, CountingTripwire, VehicleFlowCounter


def run_tracking_unit_tests():
    print("=" * 68)
    print("      SMART PARKING AI - TRACKING & COUNTING VERIFICATION    ")
    print("=" * 68)

    # TEST 1: TrackedVehicle Model & Trajectory Queue
    print("\n[TEST 1] Testing TrackedVehicle Trajectory & Footprint...")
    now = time.time()
    vehicle = TrackedVehicle(
        track_id=17,
        class_name="car",
        class_id=2,
        box=(100, 200, 240, 320),
        confidence=0.94,
        first_seen=now - 5.0,
        last_seen=now
    )
    assert vehicle.track_id == 17
    assert vehicle.center == (170, 260)
    assert vehicle.bottom_center == (170, 320)  # (xc, ymax)
    assert vehicle.dwell_time_sec >= 4.9

    # Add trajectory movement points
    vehicle.update_position((110, 210, 250, 330), 0.95, now + 0.1)
    vehicle.update_position((120, 220, 260, 340), 0.96, now + 0.2)
    assert len(vehicle.trajectory) == 3
    print(f"  [PASS] TrackedVehicle ID {vehicle.track_id} validated. Footprint: {vehicle.bottom_center}, Trajectory points: {len(vehicle.trajectory)}")

    # TEST 2: Directional Tripwire Entry Crossing
    print("\n[TEST 2] Testing Directional Entry Crossing (Southward)...")
    # Horizontal tripwire across y = 350 from x=100 to x=900
    tripwire = CountingTripwire(name="GATE 1", pt1=(100, 350), pt2=(900, 350))

    # Vehicle moving from y=340 to y=360 (crossing downwards)
    v_entry = TrackedVehicle(
        track_id=101,
        class_name="car",
        class_id=2,
        box=(400, 280, 500, 340),  # footprint at y=340
        confidence=0.91
    )
    # Move across the line to y=365
    v_entry.update_position((400, 305, 500, 365), 0.92, time.time())

    result = tripwire.evaluate_crossing(v_entry)
    assert result == "ENTRY", f"Expected ENTRY, got '{result}'"
    assert tripwire.entry_count == 1
    assert tripwire.exit_count == 0
    print(f"  [PASS] Vehicle 101 crossed gate. Result: {result} (Tally: IN={tripwire.entry_count}, OUT={tripwire.exit_count})")

    # TEST 3: Duplicate Counting Prevention
    print("\n[TEST 3] Testing Duplicate Crossing Prevention...")
    # Advance vehicle further to y=380
    v_entry.update_position((400, 320, 500, 380), 0.93, time.time())
    repeat_result = tripwire.evaluate_crossing(v_entry)
    assert repeat_result == "", "Subsequent frames must NOT re-count the same vehicle!"
    assert tripwire.entry_count == 1, "Entry tally must remain 1!"
    print(f"  [PASS] Duplicate crossing prevented successfully! Entry count remains: {tripwire.entry_count}")

    # TEST 4: Directional Tripwire Exit Crossing
    print("\n[TEST 4] Testing Directional Exit Crossing (Northward)...")
    v_exit = TrackedVehicle(
        track_id=102,
        class_name="truck",
        class_id=7,
        box=(600, 310, 750, 365),  # footprint at y=365 (below line)
        confidence=0.88
    )
    # Move across the line to y=335 (above line)
    v_exit.update_position((600, 280, 750, 335), 0.89, time.time())

    exit_result = tripwire.evaluate_crossing(v_exit)
    assert exit_result == "EXIT", f"Expected EXIT, got '{exit_result}'"
    assert tripwire.exit_count == 1
    print(f"  [PASS] Vehicle 102 crossed gate. Result: {exit_result} (Tally: IN={tripwire.entry_count}, OUT={tripwire.exit_count})")

    # TEST 5: VehicleFlowCounter Multi-Line Manager
    print("\n[TEST 5] Testing VehicleFlowCounter Manager...")
    counter = VehicleFlowCounter(tripwires=[tripwire])
    assert counter.total_entry_count == 1
    assert counter.total_exit_count == 1
    assert counter.total_unique_vehicles == 0  # not yet updated through counter.update()
    counter.update([v_entry, v_exit])
    assert counter.total_unique_vehicles == 2
    print(f"  [PASS] Flow Counter verified: Unique Vehicles={counter.total_unique_vehicles}, Total Entries={counter.total_entry_count}, Total Exits={counter.total_exit_count}")

    print("\n" + "=" * 68)
    print(">> ALL TRACKING & COUNTING UNIT TESTS PASSED SUCCESSFULLY! (100% OK)")
    print("=" * 68)


if __name__ == "__main__":
    run_tracking_unit_tests()
