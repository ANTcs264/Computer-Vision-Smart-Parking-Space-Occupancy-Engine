#!/usr/bin/env python3
"""
Parking Occupancy Engine Verification Script
Validates spatial intersection math, footprint containment, and temporal hysteresis smoothing.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv_engine.detection import Detection
from cv_engine.parking import ParkingSlot, SlotManager, OccupancyEngine, OccupancyConfig


def run_occupancy_unit_tests():
    print("=" * 68)
    print("      SMART PARKING AI - OCCUPANCY ENGINE VERIFICATION      ")
    print("=" * 68)

    # 1. Initialize Test Slot (P01: 100, 100 to 200, 200 - Area = 10,000 px^2)
    test_slot = ParkingSlot(
        slot_id="P01",
        polygon=[[100, 100], [200, 100], [200, 200], [100, 200]],
        status="AVAILABLE"
    )
    manager = SlotManager()
    manager.add_slot(test_slot)

    config = OccupancyConfig(
        overlap_threshold=0.30,
        hysteresis_window=20,
        occupied_ratio=0.70,   # Requires 14 / 20 positive frames
        vacated_ratio=0.20     # Requires <= 4 / 20 positive frames to vacate
    )
    engine = OccupancyEngine(manager, config)

    # TEST 1: Spatial Geometry & Ground Footprint
    print("\n[TEST 1] Testing Spatial Footprint & Geometric Overlap...")
    # Detection inside P01 (box from 110, 110 to 190, 190 -> footprint at 150, 190)
    inside_det = Detection(box=(110, 110, 190, 190), confidence=0.92, class_id=2, class_name="car")
    is_present, conf, cls_name, overlap = engine._check_spatial_overlap(test_slot, [inside_det])
    assert is_present, "Detection inside slot footprint should evaluate to True."
    print(f"  [PASS] Inside detection matched correctly (Overlap: {overlap:.2f}, Footprint: {inside_det.bottom_center})")

    # Detection outside P01 (box at 300, 300 to 400, 400)
    outside_det = Detection(box=(300, 300, 400, 400), confidence=0.95, class_id=2, class_name="car")
    is_present, conf, cls_name, overlap = engine._check_spatial_overlap(test_slot, [outside_det])
    assert not is_present, "Detection outside slot should evaluate to False."
    print(f"  [PASS] Disjoint vehicle correctly ignored (Overlap: {overlap:.2f})")

    # TEST 2: Temporal Anti-Flicker (Transient Noise Rejection)
    print("\n[TEST 2] Testing Transient Noise Rejection (Single-frame noise)...")
    # Feed 3 momentary false-positive frames into an empty slot
    for f in range(3):
        transitions = engine.update([inside_det])
        assert len(transitions) == 0, f"Transient frame {f+1} should NOT trigger OCCUPIED!"
    assert test_slot.status == "AVAILABLE", "Slot must stay AVAILABLE during brief transient noise!"
    print(f"  [PASS] 3 transient noise frames safely rejected! Status remains: {test_slot.status}")

    # TEST 3: Sustained Presence -> OCCUPIED Transition
    print("\n[TEST 3] Testing Sustained Presence (Slot Confirmation)...")
    # Feed 15 consecutive positive frames to cross the 70% threshold (14/20 frames)
    confirmed_transition = None
    for f in range(15):
        transitions = engine.update([inside_det])
        if transitions:
            confirmed_transition = transitions[0]

    assert confirmed_transition is not None, "Sustained frames MUST trigger OCCUPIED transition!"
    assert test_slot.status == "OCCUPIED", "Slot status must be OCCUPIED."
    assert confirmed_transition.previous_status == "AVAILABLE"
    assert confirmed_transition.new_status == "OCCUPIED"
    print(f"  [PASS] Transition confirmed: {confirmed_transition.previous_status} -> {confirmed_transition.new_status}")

    # TEST 4: Temporary Occlusion Resilience (Holding State)
    print("\n[TEST 4] Testing Occlusion Resilience (Vehicle momentarily missed)...")
    # Feed 2 missing detection frames while OCCUPIED
    for f in range(2):
        transitions = engine.update([])
        assert len(transitions) == 0, "Brief occlusion must NOT falsely vacate the slot!"
    assert test_slot.status == "OCCUPIED", "Slot must hold OCCUPIED status through brief dropouts!"
    print(f"  [PASS] Brief detection dropout absorbed! Status holds: {test_slot.status}")

    # TEST 5: Sustained Departure -> AVAILABLE Transition
    print("\n[TEST 5] Testing Sustained Departure & Dwell Time Calculation...")
    # Feed 18 empty frames to drop below 20% threshold
    vacate_transition = None
    for f in range(18):
        transitions = engine.update([])
        if transitions:
            vacate_transition = transitions[0]

    assert vacate_transition is not None, "Sustained absence MUST trigger AVAILABLE transition!"
    assert test_slot.status == "AVAILABLE", "Slot status must return to AVAILABLE."
    assert vacate_transition.previous_status == "OCCUPIED"
    assert vacate_transition.new_status == "AVAILABLE"
    assert vacate_transition.dwell_duration_sec >= 0.0
    print(f"  [PASS] Transition confirmed: {vacate_transition.previous_status} -> {vacate_transition.new_status}")
    print(f"  [PASS] Dwell duration calculated: {vacate_transition.dwell_duration_sec:.2f} seconds")

    print("\n" + "=" * 68)
    print(">> ALL OCCUPANCY ENGINE UNIT TESTS PASSED SUCCESSFULLY! (100% OK)")
    print("=" * 68)


if __name__ == "__main__":
    run_occupancy_unit_tests()
