#!/usr/bin/env python3
"""
Interactive Parking Slot Annotation Tool
Allows users to define custom 4-point convex polygons over camera frames or video footage.
Saves geometry to parking_slots.json.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv_engine.camera import CameraConfig, VideoCamera
from cv_engine.parking import ParkingSlot, SlotManager, DEFAULT_SLOTS_FILE

# Global state for mouse interactions
current_points: List[Tuple[int, int]] = []
slots_list: List[ParkingSlot] = []
window_name = "Smart Parking AI - Slot Polygon Annotator (Phase 4)"
base_frame = None


def mouse_callback(event, x, y, flags, param):
    """Handles mouse click events for defining slot polygon vertices."""
    global current_points, slots_list

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        print(f"[Annotator] Clicked point {len(current_points)}/4: ({x}, {y})")

        # Once 4 points are selected, form a completed parking slot
        if len(current_points) == 4:
            next_id = f"P{len(slots_list) + 1:02d}"
            poly_coords = [[pt[0], pt[1]] for pt in current_points]
            new_slot = ParkingSlot(
                slot_id=next_id,
                polygon=poly_coords,
                camera_id="CAM01",
                zone_id="ZONE_A",
                status="AVAILABLE"
            )
            slots_list.append(new_slot)
            print(f"[Annotator] Created slot: {next_id} with vertices: {poly_coords}")
            current_points = []


def run_annotator(
    source: str | int = "datasets/sample_parking.mp4",
    output_path: Path = DEFAULT_SLOTS_FILE,
    width: int = 1280,
    height: int = 720,
    verify_only: bool = False
):
    """
    Main interactive annotation loop.
    """
    global base_frame, current_points, slots_list

    print("=" * 68)
    print("      SMART PARKING AI - PARKING SLOT ANNOTATION TOOL      ")
    print(f"   Target Source : {source}")
    print(f"   Output JSON   : {output_path}")
    print("=" * 68)

    # 1. Load existing slots if JSON already exists
    slot_manager = SlotManager(output_path)
    slots_list = list(slot_manager.slots)
    print(f"Loaded {len(slots_list)} existing slots from: {output_path}")

    # If verify_only mode is requested (for CI / automated tests), validate and exit
    if verify_only:
        print("[Verify] Validating loaded parking slots configuration...")
        for s in slots_list:
            assert len(s.polygon) == 4, f"Slot {s.slot_id} must have 4 vertices."
            assert s.area > 0, f"Slot {s.slot_id} area must be > 0."
            print(f"  [OK] Slot {s.slot_id} [Centroid: {s.centroid}, Area: {s.area:.0f} px^2]")
        print(">> SUCCESS: Parking slots configuration verified!")
        return

    # 2. Grab a high-quality frame from camera or video
    cam_config = CameraConfig(source=source, width=width, height=height)
    with VideoCamera(cam_config) as camera:
        success, frame = camera.read()
        if not success or frame is None:
            print("ERROR: Failed to read reference frame from video source.")
            return

    base_frame = frame.copy()

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n--- INSTRUCTIONS ---")
    print("• LEFT-CLICK 4 corners of a parking bay to define a slot (P01, P02, ...)")
    print("• Press 's' : Save slots to JSON")
    print("• Press 'c' : Clear currently active (unfinished) points")
    print("• Press 'u' : Undo / delete last created slot")
    print("• Press 'r' : Reset / delete all slots")
    print("• Press 'q' / ESC : Save and Exit\n")

    while True:
        display = base_frame.copy()

        # Render all completed slots
        temp_manager = SlotManager()
        for s in slots_list:
            temp_manager.add_slot(s)
        temp_manager.draw_slots(display, alpha=0.35)

        # Render currently clicked unfinished points & guide lines
        if len(current_points) > 0:
            for i, pt in enumerate(current_points):
                cv2.circle(display, pt, 5, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(display, f"{i+1}", (pt[0] + 8, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            if len(current_points) > 1:
                pts_arr = np.array(current_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display, [pts_arr], isClosed=False, color=(0, 255, 255), thickness=2)

        # Top Instruction Bar
        h, w = display.shape[:2]
        info_banner = display.copy()
        cv2.rectangle(info_banner, (0, 0), (w, 38), (15, 23, 42), -1)
        cv2.addWeighted(info_banner, 0.85, display, 0.15, 0, display)
        info_text = f"Slots Defined: {len(slots_list)} | Points: {len(current_points)}/4 | [s] Save | [u] Undo | [c] Clear | [q] Quit"
        cv2.putText(display, info_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (241, 245, 249), 1, cv2.LINE_AA)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):  # 'q' or ESC
            save_slots(output_path, slots_list)
            break
        elif key == ord('s'):  # 's' Save
            save_slots(output_path, slots_list)
        elif key == ord('c'):  # 'c' Clear active points
            current_points = []
            print("[Annotator] Unfinished points cleared.")
        elif key == ord('u'):  # 'u' Undo last slot
            if slots_list:
                removed = slots_list.pop()
                print(f"[Annotator] Removed last slot: {removed.slot_id}")
        elif key == ord('r'):  # 'r' Reset all
            slots_list = []
            current_points = []
            print("[Annotator] All slots cleared.")

    cv2.destroyAllWindows()


def save_slots(path: Path, slots: List[ParkingSlot]):
    """Persists slots to JSON."""
    data = {
        "camera_id": "CAM01",
        "zone_id": "ZONE_A",
        "resolution": [1280, 720],
        "description": "Custom User-Annotated Parking Slots",
        "slots": [s.to_dict() for s in slots]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[Annotator] Successfully saved {len(slots)} parking slots to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Smart Parking AI - Slot Polygon Annotator")
    parser.add_argument("--source", type=str, default="datasets/sample_parking.mp4", help="Camera index or video file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_SLOTS_FILE), help="Output JSON path")
    parser.add_argument("--width", type=int, default=1280, help="Target width")
    parser.add_argument("--height", type=int, default=720, help="Target height")
    parser.add_argument("--verify", action="store_true", help="Non-interactive verification mode")

    args = parser.parse_args()
    raw_source = args.source
    source = int(raw_source) if raw_source.isdigit() else raw_source

    run_annotator(
        source=source,
        output_path=Path(args.output),
        width=args.width,
        height=args.height,
        verify_only=args.verify
    )


if __name__ == "__main__":
    main()
