#!/usr/bin/env python3
"""
Synthetic Parking Lot Video Generator
Generates a realistic 1280x720 @ 30 FPS MP4 video with parking bays and moving vehicles.
Ensures developers can test the CV pipeline immediately without needing a physical webcam.
"""

import math
import os
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "datasets" / "sample_parking.mp4"


def generate_sample_parking_video(
    output_path: Path = OUTPUT_PATH,
    duration_sec: int = 10,
    fps: int = 30,
    width: int = 1280,
    height: int = 720
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    print(f"Generating {duration_sec}s ({total_frames} frames) synthetic parking video: {output_path}...")

    # Parking slot definitions (6 slots: 3 top row, 3 bottom row)
    slots = [
        {"id": "P01", "box": (120, 100, 260, 260)},
        {"id": "P02", "box": (300, 100, 440, 260)},
        {"id": "P03", "box": (480, 100, 620, 260)},
        {"id": "P04", "box": (120, 460, 260, 620)},
        {"id": "P05", "box": (300, 460, 440, 620)},
        {"id": "P06", "box": (480, 460, 620, 620)},
    ]

    for frame_idx in range(total_frames):
        # 1. Base dark asphalt background
        frame = np.full((height, width, 3), (38, 38, 38), dtype=np.uint8)

        # Drive lane markings
        cv2.line(frame, (50, 360), (width - 50, 360), (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, "CENTRAL DRIVING LANE", (700, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)

        # 2. Draw Parking Slot Bays
        for slot in slots:
            x1, y1, x2, y2 = slot["box"]
            # Draw slot boundary
            cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 2)
            # Label
            cv2.putText(frame, slot["id"], (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

        # 3. Parked Car 1 (Static in P01)
        cv2.rectangle(frame, (135, 115), (245, 245), (180, 50, 50), -1)  # Blue Car
        cv2.rectangle(frame, (135, 115), (245, 245), (255, 255, 255), 2)
        cv2.putText(frame, "PARKED", (150, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # 4. Parked Car 2 (Static in P06)
        cv2.rectangle(frame, (495, 475), (605, 605), (50, 140, 50), -1)  # Green Car
        cv2.rectangle(frame, (495, 475), (605, 605), (255, 255, 255), 2)
        cv2.putText(frame, "PARKED", (510, 545), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # 5. Moving Car: Drives along center lane and maneuvers into slot P05!
        t = frame_idx / total_frames  # normalized 0.0 -> 1.0
        if t < 0.45:
            # Driving rightwards down lane
            car_x = int(1200 - t * 1600)
            car_y = 310
            cv2.rectangle(frame, (car_x, car_y), (car_x + 130, car_y + 80), (50, 50, 190), -1)  # Red car
            cv2.rectangle(frame, (car_x, car_y), (car_x + 130, car_y + 80), (255, 255, 255), 2)
            cv2.putText(frame, "MOVING", (car_x + 15, car_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        elif t < 0.70:
            # Turning into slot P05
            fraction = (t - 0.45) / 0.25
            car_x = int(480 - fraction * 170)
            car_y = int(310 + fraction * 170)
            cv2.rectangle(frame, (car_x, car_y), (car_x + 110, car_y + 120), (50, 50, 190), -1)
            cv2.rectangle(frame, (car_x, car_y), (car_x + 110, car_y + 120), (255, 255, 255), 2)
            cv2.putText(frame, "PARKING", (car_x + 10, car_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        else:
            # Parked completely inside slot P05
            cv2.rectangle(frame, (315, 475), (425, 605), (50, 50, 190), -1)
            cv2.rectangle(frame, (315, 475), (425, 605), (255, 255, 255), 2)
            cv2.putText(frame, "PARKED", (330, 545), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Simulation watermark
        cv2.putText(frame, f"TEST SCENARIO - FRAME {frame_idx:04d} / {total_frames}", (820, 680),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"Sample parking video created successfully: {output_path} ({os.path.getsize(output_path)} bytes)")


if __name__ == "__main__":
    generate_sample_parking_video()
