"""
Parking Slot Manager Module
Manages collection of parking bays, file persistence (parking_slots.json),
and semi-transparent HUD rendering.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .slot import ParkingSlot

logger = logging.getLogger("cv_engine.parking")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SLOTS_FILE = PROJECT_ROOT / "cv_engine" / "parking" / "parking_slots.json"

# BGR Status Colors
STATUS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "AVAILABLE": (34, 197, 94),   # Bright Green (BGR)
    "OCCUPIED": (50, 50, 230),    # Bright Red (BGR)
    "RESERVED": (230, 140, 30),   # Blue/Orange (BGR)
}
DEFAULT_COLOR = (180, 180, 180)


class SlotManager:
    """
    Manages loading, updating, saving, and rendering of all parking slot boundaries.
    """

    def __init__(self, config_path: Optional[Path | str] = None):
        self.config_path = Path(config_path) if config_path else DEFAULT_SLOTS_FILE
        self.camera_id = "CAM01"
        self.zone_id = "ZONE_A"
        self.resolution: Tuple[int, int] = (1280, 720)
        self.description = ""
        self._slots: Dict[str, ParkingSlot] = {}

        if self.config_path.is_file():
            self.load()
        else:
            logger.warning(f"Slots configuration file not found at: {self.config_path}")

    @property
    def slots(self) -> List[ParkingSlot]:
        """Returns ordered list of all active parking slots."""
        return list(self._slots.values())

    @property
    def total_count(self) -> int:
        return len(self._slots)

    @property
    def occupied_count(self) -> int:
        return sum(1 for s in self._slots.values() if s.status == "OCCUPIED")

    @property
    def available_count(self) -> int:
        return sum(1 for s in self._slots.values() if s.status == "AVAILABLE")

    @property
    def occupancy_rate(self) -> float:
        return (self.occupied_count / self.total_count * 100.0) if self.total_count > 0 else 0.0

    def get_slot(self, slot_id: str) -> Optional[ParkingSlot]:
        return self._slots.get(slot_id)

    def add_slot(self, slot: ParkingSlot) -> None:
        self._slots[slot.slot_id] = slot

    def remove_slot(self, slot_id: str) -> bool:
        if slot_id in self._slots:
            del self._slots[slot_id]
            return True
        return False

    def load(self, path: Optional[Path | str] = None) -> bool:
        """Loads slot configuration from JSON file."""
        target_path = Path(path) if path else self.config_path
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.camera_id = data.get("camera_id", "CAM01")
            self.zone_id = data.get("zone_id", "ZONE_A")
            self.resolution = tuple(data.get("resolution", [1280, 720]))
            self.description = data.get("description", "")

            self._slots.clear()
            for item in data.get("slots", []):
                slot = ParkingSlot.from_dict(item)
                self._slots[slot.slot_id] = slot

            logger.info(f"Loaded {len(self._slots)} parking slots from: {target_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load parking slots from {target_path}: {e}")
            return False

    def save(self, path: Optional[Path | str] = None) -> bool:
        """Saves current slots configuration to JSON file."""
        target_path = Path(path) if path else self.config_path
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "camera_id": self.camera_id,
                "zone_id": self.zone_id,
                "resolution": list(self.resolution),
                "description": self.description,
                "slots": [slot.to_dict() for slot in self.slots]
            }
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self._slots)} parking slots to: {target_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save parking slots to {target_path}: {e}")
            return False

    def draw_slots(self, frame: np.ndarray, alpha: float = 0.35) -> np.ndarray:
        """
        Renders color-coded semi-transparent polygon fills, crisp borders, and ID badges.

        Args:
            frame: OpenCV BGR image (H, W, 3).
            alpha: Transparency factor for slot fill (0.0 = invisible, 1.0 = solid).

        Returns:
            Annotated frame.
        """
        if not self._slots:
            return frame

        overlay = frame.copy()

        # Step 1: Draw filled polygons on overlay
        for slot in self.slots:
            color = STATUS_COLORS.get(slot.status.upper(), DEFAULT_COLOR)
            pts = slot.np_polygon.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)

        # Blend overlay with original frame for semi-transparency
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        # Step 2: Draw crisp borders and ID badges on top of blended frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        for slot in self.slots:
            color = STATUS_COLORS.get(slot.status.upper(), DEFAULT_COLOR)
            pts = slot.np_polygon.reshape((-1, 1, 2))

            # Crisp outline border
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

            # Slot ID badge centered on centroid
            cx, cy = slot.centroid
            badge_text = slot.slot_id
            font_scale = 0.52
            thick = 1
            (tw, th), _ = cv2.getTextSize(badge_text, font, font_scale, thick)

            # Pill badge background
            bx1 = cx - tw // 2 - 6
            by1 = cy - th // 2 - 4
            bx2 = cx + tw // 2 + 6
            by2 = cy + th // 2 + 4

            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (15, 23, 42), -1)  # Dark slate
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)

            cv2.putText(
                frame,
                badge_text,
                (cx - tw // 2, cy + th // 2 - 1),
                font,
                font_scale,
                (241, 245, 249),
                thick,
                cv2.LINE_AA
            )

        return frame

    def draw_slot_kpi_banner(self, frame: np.ndarray) -> np.ndarray:
        """
        Renders a floating KPI badge showing slot capacity metrics.
        """
        h, w = frame.shape[:2]
        banner_w = 340
        banner_h = 32
        x1 = w - banner_w - 15
        y1 = 80

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x1 + banner_w, y1 + banner_h), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x1 + banner_w, y1 + banner_h), (100, 116, 139), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        kpi_text = f"SLOTS: {self.total_count} | AVAIL: {self.available_count} | OCC: {self.occupied_count} ({self.occupancy_rate:.0f}%)"
        cv2.putText(frame, kpi_text, (x1 + 10, y1 + 21), font, 0.44, (241, 245, 249), 1, cv2.LINE_AA)
        return frame
