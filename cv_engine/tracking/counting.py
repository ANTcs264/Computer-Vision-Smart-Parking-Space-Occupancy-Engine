"""
Vehicle Directional Counting & Tripwire Module
Tracks vehicle line-crossings using 2D geometric vector intersections.
Calculates Entry Count, Exit Count, Active Vehicles, and prevents duplicate tallies.
"""

from dataclasses import dataclass, field
import logging
import time
from typing import List, Optional, Set, Tuple
import cv2
import numpy as np

from .vehicle import TrackedVehicle

logger = logging.getLogger("cv_engine.counting")


def ccw(A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int]) -> bool:
    """Tests if three points are listed in counterclockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def lines_intersect(
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    p3: Tuple[int, int],
    p4: Tuple[int, int]
) -> bool:
    """Determines whether line segment (p1-p2) intersects line segment (p3-p4)."""
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


@dataclass
class CountingTripwire:
    """
    Virtual gate tripwire for measuring vehicle entry and exit flow.
    Defined by a directed 2D line segment from pt1 to pt2.
    """
    name: str
    pt1: Tuple[int, int]
    pt2: Tuple[int, int]
    entry_count: int = 0
    exit_count: int = 0
    counted_entry_ids: Set[int] = field(default_factory=set)
    counted_exit_ids: Set[int] = field(default_factory=set)

    def evaluate_crossing(self, vehicle: TrackedVehicle) -> str:
        """
        Tests whether the vehicle's recent motion crossed this tripwire line.
        
        Returns:
            "ENTRY" if crossed in entry direction,
            "EXIT" if crossed in exit direction,
            "" if no crossing.
        """
        if len(vehicle.trajectory) < 2:
            return ""

        # Last two trajectory coordinates
        prev_pt = vehicle.trajectory[-2]
        curr_pt = vehicle.trajectory[-1]

        # Check line intersection
        if lines_intersect(prev_pt, curr_pt, self.pt1, self.pt2):
            # Compute cross product to determine crossing direction relative to tripwire vector
            v_wire = (self.pt2[0] - self.pt1[0], self.pt2[1] - self.pt1[1])
            v_motion = (curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1])

            # 2D cross product: v_wire_x * v_motion_y - v_wire_y * v_motion_x
            cross = v_wire[0] * v_motion[1] - v_wire[1] * v_motion[0]

            if cross > 0:
                if vehicle.track_id not in self.counted_entry_ids:
                    self.entry_count += 1
                    self.counted_entry_ids.add(vehicle.track_id)
                    vehicle.counted_entry = True
                    logger.info(f"🚗 TRIPWIRE [{self.name}]: Vehicle ID {vehicle.track_id} crossed [ENTRY] (Total: {self.entry_count})")
                    return "ENTRY"
            else:
                if vehicle.track_id not in self.counted_exit_ids:
                    self.exit_count += 1
                    self.counted_exit_ids.add(vehicle.track_id)
                    vehicle.counted_exit = True
                    logger.info(f"🚙 TRIPWIRE [{self.name}]: Vehicle ID {vehicle.track_id} crossed [EXIT] (Total: {self.exit_count})")
                    return "EXIT"

        return ""

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Renders the tripwire line, orientation arrow, and tally counter on the frame."""
        # Main tripwire line
        cv2.line(frame, self.pt1, self.pt2, (234, 179, 8), 3, cv2.LINE_AA)  # Yellow Amber
        cv2.circle(frame, self.pt1, 5, (234, 179, 8), -1)
        cv2.circle(frame, self.pt2, 5, (234, 179, 8), -1)

        # Label badge at line midpoint
        mid_x = (self.pt1[0] + self.pt2[0]) // 2
        mid_y = (self.pt1[1] + self.pt2[1]) // 2

        label = f"{self.name} | IN: {self.entry_count}  OUT: {self.exit_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)

        bx1 = mid_x - tw // 2 - 6
        by1 = mid_y - th - 8
        bx2 = mid_x + tw // 2 + 6
        by2 = mid_y + 4

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (15, 23, 42), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (234, 179, 8), 1, cv2.LINE_AA)
        cv2.putText(frame, label, (mid_x - tw // 2, by1 + th + 4), font, font_scale, (241, 245, 249), 1, cv2.LINE_AA)

        return frame


class VehicleFlowCounter:
    """
    Manages multi-tripwire vehicle entry/exit counting and turnover velocity.
    """

    def __init__(self, tripwires: Optional[List[CountingTripwire]] = None):
        self.tripwires: List[CountingTripwire] = tripwires or [
            # Default tripwire across the central driveway of 1280x720 video
            CountingTripwire("MAIN GATE", pt1=(100, 360), pt2=(1180, 360))
        ]
        self.seen_vehicle_ids: Set[int] = set()

    @property
    def total_entry_count(self) -> int:
        return sum(tw.entry_count for tw in self.tripwires)

    @property
    def total_exit_count(self) -> int:
        return sum(tw.exit_count for tw in self.tripwires)

    @property
    def total_unique_vehicles(self) -> int:
        return len(self.seen_vehicle_ids)

    def update(self, vehicles: List[TrackedVehicle]) -> List[Tuple[CountingTripwire, TrackedVehicle, str]]:
        """
        Updates line crossing statistics for all tracked vehicles in the frame.
        """
        crossings = []
        for v in vehicles:
            self.seen_vehicle_ids.add(v.track_id)
            for tw in self.tripwires:
                event = tw.evaluate_crossing(v)
                if event:
                    crossings.append((tw, v, event))
        return crossings

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Renders all tripwire lines on the video frame."""
        for tw in self.tripwires:
            tw.draw(frame)
        return frame
