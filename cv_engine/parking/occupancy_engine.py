"""
Parking Occupancy Engine Module
Evaluates geometric spatial intersections, ground footprint containment,
and dual-threshold temporal hysteresis smoothing to determine slot occupancy.
"""

from collections import deque
from dataclasses import dataclass, field
import logging
import time
from typing import Dict, List, Optional, Tuple
from shapely.geometry import Point, box

from cv_engine.detection import Detection
from .slot import ParkingSlot
from .manager import SlotManager

logger = logging.getLogger("cv_engine.occupancy")


@dataclass
class OccupancyConfig:
    """Configuration schema for Parking Occupancy Logic."""
    overlap_threshold: float = 0.28
    hysteresis_window: int = 30
    occupied_ratio: float = 0.65  # 65% of window required to confirm OCCUPIED
    vacated_ratio: float = 0.20   # <= 20% of window required to confirm VACATED
    use_footprint_priority: bool = True


@dataclass
class SlotStateTransition:
    """Represents a validated transition event for a parking slot."""
    slot_id: str
    previous_status: str
    new_status: str
    timestamp: float
    confidence: float
    dwell_duration_sec: float = 0.0
    matched_class: str = ""


class OccupancyEngine:
    """
    Core Intelligence Engine.
    Maps localized vehicle detections to parking slots and applies temporal smoothing
    to eliminate flicker.
    """

    def __init__(self, slot_manager: SlotManager, config: Optional[OccupancyConfig] = None):
        self.slot_manager = slot_manager
        self.config = config or OccupancyConfig()

        # State tracking per slot: {slot_id: deque(maxlen=hysteresis_window)}
        self._history: Dict[str, deque] = {}
        # Timestamps of last status change: {slot_id: float}
        self._status_change_time: Dict[str, float] = {}
        # Timestamps when slot became occupied: {slot_id: float}
        self._occupied_since: Dict[str, Optional[float]] = {}
        # Active matched vehicle info: {slot_id: Dict}
        self._active_matches: Dict[str, Dict] = {}

        now = time.time()
        for slot in self.slot_manager.slots:
            self._history[slot.slot_id] = deque(maxlen=self.config.hysteresis_window)
            self._status_change_time[slot.slot_id] = now
            self._occupied_since[slot.slot_id] = now if slot.status == "OCCUPIED" else None
            self._active_matches[slot.slot_id] = {"class": "", "conf": 0.0}

    def update(self, detections: List[Detection]) -> List[SlotStateTransition]:
        """
        Evaluates current frame detections against all parking slots.
        
        Args:
            detections: List of localized vehicles detected by YOLO.
            
        Returns:
            List of confirmed state transition events (e.g. AVAILABLE -> OCCUPIED).
        """
        transitions: List[SlotStateTransition] = []
        now = time.time()

        for slot in self.slot_manager.slots:
            slot_id = slot.slot_id
            if slot_id not in self._history:
                self._history[slot_id] = deque(maxlen=self.config.hysteresis_window)
                self._status_change_time[slot_id] = now
                self._occupied_since[slot_id] = None
                self._active_matches[slot_id] = {"class": "", "conf": 0.0}

            # 1. Spatial Intersection Check
            is_raw_present, best_conf, best_class, best_overlap = self._check_spatial_overlap(slot, detections)

            # 2. Record raw frame observation into rolling hysteresis buffer
            history = self._history[slot_id]
            history.append(1 if is_raw_present else 0)

            # 3. Calculate positive observation ratio normalized over the fixed hysteresis window
            # Normalizing by hysteresis_window prevents single-frame cold-start false positives
            positives = sum(history)
            presence_ratio = positives / self.config.hysteresis_window

            current_status = slot.status.upper()
            target_status = current_status
            dwell_duration = 0.0

            # 4. Temporal State Machine (Dual-Threshold Hysteresis)
            if current_status == "AVAILABLE":
                # Only transition to OCCUPIED if positive detections exceed the occupied ratio
                if presence_ratio >= self.config.occupied_ratio:
                    target_status = "OCCUPIED"
                    self._occupied_since[slot_id] = now
                    dwell_duration = 0.0
                    self._active_matches[slot_id] = {"class": best_class, "conf": best_conf}

            elif current_status == "OCCUPIED":
                # Only transition back to AVAILABLE if presence drops below vacated ratio
                if presence_ratio <= self.config.vacated_ratio:
                    target_status = "AVAILABLE"
                    if self._occupied_since[slot_id] is not None:
                        dwell_duration = now - self._occupied_since[slot_id]
                    self._occupied_since[slot_id] = None
                    self._active_matches[slot_id] = {"class": "", "conf": 0.0}
                else:
                    # Maintain occupancy & update dwell info
                    if is_raw_present:
                        self._active_matches[slot_id] = {"class": best_class, "conf": best_conf}

            # 5. Handle State Transition
            if target_status != current_status:
                logger.info(
                    f"[Occupancy] Slot {slot_id}: {current_status} -> {target_status} "
                    f"(Positive ratio: {presence_ratio:.2f}, Overlap: {best_overlap:.2f})"
                )
                slot.status = target_status
                slot.confidence = best_conf
                self._status_change_time[slot_id] = now

                transition = SlotStateTransition(
                    slot_id=slot_id,
                    previous_status=current_status,
                    new_status=target_status,
                    timestamp=now,
                    confidence=best_conf,
                    dwell_duration_sec=dwell_duration,
                    matched_class=best_class
                )
                transitions.append(transition)

        return transitions

    def _check_spatial_overlap(
        self,
        slot: ParkingSlot,
        detections: List[Detection]
    ) -> Tuple[bool, float, str, float]:
        """
        Determines whether any detection matches the slot using:
        1. Ground Footprint Point Containment (primary)
        2. Polygonal Overlap Ratio (secondary fallback)
        """
        best_overlap = 0.0
        best_conf = 0.0
        best_class = ""
        is_present = False

        slot_poly = slot.shapely_polygon

        for det in detections:
            # 1. Ground Footprint Anchor Check (xc, ymax)
            fx, fy = det.bottom_center
            footprint_inside = slot_poly.contains(Point(fx, fy))

            # 2. Polygon-to-Bounding-Box Overlap Ratio
            x1, y1, x2, y2 = det.box
            det_box = box(x1, y1, x2, y2)

            overlap_ratio = 0.0
            if slot_poly.intersects(det_box):
                inter_area = slot_poly.intersection(det_box).area
                overlap_ratio = inter_area / slot.area if slot.area > 0 else 0.0

            # Match criteria
            if footprint_inside or (overlap_ratio >= self.config.overlap_threshold):
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_conf = det.confidence
                    best_class = det.class_name
                    is_present = True

        return is_present, best_conf, best_class, best_overlap

    def get_slot_telemetry(self, slot_id: str) -> Dict:
        """Returns diagnostic telemetry for an individual slot."""
        history = self._history.get(slot_id, deque())
        positives = sum(history)
        ratio = positives / self.config.hysteresis_window
        match = self._active_matches.get(slot_id, {})
        occupied_since = self._occupied_since.get(slot_id)
        dwell = (time.time() - occupied_since) if occupied_since else 0.0

        return {
            "slot_id": slot_id,
            "presence_ratio": ratio,
            "window_size": len(history),
            "matched_class": match.get("class", ""),
            "confidence": match.get("conf", 0.0),
            "dwell_sec": dwell
        }
