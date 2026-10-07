"""
Tracked Vehicle Model Module
Represents an individual tracked vehicle instance with identity, kinematics,
trajectory history, and parking slot associations.
"""

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Deque, List, Optional, Tuple


@dataclass
class TrackedVehicle:
    """
    State container for a vehicle tracked across video frames by ByteTrack.
    """
    track_id: int
    class_name: str
    class_id: int
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    trajectory: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=45))
    assigned_slot_id: Optional[str] = None
    is_parked: bool = False
    counted_entry: bool = False
    counted_exit: bool = False

    def __post_init__(self):
        # Seed initial trajectory with the bottom footprint anchor
        self.trajectory.append(self.bottom_center)

    @property
    def x1(self) -> int:
        return self.box[0]

    @property
    def y1(self) -> int:
        return self.box[1]

    @property
    def x2(self) -> int:
        return self.box[2]

    @property
    def y2(self) -> int:
        return self.box[3]

    @property
    def center(self) -> Tuple[int, int]:
        """Geometric box center (xc, yc)."""
        return int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2)

    @property
    def bottom_center(self) -> Tuple[int, int]:
        """Pavement contact patch footprint (xc, ymax)."""
        return int((self.x1 + self.x2) / 2), self.y2

    @property
    def dwell_time_sec(self) -> float:
        """Elapsed time since vehicle was first tracked in seconds."""
        return max(0.0, self.last_seen - self.first_seen)

    def update_position(self, box: Tuple[int, int, int, int], confidence: float, current_time: float):
        """Updates vehicle bounding box, timestamp, and trajectory history."""
        self.box = box
        self.confidence = confidence
        self.last_seen = current_time
        self.trajectory.append(self.bottom_center)
