"""
Parking Slot Geometry & Model
Encapsulates individual parking slot boundary geometry, status, and spatial intersections.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
from shapely.geometry import Point, Polygon, box


@dataclass
class ParkingSlot:
    """
    Represents a geometrically defined parking slot bay.
    """
    slot_id: str
    polygon: List[List[int]]  # List of [x, y] corner vertices
    camera_id: str = "CAM01"
    zone_id: str = "ZONE_A"
    status: str = "AVAILABLE"  # AVAILABLE, OCCUPIED, RESERVED
    assigned_vehicle_id: int | None = None
    confidence: float = 0.0

    _shapely_poly: Polygon | None = field(init=False, default=None, repr=False)
    _np_poly: np.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self._init_geometry()

    def _init_geometry(self):
        """Initializes and caches geometric representations."""
        # Convert to tuple coordinates for Shapely
        coords = [tuple(p) for p in self.polygon]
        self._shapely_poly = Polygon(coords)
        self._np_poly = np.array(self.polygon, dtype=np.int32)

    @property
    def shapely_polygon(self) -> Polygon:
        """Returns cached Shapely Polygon for geometric queries."""
        if self._shapely_poly is None:
            self._init_geometry()
        return self._shapely_poly

    @property
    def np_polygon(self) -> np.ndarray:
        """Returns OpenCV compatible polygon array shape (N, 2)."""
        if self._np_poly is None:
            self._init_geometry()
        return self._np_poly

    @property
    def area(self) -> float:
        """Calculates planar surface area of the slot in square pixels."""
        return self.shapely_polygon.area

    @property
    def centroid(self) -> Tuple[int, int]:
        """Calculates center point (xc, yc) of the slot polygon."""
        c = self.shapely_polygon.centroid
        return int(c.x), int(c.y)

    def contains_point(self, x: int, y: int) -> bool:
        """Checks if a point (x, y), e.g. vehicle footprint, is strictly inside the slot."""
        return self.shapely_polygon.contains(Point(x, y))

    def intersection_ratio(self, bbox: Tuple[int, int, int, int]) -> float:
        """
        Calculates Intersection Over Slot Area ratio with a bounding box (x1, y1, x2, y2).
        Ratio = Area(Slot ∩ Bbox) / Area(Slot)
        """
        x1, y1, x2, y2 = bbox
        box_poly = box(x1, y1, x2, y2)
        if not self.shapely_polygon.intersects(box_poly):
            return 0.0

        intersection = self.shapely_polygon.intersection(box_poly)
        return float(intersection.area / self.area) if self.area > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes slot to JSON-compatible dictionary."""
        return {
            "slot_id": self.slot_id,
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "status": self.status,
            "polygon": self.polygon
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParkingSlot":
        """Deserializes slot from JSON dictionary."""
        return cls(
            slot_id=str(data["slot_id"]),
            polygon=data["polygon"],
            camera_id=data.get("camera_id", "CAM01"),
            zone_id=data.get("zone_id", "ZONE_A"),
            status=data.get("status", "AVAILABLE")
        )
