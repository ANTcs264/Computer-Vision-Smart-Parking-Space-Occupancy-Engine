"""
Parking Event Schemas Module
Defines standard domain event types, event structures, and serialization.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Any, Dict, Optional
import uuid


class EventType(str, Enum):
    """Enumeration of all system domain events."""
    VEHICLE_ENTERED = "VEHICLE_ENTERED"
    VEHICLE_EXITED = "VEHICLE_EXITED"
    SLOT_OCCUPIED = "SLOT_OCCUPIED"
    SLOT_AVAILABLE = "SLOT_AVAILABLE"
    PARKING_FULL = "PARKING_FULL"
    PARKING_WARNING = "PARKING_WARNING"
    CAMERA_OFFLINE = "CAMERA_OFFLINE"
    LOW_DETECTION_CONFIDENCE = "LOW_DETECTION_CONFIDENCE"


@dataclass
class ParkingEvent:
    """
    Standard Immutable Domain Event Record.
    """
    event_type: EventType
    camera_id: str = "CAM01"
    vehicle_id: Optional[int] = None
    slot_id: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    timestamp_iso: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes event to clean dictionary with stringified enum."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "timestamp_iso": self.timestamp_iso,
            "camera_id": self.camera_id,
            "event_type": self.event_type.value if isinstance(self.event_type, EventType) else str(self.event_type),
            "vehicle_id": self.vehicle_id,
            "slot_id": self.slot_id,
            "confidence": round(self.confidence, 4),
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        """Serializes event to compact JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParkingEvent":
        """Deserializes event from dictionary."""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).timestamp()),
            timestamp_iso=data.get("timestamp_iso", datetime.now(timezone.utc).isoformat()),
            camera_id=data.get("camera_id", "CAM01"),
            event_type=EventType(data["event_type"]),
            vehicle_id=data.get("vehicle_id"),
            slot_id=data.get("slot_id"),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {})
        )
