"""
Parking Package Interface
Exports ParkingSlot model, SlotManager, and OccupancyEngine.
"""

from .slot import ParkingSlot
from .manager import SlotManager, STATUS_COLORS, DEFAULT_SLOTS_FILE
from .occupancy_engine import OccupancyEngine, OccupancyConfig, SlotStateTransition

__all__ = [
    "ParkingSlot",
    "SlotManager",
    "OccupancyEngine",
    "OccupancyConfig",
    "SlotStateTransition",
    "STATUS_COLORS",
    "DEFAULT_SLOTS_FILE"
]
