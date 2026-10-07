"""
Parking Package Interface
Exports ParkingSlot model, SlotManager, and configuration utilities.
"""

from .slot import ParkingSlot
from .manager import SlotManager, STATUS_COLORS, DEFAULT_SLOTS_FILE

__all__ = [
    "ParkingSlot",
    "SlotManager",
    "STATUS_COLORS",
    "DEFAULT_SLOTS_FILE"
]
