"""
Tracking Package Interface
Exports TrackedVehicle, VehicleTracker (ByteTrack), and VehicleFlowCounter.
"""

from .vehicle import TrackedVehicle
from .tracker import VehicleTracker, get_color_for_id
from .counting import CountingTripwire, VehicleFlowCounter

__all__ = [
    "TrackedVehicle",
    "VehicleTracker",
    "CountingTripwire",
    "VehicleFlowCounter",
    "get_color_for_id"
]
