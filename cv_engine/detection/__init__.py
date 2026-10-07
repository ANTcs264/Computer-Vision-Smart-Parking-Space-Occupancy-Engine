"""
Detection Package Interface
Exports YOLO detector, detection data classes, and visualization helpers.
"""

from .config import DetectionConfig, DEFAULT_VEHICLE_CLASSES, DEFAULT_CLASS_NAMES
from .detector import Detection, YOLOVehicleDetector
from .utils import draw_detections, get_detection_summary, CLASS_COLORS

__all__ = [
    "DetectionConfig",
    "Detection",
    "YOLOVehicleDetector",
    "draw_detections",
    "get_detection_summary",
    "DEFAULT_VEHICLE_CLASSES",
    "DEFAULT_CLASS_NAMES",
    "CLASS_COLORS"
]
