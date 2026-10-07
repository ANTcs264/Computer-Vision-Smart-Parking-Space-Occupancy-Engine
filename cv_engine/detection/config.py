"""
YOLO Vehicle Detection Configuration Module
Manages model paths, inference thresholds, image sizing, and class filters.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Standard COCO transportation class IDs
# 2: car, 3: motorcycle, 5: bus, 7: truck
DEFAULT_VEHICLE_CLASSES = [2, 3, 5, 7]
DEFAULT_CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


@dataclass
class DetectionConfig:
    """Configuration schema for YOLO Vehicle Detection."""
    model_path: str = "models/yolo/yolov8n.pt"
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.45
    image_size: int = 640
    device: str = "cpu"
    target_classes: List[int] = field(default_factory=lambda: list(DEFAULT_VEHICLE_CLASSES))

    @classmethod
    def from_env(cls) -> "DetectionConfig":
        """Factory method to construct config directly from system environment / .env."""
        model_name = os.getenv("YOLO_MODEL", "yolov8n.pt").strip()
        
        # Check if model exists inside models/yolo/ or root
        local_model_path = PROJECT_ROOT / "models" / "yolo" / model_name
        if local_model_path.is_file():
            model_path = str(local_model_path)
        else:
            model_path = model_name

        conf = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.45"))
        iou = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))
        imgsz = int(os.getenv("IMAGE_SIZE", "640"))

        # Determine optimal compute device
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        return cls(
            model_path=model_path,
            confidence_threshold=conf,
            iou_threshold=iou,
            image_size=imgsz,
            device=device,
            target_classes=list(DEFAULT_VEHICLE_CLASSES)
        )
