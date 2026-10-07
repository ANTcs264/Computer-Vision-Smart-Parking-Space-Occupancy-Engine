"""
YOLO Vehicle Detector Module
Wraps Ultralytics YOLO with vehicle class filtering, tensor post-processing,
and high-precision inference latency tracking.
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from .config import DEFAULT_CLASS_NAMES, DetectionConfig

logger = logging.getLogger("cv_engine.detection")


@dataclass
class Detection:
    """Represents a single localized vehicle detection."""
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str

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
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        """Geometric center of the bounding box (xc, yc)."""
        return int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2)

    @property
    def bottom_center(self) -> Tuple[int, int]:
        """
        Ground footprint anchor point (xc, ymax).
        Crucial for parking slot assignment to avoid false overlap from high vehicle roofs.
        """
        return int((self.x1 + self.x2) / 2), self.y2


class YOLOVehicleDetector:
    """
    Production-grade YOLO Vehicle Detection Wrapper.
    Filters detections strictly to transportation classes and tracks inference latency.
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig.from_env()
        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads YOLO weights into memory and warms up inference engine."""
        logger.info(f"Loading YOLO model from: {self.config.model_path} on device: {self.config.device}...")
        try:
            self.model = YOLO(self.config.model_path)
            # Ensure model saves to models/yolo directory if auto-downloaded
            logger.info("YOLO model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise e

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """
        Runs object detection on a single frame.

        Args:
            frame: BGR image from OpenCV (H, W, 3).

        Returns:
            Tuple of (detections: List[Detection], inference_latency_ms: float)
        """
        if self.model is None or frame is None:
            return [], 0.0

        start_time = time.perf_counter()

        # Run inference via Ultralytics engine
        # classes=[2, 3, 5, 7] filters to car, motorcycle, bus, truck at engine level
        results = self.model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=self.config.target_classes,
            imgsz=self.config.image_size,
            device=self.config.device,
            verbose=False
        )

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections, inference_time_ms

        result = results[0]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = map(int, box)
                class_name = DEFAULT_CLASS_NAMES.get(cls_id, self.model.names.get(cls_id, f"cls_{cls_id}"))

                detections.append(
                    Detection(
                        box=(x1, y1, x2, y2),
                        confidence=float(conf),
                        class_id=cls_id,
                        class_name=class_name
                    )
                )

        return detections, inference_time_ms
