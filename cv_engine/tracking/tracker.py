"""
ByteTrack Multi-Vehicle Tracking Module
Wraps Ultralytics ByteTrack to maintain persistent vehicle IDs, kinematics,
and trajectory trails across video frames.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from cv_engine.detection import DetectionConfig, DEFAULT_CLASS_NAMES
from .vehicle import TrackedVehicle

logger = logging.getLogger("cv_engine.tracking")

# Deterministic color palette generator for persistent track IDs
PALETTE = [
    (56, 189, 248),   # Sky Blue
    (251, 146, 60),   # Orange
    (168, 85, 247),   # Purple
    (74, 222, 128),   # Green
    (244, 63, 94),    # Rose
    (250, 204, 21),   # Yellow
    (45, 212, 191),   # Teal
    (192, 132, 252),  # Lavender
]


def get_color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Deterministically assigns a vibrant BGR color based on integer track ID."""
    return PALETTE[track_id % len(PALETTE)]


class VehicleTracker:
    """
    Production ByteTrack Tracker.
    Associates vehicle detections between frames using motion prediction
    and geometric overlap, preserving identities through brief occlusions.
    """

    def __init__(self, yolo_model: YOLO, config: Optional[DetectionConfig] = None):
        self.model = yolo_model
        self.config = config or DetectionConfig.from_env()
        self.active_vehicles: Dict[int, TrackedVehicle] = {}
        self.max_lost_time_sec = 4.0  # Time before inactive vehicle is purged

    def track(self, frame: np.ndarray) -> Tuple[List[TrackedVehicle], float]:
        """
        Runs ByteTrack on a video frame and updates vehicle histories.
        
        Args:
            frame: OpenCV BGR image (H, W, 3).
            
        Returns:
            Tuple of (active_tracked_vehicles: List[TrackedVehicle], inference_latency_ms: float)
        """
        start_time = time.perf_counter()
        now = time.time()

        # Run Ultralytics YOLO with ByteTrack tracker enabled
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=self.config.target_classes,
            imgsz=self.config.image_size,
            device=self.config.device,
            verbose=False
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        current_frame_ids = set()

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0 and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                track_ids = boxes.id.cpu().numpy().astype(int)

                for box, conf, cls_id, t_id in zip(xyxy, confs, clss, track_ids):
                    current_frame_ids.add(t_id)
                    x1, y1, x2, y2 = map(int, box)
                    cls_name = DEFAULT_CLASS_NAMES.get(cls_id, self.model.names.get(cls_id, "vehicle"))

                    if t_id in self.active_vehicles:
                        # Existing vehicle - update position & trajectory
                        self.active_vehicles[t_id].update_position(
                            box=(x1, y1, x2, y2),
                            confidence=float(conf),
                            current_time=now
                        )
                    else:
                        # Newly detected vehicle - create new instance
                        vehicle = TrackedVehicle(
                            track_id=t_id,
                            class_name=cls_name,
                            class_id=cls_id,
                            box=(x1, y1, x2, y2),
                            confidence=float(conf),
                            first_seen=now,
                            last_seen=now
                        )
                        self.active_vehicles[t_id] = vehicle
                        logger.debug(f"[Tracker] New Vehicle Registered: ID {t_id} ({cls_name.upper()})")

        # Cleanup lost vehicles older than max_lost_time_sec
        lost_ids = [
            t_id for t_id, v in self.active_vehicles.items()
            if (now - v.last_seen) > self.max_lost_time_sec
        ]
        for t_id in lost_ids:
            del self.active_vehicles[t_id]

        # Return vehicles seen in the current frame
        active_list = [v for t_id, v in self.active_vehicles.items() if t_id in current_frame_ids]
        return active_list, latency_ms

    def draw_tracked_vehicles(self, frame: np.ndarray, vehicles: List[TrackedVehicle]) -> np.ndarray:
        """
        Renders bounding boxes with ByteTrack IDs and fading trajectory trails.
        """
        for v in vehicles:
            color = get_color_for_id(v.track_id)
            x1, y1, x2, y2 = v.box

            # 1. Draw Trajectory Ribbon (Connecting footprint centers)
            if len(v.trajectory) > 1:
                pts = list(v.trajectory)
                for i in range(1, len(pts)):
                    # Older points are thinner, recent points are thicker
                    thickness = max(1, int(3 * (i / len(pts))))
                    cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

            # 2. Draw Bounding Box with Corner Accents
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # 3. Draw Tracked Vehicle Badge: [ID: 17 CAR 0.94]
            label = f"ID: {v.track_id} {v.class_name.upper()} {v.confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.46
            thick = 1

            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thick)
            pill_y1 = max(0, y1 - th - 8)
            pill_y2 = pill_y1 + th + 8
            pill_x2 = min(frame.shape[1], x1 + tw + 12)

            cv2.rectangle(frame, (x1, pill_y1), (pill_x2, pill_y2), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 6, pill_y2 - 5),
                font,
                font_scale,
                (15, 23, 42),  # Contrasting dark slate text
                thick,
                cv2.LINE_AA
            )

            # 4. Footprint Contact Point
            fx, fy = v.bottom_center
            cv2.circle(frame, (fx, fy), 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (fx, fy), 6, color, 1, cv2.LINE_AA)

        return frame
