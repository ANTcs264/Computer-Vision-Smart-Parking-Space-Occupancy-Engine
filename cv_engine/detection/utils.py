"""
Detection Visualization & Overlay Utilities
Renders sleek bounding boxes, class pills, confidence tags, and ground footprint anchors.
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np

from .detector import Detection

# Professional BGR color palette for vehicle classes
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "car": (255, 180, 0),        # Cyan-Blue
    "motorcycle": (0, 215, 255),  # Yellow-Amber
    "bus": (210, 50, 210),        # Violet
    "truck": (80, 220, 100)       # Emerald Green
}
DEFAULT_BOX_COLOR = (200, 200, 200)


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    show_footprint: bool = True
) -> np.ndarray:
    """
    Renders bounding boxes, class labels, confidence scores, and ground footprints.

    Args:
        frame: OpenCV BGR image.
        detections: List of Detection instances.
        show_footprint: Whether to render ground footprint anchor point.

    Returns:
        Annotated frame.
    """
    for det in detections:
        color = CLASS_COLORS.get(det.class_name.lower(), DEFAULT_BOX_COLOR)
        x1, y1, x2, y2 = det.box

        # 1. Draw Bounding Box with corner accent highlights
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Draw corner accents for high-tech aesthetic
        corner_len = min(16, int((x2 - x1) * 0.2), int((y2 - y1) * 0.2))
        thick = 3
        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thick)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thick)
        # Top-Right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thick)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thick)
        # Bottom-Left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thick)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thick)
        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thick)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thick)

        # 2. Pill Badge Label: [CLASS CONF]
        label = f"{det.class_name.upper()} {det.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        label_thick = 1

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, label_thick)
        pill_y1 = max(0, y1 - text_h - 8)
        pill_y2 = pill_y1 + text_h + 8
        pill_x2 = min(frame.shape[1], x1 + text_w + 12)

        # Draw pill background rectangle
        cv2.rectangle(frame, (x1, pill_y1), (pill_x2, pill_y2), color, -1)
        # Draw pill text in dark contrasting slate
        cv2.putText(
            frame,
            label,
            (x1 + 6, pill_y2 - 5),
            font,
            font_scale,
            (15, 23, 42),  # Dark Slate
            label_thick,
            cv2.LINE_AA
        )

        # 3. Ground Footprint Anchor Point (for parking slot intersection)
        if show_footprint:
            fx, fy = det.bottom_center
            cv2.circle(frame, (fx, fy), 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (fx, fy), 6, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def get_detection_summary(detections: List[Detection]) -> Dict[str, int]:
    """Generates counts grouped by vehicle class."""
    summary: Dict[str, int] = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "total": 0}
    for det in detections:
        name = det.class_name.lower()
        if name in summary:
            summary[name] += 1
        summary["total"] += 1
    return summary
