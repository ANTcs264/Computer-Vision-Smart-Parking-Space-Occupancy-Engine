#!/usr/bin/env python3
"""
Single-Frame YOLO Vehicle Detection Verification Script
Tests YOLO detection on sample imagery to verify detection bounding boxes,
confidence ratings, and class categorization.
"""

import sys
from pathlib import Path
import cv2
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv_engine.detection import DetectionConfig, YOLOVehicleDetector, draw_detections, get_detection_summary

def main():
    test_img_path = PROJECT_ROOT / "datasets" / "test_vehicles.jpg"
    
    # Download a high-res sample traffic / vehicles image if not present
    if not test_img_path.is_file():
        url = "https://ultralytics.com/images/bus.jpg"
        print(f"Downloading sample vehicle image from {url}...")
        try:
            urllib.request.urlretrieve(url, test_img_path)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download image: {e}")
            return

    image = cv2.imread(str(test_img_path))
    if image is None:
        print("Failed to read test image.")
        return

    config = DetectionConfig(
        model_path=str(PROJECT_ROOT / "models" / "yolo" / "yolov8n.pt"),
        confidence_threshold=0.35,
        iou_threshold=0.45,
        image_size=640
    )
    detector = YOLOVehicleDetector(config)

    print("\nRunning YOLO Vehicle Detection...")
    detections, inf_ms = detector.detect(image)
    summary = get_detection_summary(detections)

    print(f"\nDetection Results (Inference Time: {inf_ms:.2f} ms):")
    print(f"Total Vehicles Detected: {len(detections)}")
    print(f"Breakdown: {summary}\n")
    print(f"{'INDEX':<6} {'CLASS':<12} {'CONFIDENCE':<12} {'BOUNDING BOX (x1, y1, x2, y2)':<30} {'FOOTPRINT (xc, ymax)'}")
    print("-" * 80)

    for i, det in enumerate(detections, 1):
        box_str = f"({det.x1}, {det.y1}, {det.x2}, {det.y2})"
        foot_str = f"{det.bottom_center}"
        print(f"{i:<6} {det.class_name.upper():<12} {det.confidence:<12.2f} {box_str:<30} {foot_str}")

    annotated = draw_detections(image, detections, show_footprint=True)
    out_path = PROJECT_ROOT / "datasets" / "test_vehicles_annotated.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"\nAnnotated visualization saved: {out_path}")

if __name__ == "__main__":
    main()
