#!/usr/bin/env python3
"""
Computer Vision Engine - Phase 4 Unified Pipeline
Camera Ingestion + YOLO Multi-Class Detection + Parking Slot Map Overlay.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv_engine.camera import CameraConfig, FPSMeter, VideoCamera, draw_telemetry_hud
from cv_engine.detection import (
    DetectionConfig,
    YOLOVehicleDetector,
    draw_detections,
    get_detection_summary
)
from cv_engine.parking import SlotManager, DEFAULT_SLOTS_FILE

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("cv_engine")


def run_pipeline(
    source: str | int | None = None,
    width: int = 1280,
    height: int = 720,
    frame_skip: int = 0,
    model_path: str = "models/yolo/yolov8n.pt",
    conf_thresh: float = 0.45,
    iou_thresh: float = 0.45,
    image_size: int = 640,
    slots_path: str | None = str(DEFAULT_SLOTS_FILE),
    enable_detection: bool = True,
    headless: bool = False,
    max_frames: int = 0
):
    """
    Main CV processing loop integrating Camera, YOLO Detection, and Parking Slot Overlays.
    """
    # 1. Initialize Camera Configuration
    cam_config = CameraConfig.from_env()
    if source is not None:
        cam_config.source = int(source) if str(source).isdigit() else str(source)
    if width > 0:
        cam_config.width = width
    if height > 0:
        cam_config.height = height
    cam_config.frame_skip = frame_skip

    # Fallback to sample parking video if webcam 0 is requested but not available
    sample_video_path = PROJECT_ROOT / "datasets" / "sample_parking.mp4"
    if cam_config.source == 0:
        test_cap = cv2.VideoCapture(0)
        is_webcam_open = test_cap.isOpened()
        test_cap.release()
        if not is_webcam_open and sample_video_path.is_file():
            logger.warning(
                "Webcam 0 is not accessible. Falling back to synthetic test video: datasets/sample_parking.mp4"
            )
            cam_config.source = str(sample_video_path)

    # 2. Initialize YOLO Detector
    detector = None
    if enable_detection:
        det_config = DetectionConfig.from_env()
        if model_path:
            det_config.model_path = model_path
        det_config.confidence_threshold = conf_thresh
        det_config.iou_threshold = iou_thresh
        det_config.image_size = image_size
        detector = YOLOVehicleDetector(det_config)

    # 3. Initialize Parking Slot Manager
    slot_manager = None
    if slots_path and os.path.isfile(slots_path):
        slot_manager = SlotManager(slots_path)
    elif DEFAULT_SLOTS_FILE.is_file():
        slot_manager = SlotManager(DEFAULT_SLOTS_FILE)

    logger.info("=" * 68)
    logger.info("   SMART PARKING AI - PHASE 4 CAMERA & PARKING SLOT MAP   ")
    logger.info(f"   Target Source       : {cam_config.source}")
    logger.info(f"   Target Resolution   : {cam_config.width}x{cam_config.height}")
    logger.info(f"   Detection Enabled   : {enable_detection}")
    if slot_manager:
        logger.info(f"   Parking Slots Loaded: {slot_manager.total_count} slots from {slot_manager.config_path.name}")
    logger.info(f"   Display Mode        : {'Headless' if headless else 'Interactive GUI'}")
    logger.info("=" * 68)

    fps_meter = FPSMeter(window_size=30)
    window_name = "Smart Parking AI - Camera & Parking Slot Map (Phase 4)"

    total_inf_time = 0.0
    detected_frames_count = 0

    try:
        with VideoCamera(cam_config) as camera:
            frame_idx = 0
            is_paused = False

            while True:
                if not is_paused:
                    success, frame = camera.read()
                    if not success or frame is None:
                        logger.info("Stream ended or failed to capture frame. Exiting loop.")
                        break

                    inference_ms = 0.0
                    detections = []
                    summary = None

                    # Step 1: Draw Configured Parking Slot Boundaries
                    if slot_manager is not None:
                        slot_manager.draw_slots(frame, alpha=0.30)
                        slot_manager.draw_slot_kpi_banner(frame)

                    # Step 2: Run YOLO Vehicle Detection
                    if detector is not None:
                        detections, inference_ms = detector.detect(frame)
                        total_inf_time += inference_ms
                        detected_frames_count += 1
                        summary = get_detection_summary(detections)

                        # Draw Bounding Boxes and Ground Footprints
                        draw_detections(frame, detections, show_footprint=True)

                    # Tick FPS meter & calculate performance
                    rolling_fps = fps_meter.tick()
                    latency_ms = fps_meter.latency_ms
                    frame_idx += 1

                    # Overlay engineering telemetry HUD
                    annotated_frame = draw_telemetry_hud(
                        frame=frame,
                        fps=fps_meter.fps,
                        latency_ms=latency_ms,
                        frame_idx=frame_idx,
                        source_name=camera.source_type,
                        target_res=(cam_config.width, cam_config.height),
                        inference_ms=inference_ms,
                        vehicle_counts=summary
                    )

                # Check max_frames termination
                if 0 < max_frames <= frame_idx:
                    logger.info(f"Reached max frame limit ({max_frames} frames). Terminating test.")
                    break

                # Headless Mode (for tests / CI)
                if headless:
                    if frame_idx % 10 == 0 or frame_idx == 1:
                        slot_info = f"Slots: {slot_manager.total_count}" if slot_manager else "No Slots"
                        logger.info(
                            f"[Headless] Frame: {frame_idx:04d} | FPS: {fps_meter.fps:4.1f} | "
                            f"Inf: {inference_ms:4.1f}ms | {slot_info}"
                        )
                    continue

                # Interactive Mode: Display window via OpenCV
                cv2.imshow(window_name, annotated_frame)

                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):  # 'q' or ESC
                    logger.info("Exit requested by user.")
                    break
                elif key == ord('p'):  # 'p' toggles pause
                    is_paused = not is_paused
                    logger.info(f"Stream {'PAUSED' if is_paused else 'RESUMED'}")
                elif key == ord('s'):  # 's' saves snapshot
                    snap_path = PROJECT_ROOT / "datasets" / f"snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(str(snap_path), annotated_frame)
                    logger.info(f"Annotated snapshot saved: {snap_path}")

    except KeyboardInterrupt:
        logger.info("Pipeline terminated by KeyboardInterrupt.")
    finally:
        if not headless:
            cv2.destroyAllWindows()

    avg_inf = (total_inf_time / detected_frames_count) if detected_frames_count > 0 else 0.0
    logger.info("=" * 68)
    logger.info(f"Pipeline Completed: {frame_idx} total frames processed.")
    logger.info(f"Average Rolling FPS       : {fps_meter.fps:.2f}")
    if enable_detection:
        logger.info(f"Average YOLO Inference    : {avg_inf:.2f} ms")
    logger.info(f"Total Session Time        : {fps_meter.uptime_sec:.2f} seconds")
    logger.info("=" * 68)


def main():
    parser = argparse.ArgumentParser(description="Smart Parking AI - Phase 4 Parking Slot Map Pipeline")
    parser.add_argument("--source", type=str, default=None, help="Camera device index (0), video path, or RTSP URL")
    parser.add_argument("--width", type=int, default=1280, help="Target frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Target frame height (default: 720)")
    parser.add_argument("--skip", type=int, default=0, help="Frames to skip between inferences (default: 0)")
    parser.add_argument("--model", type=str, default="models/yolo/yolov8n.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold (default: 0.45)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input image size (default: 640)")
    parser.add_argument("--slots", type=str, default=str(DEFAULT_SLOTS_FILE), help="Path to parking_slots.json")
    parser.add_argument("--no-detect", action="store_true", help="Disable YOLO detection (slot-map-only mode)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI window")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (default: 0 = infinite)")

    args = parser.parse_args()
    run_pipeline(
        source=args.source,
        width=args.width,
        height=args.height,
        frame_skip=args.skip,
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        image_size=args.imgsz,
        slots_path=args.slots,
        enable_detection=not args.no_detect,
        headless=args.headless,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
