#!/usr/bin/env python3
"""
Computer Vision Engine - Phase 7 Unified Pipeline
Camera Ingestion + YOLO Detection + ByteTrack Multi-Vehicle Tracking
+ Directional Entry/Exit Counting + Real-time Parking Occupancy Engine
+ Pluggable Domain Event Engine (Audit Logging & Event Broadcasting).
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
from cv_engine.detection import DetectionConfig, YOLOVehicleDetector
from cv_engine.parking import (
    SlotManager,
    OccupancyEngine,
    OccupancyConfig,
    DEFAULT_SLOTS_FILE
)
from cv_engine.tracking import (
    VehicleTracker,
    CountingTripwire,
    VehicleFlowCounter
)
from cv_engine.events import (
    EventDispatcher,
    LoggingEventHandler,
    FileAuditEventHandler,
    MemoryQueueHandler
)

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
    overlap_thresh: float = 0.28,
    hysteresis_window: int = 30,
    enable_tracking: bool = True,
    audit_log_path: str = "logs/parking_events.jsonl",
    headless: bool = False,
    max_frames: int = 0
):
    """
    Main unified CV pipeline with integrated domain event dispatching.
    """
    # 1. Initialize Event Dispatcher & Handlers
    dispatcher = EventDispatcher(camera_id="CAM01")
    dispatcher.register_handler(LoggingEventHandler())
    dispatcher.register_handler(FileAuditEventHandler(log_path=PROJECT_ROOT / audit_log_path))
    mem_handler = MemoryQueueHandler(maxsize=500)
    dispatcher.register_handler(mem_handler)

    # 2. Initialize Camera Configuration
    cam_config = CameraConfig.from_env()
    if source is not None:
        cam_config.source = int(source) if str(source).isdigit() else str(source)
    if width > 0:
        cam_config.width = width
    if height > 0:
        cam_config.height = height
    cam_config.frame_skip = frame_skip

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

    # 3. Initialize YOLO & ByteTrack
    det_config = DetectionConfig.from_env()
    if model_path:
        det_config.model_path = model_path
    det_config.confidence_threshold = conf_thresh
    det_config.iou_threshold = iou_thresh
    det_config.image_size = image_size

    detector = YOLOVehicleDetector(det_config)
    tracker = VehicleTracker(detector.model, det_config) if enable_tracking else None

    # 4. Initialize Flow Counter
    flow_counter = VehicleFlowCounter(tripwires=[
        CountingTripwire("MAIN DRIVEWAY", pt1=(80, 360), pt2=(1200, 360))
    ])

    # 5. Initialize Slot Manager & Occupancy Engine
    slot_manager = None
    occupancy_engine = None
    active_slots_file = slots_path if (slots_path and os.path.isfile(slots_path)) else DEFAULT_SLOTS_FILE

    if os.path.isfile(active_slots_file):
        slot_manager = SlotManager(active_slots_file)
        occ_config = OccupancyConfig(
            overlap_threshold=overlap_thresh,
            hysteresis_window=hysteresis_window,
            occupied_ratio=0.65,
            vacated_ratio=0.20
        )
        occupancy_engine = OccupancyEngine(slot_manager, occ_config)

    logger.info("=" * 68)
    logger.info("   SMART PARKING AI - PHASE 7 UNIFIED PIPELINE & EVENT ENGINE   ")
    logger.info(f"   Target Source       : {cam_config.source}")
    logger.info(f"   Target Resolution   : {cam_config.width}x{cam_config.height}")
    logger.info(f"   ByteTrack Tracker   : {'Active' if tracker else 'Disabled'}")
    if slot_manager:
        logger.info(f"   Occupancy Engine    : Active ({slot_manager.total_count} slots loaded)")
    logger.info(f"   Audit Event Log     : {audit_log_path}")
    logger.info(f"   Display Mode        : {'Headless' if headless else 'Interactive GUI'}")
    logger.info("=" * 68)

    fps_meter = FPSMeter(window_size=30)
    window_name = "Smart Parking AI - Event Engine (Phase 7)"

    total_track_time = 0.0
    tracked_frames_count = 0
    was_full_notified = False
    was_warning_notified = False

    try:
        with VideoCamera(cam_config) as camera:
            frame_idx = 0
            is_paused = False

            while True:
                if not is_paused:
                    success, frame = camera.read()
                    if not success or frame is None:
                        dispatcher.emit_camera_offline(last_frame_sec=1.0)
                        logger.info("Stream ended or failed to capture frame. Exiting loop.")
                        break

                    inference_ms = 0.0
                    tracked_vehicles = []

                    # Step 1: Run ByteTrack Tracking
                    if tracker is not None:
                        tracked_vehicles, inference_ms = tracker.track(frame)
                        total_track_time += inference_ms
                        tracked_frames_count += 1

                        # Step 2: Line Crossing Detection & Event Dispatch
                        crossings = flow_counter.update(tracked_vehicles)
                        for tw, v, cross_type in crossings:
                            if cross_type == "ENTRY":
                                dispatcher.emit_vehicle_entered(v.track_id, tw.name, {"class": v.class_name})
                            elif cross_type == "EXIT":
                                dispatcher.emit_vehicle_exited(v.track_id, tw.name, {"class": v.class_name})

                    # Step 3: Run Parking Occupancy Engine
                    if occupancy_engine is not None:
                        transitions = occupancy_engine.update(tracked_vehicles)
                        for t in transitions:
                            if t.new_status == "OCCUPIED":
                                dispatcher.emit_slot_occupied(
                                    slot_id=t.slot_id,
                                    vehicle_id=t.vehicle_id,
                                    confidence=t.confidence,
                                    metadata={"class": t.matched_class}
                                )
                            elif t.new_status == "AVAILABLE":
                                dispatcher.emit_slot_available(
                                    slot_id=t.slot_id,
                                    dwell_duration_sec=t.dwell_duration_sec,
                                    metadata={"previous_vehicle_id": t.vehicle_id}
                                )

                        # System-level Capacity Events
                        occ_rate = slot_manager.occupancy_rate
                        if occ_rate >= 100.0 and not was_full_notified:
                            dispatcher.emit_parking_full(slot_manager.total_count)
                            was_full_notified = True
                        elif occ_rate < 100.0:
                            was_full_notified = False

                        if occ_rate >= 85.0 and not was_warning_notified:
                            dispatcher.emit_parking_warning(occ_rate)
                            was_warning_notified = True
                        elif occ_rate < 85.0:
                            was_warning_notified = False

                    # Step 4: Render Overlays & HUD
                    if slot_manager is not None:
                        slot_manager.draw_slots(frame, alpha=0.30)
                        slot_manager.draw_slot_kpi_banner(frame)

                    flow_counter.draw(frame)

                    if tracker is not None:
                        tracker.draw_tracked_vehicles(frame, tracked_vehicles)

                    rolling_fps = fps_meter.tick()
                    latency_ms = fps_meter.latency_ms
                    frame_idx += 1

                    summary = {
                        "total": len(tracked_vehicles),
                        "car": sum(1 for v in tracked_vehicles if v.class_name == "car"),
                        "truck": sum(1 for v in tracked_vehicles if v.class_name == "truck"),
                        "motorcycle": sum(1 for v in tracked_vehicles if v.class_name == "motorcycle"),
                        "bus": sum(1 for v in tracked_vehicles if v.class_name == "bus"),
                    }

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

                # Headless Mode
                if headless:
                    if frame_idx % 10 == 0 or frame_idx == 1:
                        logger.info(
                            f"[Headless] Frame: {frame_idx:04d} | FPS: {fps_meter.fps:4.1f} | "
                            f"Events Dispatched: {dispatcher.total_dispatched}"
                        )
                    continue

                # Interactive Mode
                cv2.imshow(window_name, annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    logger.info("Exit requested by user.")
                    break
                elif key == ord('p'):
                    is_paused = not is_paused
                    logger.info(f"Stream {'PAUSED' if is_paused else 'RESUMED'}")
                elif key == ord('s'):
                    snap_path = PROJECT_ROOT / "datasets" / f"snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(str(snap_path), annotated_frame)
                    logger.info(f"Annotated snapshot saved: {snap_path}")

    except KeyboardInterrupt:
        logger.info("Pipeline terminated by KeyboardInterrupt.")
    finally:
        if not headless:
            cv2.destroyAllWindows()

    avg_track = (total_track_time / tracked_frames_count) if tracked_frames_count > 0 else 0.0
    logger.info("=" * 68)
    logger.info(f"Pipeline Completed: {frame_idx} total frames processed.")
    logger.info(f"Average Rolling FPS          : {fps_meter.fps:.2f}")
    logger.info(f"Average Tracking Latency     : {avg_track:.2f} ms")
    logger.info(f"Total Domain Events Emitted  : {dispatcher.total_dispatched}")
    logger.info(f"Events Buffered in Memory    : {mem_handler.queue.qsize()}")
    logger.info(f"Audit Log File Location      : {PROJECT_ROOT / audit_log_path}")
    logger.info(f"Total Session Time           : {fps_meter.uptime_sec:.2f} seconds")
    logger.info("=" * 68)


def main():
    parser = argparse.ArgumentParser(description="Smart Parking AI - Phase 7 Event Engine Pipeline")
    parser.add_argument("--source", type=str, default=None, help="Camera device index (0), video path, or RTSP URL")
    parser.add_argument("--width", type=int, default=1280, help="Target frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Target frame height (default: 720)")
    parser.add_argument("--skip", type=int, default=0, help="Frames to skip between inferences (default: 0)")
    parser.add_argument("--model", type=str, default="models/yolo/yolov8n.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold (default: 0.45)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input image size (default: 640)")
    parser.add_argument("--slots", type=str, default=str(DEFAULT_SLOTS_FILE), help="Path to parking_slots.json")
    parser.add_argument("--overlap", type=float, default=0.28, help="Slot overlap threshold (default: 0.28)")
    parser.add_argument("--hysteresis", type=int, default=30, help="Temporal hysteresis window size (default: 30)")
    parser.add_argument("--log-path", type=str, default="logs/parking_events.jsonl", help="Audit log path")
    parser.add_argument("--no-track", action="store_true", help="Disable vehicle tracking")
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
        overlap_thresh=args.overlap,
        hysteresis_window=args.hysteresis,
        enable_tracking=not args.no_track,
        audit_log_path=args.log_path,
        headless=args.headless,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
