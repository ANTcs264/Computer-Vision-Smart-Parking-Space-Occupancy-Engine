#!/usr/bin/env python3
"""
Computer Vision Engine - Phase 2 Entry Point
Demonstrates Camera Ingestion, Live Video Playback, FPS Metering,
and Telemetry Rendering via OpenCV.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv_engine.camera import CameraConfig, FPSMeter, VideoCamera, draw_telemetry_hud

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
    headless: bool = False,
    max_frames: int = 0
):
    """
    Main camera processing loop.
    Reads frames, calculates FPS, overlays telemetry, and renders to screen or headless sink.
    """
    # 1. Initialize Configuration
    config = CameraConfig.from_env()
    if source is not None:
        config.source = int(source) if str(source).isdigit() else str(source)
    if width > 0:
        config.width = width
    if height > 0:
        config.height = height
    config.frame_skip = frame_skip

    # Fallback to sample video if webcam 0 requested but not available
    sample_video_path = PROJECT_ROOT / "datasets" / "sample_parking.mp4"
    if config.source == 0:
        test_cap = cv2.VideoCapture(0)
        is_webcam_open = test_cap.isOpened()
        test_cap.release()
        if not is_webcam_open and sample_video_path.is_file():
            logger.warning(
                "Webcam 0 is not accessible. Falling back to synthetic test video: datasets/sample_parking.mp4"
            )
            config.source = str(sample_video_path)

    logger.info("=" * 65)
    logger.info("   SMART PARKING AI - COMPUTER VISION CAMERA PIPELINE   ")
    logger.info(f"   Target Source     : {config.source}")
    logger.info(f"   Target Resolution : {config.width}x{config.height}")
    logger.info(f"   Frame Skip Factor : {config.frame_skip}")
    logger.info(f"   Display Mode      : {'Headless' if headless else 'Interactive GUI'}")
    logger.info("=" * 65)

    fps_meter = FPSMeter(window_size=30)
    window_name = "Smart Parking AI - Live Camera Stream (Phase 2)"

    try:
        with VideoCamera(config) as camera:
            frame_idx = 0
            is_paused = False

            while True:
                if not is_paused:
                    success, frame = camera.read()
                    if not success or frame is None:
                        logger.info("Stream ended or failed to capture frame. Exiting loop.")
                        break

                    # Tick FPS meter & calculate performance
                    rolling_fps = fps_meter.tick()
                    latency_ms = fps_meter.latency_ms
                    frame_idx += 1

                    # Overlay engineering HUD
                    annotated_frame = draw_telemetry_hud(
                        frame=frame,
                        fps=fps_meter.fps,
                        latency_ms=latency_ms,
                        frame_idx=frame_idx,
                        source_name=camera.source_type,
                        target_res=(config.width, config.height)
                    )

                # Check max_frames termination (for testing / benchmarking)
                if 0 < max_frames <= frame_idx:
                    logger.info(f"Reached max frame limit ({max_frames} frames). Terminating test.")
                    break

                # Headless Mode (no OpenCV GUI window created)
                if headless:
                    if frame_idx % 30 == 0 or frame_idx == 1:
                        logger.info(
                            f"[Headless] Frame: {frame_idx:05d} | FPS: {fps_meter.fps:5.1f} | Latency: {latency_ms:4.1f}ms | Shape: {frame.shape}"
                        )
                    # Yield slightly to avoid 100% CPU lock in headless test
                    time.sleep(0.005)
                    continue

                # Interactive Mode: Display window via OpenCV
                cv2.imshow(window_name, annotated_frame)

                # WaitKey timing: ~30 FPS sync or keypress capture
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):  # 'q' or ESC
                    logger.info("Exit requested by user.")
                    break
                elif key == ord('p'):  # 'p' toggles pause
                    is_paused = not is_paused
                    logger.info(f"Stream {'PAUSED' if is_paused else 'RESUMED'}")
                elif key == ord('s'):  # 's' saves snapshot
                    snap_path = PROJECT_ROOT / "datasets" / f"snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(str(snap_path), frame)
                    logger.info(f"Snapshot saved: {snap_path}")

    except KeyboardInterrupt:
        logger.info("Pipeline terminated by KeyboardInterrupt.")
    finally:
        if not headless:
            cv2.destroyAllWindows()

    logger.info("=" * 65)
    logger.info(f"Pipeline Completed: {frame_idx} total frames processed.")
    logger.info(f"Average Rolling FPS : {fps_meter.fps:.2f}")
    logger.info(f"Total Session Time  : {fps_meter.uptime_sec:.2f} seconds")
    logger.info("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Smart Parking AI - Phase 2 Camera Pipeline")
    parser.add_argument("--source", type=str, default=None, help="Camera device index (0), video path, or RTSP URL")
    parser.add_argument("--width", type=int, default=1280, help="Target frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Target frame height (default: 720)")
    parser.add_argument("--skip", type=int, default=0, help="Frames to skip between inferences (default: 0)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window (for testing/CI)")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (default: 0 = infinite)")

    args = parser.parse_args()
    run_pipeline(
        source=args.source,
        width=args.width,
        height=args.height,
        frame_skip=args.skip,
        headless=args.headless,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
