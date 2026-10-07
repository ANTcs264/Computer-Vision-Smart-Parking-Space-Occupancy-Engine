"""
Video Camera Ingestion Module
Handles video streaming from webcams, pre-recorded video files, and RTSP streams
with automatic reconnection, resolution scaling, frame regulation, and failure recovery.
"""

import logging
import os
import time
from typing import Generator, Optional, Tuple
import cv2
import numpy as np

from .config import CameraConfig
from .utils import resize_frame

logger = logging.getLogger("cv_engine.camera")


class VideoCamera:
    """
    Robust Video Ingestion Engine.
    Wraps OpenCV VideoCapture with connection resilience, frame-skip acceleration,
    and automatic video looping.
    """

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_file = isinstance(self.config.source, str) and os.path.isfile(self.config.source)
        self._is_network = isinstance(self.config.source, str) and (
            self.config.source.startswith("rtsp://") or self.config.source.startswith("http://")
        )
        self._frame_count = 0
        self._is_running = False

        self._connect()

    @property
    def source_type(self) -> str:
        """Returns human-readable source category."""
        if self._is_file:
            return "Video File"
        if self._is_network:
            return "RTSP/Network Stream"
        return f"Webcam Device #{self.config.source}"

    def _connect(self) -> bool:
        """Establishes or re-establishes the video capture stream."""
        if self.cap is not None:
            self.cap.release()

        logger.info(f"Connecting to video source: {self.config.source} ({self.source_type})...")

        # Select backend: CAP_DSHOW on Windows for fast webcam init, else CAP_ANY
        if isinstance(self.config.source, int) and os.name == "nt":
            self.cap = cv2.VideoCapture(self.config.source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.config.source)

        if not self.cap.isOpened():
            logger.warning(f"Failed to open video source: {self.config.source}")
            return False

        # Attempt to configure hardware capture dimensions
        if not self._is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            if self.config.target_fps > 0:
                self.cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)

        self._is_running = True
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        logger.info(
            f"Successfully connected to {self.source_type} | Native stream: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS"
        )
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads and preprocesses the next frame.
        Handles frame skipping, video looping, and reconnection on failure.
        
        Returns:
            (success: bool, frame: np.ndarray or None)
        """
        if self.cap is None or not self.cap.isOpened():
            if self.config.auto_reconnect:
                if not self._reconnect():
                    return False, None
            else:
                return False, None

        # Frame Skipping optimization
        # If frame_skip > 0, grab (without decoding) to advance buffer quickly
        if self.config.frame_skip > 0 and not self._is_file:
            for _ in range(self.config.frame_skip):
                if not self.cap.grab():
                    break

        success, frame = self.cap.read()

        # Handle End-Of-File (EOF) for video files
        if not success and self._is_file:
            if self.config.loop_video:
                logger.debug("End of video reached. Looping back to beginning...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = self.cap.read()
            else:
                logger.info("End of video file reached. Stopping playback.")
                return False, None

        # Handle Stream Interruption / Dropouts (Webcam or RTSP)
        if not success:
            logger.warning("Frame read failed or stream disconnected.")
            if self.config.auto_reconnect:
                if self._reconnect():
                    return self.read()
            return False, None

        self._frame_count += 1

        # Enforce target output resolution if different from capture dimensions
        frame = resize_frame(frame, self.config.width, self.config.height)
        return True, frame

    def _reconnect(self) -> bool:
        """Attempts exponential/linear backoff reconnection."""
        logger.warning(f"Initiating stream reconnection (max {self.config.max_reconnect_attempts} attempts)...")
        for attempt in range(1, self.config.max_reconnect_attempts + 1):
            logger.info(f"Reconnection attempt {attempt}/{self.config.max_reconnect_attempts} in {self.config.reconnect_delay_sec}s...")
            time.sleep(self.config.reconnect_delay_sec)
            if self._connect():
                logger.info("Stream successfully recovered!")
                return True

        logger.error(f"Failed to reconnect after {self.config.max_reconnect_attempts} attempts.")
        return False

    def stream(self) -> Generator[np.ndarray, None, None]:
        """Generator yielding successfully decoded frames."""
        while self._is_running:
            success, frame = self.read()
            if not success or frame is None:
                break
            yield frame

    def release(self) -> None:
        """Releases the camera hardware resources cleanly."""
        self._is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Video capture resources successfully released.")

    def __enter__(self) -> "VideoCamera":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
