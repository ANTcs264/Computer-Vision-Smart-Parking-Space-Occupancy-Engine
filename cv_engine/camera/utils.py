"""
Camera Utilities Module
Provides high-precision FPS metering, latency calculation, and telemetry rendering.
"""

import time
from collections import deque
import cv2
import numpy as np


class FPSMeter:
    """
    High-precision rolling window FPS and latency monitor.
    Uses time.perf_counter() to avoid clock drift.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._timestamps = deque(maxlen=window_size)
        self._last_time = time.perf_counter()
        self._frame_count = 0
        self._start_time = self._last_time
        self._last_latency_ms = 0.0

    def tick(self) -> float:
        """
        Record the processing of a single frame.
        Returns the instantaneous FPS.
        """
        current_time = time.perf_counter()
        delta = current_time - self._last_time
        self._last_time = current_time

        self._last_latency_ms = delta * 1000.0
        self._timestamps.append(current_time)
        self._frame_count += 1

        if len(self._timestamps) > 1:
            total_duration = self._timestamps[-1] - self._timestamps[0]
            if total_duration > 0:
                return (len(self._timestamps) - 1) / total_duration
        return 0.0

    @property
    def fps(self) -> float:
        """Calculate the smoothed rolling average FPS."""
        if len(self._timestamps) > 1:
            total_duration = self._timestamps[-1] - self._timestamps[0]
            if total_duration > 0:
                return (len(self._timestamps) - 1) / total_duration
        return 0.0

    @property
    def latency_ms(self) -> float:
        """Returns the latency of the most recent frame in milliseconds."""
        return self._last_latency_ms

    @property
    def total_frames(self) -> int:
        """Total number of frames ticked."""
        return self._frame_count

    @property
    def uptime_sec(self) -> float:
        """Total running time in seconds."""
        return time.perf_counter() - self._start_time


def draw_telemetry_hud(
    frame: np.ndarray,
    fps: float,
    latency_ms: float,
    frame_idx: int,
    source_name: str,
    target_res: tuple[int, int]
) -> np.ndarray:
    """
    Draws a semi-transparent engineering telemetry HUD on the top of the video frame.
    Displays: Status, Resolution, FPS, Latency, Frame Counter, and Timestamp.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Draw semi-transparent header bar across the top
    banner_height = 42
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (15, 23, 42), -1)  # Dark slate gray
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Accent color line
    cv2.line(frame, (0, banner_height), (w, banner_height), (34, 197, 94), 2)  # Green accent line

    # Prepare status text
    status_text = f"STREAM: [{source_name.upper()}]"
    res_text = f"RES: {w}x{h}"
    fps_color = (34, 197, 94) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
    fps_text = f"FPS: {fps:5.1f}"
    lat_text = f"LATENCY: {latency_ms:4.1f}ms"
    frame_text = f"FRAME: {frame_idx:06d}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1

    # Text placements
    cv2.putText(frame, status_text, (15, 26), font, font_scale, (241, 245, 249), thickness, cv2.LINE_AA)
    cv2.putText(frame, res_text, (240, 26), font, font_scale, (148, 163, 184), thickness, cv2.LINE_AA)
    cv2.putText(frame, fps_text, (400, 26), font, font_scale, fps_color, 2, cv2.LINE_AA)
    cv2.putText(frame, lat_text, (530, 26), font, font_scale, (203, 213, 225), thickness, cv2.LINE_AA)
    cv2.putText(frame, frame_text, (710, 26), font, font_scale, (148, 163, 184), thickness, cv2.LINE_AA)

    # Bottom helper bar
    help_overlay = frame.copy()
    cv2.rectangle(help_overlay, (0, h - 28), (w, h), (15, 23, 42), -1)
    cv2.addWeighted(help_overlay, 0.75, frame, 0.25, 0, frame)
    help_text = "Press 'q' or 'ESC' to exit | Smart Parking AI - CV Pipeline (Phase 2)"
    cv2.putText(frame, help_text, (15, h - 9), font, 0.44, (203, 213, 225), 1, cv2.LINE_AA)

    return frame


def resize_frame(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    Safely resizes the frame to target dimensions.
    Uses INTER_AREA for downsampling (anti-aliasing) and INTER_LINEAR for upsampling.
    """
    curr_h, curr_w = frame.shape[:2]
    if curr_w == target_w and curr_h == target_h:
        return frame

    interpolation = cv2.INTER_AREA if (target_w < curr_w and target_h < curr_h) else cv2.INTER_LINEAR
    return cv2.resize(frame, (target_w, target_h), interpolation=interpolation)
