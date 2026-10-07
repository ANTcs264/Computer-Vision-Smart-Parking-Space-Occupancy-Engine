"""
Camera Pipeline Configuration Module
Loads camera source, target resolution, and frame regulation parameters.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Find and load the root .env file
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class CameraConfig:
    """Configuration schema for Video Camera Ingestion."""
    source: str | int = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 30
    frame_skip: int = 0
    loop_video: bool = True
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 5
    reconnect_delay_sec: float = 2.0

    @classmethod
    def from_env(cls) -> "CameraConfig":
        """Factory method to construct config directly from system environment / .env."""
        raw_source = os.getenv("CAMERA_SOURCE", "0").strip()
        # Parse numeric camera index (e.g. "0" -> 0) or preserve string path/URL
        source: str | int = int(raw_source) if raw_source.isdigit() else raw_source

        # If it's a relative path to a video file, resolve relative to project root
        if isinstance(source, str) and not source.startswith("rtsp://") and not source.startswith("http://"):
            resolved = PROJECT_ROOT / source
            if resolved.is_file():
                source = str(resolved)

        width = int(os.getenv("CAMERA_WIDTH", "1280"))
        height = int(os.getenv("CAMERA_HEIGHT", "720"))
        target_fps = int(os.getenv("CAMERA_TARGET_FPS", "30"))
        frame_skip = int(os.getenv("FRAME_SKIP", "0"))

        return cls(
            source=source,
            width=width,
            height=height,
            target_fps=target_fps,
            frame_skip=frame_skip,
            loop_video=True,
            auto_reconnect=True,
            max_reconnect_attempts=5,
            reconnect_delay_sec=2.0
        )
