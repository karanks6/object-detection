"""
recorder.py
Video recording module for Object Detection Pro.
Wraps OpenCV VideoWriter with start/stop/toggle controls.
"""

import cv2
import os
import time
from pathlib import Path


class VideoRecorder:
    """
    Records annotated frames to an .avi file.

    Usage:
        recorder = VideoRecorder(output_dir="recordings")
        recorder.start(frame_width, frame_height, fps=20)
        recorder.write(frame)
        recorder.stop()
    """

    def __init__(self, output_dir: str = "recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._writer: cv2.VideoWriter | None = None
        self._filepath: str = ""
        self.is_recording: bool = False

    def start(self, width: int, height: int, fps: float = 20.0) -> str:
        """Start recording. Returns the output file path."""
        if self.is_recording:
            self.stop()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.avi"
        self._filepath = str(self.output_dir / filename)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self._writer = cv2.VideoWriter(self._filepath, fourcc, fps, (width, height))
        self.is_recording = True
        return self._filepath

    def write(self, frame):
        """Write a single frame. No-op if not recording."""
        if self.is_recording and self._writer is not None:
            self._writer.write(frame)

    def stop(self) -> str:
        """Stop recording and release the writer. Returns the saved file path."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self.is_recording = False
        return self._filepath

    def toggle(self, width: int, height: int, fps: float = 20.0) -> tuple[bool, str]:
        """
        Toggle recording on/off.
        Returns (is_now_recording, filepath)
        """
        if self.is_recording:
            path = self.stop()
            return False, path
        else:
            path = self.start(width, height, fps)
            return True, path

    @property
    def current_file(self) -> str:
        return self._filepath


class SnapshotSaver:
    """
    Saves individual annotated frames as PNG snapshots.
    """

    def __init__(self, output_dir: str = "snapshots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, frame) -> str:
        """Save a frame as PNG. Returns the saved file path."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.png"
        filepath = str(self.output_dir / filename)
        cv2.imwrite(filepath, frame)
        return filepath
