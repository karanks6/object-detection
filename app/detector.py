"""
detector.py
Detection & tracking thread for Object Detection Pro.

Runs YOLO inference (with ByteTrack object tracking) in a background thread.
Outputs annotated frames and detection metadata via thread-safe queues.

Key design:
  - Uses model.track() for persistent object IDs across frames
  - Emits DetectionFrame objects containing the annotated frame + metadata
  - Supports hot-swapping the YOLO model at runtime
  - Supports per-class enable/disable filtering
  - Computes FPS from a rolling window
  - Thread-safe via threading.Event and queue.Queue
"""

import cv2
import time
import threading
import queue
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from ultralytics import YOLO

from app.color_names import get_color_name
from app.alerts import AlertSystem
from app.recorder import VideoRecorder, SnapshotSaver


@dataclass
class Detection:
    """Represents a single detected object in one frame."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    color_name: str
    avg_r: int
    avg_g: int
    avg_b: int


@dataclass
class DetectionFrame:
    """Represents one processed frame with all detection data."""
    frame: np.ndarray                       # Annotated BGR frame
    raw_frame: np.ndarray                   # Unannotated frame (for snapshot)
    detections: list[Detection]             # All detections in this frame
    fps: float                              # Current rolling FPS
    timestamp: float                        # epoch timestamp
    class_counts: dict[str, int] = field(default_factory=dict)  # class -> count


# ─── Color palette for bounding boxes (per track_id) ─────────────────────────
_BOX_COLORS = [
    (0, 200, 255),   # Vivid Cyan
    (0, 255, 127),   # Spring Green
    (255, 80, 80),   # Coral Red
    (200, 0, 255),   # Purple
    (255, 200, 0),   # Amber
    (0, 150, 255),   # Blue
    (255, 100, 200), # Pink
    (100, 255, 50),  # Lime
    (255, 165, 0),   # Orange
    (64, 224, 208),  # Turquoise
]

def _track_color(track_id: int) -> tuple:
    return _BOX_COLORS[track_id % len(_BOX_COLORS)]


def _get_avg_rgb(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int]:
    """Returns average (R, G, B) of a bounding box region."""
    roi = frame[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
    if roi.size == 0:
        return 128, 128, 128
    avg = np.mean(roi, axis=(0, 1))
    b, g, r = int(avg[0]), int(avg[1]), int(avg[2])
    return r, g, b


def _draw_rounded_rect(img, pt1, pt2, color, radius=8, thickness=2):
    """Draws a rounded-corner rectangle on the frame."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    # Draw four arcs at corners
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r),  90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r),   0, 0, 90, color, thickness)
    # Draw straight edges
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)


def _draw_label_pill(img, text: str, x: int, y: int, color: tuple, font_scale=0.55, thickness=1):
    """Draws a filled pill-shaped label background with text."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 5
    x2, y2 = x + w + pad * 2, y + h + pad * 2
    # Semi-transparent fill
    overlay = img.copy()
    cv2.rectangle(overlay, (x - pad, y - h - pad), (x2 - pad, y2 - h - pad), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    # Colored text
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _draw_alert_banner(img, message: str):
    """Draws a full-width alert banner at the top of the frame."""
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (20, 20, 180), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.putText(img, message, (12, 36), cv2.FONT_HERSHEY_DUPLEX, 0.75,
                (255, 255, 80), 1, cv2.LINE_AA)


class ObjectDetector:
    """
    Manages the YOLO detection/tracking loop in a background thread.

    Args:
        model_path (str): Path to the YOLO model weights file.
        source (int|str): Camera index or video file path.
        output_queue (queue.Queue): Queue to push DetectionFrame objects into.
        alert_system (AlertSystem): Shared alert system instance.
        recorder (VideoRecorder): Shared video recorder instance.
        snapshot_saver (SnapshotSaver): Shared snapshot saver instance.
        conf_threshold (float): Minimum confidence to show a detection.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        source: int = 0,
        output_queue: queue.Queue | None = None,
        alert_system: AlertSystem | None = None,
        recorder: VideoRecorder | None = None,
        snapshot_saver: SnapshotSaver | None = None,
        conf_threshold: float = 0.40,
    ):
        self.model_path = model_path
        self.source = source
        self.output_queue = output_queue or queue.Queue(maxsize=2)
        self.alert_system = alert_system or AlertSystem()
        self.recorder = recorder or VideoRecorder()
        self.snapshot_saver = snapshot_saver or SnapshotSaver()
        self.conf_threshold = conf_threshold

        self._model: YOLO | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._swap_model_path: str | None = None  # pending model swap
        self._swap_lock = threading.Lock()

        # Class filter: None = show all; set of class names (lowercase) = only show these
        self.enabled_classes: set[str] | None = None
        self._class_lock = threading.Lock()

        # FPS tracking
        self._fps_window: deque[float] = deque(maxlen=30)
        self._last_log_time = 0.0

        # Detection event log: list of (timestamp, class_name, confidence, track_id)
        self.detection_log: list[tuple] = []
        self._log_lock = threading.Lock()
        self.max_log_entries = 200

        # Snapshot request flag
        self._snapshot_requested = threading.Event()

    # ── Public control API ────────────────────────────────────────────────────

    def start(self):
        """Start the detection thread."""
        self._stop_event.clear()
        self._model = YOLO(self.model_path)
        self._thread = threading.Thread(target=self._run, daemon=True, name="DetectorThread")
        self._thread.start()

    def stop(self):
        """Signal the detection thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def request_snapshot(self):
        """Request a snapshot to be saved on the next frame."""
        self._snapshot_requested.set()

    def swap_model(self, new_model_path: str):
        """Hot-swap to a different YOLO model (applied on next loop iteration)."""
        with self._swap_lock:
            self._swap_model_path = new_model_path

    def set_enabled_classes(self, classes: set[str] | None):
        """
        Set the enabled class filter.
        Pass None to show all classes.
        Pass a set of lowercase class names to restrict.
        """
        with self._class_lock:
            self.enabled_classes = classes

    def get_class_names(self) -> list[str]:
        """Return all class names from the loaded model."""
        if self._model:
            return list(self._model.names.values())
        return []

    def get_log(self) -> list[tuple]:
        """Return a copy of the detection event log."""
        with self._log_lock:
            return list(self.detection_log)

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[Detector] ERROR: Cannot open source: {self.source}")
            return

        print(f"[Detector] Started with model: {self.model_path}")

        while not self._stop_event.is_set():
            # Check for pending model swap
            with self._swap_lock:
                if self._swap_model_path:
                    print(f"[Detector] Swapping model -> {self._swap_model_path}")
                    self._model = YOLO(self._swap_model_path)
                    self.model_path = self._swap_model_path
                    self._swap_model_path = None

            ret, frame = cap.read()
            if not ret:
                print("[Detector] Frame read failed. Stopping.")
                break

            t0 = time.perf_counter()

            # Run tracking
            try:
                results = self._model.track(
                    frame,
                    persist=True,
                    conf=self.conf_threshold,
                    verbose=False,
                )[0]
            except Exception as e:
                print(f"[Detector] Inference error: {e}")
                continue

            annotated = frame.copy()
            detections: list[Detection] = []
            alert_inputs: list[dict] = []
            now = time.time()

            with self._class_lock:
                enabled = self.enabled_classes

            for box in results.boxes:
                cls_id = int(box.cls[0])
                class_name = self._model.names[cls_id]

                # Class filter
                if enabled is not None and class_name.lower() not in enabled:
                    continue

                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else 0
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                r, g, b = _get_avg_rgb(frame, x1, y1, x2, y2)
                color_name = get_color_name(r, g, b)
                box_color = _track_color(track_id)

                det = Detection(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    color_name=color_name,
                    avg_r=r, avg_g=g, avg_b=b,
                )
                detections.append(det)
                alert_inputs.append({"class_name": class_name, "confidence": conf})

                # ── Draw bounding box ──────────────────────────────────────
                _draw_rounded_rect(annotated, (x1, y1), (x2, y2), box_color, radius=10, thickness=2)

                # Primary label: ID + class + confidence
                label = f"#{track_id} {class_name}  {conf:.0%}"
                _draw_label_pill(annotated, label, x1 + 6, y1 - 8, box_color)

                # Secondary label: color name
                color_label = f"● {color_name}"
                _draw_label_pill(annotated, color_label, x1 + 6, y2 + 20, box_color, font_scale=0.45)

            # Alert check
            self.alert_system.check(alert_inputs)

            # Alert banner overlay
            banner = self.alert_system.get_banner()
            if banner:
                _draw_alert_banner(annotated, banner)

            # FPS overlay (top-right)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            self._fps_window.append(elapsed)
            fps = 1.0 / (sum(self._fps_window) / len(self._fps_window)) if self._fps_window else 0.0

            fps_text = f"FPS: {fps:.1f}"
            (fw, fh), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            h_frame, w_frame = annotated.shape[:2]
            _draw_label_pill(annotated, fps_text, w_frame - fw - 16, 28, (0, 255, 200), font_scale=0.6)

            # Class counts
            class_counts: dict[str, int] = {}
            for d in detections:
                class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1

            # Detection log
            with self._log_lock:
                for d in detections:
                    self.detection_log.append((now, d.class_name, d.confidence, d.track_id))
                if len(self.detection_log) > self.max_log_entries:
                    self.detection_log = self.detection_log[-self.max_log_entries:]

            # Recording
            if self.recorder.is_recording:
                self.recorder.write(annotated)

            # Snapshot
            if self._snapshot_requested.is_set():
                self._snapshot_requested.clear()
                path = self.snapshot_saver.save(annotated)
                print(f"[Detector] Snapshot saved: {path}")

            # Push frame to GUI queue (drop if full to avoid lag)
            df = DetectionFrame(
                frame=annotated,
                raw_frame=frame,
                detections=detections,
                fps=fps,
                timestamp=now,
                class_counts=class_counts,
            )
            try:
                self.output_queue.put_nowait(df)
            except queue.Full:
                pass  # GUI is behind; drop frame

        cap.release()
        if self.recorder.is_recording:
            self.recorder.stop()
        print("[Detector] Thread stopped.")
