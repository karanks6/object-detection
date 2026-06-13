"""
gui.py
Object Detection Pro — Main GUI
Built with customtkinter for a modern dark-mode dashboard.

Layout:
  ┌─────────────────────────────────────────────────────┐
  │ Title Bar                                           │
  ├────────────────────────┬────────────────────────────┤
  │                        │  Sidebar                   │
  │   Video Feed           │  • Live Stats              │
  │   (main panel)         │  • Class Counts            │
  │                        │  • Controls                │
  │                        │  • Model Switcher          │
  │                        │  • Alert Setup             │
  │                        │  • Class Filter            │
  ├────────────────────────┴────────────────────────────┤
  │  Detection Log (scrollable)                         │
  └─────────────────────────────────────────────────────┘
"""

import queue
import time
import threading
import os

import customtkinter as ctk
from PIL import Image, ImageTk

from app.detector import ObjectDetector, DetectionFrame
from app.alerts import AlertSystem
from app.recorder import VideoRecorder, SnapshotSaver


# ─── Theme Configuration ──────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT       = "#00C8FF"      # Vivid cyan accent
ACCENT2      = "#7B2FFF"      # Purple secondary
SUCCESS      = "#00FF99"      # Green for recording active
WARNING      = "#FF9900"      # Amber
DANGER       = "#FF4444"      # Red
BG_DARK      = "#0D0D0F"      # Near-black background
BG_CARD      = "#161620"      # Card background
BG_CARD2     = "#1E1E2A"      # Lighter card
TEXT_PRIMARY = "#EAEAF0"
TEXT_MUTED   = "#8888A8"


class ObjectDetectionApp(ctk.CTk):
    """
    Main application window for Object Detection Pro.
    """

    MODELS = {
        "YOLOv8 Nano  (Fast)":   "yolov8n.pt",
        "YOLOv8 Medium (Accurate)": "yolov8m.pt",
    }
    LOG_MAX_ROWS = 150

    def __init__(self):
        super().__init__()

        # ── Window setup ──────────────────────────────────────────────────
        self.title("Object Detection Pro")
        self.geometry("1280x780")
        self.minsize(1100, 680)
        self.configure(fg_color=BG_DARK)

        # ── Shared subsystems ─────────────────────────────────────────────
        self._frame_queue:  queue.Queue        = queue.Queue(maxsize=2)
        self._alert_system: AlertSystem        = AlertSystem(sound_enabled=True)
        self._recorder:     VideoRecorder      = VideoRecorder("recordings")
        self._snapshot_saver: SnapshotSaver    = SnapshotSaver("snapshots")
        self._detector:     ObjectDetector | None = None

        # ── State ─────────────────────────────────────────────────────────
        self._running           = False
        self._selected_model    = list(self.MODELS.keys())[0]
        self._class_vars:  dict[str, ctk.BooleanVar] = {}
        self._alert_vars:  dict[str, ctk.BooleanVar] = {}
        self._status_msg        = ""
        self._total_detections  = 0
        self._session_start     = 0.0

        # ── Build UI ──────────────────────────────────────────────────────
        self._build_ui()
        self._set_status("Ready — press ▶ Start to begin", color=TEXT_MUTED)

        # ── Poll the frame queue ──────────────────────────────────────────
        self.after(33, self._poll_frame_queue)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ═════════════════════════════════════════════════════════════════════════
    # UI Construction
    # ═════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        # Title bar
        self._build_title_bar()

        # Main content area
        content = ctk.CTkFrame(self, fg_color=BG_DARK, corner_radius=0)
        content.pack(fill="both", expand=True, padx=0, pady=0)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=1, minsize=320)
        content.rowconfigure(0, weight=3)
        content.rowconfigure(1, weight=1, minsize=160)

        self._build_video_panel(content)
        self._build_sidebar(content)
        self._build_log_panel(content)

    def _build_title_bar(self):
        bar = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=0, height=52)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        # Logo / title
        title_lbl = ctk.CTkLabel(
            bar,
            text="🎯  Object Detection Pro",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=ACCENT,
        )
        title_lbl.pack(side="left", padx=18, pady=8)

        # Status message (right-aligned)
        self._status_label = ctk.CTkLabel(
            bar,
            text="",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=TEXT_MUTED,
        )
        self._status_label.pack(side="right", padx=18)

        # Model dropdown (right of title)
        model_frame = ctk.CTkFrame(bar, fg_color="transparent")
        model_frame.pack(side="right", padx=12)

        ctk.CTkLabel(
            model_frame, text="Model:", text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=(0, 6))

        self._model_var = ctk.StringVar(value=self._selected_model)
        self._model_dropdown = ctk.CTkOptionMenu(
            model_frame,
            values=list(self.MODELS.keys()),
            variable=self._model_var,
            command=self._on_model_change,
            width=220,
            fg_color=BG_CARD2,
            button_color=ACCENT2,
            button_hover_color="#5A1FCC",
            dropdown_fg_color=BG_CARD2,
            text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=12),
        )
        self._model_dropdown.pack(side="left")

    def _build_video_panel(self, parent):
        """Left video panel with the webcam feed."""
        panel = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=12)
        panel.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=(10, 5))
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)

        # Header
        hdr = ctk.CTkFrame(panel, fg_color="transparent", height=36)
        hdr.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 0))
        ctk.CTkLabel(
            hdr, text="📹  Live Feed",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=TEXT_PRIMARY,
        ).pack(side="left")

        self._rec_badge = ctk.CTkLabel(
            hdr, text="", font=ctk.CTkFont(size=11),
            text_color=DANGER,
        )
        self._rec_badge.pack(side="right")

        # Video label
        self._video_label = ctk.CTkLabel(panel, text="", fg_color="#080810", corner_radius=8)
        self._video_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 10))

        # Control toolbar below video
        self._build_control_toolbar(panel)

    def _build_control_toolbar(self, parent):
        """Toolbar with Start/Stop, Record, Snapshot."""
        toolbar = ctk.CTkFrame(parent, fg_color=BG_CARD2, corner_radius=8, height=52)
        toolbar.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        toolbar.pack_propagate(False)

        btn_cfg = dict(
            height=36,
            corner_radius=8,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
        )

        self._start_btn = ctk.CTkButton(
            toolbar,
            text="▶  Start",
            fg_color="#1A6B35",
            hover_color="#27A850",
            text_color="white",
            command=self._toggle_detection,
            **btn_cfg,
            width=120,
        )
        self._start_btn.pack(side="left", padx=10, pady=8)

        self._record_btn = ctk.CTkButton(
            toolbar,
            text="⏺  Record",
            fg_color=BG_CARD,
            hover_color="#3A1A1A",
            text_color=DANGER,
            border_color=DANGER,
            border_width=1,
            command=self._toggle_recording,
            **btn_cfg,
            width=120,
        )
        self._record_btn.pack(side="left", padx=4, pady=8)

        self._snap_btn = ctk.CTkButton(
            toolbar,
            text="📷  Snapshot",
            fg_color=BG_CARD,
            hover_color=BG_CARD2,
            text_color=ACCENT,
            border_color=ACCENT,
            border_width=1,
            command=self._take_snapshot,
            **btn_cfg,
            width=130,
        )
        self._snap_btn.pack(side="left", padx=4, pady=8)

        # Confidence slider
        ctk.CTkLabel(
            toolbar, text="Conf:", text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(20, 2))

        self._conf_var = ctk.DoubleVar(value=0.40)
        self._conf_slider = ctk.CTkSlider(
            toolbar,
            from_=0.1, to=0.95, number_of_steps=85,
            variable=self._conf_var,
            command=self._on_conf_change,
            width=120,
            button_color=ACCENT,
            progress_color=ACCENT2,
        )
        self._conf_slider.pack(side="left", padx=2)

        self._conf_label = ctk.CTkLabel(
            toolbar, text="40%", text_color=ACCENT,
            font=ctk.CTkFont(size=11, weight="bold")
        )
        self._conf_label.pack(side="left", padx=(2, 16))

    def _build_sidebar(self, parent):
        """Right sidebar with stats, filters, alerts."""
        sidebar = ctk.CTkScrollableFrame(
            parent, fg_color=BG_DARK, corner_radius=0,
            scrollbar_button_color=BG_CARD2,
        )
        sidebar.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(5, 10), pady=(10, 10))
        sidebar.columnconfigure(0, weight=1)

        self._build_stats_card(sidebar)
        self._build_class_count_card(sidebar)
        self._build_alert_card(sidebar)
        self._build_filter_card(sidebar)

    def _card(self, parent, title: str, icon: str = "") -> ctk.CTkFrame:
        """Helper: creates a titled card widget and returns its content frame."""
        outer = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=12)
        outer.pack(fill="x", padx=0, pady=(0, 10))

        hdr = ctk.CTkFrame(outer, fg_color="transparent", height=32)
        hdr.pack(fill="x", padx=12, pady=(8, 0))
        ctk.CTkLabel(
            hdr,
            text=f"{icon}  {title}" if icon else title,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=ACCENT,
        ).pack(side="left")

        sep = ctk.CTkFrame(outer, height=1, fg_color="#2A2A3A")
        sep.pack(fill="x", padx=12, pady=(4, 0))

        body = ctk.CTkFrame(outer, fg_color="transparent")
        body.pack(fill="x", padx=12, pady=(6, 10))
        return body

    def _build_stats_card(self, parent):
        body = self._card(parent, "Live Stats", "📊")
        body.columnconfigure((0, 1), weight=1)

        def stat(row, label, value_text, color=TEXT_PRIMARY):
            ctk.CTkLabel(
                body, text=label, text_color=TEXT_MUTED,
                font=ctk.CTkFont(size=11)
            ).grid(row=row, column=0, sticky="w", pady=2)
            lbl = ctk.CTkLabel(
                body, text=value_text, text_color=color,
                font=ctk.CTkFont(size=12, weight="bold")
            )
            lbl.grid(row=row, column=1, sticky="e", pady=2)
            return lbl

        self._fps_lbl       = stat(0, "FPS",            "—",   ACCENT)
        self._det_count_lbl = stat(1, "Objects Now",    "0",   SUCCESS)
        self._total_lbl     = stat(2, "Total Detected", "0",   TEXT_PRIMARY)
        self._session_lbl   = stat(3, "Session Time",   "00:00", TEXT_MUTED)
        self._model_lbl     = stat(4, "Model",          "—",   ACCENT2)

    def _build_class_count_card(self, parent):
        body = self._card(parent, "Object Counts", "🏷️")
        self._class_count_frame = body

        self._class_count_labels: dict[str, ctk.CTkLabel] = {}
        ctk.CTkLabel(
            body, text="Start detection to see counts.",
            text_color=TEXT_MUTED, font=ctk.CTkFont(size=11)
        ).pack(anchor="w")

    def _build_alert_card(self, parent):
        body = self._card(parent, "Smart Alerts", "🔔")

        ctk.CTkLabel(
            body,
            text="Alert when class is detected:",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=11),
        ).pack(anchor="w", pady=(0, 4))

        # Placeholder — populated after model loads
        self._alert_inner = ctk.CTkFrame(body, fg_color="transparent")
        self._alert_inner.pack(fill="x")

        ctk.CTkLabel(
            self._alert_inner,
            text="Start detection to configure alerts.",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=11),
        ).pack(anchor="w")

        # Sound toggle
        self._sound_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            body,
            text="Sound alert (Windows beep)",
            variable=self._sound_var,
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=11),
            checkbox_height=16, checkbox_width=16,
            border_color=ACCENT2,
            fg_color=ACCENT2,
            hover_color="#5A1FCC",
        ).pack(anchor="w", pady=(6, 0))

    def _build_filter_card(self, parent):
        body = self._card(parent, "Class Filter", "🔍")

        ctk.CTkLabel(
            body,
            text="Only show selected classes:",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=11),
        ).pack(anchor="w", pady=(0, 4))

        self._filter_inner = ctk.CTkScrollableFrame(
            body, fg_color="transparent", height=120,
            scrollbar_button_color=BG_CARD2,
        )
        self._filter_inner.pack(fill="x")

        ctk.CTkLabel(
            self._filter_inner,
            text="Start detection to configure filters.",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=11),
        ).pack(anchor="w")

        # Select all / none buttons
        btn_row = ctk.CTkFrame(body, fg_color="transparent")
        btn_row.pack(fill="x", pady=(6, 0))
        ctk.CTkButton(
            btn_row, text="All", width=60, height=24,
            font=ctk.CTkFont(size=11),
            fg_color=BG_CARD2, hover_color="#2A2A4A",
            text_color=ACCENT,
            command=self._select_all_classes,
        ).pack(side="left", padx=(0, 6))
        ctk.CTkButton(
            btn_row, text="None", width=60, height=24,
            font=ctk.CTkFont(size=11),
            fg_color=BG_CARD2, hover_color="#2A2A4A",
            text_color=TEXT_MUTED,
            command=self._deselect_all_classes,
        ).pack(side="left")

    def _build_log_panel(self, parent):
        """Bottom detection log panel."""
        panel = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=12)
        panel.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=(5, 10))
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(panel, fg_color="transparent", height=30)
        hdr.grid(row=0, column=0, sticky="ew", padx=12, pady=(6, 0))
        ctk.CTkLabel(
            hdr, text="📋  Detection Log",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=TEXT_PRIMARY,
        ).pack(side="left")

        clear_btn = ctk.CTkButton(
            hdr, text="Clear", width=50, height=22,
            font=ctk.CTkFont(size=10),
            fg_color=BG_CARD2, hover_color="#2A2A4A",
            text_color=TEXT_MUTED,
            command=self._clear_log,
        )
        clear_btn.pack(side="right")

        self._log_box = ctk.CTkTextbox(
            panel,
            fg_color=BG_CARD2,
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(family="Consolas", size=10),
            state="disabled",
            corner_radius=8,
            wrap="none",
        )
        self._log_box.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 10))

    # ═════════════════════════════════════════════════════════════════════════
    # Control Handlers
    # ═════════════════════════════════════════════════════════════════════════

    def _toggle_detection(self):
        if not self._running:
            self._start_detection()
        else:
            self._stop_detection()

    def _start_detection(self):
        model_key = self._model_var.get()
        model_path = self.MODELS.get(model_key, "yolov8n.pt")
        self._alert_system.sound_enabled = self._sound_var.get()

        self._detector = ObjectDetector(
            model_path=model_path,
            source=0,
            output_queue=self._frame_queue,
            alert_system=self._alert_system,
            recorder=self._recorder,
            snapshot_saver=self._snapshot_saver,
            conf_threshold=self._conf_var.get(),
        )
        self._detector.start()
        self._running = True
        self._session_start = time.time()

        self._start_btn.configure(
            text="⏹  Stop",
            fg_color="#6B1A1A",
            hover_color="#A82727",
        )
        self._model_dropdown.configure(state="disabled")
        self._set_status("Detection running…", color=SUCCESS)
        self._model_lbl.configure(text=model_key.split("(")[0].strip())

        # Populate class filter and alert panels after model loads (slight delay)
        self.after(1200, self._populate_class_panels)

    def _stop_detection(self):
        if self._detector:
            self._detector.stop()
            self._detector = None
        if self._recorder.is_recording:
            path = self._recorder.stop()
            self._set_status(f"Recording saved: {os.path.basename(path)}", color=WARNING)
            self._rec_badge.configure(text="")
        else:
            self._set_status("Detection stopped.", color=TEXT_MUTED)

        self._running = False
        self._start_btn.configure(
            text="▶  Start",
            fg_color="#1A6B35",
            hover_color="#27A850",
        )
        self._model_dropdown.configure(state="normal")
        self._video_label.configure(image=None, text="Feed stopped.")
        self._fps_lbl.configure(text="—")

    def _toggle_recording(self):
        if not self._running:
            self._set_status("Start detection first.", color=WARNING)
            return

        # Get latest frame size
        w, h = 640, 480
        is_rec, path = self._recorder.toggle(w, h, fps=20.0)
        if is_rec:
            self._record_btn.configure(
                text="⏹  Stop Rec",
                fg_color="#6B1A1A",
                border_color=DANGER,
                text_color=DANGER,
            )
            self._rec_badge.configure(text="● REC")
            self._set_status(f"Recording: {os.path.basename(path)}", color=DANGER)
        else:
            self._record_btn.configure(
                text="⏺  Record",
                fg_color=BG_CARD,
                border_color=DANGER,
                text_color=DANGER,
            )
            self._rec_badge.configure(text="")
            self._set_status(f"Saved: {os.path.basename(path)}", color=SUCCESS)

    def _take_snapshot(self):
        if not self._running or not self._detector:
            self._set_status("Start detection first.", color=WARNING)
            return
        self._detector.request_snapshot()
        self._set_status("📷 Snapshot saved to /snapshots", color=ACCENT)

    def _on_model_change(self, choice: str):
        if self._running and self._detector:
            new_path = self.MODELS.get(choice, "yolov8n.pt")
            self._detector.swap_model(new_path)
            self._model_lbl.configure(text=choice.split("(")[0].strip())
            self._set_status(f"Switching to {choice}…", color=WARNING)

    def _on_conf_change(self, value):
        pct = int(float(value) * 100)
        self._conf_label.configure(text=f"{pct}%")
        if self._detector:
            self._detector.conf_threshold = float(value)

    def _populate_class_panels(self):
        """Populate class filter checkboxes and alert toggles after model loads."""
        if not self._detector:
            return
        class_names = sorted(self._detector.get_class_names())
        if not class_names:
            self.after(500, self._populate_class_panels)
            return

        # ── Class filter panel ────────────────────────────────────────────
        for w in self._filter_inner.winfo_children():
            w.destroy()

        self._class_vars = {}
        for name in class_names:
            var = ctk.BooleanVar(value=True)
            self._class_vars[name] = var

            cb = ctk.CTkCheckBox(
                self._filter_inner,
                text=name,
                variable=var,
                command=self._apply_class_filter,
                font=ctk.CTkFont(size=10),
                text_color=TEXT_PRIMARY,
                checkbox_height=14, checkbox_width=14,
                border_color=ACCENT,
                fg_color=ACCENT,
                hover_color=ACCENT2,
            )
            cb.pack(anchor="w", pady=1)

        # ── Alert panel ───────────────────────────────────────────────────
        # Show only common/interesting COCO classes for brevity
        ALERT_CLASSES = [
            "person", "car", "truck", "bus", "motorcycle", "bicycle",
            "dog", "cat", "knife", "cell phone", "laptop", "backpack",
            "fire hydrant", "stop sign",
        ]
        alert_targets = [c for c in ALERT_CLASSES if c in class_names]

        for w in self._alert_inner.winfo_children():
            w.destroy()

        self._alert_vars = {}
        for name in alert_targets:
            var = ctk.BooleanVar(value=False)
            self._alert_vars[name] = var

            row = ctk.CTkFrame(self._alert_inner, fg_color="transparent")
            row.pack(fill="x", pady=1)

            cb = ctk.CTkCheckBox(
                row,
                text=name.title(),
                variable=var,
                command=lambda n=name, v=var: self._on_alert_toggle(n, v),
                font=ctk.CTkFont(size=11),
                text_color=TEXT_PRIMARY,
                checkbox_height=14, checkbox_width=14,
                border_color=WARNING,
                fg_color=WARNING,
                hover_color="#CC7700",
            )
            cb.pack(side="left")

    def _on_alert_toggle(self, class_name: str, var: ctk.BooleanVar):
        if var.get():
            self._alert_system.set_alert(class_name, threshold=0.45, cooldown=5.0)
        else:
            self._alert_system.remove_alert(class_name)

    def _apply_class_filter(self):
        if not self._detector:
            return
        enabled = {name for name, var in self._class_vars.items() if var.get()}
        # If all selected, pass None (no filter)
        if len(enabled) == len(self._class_vars):
            self._detector.set_enabled_classes(None)
        else:
            self._detector.set_enabled_classes({n.lower() for n in enabled})

    def _select_all_classes(self):
        for var in self._class_vars.values():
            var.set(True)
        self._apply_class_filter()

    def _deselect_all_classes(self):
        for var in self._class_vars.values():
            var.set(False)
        self._apply_class_filter()

    def _clear_log(self):
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")

    # ═════════════════════════════════════════════════════════════════════════
    # Frame Queue Polling & UI Update
    # ═════════════════════════════════════════════════════════════════════════

    def _poll_frame_queue(self):
        """Called every ~33ms by tkinter's after scheduler to consume frames."""
        try:
            df: DetectionFrame = self._frame_queue.get_nowait()
            self._update_video(df)
            self._update_stats(df)
            self._update_class_counts(df)
            self._append_log(df)
        except queue.Empty:
            pass

        if self._running:
            self._update_session_time()

        self.after(33, self._poll_frame_queue)

    def _update_video(self, df: DetectionFrame):
        """Resize and display the annotated frame in the video label."""
        label_w = self._video_label.winfo_width()
        label_h = self._video_label.winfo_height()
        if label_w < 10 or label_h < 10:
            label_w, label_h = 800, 480

        frame_rgb = df.frame[:, :, ::-1]  # BGR→RGB
        img = Image.fromarray(frame_rgb)
        img.thumbnail((label_w, label_h), Image.BILINEAR)

        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
        self._video_label.configure(image=ctk_img, text="")
        self._video_label._image_ref = ctk_img  # prevent GC

    def _update_stats(self, df: DetectionFrame):
        self._fps_lbl.configure(text=f"{df.fps:.1f}")
        n = len(df.detections)
        self._det_count_lbl.configure(text=str(n))
        self._total_detections += n
        self._total_lbl.configure(text=str(self._total_detections))

    def _update_class_counts(self, df: DetectionFrame):
        """Refresh the class count card."""
        frame = self._class_count_frame

        # Remove old widgets
        for w in frame.winfo_children():
            w.destroy()
        self._class_count_labels = {}

        if not df.class_counts:
            ctk.CTkLabel(
                frame, text="No objects detected.",
                text_color=TEXT_MUTED, font=ctk.CTkFont(size=11)
            ).pack(anchor="w")
            return

        for cls_name, count in sorted(df.class_counts.items(), key=lambda x: -x[1]):
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x", pady=1)

            ctk.CTkLabel(
                row, text=f"  {cls_name}",
                text_color=TEXT_PRIMARY, font=ctk.CTkFont(size=11),
                anchor="w",
            ).pack(side="left", fill="x", expand=True)

            bar_val = min(count / 5.0, 1.0)
            bar = ctk.CTkProgressBar(row, width=60, height=6, progress_color=ACCENT2)
            bar.set(bar_val)
            bar.pack(side="right", padx=(4, 0))

            ctk.CTkLabel(
                row, text=f"{count}",
                text_color=ACCENT, font=ctk.CTkFont(size=11, weight="bold"), width=24,
            ).pack(side="right")

    def _append_log(self, df: DetectionFrame):
        """Append new detection events to the log textbox."""
        if not df.detections:
            return

        ts = time.strftime("%H:%M:%S", time.localtime(df.timestamp))
        lines = []
        for d in df.detections:
            lines.append(
                f"[{ts}]  #{d.track_id:<3}  {d.class_name:<16}  {d.confidence:.0%}  |  {d.color_name}"
            )

        self._log_box.configure(state="normal")
        for line in lines:
            self._log_box.insert("end", line + "\n")
        # Keep log bounded
        total = int(self._log_box.index("end-1c").split(".")[0])
        if total > self.LOG_MAX_ROWS:
            self._log_box.delete("1.0", f"{total - self.LOG_MAX_ROWS}.0")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _update_session_time(self):
        elapsed = int(time.time() - self._session_start)
        m, s = divmod(elapsed, 60)
        self._session_lbl.configure(text=f"{m:02d}:{s:02d}")

    def _set_status(self, msg: str, color: str = TEXT_MUTED):
        self._status_label.configure(text=msg, text_color=color)

    # ═════════════════════════════════════════════════════════════════════════
    # Lifecycle
    # ═════════════════════════════════════════════════════════════════════════

    def _on_close(self):
        if self._running:
            self._stop_detection()
        self.destroy()
