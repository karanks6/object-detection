# 🎯 Object Detection Pro

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00CFFF?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMiIgZmlsbD0id2hpdGUiLz48L3N2Zz4=)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?style=for-the-badge&logo=opencv&logoColor=white)
![CustomTkinter](https://img.shields.io/badge/CustomTkinter-5.2%2B-7B2FFF?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A premium real-time object detection dashboard powered by YOLOv8 and ByteTrack.**  
Featuring persistent object tracking, smart alerts, color intelligence, recording, and a beautiful dark-mode UI.

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **Real-Time Object Detection** | YOLOv8 inference on live webcam feed at high FPS |
| 🔁 **Object Tracking (ByteTrack)** | Each object gets a persistent ID across frames |
| 🎨 **Color Intelligence** | Maps average object color to 110+ human-readable names |
| 🔔 **Smart Alert System** | Configurable per-class alerts with cooldown & sound |
| 📊 **Live Dashboard** | FPS counter, per-class object counts, session timer |
| 🎥 **Video Recording** | One-click toggle to save annotated video as `.avi` |
| 📸 **Snapshot Capture** | Save any frame as a timestamped `.png` |
| ⚙️ **Hot Model Swap** | Switch between YOLOv8n/m without restarting |
| 🔍 **Class Filter** | Toggle which COCO classes to detect at runtime |
| 📋 **Detection Log** | Live scrollable timestamped log of every detection |
| 🖥️ **Premium Dark UI** | Built with CustomTkinter — modern glassmorphism design |

---

## 🖼️ Screenshots

> *Launch the app and press **▶ Start** to see it in action.*

The interface consists of:
- **Left panel** — live annotated webcam feed with rounded bounding boxes, track IDs, confidence scores, and color names
- **Right sidebar** — stats card, per-class count bars, alert toggles, class filter checkboxes
- **Bottom log** — scrollable real-time detection event log

---

## 📁 Project Structure

```
object-detection/
│
├── main.py                  # 🚀 Entry point — run this
│
├── app/
│   ├── __init__.py
│   ├── gui.py               # 🖥️  CustomTkinter dark-mode dashboard
│   ├── detector.py          # 🧠  YOLO inference + ByteTrack tracking thread
│   ├── alerts.py            # 🔔  Smart alert system
│   ├── color_names.py       # 🎨  RGB → human-readable color name mapping
│   └── recorder.py          # 🎥  Video recorder & snapshot saver
│
├── snapshots/               # 📸  Auto-created; stores PNG snapshots
├── recordings/              # 🎞️  Auto-created; stores .avi recordings
│
├── yolov8n.pt               # Fast model (nano)
├── yolov8m.pt               # Accurate model (medium)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python **3.10 or later**
- A working **webcam**
- Windows / Linux / macOS

### 2. Clone the repository

```bash
git clone https://github.com/your-username/object-detection.git
cd object-detection
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> 💡 On first run, `ultralytics` may download model weights automatically if `.pt` files are missing.

### 5. Run the app

```bash
python main.py
```

---

## 🎮 Usage Guide

### Starting / Stopping Detection

Click **▶ Start** in the toolbar to begin. Click **⏹ Stop** to halt detection.

### Switching Models

Use the **Model** dropdown in the title bar:
- **YOLOv8 Nano** — Fastest, ideal for low-end hardware or high-FPS requirements
- **YOLOv8 Medium** — More accurate, better for complex scenes

You can switch models **while detection is running** — the swap is applied on the next inference cycle.

### Confidence Threshold

Drag the **Conf:** slider in the toolbar to set the minimum detection confidence (10%–95%). Lower values detect more objects but with more false positives.

### Object Tracking

Every detected object receives a **persistent track ID** (e.g., `#3 person`) that stays consistent across frames using ByteTrack. The bounding box color is unique per track ID.

### Color Intelligence

Below each bounding box you'll see the nearest **human-readable color name** for that object (e.g., `● Steel Blue`, `● Forest Green`). This is computed from the average RGB of the object's region and mapped to a palette of 110+ curated colors.

### Smart Alerts 🔔

In the **Smart Alerts** sidebar card:
1. Check any class (e.g., `Person`, `Car`) to enable an alert
2. When that class is detected with ≥45% confidence, an **alert banner** appears on the video feed
3. A **beep sound** plays (Windows only) — toggle with the "Sound alert" checkbox
4. Alerts have a **5-second cooldown** to prevent spam

### Class Filter 🔍

In the **Class Filter** sidebar card:
- Uncheck any COCO class to **hide it from detection**
- Use **All** / **None** buttons for quick selection
- Applies live — no restart needed

### Recording 🎥

Click **⏺ Record** to start saving the annotated feed as a `.avi` file in the `recordings/` folder.
- Files are timestamped: `recording_20260613_143022.avi`
- Click **⏹ Stop Rec** to finalize and save
- A **● REC** badge appears in the top-right of the video panel while active

### Snapshots 📸

Click **📷 Snapshot** to save the current annotated frame as a `.png` in the `snapshots/` folder.
- Files are timestamped: `snapshot_20260613_143045.png`

### Detection Log 📋

The bottom panel shows a scrollable, real-time log:
```
[14:30:22]  #3   person            87%  |  Charcoal
[14:30:22]  #7   car               92%  |  Dark Gray
[14:30:23]  #3   person            85%  |  Charcoal
```
Click **Clear** to reset the log.

---

## ⚙️ Configuration

All key parameters are exposed in the UI. For advanced users, constants in `app/gui.py` (colors, log size) and `app/detector.py` (FPS window size, box drawing) can be tweaked directly.

| Parameter | Location | Default | Description |
|---|---|---|---|
| `conf_threshold` | UI Slider | `0.40` | Min detection confidence |
| `cooldown` | `alerts.py` | `5.0s` | Alert cooldown per class |
| `BANNER_DURATION` | `alerts.py` | `3.0s` | Alert banner display time |
| `max_log_entries` | `detector.py` | `200` | Max detection log entries |
| `LOG_MAX_ROWS` | `gui.py` | `150` | Max visible log rows in UI |
| Recording FPS | `recorder.py` toggle | `20.0` | Output video frame rate |

---

## 🧠 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
│                   (Entry Point)                             │
└────────────────────────┬────────────────────────────────────┘
                         │ creates
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   ObjectDetectionApp                        │
│                     (app/gui.py)                            │
│                                                             │
│  Owns:  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
│         │ AlertSystem  │  │VideoRecorder │  │Snapshot  │  │
│         │ (alerts.py)  │  │(recorder.py) │  │ Saver    │  │
│         └──────────────┘  └──────────────┘  └──────────┘  │
│                                                             │
│  Spawns: ┌────────────────────────────────────────────────┐ │
│          │          ObjectDetector Thread                 │ │
│          │           (app/detector.py)                    │ │
│          │                                                │ │
│          │  1. Reads webcam frames                        │ │
│          │  2. Runs model.track() (ByteTrack)             │ │
│          │  3. Calls get_color_name() per box             │ │
│          │  4. Draws rounded boxes, labels, banners       │ │
│          │  5. Pushes DetectionFrame to queue             │ │
│          └──────────────────────────────────────────────--┘ │
│                         │ queue.Queue                       │
│                         ▼                                   │
│         GUI polls every 33ms → updates video + stats       │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Decisions:**
- **Thread-safe queue**: The detector pushes frames via `queue.Queue(maxsize=2)`. If the GUI is slow, frames are dropped (not buffered) to prevent memory growth.
- **ByteTrack (`persist=True`)**: YOLOv8's built-in tracker assigns stable IDs without requiring a separate library.
- **Model hot-swap**: The detector checks a `_swap_model_path` flag on each loop iteration — atomic swap with no thread locks needed on inference.
- **Alert cooldown**: Prevents alert spam when an object stays in frame for multiple seconds.

---

## 🗂️ COCO Classes Supported

The models are trained on the [COCO dataset](https://cocodataset.org/) with **80 object classes**, including:

`person`, `bicycle`, `car`, `motorcycle`, `airplane`, `bus`, `train`, `truck`, `boat`, `traffic light`, `fire hydrant`, `stop sign`, `parking meter`, `bench`, `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`, `backpack`, `umbrella`, `handbag`, `tie`, `suitcase`, `frisbee`, `skis`, `snowboard`, `sports ball`, `kite`, `baseball bat`, `baseball glove`, `skateboard`, `surfboard`, `tennis racket`, `bottle`, `wine glass`, `cup`, `fork`, `knife`, `spoon`, `bowl`, `banana`, `apple`, `sandwich`, `orange`, `broccoli`, `carrot`, `hot dog`, `pizza`, `donut`, `cake`, `chair`, `couch`, `potted plant`, `bed`, `dining table`, `toilet`, `tv`, `laptop`, `mouse`, `remote`, `keyboard`, `cell phone`, `microwave`, `oven`, `toaster`, `sink`, `refrigerator`, `book`, `clock`, `vase`, `scissors`, `teddy bear`, `hair drier`, `toothbrush`

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `ultralytics` | ≥8.0 | YOLOv8 model inference and ByteTrack tracking |
| `opencv-python` | ≥4.8 | Webcam capture, frame drawing, video writing |
| `customtkinter` | ≥5.2 | Modern dark-mode GUI framework |
| `Pillow` | ≥10.0 | Frame → PIL Image conversion for CTkImage |
| `numpy` | ≥1.24 | Array operations, average color computation |

---

## 🛠️ Troubleshooting

### Webcam not found
```
[Detector] ERROR: Cannot open source: 0
```
- Ensure no other app is using the webcam
- Try changing `source=0` to `source=1` in `gui.py` if you have multiple cameras

### Model loading is slow
- First-run model loading can take 5–15 seconds. The UI will be responsive but the video feed will appear after loading completes.

### `customtkinter` not found
```bash
pip install customtkinter
```

### Low FPS
- Switch to **YOLOv8 Nano** in the model dropdown
- Raise the confidence threshold slider (filters out more boxes = less drawing)
- Close other GPU-heavy applications

### Sound not working on non-Windows
- The `winsound` beep is Windows-only. On Linux/macOS, alerts are visual-only. The "Sound alert" checkbox will be ignored silently.

---

## 🤝 Contributing

Contributions, ideas, and PRs are welcome!

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m "Add amazing feature"`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Ideas for Future Features
- [ ] Support for video file input (not just webcam)
- [ ] Zone-based detection (alert only when object enters a defined region)
- [ ] Export detection log as CSV
- [ ] Heatmap overlay of object positions over time
- [ ] Multi-camera support
- [ ] Custom model loading (bring your own `.pt`)
- [ ] Face blur / anonymization mode

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — state-of-the-art object detection
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — multi-object tracking algorithm
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) — modern Python GUI framework
- [COCO Dataset](https://cocodataset.org/) — training data for YOLO models

---

<div align="center">

Made with ❤️ using Python, YOLOv8, and CustomTkinter

</div>
