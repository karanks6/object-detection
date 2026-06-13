import cv2
import numpy as np
from ultralytics import YOLO
import threading
import tkinter as tk
from PIL import Image, ImageTk

# Load YOLO model
model = YOLO("yolov8m.pt")

# Create main Tkinter window
root = tk.Tk()
root.title("Object and Color Detection")
root.geometry("800x600")

# Stop flag
stop_flag = False

# Label to display video
video_label = tk.Label(root)
video_label.pack()

def get_average_rgb(image, x1, y1, x2, y2):
    roi = image[y1:y2, x1:x2]
    avg_color = np.mean(roi, axis=(0, 1))
    return tuple(map(int, avg_color))

def detect_objects():
    global stop_flag
    cap = cv2.VideoCapture(0)

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            b, g, r = get_average_rgb(frame, x1, y1, x2, y2)
            color_text = f"RGB: ({r}, {g}, {b})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (r, g, b), 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (r, g, b), 2)
            cv2.putText(frame, color_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (r, g, b), 2)

        # Convert frame to Tkinter image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        root.update_idletasks()
        root.update()

    cap.release()
    cv2.destroyAllWindows()

# Function to stop detection
def stop_detection():
    global stop_flag
    stop_flag = True
    root.destroy()

# Stop button
stop_button = tk.Button(root, text="Stop Detection", font=("Arial", 14), bg="red", fg="white", command=stop_detection)
stop_button.pack(pady=10)

# Start object detection in a separate thread
threading.Thread(target=detect_objects, daemon=True).start()

# Run the GUI
root.mainloop()
