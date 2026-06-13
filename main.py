"""
Object Detection Pro
====================
Entry point. Launches the CustomTkinter GUI.

Run with:
    python main.py
"""

from app.gui import ObjectDetectionApp


def main():
    app = ObjectDetectionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
