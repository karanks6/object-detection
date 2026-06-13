"""
alerts.py
Smart alert system for Object Detection Pro.
Allows users to configure class-based triggers.
When a trigger fires, a banner message is set and optionally a sound plays.
"""

import time
import threading
try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False


class AlertSystem:
    """
    Manages user-defined detection alerts.

    An alert fires when:
      - A target class is detected
      - Confidence >= threshold
      - Cooldown period has elapsed since last alert

    Attributes:
        active_alerts (dict): class_name -> {"threshold": float, "cooldown": float, "last_fired": float}
        banner_message (str): Current message to display on screen
        banner_until (float): Timestamp until which banner should show
        sound_enabled (bool): Whether to play a beep on alert
        on_alert_callback: Optional callback(class_name) called when alert fires
    """

    BANNER_DURATION = 3.0  # seconds the banner stays visible

    def __init__(self, sound_enabled: bool = True):
        self.active_alerts: dict = {}
        self.banner_message: str = ""
        self.banner_until: float = 0.0
        self.sound_enabled: bool = sound_enabled and SOUND_AVAILABLE
        self.on_alert_callback = None
        self._lock = threading.Lock()

    def set_alert(self, class_name: str, threshold: float = 0.5, cooldown: float = 5.0):
        """Add or update an alert for a given class."""
        with self._lock:
            self.active_alerts[class_name.lower()] = {
                "threshold": threshold,
                "cooldown": cooldown,
                "last_fired": 0.0,
            }

    def remove_alert(self, class_name: str):
        """Remove an alert for a given class."""
        with self._lock:
            self.active_alerts.pop(class_name.lower(), None)

    def clear_alerts(self):
        """Remove all alerts."""
        with self._lock:
            self.active_alerts.clear()

    def get_alert_classes(self) -> list:
        """Return the list of classes that have alerts configured."""
        with self._lock:
            return list(self.active_alerts.keys())

    def check(self, detections: list):
        """
        Check a list of detections against configured alerts.

        Args:
            detections: list of dicts with keys: 'class_name', 'confidence'
        """
        now = time.time()
        with self._lock:
            for det in detections:
                name = det.get("class_name", "").lower()
                conf = det.get("confidence", 0.0)
                if name in self.active_alerts:
                    alert = self.active_alerts[name]
                    if conf >= alert["threshold"] and (now - alert["last_fired"]) >= alert["cooldown"]:
                        alert["last_fired"] = now
                        self._fire(name, conf)

    def _fire(self, class_name: str, confidence: float):
        """Internal: fire the alert (sets banner, plays sound, calls callback)."""
        display_name = class_name.title()
        self.banner_message = f"[!] ALERT: {display_name} detected! ({confidence:.0%})"
        self.banner_until = time.time() + self.BANNER_DURATION

        if self.sound_enabled:
            threading.Thread(
                target=lambda: winsound.Beep(1000, 300),
                daemon=True
            ).start()

        if self.on_alert_callback:
            try:
                self.on_alert_callback(class_name)
            except Exception:
                pass

    def get_banner(self) -> str | None:
        """
        Returns the current banner message if it should be displayed, else None.
        """
        if time.time() < self.banner_until:
            return self.banner_message
        return None

    def is_sound_available(self) -> bool:
        return SOUND_AVAILABLE
