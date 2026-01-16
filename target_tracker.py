import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class TargetState:
    seen: bool = False
    cx_norm: float = 0.0  # -1..+1 (left..right)
    bbox: Optional[Tuple[int, int, int, int]] = None  # x,y,w,h
    last_ts: float = 0.0
    method: str = "none"  # "tracker" or "detect"


class TargetTracker:
    """
    Lightweight camera tracker that gives left/right target direction.
    - Uses Haar face detection to acquire a target
    - Uses an OpenCV tracker (KCF/MOSSE/CSRT if available) to keep tracking even if face turns away briefly

    Output:
      cx_norm in [-1..+1] where:
        - negative = target is left of image center
        - positive = target is right of image center
    """
    def __init__(self, cam, prefer_tracker: bool = True, detect_every_n: int = 6):
        self.cam = cam
        self.prefer_tracker = prefer_tracker
        self.detect_every_n = max(1, int(detect_every_n))

        self._lock = threading.Lock()
        self._state = TargetState()

        self._running = False
        self._thread = None

        self._tracker = None
        self._frames = 0

        # Haar cascade (bundled with opencv-python)
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face = cv2.CascadeClassifier(cascade_path)
            if self._face.empty():
                self._face = None
        except Exception:
            self._face = None

        # pick a tracker implementation if available
        self._tracker_factory = self._pick_tracker_factory() if prefer_tracker else None

    def _pick_tracker_factory(self):
        # Try common trackers depending on OpenCV build
        factories = []
        # OpenCV 4.x sometimes has these under cv2.legacy
        if hasattr(cv2, "legacy"):
            leg = cv2.legacy
            if hasattr(leg, "TrackerCSRT_create"):
                factories.append(("CSRT", leg.TrackerCSRT_create))
            if hasattr(leg, "TrackerKCF_create"):
                factories.append(("KCF", leg.TrackerKCF_create))
            if hasattr(leg, "TrackerMOSSE_create"):
                factories.append(("MOSSE", leg.TrackerMOSSE_create))
        # Or directly on cv2
        if hasattr(cv2, "TrackerCSRT_create"):
            factories.append(("CSRT", cv2.TrackerCSRT_create))
        if hasattr(cv2, "TrackerKCF_create"):
            factories.append(("KCF", cv2.TrackerKCF_create))
        if hasattr(cv2, "TrackerMOSSE_create"):
            factories.append(("MOSSE", cv2.TrackerMOSSE_create))

        return factories[0] if factories else None  # best available

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.8)
        self._thread = None

    def get_state(self) -> TargetState:
        with self._lock:
            return TargetState(
                seen=self._state.seen,
                cx_norm=self._state.cx_norm,
                bbox=self._state.bbox,
                last_ts=self._state.last_ts,
                method=self._state.method,
            )

    def _set_state(self, seen: bool, cx_norm: float, bbox, method: str):
        with self._lock:
            self._state.seen = bool(seen)
            self._state.cx_norm = float(cx_norm)
            self._state.bbox = bbox
            self._state.last_ts = time.time()
            self._state.method = method

    def _decode(self, jpg: bytes):
        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img

    def _bbox_to_cx_norm(self, bbox, w_img: int):
        x, y, w, h = bbox
        cx = x + w * 0.5
        cx_norm = (cx - (w_img * 0.5)) / (w_img * 0.5)
        return max(-1.0, min(1.0, float(cx_norm)))

    def _detect_face(self, img):
        if self._face is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._face.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(40, 40),
        )
        if faces is None or len(faces) == 0:
            return None

        # choose largest face
        best = None
        best_area = 0
        for (x, y, w, h) in faces:
            area = int(w * h)
            if area > best_area:
                best_area = area
                best = (int(x), int(y), int(w), int(h))

        return best

    def _init_tracker(self, img, bbox):
        self._tracker = None
        if self._tracker_factory is None:
            return
        try:
            _name, factory = self._tracker_factory
            trk = factory()
            ok = trk.init(img, tuple(bbox))
            if ok:
                self._tracker = trk
        except Exception:
            self._tracker = None

    def _update_tracker(self, img):
        if self._tracker is None:
            return None
        try:
            ok, box = self._tracker.update(img)
            if not ok:
                return None
            x, y, w, h = box
            return (int(x), int(y), int(w), int(h))
        except Exception:
            return None

    def _run(self):
        while self._running:
            jpg = self.cam.get_frame(timeout=0.25)
            if not jpg:
                continue

            img = self._decode(jpg)
            if img is None:
                continue

            h_img, w_img = img.shape[:2]
            self._frames += 1

            # 1) Try tracker first
            bbox = self._update_tracker(img)
            if bbox is not None:
                cx_norm = self._bbox_to_cx_norm(bbox, w_img)
                self._set_state(True, cx_norm, bbox, method="tracker")
                continue

            # If tracker failed, drop it so we re-acquire
            self._tracker = None

            # 2) Periodically detect (or if no tracker)
            if (self._frames % self.detect_every_n) != 0:
                # keep state as "not seen" but don't spam
                # (autonomous.py uses seen=False to start search)
                self._set_state(False, 0.0, None, method="none")
                continue

            bbox = self._detect_face(img)
            if bbox is None:
                self._set_state(False, 0.0, None, method="detect")
                continue

            # Initialize tracker on the detected box (helps survive turning away briefly)
            self._init_tracker(img, bbox)

            cx_norm = self._bbox_to_cx_norm(bbox, w_img)
            self._set_state(True, cx_norm, bbox, method="detect")
