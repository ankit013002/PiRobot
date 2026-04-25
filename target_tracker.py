import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class TargetState:
    seen:     bool  = False
    cx_norm:  float = 0.0   # −1 (left) … +1 (right)
    bbox:     Optional[Tuple[int, int, int, int]] = None
    last_ts:  float = 0.0
    method:   str   = "none"


class TargetTracker:
    """
    Camera-based target tracker.
    Acquires targets via Haar face detection, then holds them with an
    OpenCV tracker (CSRT > KCF > MOSSE, whichever is available).

    Thread-safe; run start() / stop() from any thread.
    Poll get_state() for the latest TargetState.
    """

    def __init__(self, cam, prefer_tracker: bool = True, detect_every_n: int = 6):
        self._cam            = cam
        self._prefer_tracker = prefer_tracker
        self._detect_every   = max(1, int(detect_every_n))

        self._lock    = threading.Lock()
        self._state   = TargetState()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._tracker = None
        self._frames  = 0

        # Haar face cascade (bundled with opencv-python)
        try:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face = cv2.CascadeClassifier(path)
            if self._face.empty():
                self._face = None
        except Exception:
            self._face = None

        self._tracker_factory = self._pick_tracker() if prefer_tracker else None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.8)
        self._thread = None

    def get_state(self) -> TargetState:
        with self._lock:
            s = self._state
            return TargetState(seen=s.seen, cx_norm=s.cx_norm,
                               bbox=s.bbox, last_ts=s.last_ts, method=s.method)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _pick_tracker(self):
        candidates = []
        for ns in (cv2, getattr(cv2, 'legacy', None)):
            if ns is None:
                continue
            for name in ('TrackerCSRT_create', 'TrackerKCF_create', 'TrackerMOSSE_create'):
                fn = getattr(ns, name, None)
                if fn:
                    candidates.append((name.split('_')[0][7:], fn))
        return candidates[0] if candidates else None

    def _set_state(self, seen: bool, cx_norm: float, bbox, method: str) -> None:
        with self._lock:
            self._state = TargetState(seen=seen, cx_norm=cx_norm,
                                      bbox=bbox, last_ts=time.time(), method=method)

    def _cx_norm(self, bbox, w_img: int) -> float:
        x, _, w, _ = bbox
        return max(-1.0, min(1.0, ((x + w * 0.5) - w_img * 0.5) / (w_img * 0.5)))

    def _detect(self, img) -> Optional[tuple]:
        if self._face is None:
            return None
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._face.detectMultiScale(gray, scaleFactor=1.15,
                                            minNeighbors=5, minSize=(40, 40))
        if faces is None or len(faces) == 0:
            return None
        return max(((int(x), int(y), int(w), int(h))
                    for x, y, w, h in faces), key=lambda b: b[2] * b[3])

    def _init_tracker(self, img, bbox) -> None:
        self._tracker = None
        if self._tracker_factory is None:
            return
        try:
            _, factory = self._tracker_factory
            trk = factory()
            if trk.init(img, tuple(bbox)):
                self._tracker = trk
        except Exception:
            pass

    def _update_tracker(self, img) -> Optional[tuple]:
        if self._tracker is None:
            return None
        try:
            ok, box = self._tracker.update(img)
            return (int(box[0]), int(box[1]), int(box[2]), int(box[3])) if ok else None
        except Exception:
            return None

    def _run(self) -> None:
        while self._running:
            jpg = self._cam.get_frame(timeout=0.25)
            if not jpg:
                continue
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            _, w_img = img.shape[:2]
            self._frames += 1

            bbox = self._update_tracker(img)
            if bbox is not None:
                self._set_state(True, self._cx_norm(bbox, w_img), bbox, "tracker")
                continue

            self._tracker = None  # tracker lost — re-acquire

            if self._frames % self._detect_every != 0:
                self._set_state(False, 0.0, None, "none")
                continue

            bbox = self._detect(img)
            if bbox is None:
                self._set_state(False, 0.0, None, "detect")
                continue

            self._init_tracker(img, bbox)
            self._set_state(True, self._cx_norm(bbox, w_img), bbox, "detect")
