import time
import threading
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VisionState:
    ts: float = 0.0
    collision_risk: float = 0.0   # 0..1
    free_dir: str = "C"           # "L" | "R" | "C"
    motion_score: float = 0.0     # bigger = more motion
    face_count: int = 0

    # NEW
    person_count: int = 0
    person_close: bool = False
    head_tilt_deg: int = 120


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class VisionPerception:
    def __init__(
        self,
        cam,
        flow_size=(160, 120),
        face_every_n=5,
        person_every_n=8,
        debug=False,
        # NEW head sweep support:
        servo=None,
        head_lock=None,
        tilt_channel="1",
        tilt_center=120,
        tilt_angles=(85, 120, 155),
        tilt_settle=0.12,
        head_sweep_period=1.2,
    ):
        self.cam = cam
        self.flow_size = flow_size
        self.face_every_n = max(1, int(face_every_n))
        self.person_every_n = max(1, int(person_every_n))
        self.debug = debug

        self.servo = servo
        self.head_lock = head_lock
        self.tilt_channel = str(tilt_channel)
        self.tilt_center = int(tilt_center)
        self.tilt_angles = [int(a) for a in tilt_angles]
        self.tilt_settle = float(tilt_settle)
        self.head_sweep_period = float(head_sweep_period)

        self._state = VisionState(head_tilt_deg=self.tilt_center)
        self._state_lock = threading.Lock()

        self._thread = None
        self._running = False

        # for optical flow / motion
        self._prev_gray = None
        self._frame_i = 0

        # face detector (ships with opencv)
        try:
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        except Exception:
            self._face_cascade = None

        # person detector (no downloads)
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # head sweep state
        self._tilt_idx = 0
        self._next_tilt_ts = time.time() + 0.5
        self._pause_head_until = 0.0

    def pause_head_motion(self, seconds: float):
        """Temporarily stop tilt sweeping (useful during ultrasonic scans)."""
        self._pause_head_until = max(self._pause_head_until, time.time() + float(seconds))

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        t = self._thread
        if t:
            t.join(timeout=1.0)
        self._thread = None

    def get_state(self) -> VisionState:
        with self._state_lock:
            return VisionState(**self._state.__dict__)

    def _set_tilt(self, angle: int):
        if self.servo is None:
            return
        try:
            if self.head_lock is None:
                self.servo.set_servo_pwm(self.tilt_channel, angle)
            else:
                with self.head_lock:
                    self.servo.set_servo_pwm(self.tilt_channel, angle)
        except Exception:
            pass

        with self._state_lock:
            self._state.head_tilt_deg = int(angle)

    def _maybe_head_sweep(self):
        if self.servo is None:
            return
        now = time.time()
        if now < self._pause_head_until:
            return
        if now < self._next_tilt_ts:
            return

        self._tilt_idx = (self._tilt_idx + 1) % len(self.tilt_angles)
        ang = self.tilt_angles[self._tilt_idx]
        self._set_tilt(ang)
        time.sleep(self.tilt_settle)
        self._next_tilt_ts = now + self.head_sweep_period

    def _compute_risk_and_free_dir(self, gray_small: np.ndarray) -> tuple[float, str]:
        # Edge-based "clutter" proxy: more edges in the center => more risk
        edges = cv2.Canny(gray_small, 80, 160)

        h, w = edges.shape[:2]
        third = w // 3
        left = edges[:, :third]
        center = edges[:, third : 2 * third]
        right = edges[:, 2 * third :]

        l = float(np.mean(left))
        c = float(np.mean(center))
        r = float(np.mean(right))

        # normalize roughly (Canny mean is 0..255)
        risk = _clamp01((c - 15.0) / 60.0)

        # free direction: pick side with fewer edges
        if l + 3.0 < r:
            free = "L"
        elif r + 3.0 < l:
            free = "R"
        else:
            free = "C"

        return risk, free

    def _loop(self):
        while self._running:
            self._maybe_head_sweep()

            jpg = self.cam.get_frame(timeout=0.5)
            if not jpg:
                continue

            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            self._frame_i += 1

            # --- motion ---
            small = cv2.resize(img, self.flow_size)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            motion_score = 0.0
            if self._prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    self._prev_gray, gray, None,
                    0.5, 2, 15, 2, 5, 1.2, 0
                )
                mag, _ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_score = float(np.mean(mag))
            self._prev_gray = gray

            # --- collision-ish risk + free dir ---
            risk, free_dir = self._compute_risk_and_free_dir(gray)

            # --- faces ---
            face_count = 0
            if self._face_cascade is not None and (self._frame_i % self.face_every_n == 0):
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                g = cv2.equalizeHist(g)
                faces = self._face_cascade.detectMultiScale(g, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
                face_count = 0 if faces is None else len(faces)
            else:
                with self._state_lock:
                    face_count = self._state.face_count  # keep last

            # --- PEOPLE (humans) ---
            person_count = 0
            person_close = False
            if self._frame_i % self.person_every_n == 0:
                # HOG works better on somewhat larger width
                frame = cv2.resize(img, (320, 240))
                rects, weights = self._hog.detectMultiScale(
                    frame,
                    winStride=(8, 8),
                    padding=(16, 16),
                    scale=1.05
                )
                if rects is not None:
                    person_count = len(rects)
                    if person_count > 0:
                        h, w = frame.shape[:2]
                        best_area = 0.0
                        for (x, y, rw, rh) in rects:
                            best_area = max(best_area, (rw * rh) / float(w * h))
                        # "close" heuristic: big box fraction
                        person_close = best_area > 0.18
            else:
                with self._state_lock:
                    person_count = self._state.person_count
                    person_close = self._state.person_close

            with self._state_lock:
                self._state.ts = time.time()
                self._state.collision_risk = risk
                self._state.free_dir = free_dir
                self._state.motion_score = motion_score
                self._state.face_count = face_count
                self._state.person_count = person_count
                self._state.person_close = person_close

            # small sleep to avoid pegging CPU
            time.sleep(0.01)
