#!/usr/bin/env python3
from __future__ import annotations

"""
autonomous3.py
--------------
Server-driven autonomous loop for Freenove 4WD Smart Car (Pi side).

Reads and USES (locally + sends to server) all sensors:
- Battery voltage (ADC ch2)
- Ultrasonic forward distance (head parked forward)
- Ultrasonic triplet scan (right/center/left via servo pan)
- Infrared line sensors (3-bit mask)
- Photoresistors (ADC ch0/ch1)
- Dead-reckoned pose + coarse occupancy/visited memory (from motor commands)
- Stuck detection (distance change while commanding forward)

Sends sensor snapshot to:
    POST {PET_SERVER_URL}/pet/step

Applies returned plan:
    { mode, action, duration_s, speed_pwm, head_pan, chirp, sequence }

Also has a LOCAL reflex/safety layer:
- Low battery -> stop/sleep
- "Cliff-ish" IR all-zero while moving forward -> back+turn
- Ultrasonic too close -> stop + scan_and_avoid_with_memory()
- Stuck -> escape_stuck_with_memory()

IMPORTANT FIX for "STOP_WHILE_THINKING = False":
- Do NOT scan triplet while driving forward (sonar is on the head)
- Keep head parked forward any time you drive
- Request new plans near action end (avoid spam / jitter)

Voice is NOT used. Only optional buzzer chirps.
Camera/vision is OFF by default for simple autonomous.
"""

import os
import time
import json
import random
import threading
import logging
import traceback
from dataclasses import dataclass
from typing import Optional, Deque, Tuple, Any, Dict
from collections import deque
from queue import Queue, Empty

from car import Car
from led import Led
from buzzer import Buzzer

# ============================================================
# Optional .env load
# ============================================================
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ============================================================
# LOGGING
# ============================================================
LOG_LEVEL = logging.INFO
LOG_TO_FILE = True
LOG_FILE_PATH = "autonomous3_ai.log"

log = logging.getLogger("autonomous3")
log.setLevel(LOG_LEVEL)

_fmt = logging.Formatter(
    fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

_sh = logging.StreamHandler()
_sh.setFormatter(_fmt)
_sh.setLevel(LOG_LEVEL)
log.addHandler(_sh)

if LOG_TO_FILE:
    try:
        _fh = logging.FileHandler(LOG_FILE_PATH)
        _fh.setFormatter(_fmt)
        _fh.setLevel(LOG_LEVEL)
        log.addHandler(_fh)
    except Exception:
        log.exception("Failed to create log file handler")


def log_exc(prefix: str):
    log.error("%s\n%s", prefix, traceback.format_exc())


# ============================================================
# ENV / SERVER CONFIG
# ============================================================
PET_SERVER_URL = os.environ.get("PET_SERVER_URL", "").strip().rstrip("/")
PI_API_KEY = os.environ.get("PI_API_KEY", "").strip()
ROBOT_ID = os.environ.get("PET_ROBOT_ID", "sparky").strip()[:32]

# Optional: server step timeout (seconds)
PET_STEP_TIMEOUT_S = float(os.environ.get("PET_STEP_TIMEOUT_S", "6").strip() or "6")
PET_STEP_TIMEOUT_S = max(3.0, min(30.0, PET_STEP_TIMEOUT_S))

# Privacy / identity
PET_PRIVACY_LEVEL = os.environ.get("PET_PRIVACY_LEVEL", "normal").strip().lower()  # normal|minimal|offload
PET_ANON = os.environ.get("PET_ANON", "1").strip() == "1"

# Camera/target tracking (optional) — OFF by default for "simple autonomous"
PET_USE_CAMERA = os.environ.get("PET_USE_CAMERA", "0").strip() == "1"
PET_VISION_FPS = float(os.environ.get("PET_VISION_FPS", "3").strip() or "3")
PET_VISION_FPS = max(0.5, min(10.0, PET_VISION_FPS))

# ============================================================
# CORE LOOP / SAFETY
# ============================================================
LOOP_DT = 0.05  # ~20 Hz

LOW_BATTERY_THRESHOLD = float(os.environ.get("LOW_BATTERY_THRESHOLD", "5.0"))

BATTERY_UPDATE_DT = 0.60
SONAR_UPDATE_DT = 0.18
SCAN_TRIPLET_DT = 0.90        # right/center/left scan interval while stopped/thinking
LIGHT_UPDATE_DT = 0.50        # photoresistors via ADC 0/1
IR_UPDATE_DT = 0.12           # line sensors
POSE_UPDATE_DT = 0.35

ROAM_OBS_TRIGGER_CM = 45.0
FORWARD_HARD_STOP_CM = 18.0

# IR cliff reflex: disabled by default — the line sensors read 0 on most floors,
# causing constant false triggers. Enable only if the robot runs near table edges.
# Set PET_CLIFF_DETECT=1 to enable.
PET_CLIFF_DETECT = os.environ.get("PET_CLIFF_DETECT", "0").strip() == "1"
CLIFF_IR_MASK_VALUE = 0

STUCK_MIN_DELTA_CM = 4.0
STUCK_TIMEOUT_S = 1.6

DEFAULT_FWD_PWM = 1100
DEFAULT_TURN_BIAS = 250

# Head
HEAD_CENTER = 90
HEAD_LEFT = 150
HEAD_RIGHT = 30

# ============================================================
# "STOP AND THINK" (while waiting on server)
# ============================================================
# You *can* set this False for smoother motion, because we now:
# - never scan triplet while driving forward
# - keep the head parked forward while driving
STOP_WHILE_THINKING = False

THINK_MAX_WAIT_S = 6.0
THINK_HEAD_SWEEP_DT = 0.60
THINK_CHIRP_ON_REQUEST = 0  # keep 0 for silent

# ============================================================
# MODES + ACTIONS
# ============================================================
MODE_ROAM = "ROAM"
MODE_IDLE = "IDLE"
MODE_PLAY = "PLAY"
MODE_SLEEP = "SLEEP"

ALLOWED_MODES = {MODE_ROAM, MODE_IDLE, MODE_PLAY, MODE_SLEEP}
ALLOWED_ACTIONS = {
    "stop",
    "forward",
    "reverse",
    "turn_left",
    "turn_right",
    "spin_left",
    "spin_right",
    "wander",
    "sequence",
}

# ============================================================
# LED / BUZZER helpers
# ============================================================
def set_mode_led(led: Led, mode: str):
    try:
        if mode == MODE_ROAM:
            led.ledIndex(0xFF, 0, 255, 0)
        elif mode == MODE_IDLE:
            led.ledIndex(0xFF, 255, 200, 80)
        elif mode == MODE_PLAY:
            led.ledIndex(0xFF, 180, 0, 255)
        elif mode == MODE_SLEEP:
            led.ledIndex(0xFF, 0, 0, 80)
        else:
            led.ledIndex(0xFF, 0, 255, 0)
    except Exception:
        log_exc("LED update failed")


def chirp(buzzer: Buzzer, n: int = 1, on: float = 0.06, off: float = 0.05):
    for _ in range(max(0, int(n))):
        try:
            buzzer.set_state(True)
            time.sleep(on)
            buzzer.set_state(False)
            time.sleep(off)
        except Exception:
            log_exc("Buzzer chirp failed")


def clamp_int(v: Any, lo: int, hi: int, default: int) -> int:
    try:
        x = int(v)
        return max(lo, min(hi, x))
    except Exception:
        return default


def clamp_float(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        x = float(v)
        if x != x:
            return default
        return max(lo, min(hi, x))
    except Exception:
        return default


# ============================================================
# Local play sequences (if server returns action="sequence")
# ============================================================
@dataclass
class Step:
    cmd: Tuple[int, int, int, int]
    dur: float
    head_pan: Optional[int] = None
    head_tilt: Optional[int] = None


class StepRunner:
    def __init__(self):
        self.steps: Deque[Step] = deque()
        self.step_end: float = 0.0

    def load(self, steps: list[Step], now: float):
        self.steps = deque(steps)
        self.step_end = now
        log.info("[AI][SEQ] loaded %d steps", len(steps))

    def active(self) -> bool:
        return len(self.steps) > 0

    def tick(self, car: Car, now: float) -> bool:
        if not self.steps:
            return False

        if now >= self.step_end:
            st = self.steps.popleft()

            try:
                if st.head_pan is not None or st.head_tilt is not None:
                    car.set_head_pose(pan=st.head_pan, tilt=st.head_tilt, settle=0.01)
            except Exception:
                log_exc("[AI][SEQ] head pose failed")

            try:
                car.set_motors(*st.cmd)
            except Exception:
                log_exc("[AI][SEQ] set_motors failed")

            self.step_end = now + max(0.02, float(st.dur))

        return True


def play_sequence_zoomies() -> list[Step]:
    return [
        Step((0, 0, 0, 0), 0.15, head_pan=HEAD_CENTER),
        Step((1200, 1200, 1200, 1200), 0.35, head_pan=HEAD_RIGHT),
        Step((0, 0, 0, 0), 0.12, head_pan=HEAD_LEFT),
        Step((1200, 1200, 1200, 1200), 0.30, head_pan=HEAD_CENTER),
        Step((1200, 1200, -1200, -1200), 0.45, head_pan=HEAD_RIGHT),
        Step((0, 0, 0, 0), 0.10, head_pan=HEAD_LEFT),
        Step((1000, 1000, 1000, 1000), 0.28, head_pan=HEAD_CENTER),
        Step((0, 0, 0, 0), 0.15, head_pan=HEAD_CENTER),
    ]


def play_sequence_wiggle() -> list[Step]:
    return [
        Step((0, 0, 0, 0), 0.12, head_pan=HEAD_RIGHT),
        Step((900, 900, -900, -900), 0.20, head_pan=HEAD_LEFT),
        Step((-900, -900, 900, 900), 0.20, head_pan=HEAD_RIGHT),
        Step((900, 900, -900, -900), 0.20, head_pan=HEAD_LEFT),
        Step((0, 0, 0, 0), 0.10, head_pan=HEAD_CENTER),
        Step((950, 950, 950, 950), 0.22, head_pan=HEAD_RIGHT),
        Step((0, 0, 0, 0), 0.10, head_pan=HEAD_LEFT),
        Step((950, 950, 950, 950), 0.22, head_pan=HEAD_CENTER),
        Step((0, 0, 0, 0), 0.18, head_pan=HEAD_CENTER),
    ]


def play_sequence_spin_showoff() -> list[Step]:
    return [
        Step((0, 0, 0, 0), 0.18, head_pan=HEAD_CENTER),
        Step((1200, 1200, -1200, -1200), 0.70, head_pan=HEAD_RIGHT),
        Step((0, 0, 0, 0), 0.12, head_pan=HEAD_LEFT),
        Step((-1200, -1200, 1200, 1200), 0.55, head_pan=HEAD_LEFT),
        Step((0, 0, 0, 0), 0.20, head_pan=HEAD_CENTER),
    ]


def get_sequence_by_name(name: str) -> list[Step]:
    n = (name or "").strip().lower()
    if n == "zoomies":
        return play_sequence_zoomies()
    if n == "wiggle":
        return play_sequence_wiggle()
    if n == "spin":
        return play_sequence_spin_showoff()
    return random.choice([play_sequence_zoomies, play_sequence_wiggle, play_sequence_spin_showoff])()


# ============================================================
# Optional camera + target tracker (OFF by default)
# ============================================================
class VisionTracker:
    def __init__(self):
        self.ok = False
        self._lock = threading.Lock()
        self.seen = False
        self.cx_norm = 0.0
        self.last_seen_ts = 0.0
        self.method = "none"

        self._cam = None
        self._tracker = None
        self._thread = None
        self._running = False
        self._poll_dt = 1.0 / PET_VISION_FPS

        try:
            from camera import Camera  # your file
            from target_tracker import TargetTracker  # your file
            self._cam = Camera(stream_size=(400, 300))
            self._cam.start_stream()
            self._tracker = TargetTracker(self._cam, prefer_tracker=True, detect_every_n=6)
            self._tracker.start()
            self.ok = True
        except Exception:
            self.ok = False
            return

    def start(self):
        if not self.ok or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        try:
            if self._tracker:
                self._tracker.stop()
        except Exception:
            pass
        try:
            if self._cam:
                self._cam.stop_stream()
                self._cam.close()
        except Exception:
            pass

    def _run(self):
        while self._running:
            try:
                st = self._tracker.get_state()
                now = time.time()
                with self._lock:
                    self.seen = bool(getattr(st, "seen", False))
                    self.cx_norm = float(getattr(st, "cx_norm", 0.0) or 0.0)
                    self.method = str(getattr(st, "method", "none") or "none")
                    if self.seen:
                        self.last_seen_ts = now
            except Exception:
                pass
            time.sleep(self._poll_dt)

    def snapshot(self) -> dict:
        with self._lock:
            age_s = (time.time() - self.last_seen_ts) if self.last_seen_ts else 1e9
            return {
                "target_seen": bool(self.seen),
                "target_x": float(self.cx_norm),
                "target_age_s": float(age_s),
                "target_method": self.method,
            }


# ============================================================
# Sensor hub
# ============================================================
@dataclass
class SensorState:
    battery_v: Optional[float] = None
    battery_ts: float = 0.0

    ahead_cm: Optional[float] = None
    ahead_ts: float = 0.0

    scan_right_cm: Optional[float] = None
    scan_center_cm: Optional[float] = None
    scan_left_cm: Optional[float] = None
    scan_ts: float = 0.0

    ir_bits: Optional[int] = None
    ir_ts: float = 0.0

    light_l_v: Optional[float] = None
    light_r_v: Optional[float] = None
    light_ts: float = 0.0

    pose_x_cm: Optional[float] = None
    pose_y_cm: Optional[float] = None
    pose_th_deg: Optional[float] = None
    pose_ts: float = 0.0

    target_seen: Optional[bool] = None
    target_x: Optional[float] = None
    target_age_s: Optional[float] = None
    target_method: Optional[str] = None
    target_ts: float = 0.0


class SensorHub:
    def __init__(self, car: Car, vision: Optional[VisionTracker]):
        self.car = car
        self.vision = vision
        self.s = SensorState()

        self._last_batt_ts = 0.0
        self._last_sonar_ts = 0.0
        self._last_light_ts = 0.0
        self._last_ir_ts = 0.0
        self._last_pose_ts = 0.0

    def update_battery(self, now: float) -> Optional[float]:
        if (now - self._last_batt_ts) < BATTERY_UPDATE_DT:
            return self.s.battery_v
        self._last_batt_ts = now

        power_raw = None
        try:
            power_raw = self.car.adc.read_adc(2)
        except Exception:
            power_raw = None

        if power_raw is None:
            self.s.battery_v = None
            return None

        try:
            self.s.battery_v = float(power_raw) * (3 if self.car.adc.pcb_version == 1 else 2)
            self.s.battery_ts = now
        except Exception:
            self.s.battery_v = None
        return self.s.battery_v

    def update_forward_sonar(self, now: float) -> Optional[float]:
        if (now - self._last_sonar_ts) < SONAR_UPDATE_DT:
            return self.s.ahead_cm
        self._last_sonar_ts = now
        try:
            self.s.ahead_cm = self.car.get_forward_distance()
            self.s.ahead_ts = now
        except Exception:
            self.s.ahead_cm = None
        return self.s.ahead_cm

    def update_light(self, now: float):
        if (now - self._last_light_ts) < LIGHT_UPDATE_DT:
            return
        self._last_light_ts = now
        try:
            self.s.light_l_v = float(self.car.adc.read_adc(0))
            self.s.light_r_v = float(self.car.adc.read_adc(1))
            self.s.light_ts = now
        except Exception:
            pass

    def update_ir(self, now: float):
        if (now - self._last_ir_ts) < IR_UPDATE_DT:
            return
        self._last_ir_ts = now
        try:
            self.s.ir_bits = int(self.car.infrared.read_all_infrared())
            self.s.ir_ts = now
        except Exception:
            pass

    def update_pose(self, now: float):
        if (now - self._last_pose_ts) < POSE_UPDATE_DT:
            return
        self._last_pose_ts = now
        try:
            self.s.pose_x_cm = float(self.car.pose.x_cm)
            self.s.pose_y_cm = float(self.car.pose.y_cm)
            self.s.pose_th_deg = float(self.car.pose.th_deg)
            self.s.pose_ts = now
        except Exception:
            pass

    def update_target(self, now: float):
        if not self.vision or not self.vision.ok:
            return
        snap = self.vision.snapshot()
        self.s.target_seen = bool(snap.get("target_seen", False))
        self.s.target_x = float(snap.get("target_x", 0.0))
        self.s.target_age_s = float(snap.get("target_age_s", 1e9))
        self.s.target_method = str(snap.get("target_method", "none"))
        self.s.target_ts = now

    def maybe_scan_triplet(self, now: float, allow: bool) -> None:
        """
        Ultrasonic scan at pan angles (30, 90, 150) -> right, center, left.
        Only do when stopped/thinking (and NOT driving forward).
        """
        if not allow:
            return
        if (now - self.s.scan_ts) < SCAN_TRIPLET_DT:
            return

        try:
            angles, dists = self.car.scan_distances(angles=(30, 90, 150), settle=0.07, samples=2)
            self.s.scan_right_cm = float(dists[0]) if dists[0] is not None else None
            self.s.scan_center_cm = float(dists[1]) if dists[1] is not None else None
            self.s.scan_left_cm = float(dists[2]) if dists[2] is not None else None
            self.s.scan_ts = now

            # update memory map
            try:
                self.car.mem.update_from_scan(self.car.pose, angles, dists)
            except Exception:
                pass

            # park head
            try:
                self.car.park_head_for_drive()
            except Exception:
                pass
        except Exception:
            pass

    def snapshot_for_server(self) -> Dict[str, Any]:
        s = self.s
        snap: Dict[str, Any] = {
            "id": ROBOT_ID if not PET_ANON else "pet",
            "ts": time.time(),
            "battery_v": float(s.battery_v) if isinstance(s.battery_v, (int, float)) else None,
            "ahead_cm": float(s.ahead_cm) if isinstance(s.ahead_cm, (int, float)) else None,
        }

        if PET_PRIVACY_LEVEL in ("minimal",):
            return snap

        snap.update({
            "ir_bits": int(s.ir_bits) if isinstance(s.ir_bits, int) else None,
            "light_l_v": float(s.light_l_v) if isinstance(s.light_l_v, (int, float)) else None,
            "light_r_v": float(s.light_r_v) if isinstance(s.light_r_v, (int, float)) else None,

            "scan_right_cm": float(s.scan_right_cm) if isinstance(s.scan_right_cm, (int, float)) else None,
            "scan_center_cm": float(s.scan_center_cm) if isinstance(s.scan_center_cm, (int, float)) else None,
            "scan_left_cm": float(s.scan_left_cm) if isinstance(s.scan_left_cm, (int, float)) else None,

            "pose_x_cm": float(s.pose_x_cm) if isinstance(s.pose_x_cm, (int, float)) else None,
            "pose_y_cm": float(s.pose_y_cm) if isinstance(s.pose_y_cm, (int, float)) else None,
            "pose_th_deg": float(s.pose_th_deg) if isinstance(s.pose_th_deg, (int, float)) else None,

            # freshness (useful for debugging)
            "age_batt_s": (time.time() - s.battery_ts) if s.battery_ts else None,
            "age_ahead_s": (time.time() - s.ahead_ts) if s.ahead_ts else None,
            "age_scan_s": (time.time() - s.scan_ts) if s.scan_ts else None,
            "age_ir_s": (time.time() - s.ir_ts) if s.ir_ts else None,
            "age_light_s": (time.time() - s.light_ts) if s.light_ts else None,
        })

        if s.target_seen is not None:
            snap.update({
                "target_seen": bool(s.target_seen),
                "target_x": float(s.target_x) if isinstance(s.target_x, (int, float)) else 0.0,
                "target_age_s": float(s.target_age_s) if isinstance(s.target_age_s, (int, float)) else 1e9,
                "target_method": str(s.target_method or "none"),
            })

        return snap


# ============================================================
# Server worker: /pet/step
# ============================================================
class ServerStepWorker:
    """
    Background worker that calls POST {PET_SERVER_URL}/pet/step and stores latest plan.
    """
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

        self.q: Queue = Queue(maxsize=1)
        self.latest: Optional[dict] = None
        self.latest_ts: float = 0.0
        self.busy: bool = False

        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def request(self, snap: Dict[str, Any]) -> None:
        if self.busy:
            return

        # drop any pending request; keep only newest
        try:
            while True:
                self.q.get_nowait()
        except Empty:
            pass
        except Exception:
            pass

        try:
            self.q.put_nowait((snap, time.time()))
        except Exception:
            pass

    def _http_post_json(self, url: str, obj: dict, timeout_s: float) -> dict:
        import urllib.request
        body = json.dumps(obj).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8", errors="ignore"))

    def _run(self):
        step_url = f"{self.base_url}/pet/step"
        while True:
            snap, _enq_ts = self.q.get()
            self.busy = True
            t0 = time.monotonic()
            try:
                log.info(
                    "[AI][STEP] -> bat=%sV ahead=%scm ir=%s light=(%s,%s) scan=(%s,%s,%s) stuck=%s",
                    ("NA" if snap.get("battery_v") is None else f"{float(snap['battery_v']):.2f}"),
                    ("NA" if snap.get("ahead_cm") is None else f"{float(snap['ahead_cm']):.1f}"),
                    ("NA" if snap.get("ir_bits") is None else str(snap.get("ir_bits"))),
                    ("NA" if snap.get("light_l_v") is None else f"{float(snap['light_l_v']):.2f}"),
                    ("NA" if snap.get("light_r_v") is None else f"{float(snap['light_r_v']):.2f}"),
                    ("NA" if snap.get("scan_right_cm") is None else f"{float(snap['scan_right_cm']):.0f}"),
                    ("NA" if snap.get("scan_center_cm") is None else f"{float(snap['scan_center_cm']):.0f}"),
                    ("NA" if snap.get("scan_left_cm") is None else f"{float(snap['scan_left_cm']):.0f}"),
                    str(bool(snap.get("stuck", False))),
                )

                data = self._http_post_json(step_url, snap, timeout_s=PET_STEP_TIMEOUT_S)
                dt = time.monotonic() - t0

                if isinstance(data, dict) and data.get("ok") is True:
                    self.latest = data
                    self.latest_ts = time.time()
                    log.info(
                        "[AI][STEP] <- plan in %.2fs src=%s server_ms=%s mode=%s act=%s dur=%.2f spd=%s head=%s chirp=%s seq=%s",
                        dt,
                        data.get("source", "unknown"),
                        data.get("ms", None),
                        data.get("mode"),
                        data.get("action"),
                        float(data.get("duration_s", 1.0)),
                        data.get("speed_pwm", None),
                        data.get("head_pan", None),
                        data.get("chirp", None),
                        data.get("sequence", None),
                    )
                else:
                    log.warning("[AI][STEP] <- non-ok in %.2fs: %s", dt, data)
            except Exception:
                log_exc("[AI][STEP] request exception")
            finally:
                self.busy = False


# ============================================================
# Server-driven brain
# ============================================================
class ServerPetBrain:
    def __init__(self, sensors: SensorHub):
        self.sensors = sensors

        self.mode: str = MODE_ROAM
        self.energy: float = 0.85

        self._reflex_cooldown_until = 0.0

        self.action: str = "wander"
        self.action_until: float = time.time() + 1.0
        self.speed_pwm: int = DEFAULT_FWD_PWM

        self.runner = StepRunner()
        self.step_worker = ServerStepWorker(PET_SERVER_URL, PI_API_KEY)

        self._last_step_request = 0.0
        self.step_min_interval_s = 0.35

        self.thinking: bool = False
        self._think_until: float = 0.0
        self._think_head_next: float = 0.0

        self.stuck_recent: bool = False

        self._wander_next_drift = 0.0
        self._wander_bias = 0

        self._last_plan_head_ts = 0.0

    def _update_energy(self, dt: float):
        if self.mode == MODE_ROAM:
            self.energy -= 0.006 * dt
        elif self.mode == MODE_PLAY:
            self.energy -= 0.012 * dt
        elif self.mode == MODE_IDLE:
            self.energy += 0.004 * dt
        elif self.mode == MODE_SLEEP:
            self.energy += 0.020 * dt
        self.energy = max(0.0, min(1.0, self.energy))

    def _safe_set_head(self, car: Car, pan: Optional[int] = None, tilt: Optional[int] = None):
        try:
            if pan is None and tilt is None:
                return
            if tilt is None:
                tilt = getattr(car, "TILT_CENTER", None)
            car.set_head_pose(pan=pan, tilt=tilt, settle=0.01)
        except Exception:
            pass

    def _motors_stop(self, car: Car):
        try:
            car.set_motors(0, 0, 0, 0)
        except Exception:
            pass

    def _apply_action_tick(self, car: Car, now: float):
        if self.action == "sequence":
            self.runner.tick(car, now)
            if not self.runner.active() and now >= self.action_until:
                self._motors_stop(car)
            return

        p = int(self.speed_pwm)

        # IMPORTANT: keep sonar head forward while driving
        if self.action in {"forward", "wander", "turn_left", "turn_right", "spin_left", "spin_right"}:
            try:
                car.park_head_for_drive()
            except Exception:
                pass

        try:
            if self.action == "stop":
                self._motors_stop(car)
            elif self.action == "forward":
                car.set_motors(p, p, p, p)
            elif self.action == "reverse":
                try:
                    car.park_head_for_reverse()
                except Exception:
                    pass
                car.set_motors(-p, -p, -p, -p)
            elif self.action == "turn_left":
                car.set_motors(p - DEFAULT_TURN_BIAS, p - DEFAULT_TURN_BIAS, p, p)
            elif self.action == "turn_right":
                car.set_motors(p, p, p - DEFAULT_TURN_BIAS, p - DEFAULT_TURN_BIAS)
            elif self.action == "spin_left":
                car.set_motors(-p, -p, p, p)
            elif self.action == "spin_right":
                car.set_motors(p, p, -p, -p)
            elif self.action == "wander":
                if now >= self._wander_next_drift:
                    self._wander_next_drift = now + random.uniform(0.25, 0.85)
                    self._wander_bias = random.choice([-1, 0, 0, 1])
                if self._wander_bias < 0:
                    car.set_motors(p - DEFAULT_TURN_BIAS, p - DEFAULT_TURN_BIAS, p, p)
                elif self._wander_bias > 0:
                    car.set_motors(p, p, p - DEFAULT_TURN_BIAS, p - DEFAULT_TURN_BIAS)
                else:
                    car.set_motors(p, p, p, p)
            else:
                self._motors_stop(car)
        except Exception:
            pass

    def _reflex_layer(self, car: Car, now: float, battery_v: Optional[float], ahead_cm: Optional[float], ir_bits: Optional[int]) -> bool:
        # Low battery
        if battery_v is not None and battery_v < LOW_BATTERY_THRESHOLD:
            log.warning("[AI][REFLEX] low battery %.2fV -> stop/sleep", battery_v)
            self.mode = MODE_SLEEP
            self.action = "stop"
            self.action_until = now + 2.0
            self.thinking = False
            self._motors_stop(car)
            return True

        # IR "cliff-ish" while commanding forward (disabled by default — see PET_CLIFF_DETECT)
        try:
            if PET_CLIFF_DETECT and ir_bits is not None and int(ir_bits) == CLIFF_IR_MASK_VALUE and car.is_commanding_forward():
                log.warning("[AI][REFLEX] IR=%s while moving -> back+turn", ir_bits)
                self.thinking = False
                self._motors_stop(car)
                try:
                    car.park_head_for_reverse()
                except Exception:
                    pass
                try:
                    car.set_motors(-1200, -1200, -1200, -1200)
                    time.sleep(0.22)
                    car.set_motors(0, 0, 0, 0)
                    time.sleep(0.03)
                    car.park_head_for_drive()
                    if random.random() < 0.5:
                        car.set_motors(1200, 1200, -1200, -1200)
                    else:
                        car.set_motors(-1200, -1200, 1200, 1200)
                    time.sleep(0.45)
                    car.set_motors(0, 0, 0, 0)
                except Exception:
                    pass

                self._reflex_cooldown_until = now + 1.2
                self.mode = MODE_ROAM
                self.action = "stop"
                self.action_until = now + 0.4
                return True
        except Exception:
            pass

        # Ultrasonic reflex
        if ahead_cm is None:
            return False

        d = float(ahead_cm)

        if d < FORWARD_HARD_STOP_CM:
            log.warning("[AI][REFLEX] HARD_STOP ahead=%.1fcm -> avoid", d)
            self.thinking = False
            self._motors_stop(car)
            try:
                car.scan_and_avoid_with_memory()
            except Exception:
                pass
            self._reflex_cooldown_until = now + 1.0
            self.mode = MODE_ROAM
            self.action = "stop"
            self.action_until = now + 0.3
            return True

        if d < ROAM_OBS_TRIGGER_CM:
            log.warning("[AI][REFLEX] avoid zone ahead=%.1fcm -> avoid", d)
            self.thinking = False
            try:
                car.scan_and_avoid_with_memory()
            except Exception:
                pass
            self._reflex_cooldown_until = now + 1.0
            self.mode = MODE_ROAM
            self.action = "stop"
            self.action_until = now + 0.2
            return True

        return False

    def _enter_thinking(self, car: Car, buzzer: Buzzer, now: float):
        self.thinking = True
        self._think_until = now + THINK_MAX_WAIT_S
        self._think_head_next = now

        if THINK_CHIRP_ON_REQUEST > 0:
            chirp(buzzer, n=THINK_CHIRP_ON_REQUEST)

        if STOP_WHILE_THINKING and self.mode != MODE_SLEEP:
            self.action = "stop"
            self.action_until = max(self.action_until, self._think_until)
            self.speed_pwm = DEFAULT_FWD_PWM
            self._motors_stop(car)

    def _thinking_tick(self, car: Car, now: float):
        if not self.thinking:
            return
        if now >= self._think_until:
            log.warning("[AI][THINK] timeout waiting for plan")
            self.thinking = False
            return

        # IMPORTANT:
        # Do NOT rotate sonar head while driving forward, or "ahead_cm" stops being "ahead".
        allow_scan = (self.action == "stop") or (not car.is_commanding_forward())

        if allow_scan and now >= self._think_head_next:
            self._think_head_next = now + THINK_HEAD_SWEEP_DT
            self.sensors.maybe_scan_triplet(now, allow=True)

    def _maybe_request_server_plan(self, car: Car, buzzer: Buzzer, now: float):
        if now < self._reflex_cooldown_until:
            return

        # Smoothness: only request when current action is about to expire
        # (prevents mid-action churn that can cause wall-plowing/jitter)
        if now < (self.action_until - 0.12):
            return

        if (now - self._last_step_request) < self.step_min_interval_s:
            return
        if self.step_worker.busy:
            return

        self._last_step_request = now

        snap = self.sensors.snapshot_for_server()
        snap.update({
            "stuck": bool(self.stuck_recent),
            "mode": self.mode,
            "action": self.action,
            "energy": float(self.energy),
        })

        self._enter_thinking(car, buzzer, now)
        self.step_worker.request(snap)

    def _maybe_apply_server_plan(self, car: Car, buzzer: Buzzer, now: float):
        if not self.step_worker.latest:
            return

        plan = self.step_worker.latest
        self.step_worker.latest = None
        self.step_worker.latest_ts = 0.0

        mode = str(plan.get("mode", self.mode)).strip().upper()
        action = str(plan.get("action", "wander")).strip()
        duration = plan.get("duration_s", 1.0)

        if mode not in ALLOWED_MODES:
            mode = self.mode
        if action not in ALLOWED_ACTIONS:
            action = "wander"

        dur_s = clamp_float(duration, 0.2, 3.0, 1.0)
        spd_pwm = clamp_int(plan.get("speed_pwm", self.speed_pwm), 800, 1400, self.speed_pwm)

        # extra local safety
        ahead_cm = self.sensors.s.ahead_cm
        if isinstance(ahead_cm, (int, float)):
            d = float(ahead_cm)
            if d < FORWARD_HARD_STOP_CM:
                action = "stop"
            elif d < ROAM_OBS_TRIGGER_CM and action in {"forward", "wander"}:
                action = random.choice(["stop", "turn_left", "turn_right", "spin_left", "spin_right"])

        self.mode = mode
        self.action = action
        self.speed_pwm = spd_pwm
        self.action_until = now + dur_s

        head_pan = plan.get("head_pan", None)
        if head_pan is not None:
            pan = clamp_int(head_pan, 30, 150, HEAD_CENTER)
            self._safe_set_head(car, pan=pan, tilt=getattr(car, "TILT_CENTER", None))
            self._last_plan_head_ts = now

        chirps = clamp_int(plan.get("chirp", 0), 0, 2, 0)
        if chirps > 0:
            chirp(buzzer, n=chirps)

        if self.action == "sequence":
            seq_name = str(plan.get("sequence", "")).strip()
            steps = get_sequence_by_name(seq_name)
            self.runner.load(steps, now)
            approx_seq_time = sum(st.dur for st in steps) + 0.1
            self.action_until = max(self.action_until, now + approx_seq_time)

        self.thinking = False
        log.info(
            "[AI][APPLY] mode=%s action=%s dur=%.2f speed=%d head=%s chirp=%d",
            self.mode, self.action, dur_s, self.speed_pwm,
            (str(head_pan) if head_pan is not None else "no"),
            chirps,
        )

    def tick(self, car: Car, led: Led, buzzer: Buzzer, now: float, dt: float):
        # Update ALL sensors
        battery_v = self.sensors.update_battery(now)
        ahead_cm = self.sensors.update_forward_sonar(now)
        self.sensors.update_ir(now)
        self.sensors.update_light(now)
        self.sensors.update_pose(now)
        self.sensors.update_target(now)

        # Reflex safety
        if self._reflex_layer(car, now, battery_v, ahead_cm, self.sensors.s.ir_bits):
            set_mode_led(led, self.mode)
            self._update_energy(dt)
            return

        # Request plan (near end of action)
        self._maybe_request_server_plan(car, buzzer, now)

        # While waiting: gather triplet scan ONLY if not driving forward
        self._thinking_tick(car, now)

        # Apply plan if arrived
        self._maybe_apply_server_plan(car, buzzer, now)

        # If action expired
        if now >= self.action_until:
            if self.thinking and STOP_WHILE_THINKING:
                self.action = "stop"
                self.action_until = max(self.action_until, self._think_until)
            else:
                if self.mode == MODE_SLEEP:
                    self.action = "stop"
                    self.action_until = now + 1.0
                elif self.mode == MODE_IDLE:
                    self.action = "stop"
                    self.action_until = now + 0.8
                else:
                    self.action = "wander"
                    self.action_until = now + 1.0
                    self.speed_pwm = DEFAULT_FWD_PWM

        # If stopped and not thinking, opportunistically scan triplet (keeps scan_* fresh)
        allow_scan = (self.action == "stop") and (not self.thinking)
        self.sensors.maybe_scan_triplet(now, allow=allow_scan)

        set_mode_led(led, self.mode)
        self._apply_action_tick(car, now)
        self._update_energy(dt)


# ============================================================
# MAIN
# ============================================================
def main():
    log.info("[BOOT] autonomous3 -> %s", PET_SERVER_URL or "(local brain)")
    log.info("[BOOT] ROBOT_ID=%s anon=%s privacy=%s camera=%s",
             ROBOT_ID, PET_ANON, PET_PRIVACY_LEVEL, PET_USE_CAMERA)
    log.info("[BOOT] PET_STEP_TIMEOUT_S=%.1f STOP_WHILE_THINKING=%s",
             PET_STEP_TIMEOUT_S, str(STOP_WHILE_THINKING))

    car = Car()
    led = Led()
    buzzer = Buzzer()

    vision = None
    if PET_USE_CAMERA:
        try:
            vision = VisionTracker()
            if vision.ok:
                vision.start()
                log.info("[BOOT] vision tracker OK (fps=%.1f)", PET_VISION_FPS)
            else:
                vision = None
                log.info("[BOOT] vision tracker disabled (init failed)")
        except Exception:
            vision = None
            log.info("[BOOT] vision tracker disabled (exception)")

    sensors = SensorHub(car, vision)

    if PET_SERVER_URL:
        brain = ServerPetBrain(sensors)
        log.info("[BOOT] brain=SERVER  url=%s", PET_SERVER_URL)
    else:
        from pet_brain_local import LocalPetBrain
        brain = LocalPetBrain(sensors)
        log.info("[BOOT] brain=LOCAL  (set PET_SERVER_URL to use AI server)")

    # short boot chirp
    chirp(buzzer, n=1)
    last_ts = time.time()

    try:
        set_mode_led(led, MODE_ROAM)

        while True:
            now = time.time()
            dt = max(0.0, now - last_ts)
            last_ts = now

            brain.tick(car, led, buzzer, now, dt)

            # Stuck detection + escape (uses forward sonar cache)
            try:
                moving_forward = car.is_commanding_forward()
                ahead = sensors.s.ahead_cm
                stuck = car.detect_stuck(
                    current_distance=ahead,
                    moving_forward=moving_forward,
                    min_delta=STUCK_MIN_DELTA_CM,
                    timeout=STUCK_TIMEOUT_S,
                )
                brain.stuck_recent = bool(stuck)

                if stuck:
                    log.warning("[AI][STUCK] detected -> escape")
                    try:
                        car.escape_stuck_with_memory()
                    except Exception:
                        # minimal fallback
                        try:
                            car.set_motors(0, 0, 0, 0)
                            time.sleep(0.05)
                            try:
                                car.park_head_for_reverse()
                            except Exception:
                                pass
                            car.set_motors(-1300, -1300, -1300, -1300)
                            time.sleep(0.35)
                            car.set_motors(0, 0, 0, 0)
                            time.sleep(0.05)
                            try:
                                car.park_head_for_drive()
                            except Exception:
                                pass
                            car.set_motors(1200, 1200, -1200, -1200)
                            time.sleep(0.65)
                            car.set_motors(0, 0, 0, 0)
                        except Exception:
                            pass
            except Exception:
                pass

            time.sleep(LOOP_DT)

    except KeyboardInterrupt:
        log.info("[SHUTDOWN] Ctrl+C received")

    finally:
        try:
            if vision:
                vision.stop()
        except Exception:
            pass
        try:
            car.set_motors(0, 0, 0, 0)
        except Exception:
            pass
        try:
            buzzer.set_state(False)
        except Exception:
            pass
        try:
            led.colorBlink(0)
        except Exception:
            pass
        try:
            car.close()
        except Exception:
            pass

        log.info("[SHUTDOWN] cleaned up")


if __name__ == "__main__":
    main()
