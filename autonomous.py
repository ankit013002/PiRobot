#!/usr/bin/env python3
from __future__ import annotations

import time
import os
import cv2
import numpy as np
import random
import subprocess
import tempfile
import shutil
import urllib.parse
import threading
from queue import Queue, Empty, Full
import math

from dotenv import load_dotenv

load_dotenv()

from voice_command import VoiceCommandListener
from car import Car
from led import Led
from buzzer import Buzzer
from camera import Camera
from vision_perception import VisionPerception
from target_tracker import TargetTracker, TargetState
from pet_server_bridge import PetServerBridge


# -------------------------
# Wake word -> Voice assistant (server STT/LLM/TTS)
# -------------------------
WAKE_WORDS = {"SPARKY"}  # what VoiceCommandListener should emit OR appear in raw text
WAKE_DEBOUNCE_S = 2.0

ASSIST_URL = os.environ.get("PET_ASSIST_URL", "").strip()  # e.g. http://192.168.1.50:3000/assist
ASSIST_API_KEY = os.environ.get("PI_API_KEY", "").strip()  # must match server.js PI_API_KEY (if used)

ASSIST_REC_SECONDS = float(os.environ.get("PET_ASSIST_REC_SECONDS", "4.0"))
ASSIST_SR = int(os.environ.get("PET_ASSIST_SR", "16000"))
ASSIST_MAX_TURNS = int(os.environ.get("PET_ASSIST_MAX_TURNS", "1"))

# Optional: specify ALSA devices (leave blank to use defaults)
ASSIST_AREC_DEVICE = os.environ.get("PET_ASSIST_AREC_DEVICE", "").strip()   # e.g. "plughw:1,0"
ASSIST_APLAY_DEVICE = os.environ.get("PET_ASSIST_APLAY_DEVICE", "").strip() # e.g. "plughw:0,0"


# -------------------------
# Display settings
# -------------------------
SHOW_VIEW = os.environ.get("DISPLAY") is not None  # auto-disable if headless

LOW_BATTERY_THRESHOLD = 5.0
STATUS_PRINT_INTERVAL = 1.0

# -------------------------
# Vision tuning
# -------------------------
VISION_RISK_TURN = 0.80
VISION_RISK_SLOW = 0.65
VISUAL_STUCK_MOTION_THRESH = 0.35
VISUAL_STUCK_TIMEOUT = 0.9  # seconds

# -------------------------
# Pet / companion modes
# -------------------------
MODE_ROAM = "ROAM"
MODE_FOLLOW = "FOLLOW"
MODE_STOP = "STOP"

FOLLOW_TARGET_CM = 55
FOLLOW_TOO_CLOSE_CM = 30

# How long we allow "no target seen" before we enter SEARCH behavior
FOLLOW_LOST_TIMEOUT = 2.0  # seconds

# Main loop sleep
FOLLOW_SEARCH_STEP_DT = 0.05

PET_PAUSE_MIN = 0.4
PET_PAUSE_MAX = 1.2
PET_CURIOUS_SCAN_EVERY = 6.0  # seconds (roam)

# -------------------------
# Camera-based following + head control
# -------------------------
FOLLOW_STEER_GAIN = 0.65

FOLLOW_ROTATE_WHEN_OFFCENTER = 0.55
FOLLOW_ROTATE_PWM = 850

# Head tracking (pan) using camera target offset
FOLLOW_PAN_GAIN_DEG = 55.0  # offset * gain -> degrees
FOLLOW_PAN_MIN = 30
FOLLOW_PAN_MAX = 150

# Slow the head down so camera can keep up
FOLLOW_PAN_DEADBAND = 5  # don't spam tiny servo moves
FOLLOW_PAN_SMOOTH = 0.20  # 0..1 (lower = smoother/slower)
FOLLOW_PAN_UPDATE_HZ = 8.0  # max servo command rate
FOLLOW_PAN_MAX_DEG_PER_SEC = 120.0  # speed cap
FOLLOW_PAN_MAX_STEP_DEG = 8.0  # cap step size even if dt is large
FOLLOW_PAN_SETTLE = 0.02  # small settle time to reduce blur

# Forward obstacle safety while head is NOT forward:
FORWARD_PROBE_PERIOD = 0.35
FORWARD_PROBE_SETTLE = 0.02
FORWARD_HARD_STOP_CM = 18

# -------------------------
# Server "brain" integration
# -------------------------
BRAIN_STEP_HZ = float(os.environ.get("PET_BRAIN_HZ", "4.0"))  # how often to POST /pet/step
BRAIN_STEP_INTERVAL = 1.0 / max(0.5, BRAIN_STEP_HZ)
BRAIN_CAN_OVERRIDE_VOICE = bool(int(os.environ.get("BRAIN_CAN_OVERRIDE_VOICE", "0")))
PET_SPEAK = bool(int(os.environ.get("PET_SPEAK", "0")))  # local espeak (optional; default off)


def chirp(buzzer, n=1, on=0.06, off=0.05):
    for _ in range(n):
        buzzer.set_state(True)
        time.sleep(on)
        buzzer.set_state(False)
        time.sleep(off)


def set_mode_visual(led, mode: str):
    if mode == MODE_FOLLOW:
        led.ledIndex(0xFF, 0, 0, 255)  # blue
    elif mode == MODE_ROAM:
        led.ledIndex(0xFF, 0, 255, 0)  # green
    elif mode == MODE_STOP:
        led.ledIndex(0xFF, 255, 255, 0)  # yellow
    else:
        led.ledIndex(0xFF, 0, 255, 0)


def target_seen(vs, ts: TargetState) -> bool:
    if ts is not None and ts.seen:
        return True
    return (getattr(vs, "person_count", 0) > 0) or (getattr(vs, "face_count", 0) > 0)


def _move_toward(cur: float, target: float, max_step: float) -> float:
    if target > cur:
        return min(target, cur + max_step)
    else:
        return max(target, cur - max_step)


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _say_local(text: str):
    """
    Optional "cute pet voice" on the Pi using espeak (best-effort).
    Enable with: export PET_SPEAK=1
    """
    if not text:
        return
    if not PET_SPEAK:
        return
    try:
        subprocess.run(
            ["espeak", "-s", "165", "-a", "130", text],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _record_wav(path: str, seconds: float, sr: int) -> bool:
    """
    Record mono WAV on the Pi.
    Prefers ALSA arecord. Falls back to sounddevice if installed.
    """
    seconds = max(0.5, float(seconds))
    seconds_i = max(1, int(math.ceil(seconds)))

    # Preferred: arecord (ALSA)
    if _which("arecord"):
        args = ["arecord", "-q", "-f", "S16_LE", "-r", str(sr), "-c", "1", "-d", str(seconds_i), "-t", "wav"]
        if ASSIST_AREC_DEVICE:
            args += ["-D", ASSIST_AREC_DEVICE]
        args.append(path)

        try:
            subprocess.run(args, check=True)
            return True
        except Exception as e:
            print(f"[ASSIST][REC] arecord failed: {e}")

    # Fallback: sounddevice (if available)
    try:
        import sounddevice as _sd
        import soundfile as _sf

        frames = int(seconds * sr)
        print(f"[ASSIST][REC] sounddevice recording {seconds:.1f}s @ {sr}Hz ...")
        audio = _sd.rec(frames, samplerate=sr, channels=1, dtype="float32")
        _sd.wait()
        mono = audio[:, 0].astype("float32", copy=False)
        _sf.write(path, mono, sr, subtype="PCM_16")
        return True
    except Exception as e:
        print(f"[ASSIST][REC] sounddevice fallback failed: {e}")

    return False


def _play_wav(path: str) -> bool:
    """
    Play WAV on the Pi.
    Prefers aplay. Falls back to paplay or ffplay if present.
    """
    # Preferred: aplay
    if _which("aplay"):
        args = ["aplay", "-q"]
        if ASSIST_APLAY_DEVICE:
            args += ["-D", ASSIST_APLAY_DEVICE]
        args.append(path)
        try:
            subprocess.run(args, check=True)
            return True
        except Exception as e:
            print(f"[ASSIST][PLAY] aplay failed: {e}")

    # Fallback: paplay (PulseAudio)
    if _which("paplay"):
        try:
            subprocess.run(["paplay", path], check=True)
            return True
        except Exception as e:
            print(f"[ASSIST][PLAY] paplay failed: {e}")

    # Fallback: ffplay
    if _which("ffplay"):
        try:
            subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path], check=True)
            return True
        except Exception as e:
            print(f"[ASSIST][PLAY] ffplay failed: {e}")

    return False


def _post_assist(wav_path: str) -> tuple[bytes | None, str, str]:
    """
    POST wav to server /assist. Returns (wav_bytes, transcript, reply_text).
    transcript/reply_text come from response headers when available.
    """
    if not ASSIST_URL:
        print("[ASSIST] PET_ASSIST_URL not set. Example: http://<PC_IP>:3000/assist")
        return None, "", ""

    headers = {}
    if ASSIST_API_KEY:
        headers["X-API-Key"] = ASSIST_API_KEY

    try:
        import requests

        with open(wav_path, "rb") as f:
            files = {"audio": ("audio.wav", f, "audio/wav")}
            r = requests.post(ASSIST_URL, files=files, headers=headers, timeout=30)

        if r.status_code != 200:
            print(f"[ASSIST] Server returned {r.status_code}: {r.text[:500]}")
            return None, "", ""

        transcript = urllib.parse.unquote(r.headers.get("X-Transcript", "") or "")
        reply_text = urllib.parse.unquote(r.headers.get("X-Reply-Text", "") or "")
        return r.content, transcript, reply_text

    except Exception as e:
        print(f"[ASSIST] requests failed: {e}")

    return None, "", ""


def run_voice_assistant_session(car: Car, led: Led, buzzer: Buzzer) -> None:
    """
    Blocking session: record -> /assist -> play, repeated until exit phrase or max turns.
    Robot remains stopped the whole time.
    """
    exit_phrases = {
        "resume",
        "continue",
        "stop listening",
        "nevermind",
        "never mind",
        "bye",
        "goodbye",
        "thanks sparky",
        "thank you sparky",
    }

    # Make it obvious we're in "listening" mode
    try:
        car.set_motors(0, 0, 0, 0)
    except Exception:
        pass

    try:
        led.ledIndex(0xFF, 180, 0, 255)  # purple-ish
    except Exception:
        pass

    chirp(buzzer, n=1, on=0.08, off=0.05)

    with tempfile.TemporaryDirectory(prefix="sparky_") as td:
        in_wav = os.path.join(td, "in.wav")
        out_wav = os.path.join(td, "out.wav")

        for turn in range(max(1, ASSIST_MAX_TURNS)):
            print(f"[ASSIST] Turn {turn+1}/{ASSIST_MAX_TURNS}: recording...")
            ok = _record_wav(in_wav, ASSIST_REC_SECONDS, ASSIST_SR)
            if not ok:
                print("[ASSIST] Recording failed. Exiting assistant mode.")
                chirp(buzzer, n=2)
                break

            wav_bytes, transcript, reply_text = _post_assist(in_wav)
            if transcript:
                print(f"[ASSIST][STT] {transcript}")

            tlow = (transcript or "").strip().lower()
            if any(p in tlow for p in exit_phrases):
                print("[ASSIST] Exit phrase detected. Leaving assistant mode.")
                chirp(buzzer, n=2)
                break

            if not wav_bytes:
                print("[ASSIST] No audio returned from server.")
                chirp(buzzer, n=2)
                break

            with open(out_wav, "wb") as f:
                f.write(wav_bytes)

            print(f"[ASSIST][TTS] {reply_text}" if reply_text else "[ASSIST] Playing response...")
            if not _play_wav(out_wav):
                print("[ASSIST] Playback failed.")
                chirp(buzzer, n=2)
                break


def _apply_brain_directives(led: Led, buzzer: Buzzer, out: dict, talker: ServerTalker | None = None):
    if not out or not out.get("ok"):
        return

    led_rgb = out.get("led", None)
    if isinstance(led_rgb, (list, tuple)) and len(led_rgb) == 3:
        r, g, b = led_rgb
        try:
            led.ledIndex(0xFF, int(r), int(g), int(b))
        except Exception:
            pass

    if out.get("buzzer", False):
        chirp(buzzer, n=1, on=0.08, off=0.04)

    say = out.get("say", None)
    if isinstance(say, str) and say.strip():
        say = say.strip()
        print(f"[BRAIN] say: {say}")
        if talker:
            talker.say(say, key=f"brain:{say.lower()}", cooldown_s=8.0)


def _clean_tts(text: str) -> str:
    """
    Best-effort fix for common mojibake sequences when something got decoded as cp1252.
    """
    if not text:
        return ""
    fixes = {
        "â€”": "—",
        "â€“": "–",
        "â€¦": "…",
        "â€œ": '"',
        "â€": '"',
        "â€\x9d": '"',
        "â€™": "'",
        "â€\x99": "'",
    }
    for a, b in fixes.items():
        text = text.replace(a, b)
    return " ".join(text.split())


class ServerTalker:
    """
    Async "think out loud" using the server's Kokoro TTS.
    Also pauses Vosk while speaking so Sparky doesn't trigger itself.
    """

    def __init__(self, bridge: PetServerBridge, enabled: bool = True):
        self.bridge = bridge
        self.enabled = bool(enabled and bridge and bridge.enabled)
        self.q: Queue[str] = Queue(maxsize=8)
        self.last_said: dict[str, float] = {}
        self._mute_until = 0.0
        self._running = True

        self.pause_voice_cb = None
        self.resume_voice_cb = None
        self._hold_voice_control = False

        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def set_voice_callbacks(self, pause_cb, resume_cb):
        self.pause_voice_cb = pause_cb
        self.resume_voice_cb = resume_cb

    def hold_voice_control(self, hold: bool):
        # when True, talker won't stop/start Vosk (used during assistant mode)
        self._hold_voice_control = bool(hold)

    def mute(self, seconds: float = 999999):
        self._mute_until = time.time() + float(seconds)

    def unmute(self):
        self._mute_until = 0.0

    def say(self, text: str, *, key: str | None = None, cooldown_s: float = 6.0, priority: bool = False):
        if not self.enabled:
            return
        text = _clean_tts(text)
        if not text:
            return

        now = time.time()
        k = key or text.lower()
        last = self.last_said.get(k, 0.0)
        if (not priority) and (now - last) < float(cooldown_s):
            return
        self.last_said[k] = now

        try:
            self.q.put_nowait(text)
        except Full:
            # drop oldest and try again
            try:
                _ = self.q.get_nowait()
            except Empty:
                pass
            try:
                self.q.put_nowait(text)
            except Full:
                pass

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                text = self.q.get(timeout=0.2)
            except Empty:
                continue

            if time.time() < self._mute_until:
                continue

            # Pause Vosk while speaking to avoid self-hearing wakeword
            if (not self._hold_voice_control) and self.pause_voice_cb:
                try:
                    self.pause_voice_cb()
                except Exception:
                    pass

            wav = None
            try:
                wav = self.bridge.tts(text, timeout=float(os.environ.get("PET_TTS_TIMEOUT", "12")))
            except Exception:
                wav = None

            if wav:
                self.bridge.play_wav_bytes(wav)

            if (not self._hold_voice_control) and self.resume_voice_cb:
                try:
                    self.resume_voice_cb()
                except Exception:
                    pass


class FollowSearchController:
    """
    When target is lost:
      - stop and scan with head (camera) first
      - then do gentle rotate bursts while continuing head scan
      - exit immediately once target is seen again
    """

    def __init__(self):
        self.searching = False
        self.started_at = 0.0

        self.step_i = 0
        self.step_end = 0.0
        self.pattern = []

        self.head_angles = []
        self.head_i = 0
        self.head_next = 0.0

        # Slower head scan so the camera can actually see things
        self.head_period = 0.35
        self.head_settle = 0.02

        self.head_only_until = 0.0

    def start(self, now: float, car: Car = None, bias_dir: str = None):
        self.searching = True
        self.started_at = now

        if bias_dir == "L":
            self.head_angles = [90, 110, 130, 150, 120, 90, 70, 50, 30, 60, 90]
            self.pattern = [
                ((0, 0, 0, 0), 0.25),
                ((-900, -900, 900, 900), 0.40),
                ((0, 0, 0, 0), 0.18),
                ((900, 900, -900, -900), 0.24),
                ((0, 0, 0, 0), 0.18),
            ]
        elif bias_dir == "R":
            self.head_angles = [90, 70, 50, 30, 60, 90, 110, 130, 150, 120, 90]
            self.pattern = [
                ((0, 0, 0, 0), 0.25),
                ((900, 900, -900, -900), 0.40),
                ((0, 0, 0, 0), 0.18),
                ((-900, -900, 900, 900), 0.24),
                ((0, 0, 0, 0), 0.18),
            ]
        else:
            self.head_angles = [90, 60, 120, 30, 150, 90]
            self.pattern = [
                ((0, 0, 0, 0), 0.25),
                ((-900, -900, 900, 900), 0.22),
                ((0, 0, 0, 0), 0.18),
                ((900, 900, -900, -900), 0.40),
                ((0, 0, 0, 0), 0.18),
            ]

        self.step_i = 0
        self.step_end = now + self.pattern[0][1]

        self.head_i = 0
        self.head_next = now
        self.head_only_until = now + 1.2

        if car:
            car.set_head_pose(pan=90, tilt=car.TILT_CENTER, settle=0.03)

    def stop(self, car: Car = None):
        self.searching = False
        if car:
            car.park_head_for_drive()

    def tick(self, now: float, car: Car = None):
        if not self.searching:
            return (0, 0, 0, 0)

        if car and now >= self.head_next and self.head_angles:
            self.head_i = (self.head_i + 1) % len(self.head_angles)
            car.set_head_pose(
                pan=self.head_angles[self.head_i],
                tilt=car.TILT_CENTER,
                settle=self.head_settle,
            )
            self.head_next = now + self.head_period

        if now < self.head_only_until:
            return (0, 0, 0, 0)

        if now >= self.step_end:
            self.step_i = (self.step_i + 1) % len(self.pattern)
            self.step_end = now + self.pattern[self.step_i][1]

        cmd, _dur = self.pattern[self.step_i]
        return cmd


def follow_drive_cmd(forward_cm):
    if forward_cm is None:
        return (650, 650, 650, 650)

    d = _safe_float(forward_cm, None)
    if d is None:
        return (650, 650, 650, 650)

    if d < FOLLOW_TOO_CLOSE_CM:
        return (-650, -650, -650, -650)

    if d > (FOLLOW_TARGET_CM + 20):
        return (950, 950, 950, 950)
    elif d > (FOLLOW_TARGET_CM + 5):
        return (750, 750, 750, 750)
    elif d < (FOLLOW_TARGET_CM - 10):
        return (0, 0, 0, 0)
    else:
        return (500, 500, 500, 500)


def apply_steering(base_cmd, steer: float):
    fl, bl, fr, br = base_cmd

    # If backing away, keep it straight
    if fl < 0 and bl < 0 and fr < 0 and br < 0:
        return base_cmd

    # If stopped but steering indicates turning, rotate in place
    if fl == 0 and bl == 0 and fr == 0 and br == 0 and abs(steer) > 0.12:
        if steer > 0:
            return (FOLLOW_ROTATE_PWM, FOLLOW_ROTATE_PWM, -FOLLOW_ROTATE_PWM, -FOLLOW_ROTATE_PWM)
        else:
            return (-FOLLOW_ROTATE_PWM, -FOLLOW_ROTATE_PWM, FOLLOW_ROTATE_PWM, FOLLOW_ROTATE_PWM)

    # Large offset -> rotate in place
    if abs(steer) >= FOLLOW_ROTATE_WHEN_OFFCENTER:
        if steer > 0:
            return (FOLLOW_ROTATE_PWM, FOLLOW_ROTATE_PWM, -FOLLOW_ROTATE_PWM, -FOLLOW_ROTATE_PWM)
        else:
            return (-FOLLOW_ROTATE_PWM, -FOLLOW_ROTATE_PWM, FOLLOW_ROTATE_PWM, FOLLOW_ROTATE_PWM)

    # Forward drive -> differential steering
    if fl > 0 and bl > 0 and fr > 0 and br > 0:
        k = FOLLOW_STEER_GAIN
        steer = max(-1.0, min(1.0, float(steer)))

        left_mul = 1.0 + k * steer
        right_mul = 1.0 - k * steer

        l = int(round(fl * left_mul))
        r = int(round(fr * right_mul))

        l = max(0, min(1600, l))
        r = max(0, min(1600, r))

        return (l, l, r, r)

    return base_cmd


def roam_step(car: Car, forward_cm, now, next_pet_pause_at, last_curious_scan):
    if now >= next_pet_pause_at:
        return (0, 0, 0, 0), (now + random.uniform(4.0, 9.0)), last_curious_scan

    if (now - last_curious_scan) > PET_CURIOUS_SCAN_EVERY and (forward_cm is None or forward_cm > 60):
        car.scan_distances(angles=(30, 60, 90, 120, 150), settle=0.04, samples=1)
        last_curious_scan = now

    return (1100, 1100, 1100, 1100), next_pet_pause_at, last_curious_scan


def avoid_ultrasonic_locked(car: Car):
    angles, dists = car.scan_distances(angles=(30, 90, 150), settle=0.06, samples=1)
    right_cm, center_cm, left_cm = dists[0], dists[1], dists[2]
    car.run_motor_ultrasonic(
        [
            right_cm if right_cm is not None else 180,
            center_cm if center_cm is not None else 180,
            left_cm if left_cm is not None else 180,
        ]
    )


def escape_stuck_basic(car: Car):
    TURN_RIGHT = (1500, 1500, -1500, -1500)
    TURN_LEFT = (-1500, -1500, 1500, 1500)

    car.set_motors(0, 0, 0, 0)
    time.sleep(0.02)

    car.park_head_for_reverse()
    car.set_motors(-1300, -1300, -1300, -1300)
    time.sleep(0.35)
    car.set_motors(0, 0, 0, 0)
    time.sleep(0.05)

    car.park_head_for_drive()
    _angs, dists = car.scan_distances(angles=(30, 90, 150), settle=0.06, samples=2)
    right_cm, _center_cm, left_cm = dists[0], dists[1], dists[2]

    ls = 180.0 if left_cm is None else float(left_cm)
    rs = 180.0 if right_cm is None else float(right_cm)

    cmd = TURN_LEFT if ls >= rs else TURN_RIGHT
    car.set_motors(*cmd)
    time.sleep(0.65)
    car.set_motors(0, 0, 0, 0)
    time.sleep(0.05)

    car.set_motors(850, 850, 850, 850)
    time.sleep(0.45)
    car.set_motors(0, 0, 0, 0)
    time.sleep(0.05)


def forward_sonar_probe(car: Car, settle: float = FORWARD_PROBE_SETTLE):
    if car.servo is None or car.sonic is None:
        return None

    with car.head_lock:
        prev_pan = getattr(car, "current_pan", 90)
        prev_tilt = getattr(car, "current_tilt", car.TILT_CENTER)

        car.servo.set_servo_pwm(car.TILT_CH, car.TILT_CENTER)
        car.current_tilt = car.TILT_CENTER
        car.servo.set_servo_pwm(car.PAN_CH, 90)
        car.current_pan = 90
        if settle:
            time.sleep(settle)

        d = car.sonic.get_distance()

        car.servo.set_servo_pwm(car.PAN_CH, int(prev_pan))
        car.current_pan = int(prev_pan)
        car.servo.set_servo_pwm(car.TILT_CH, int(prev_tilt))
        car.current_tilt = int(prev_tilt)

    return d


def apply_safety_overrides(car: Car, vs, forward_ahead_cm):
    if forward_ahead_cm is not None and float(forward_ahead_cm) < FORWARD_HARD_STOP_CM:
        car.set_motors(0, 0, 0, 0)
        avoid_ultrasonic_locked(car)
        return True
    return False


def main():
    print("Starting autonomous mode (ultrasonic + vision + camera tracking + optional server brain)...")

    car = Car()
    led = Led()
    buzzer = Buzzer()

    bridge = PetServerBridge()
    last_brain_ts = 0.0
    last_brain_out = None
    last_wake_ts = 0.0

    # If voice issues a command, we treat it as a "manual override" (sticky)
    voice_override = False

    # --- Voice listener (offline) ---
    voice = None
    model_path = os.environ.get(
        "VOICE_MODEL_PATH",
        os.path.expanduser("~/vosk_models/vosk-model-small-en-us-0.15"),
    )

    def _create_voice_listener():
        return VoiceCommandListener(model_path=model_path, sample_rate=16000)

    def _start_voice_fresh():
        nonlocal voice
        try:
            if voice is not None:
                try:
                    voice.stop()
                except Exception:
                    pass
        finally:
            voice = None

        try:
            voice = _create_voice_listener()
            voice.start()
            print(f"[VOICE] Listening... model={model_path}", flush=True)
        except Exception as e:
            voice = None
            print(f"[VOICE] Disabled (could not start): {e}")

    # Start voice initially
    _start_voice_fresh()

    talker = ServerTalker(bridge, enabled=bool(int(os.environ.get("PET_TALK", "1"))))

    def _pause_voice():
        if voice is not None:
            try:
                voice.stop()
            except Exception:
                pass

    def _resume_voice():
        # Prefer restarting same instance to avoid constant re-instantiation while talking;
        # if that fails, fall back to a clean re-create.
        nonlocal voice
        if voice is None:
            _start_voice_fresh()
            return
        try:
            voice.start()
        except Exception:
            _start_voice_fresh()

    talker.set_voice_callbacks(_pause_voice, _resume_voice)

    # Camera + perception
    cam = Camera(stream_size=(320, 240))
    cam.start_stream()

    if SHOW_VIEW:
        cv2.namedWindow("Car Camera", cv2.WINDOW_NORMAL)

    # Existing perception thread (kept) but do NOT swing the head around quickly
    vision = VisionPerception(
        cam,
        flow_size=(160, 120),
        face_every_n=5,
        person_every_n=8,
        debug=False,
        servo=car.servo,
        head_lock=car.head_lock,
        tilt_channel="1",
        tilt_center=car.TILT_CENTER,
        # prevent tilt sweeping (extra head motion hurts tracking)
        tilt_angles=(car.TILT_CENTER,),
        head_sweep_period=4.0,
    )
    vision.start()

    # Camera tracker thread
    tracker = TargetTracker(cam, prefer_tracker=True, detect_every_n=6)
    tracker.start()

    last_status_print = 0.0
    visual_stuck_start = None

    mode = MODE_ROAM
    set_mode_visual(led, mode)
    chirp(buzzer, n=2)

    next_pet_pause_at = time.time() + random.uniform(4.0, 8.0)
    last_curious_scan = 0.0

    last_person_seen = 0.0
    last_seen_dir = None
    follow_search = FollowSearchController()

    last_probe_ts = 0.0
    forward_ahead_cache = None

    smoothed_pan = 90.0
    last_pan_cmd_ts = 0.0

    try:
        led.ledIndex(0xFF, 0, 255, 0)

        while True:
            now = time.time()
            stuck_this_tick = False

            # --- Voice commands ---
            if voice is not None:
                # 1) check raw transcripts for wake word
                heard = voice.poll_text()
                if heard:
                    s = heard.strip().upper()
                    if (s in WAKE_WORDS) or ("SPARKY" in s):
                        print("LISTENING", flush=True)
                        if (now - last_wake_ts) >= WAKE_DEBOUNCE_S:
                            last_wake_ts = now

                            # stop motion/search
                            try:
                                car.set_motors(0, 0, 0, 0)
                            except Exception:
                                pass
                            follow_search.stop(car)

                            prev_mode = mode
                            mode = MODE_STOP
                            set_mode_visual(led, mode)

                            print("[ASSIST] Wake word heard. Entering assistant mode...", flush=True)

                            # Make sure talker won't fight for mic and won't speak during assistant
                            talker.mute(999999)
                            talker.hold_voice_control(True)

                            # IMPORTANT: free the mic
                            try:
                                voice.stop()
                            except Exception:
                                pass
                            time.sleep(0.10)

                            try:
                                run_voice_assistant_session(car, led, buzzer)
                            finally:
                                # Restore talker + voice (critical)
                                talker.hold_voice_control(False)
                                talker.unmute()

                                # safest: restart voice cleanly
                                _start_voice_fresh()

                                mode = prev_mode if prev_mode in (MODE_ROAM, MODE_FOLLOW, MODE_STOP) else MODE_ROAM
                                set_mode_visual(led, mode)
                                chirp(buzzer, n=1)

                            time.sleep(FOLLOW_SEARCH_STEP_DT)
                            continue

                # 2) check commands separately (FOLLOW/ROAM/STOP)
                cmd = voice.poll_cmd()
                if cmd:
                    cmd = cmd.strip().upper()
                    print(f"[DBG] cmd -> {cmd!r}", flush=True)

                    if cmd == "FOLLOW" and mode != MODE_FOLLOW:
                        mode = MODE_FOLLOW
                        voice_override = True
                        set_mode_visual(led, mode)
                        chirp(buzzer, n=1)
                        print("[MODE] FOLLOW (voice override)")
                    elif cmd == "ROAM" and mode != MODE_ROAM:
                        mode = MODE_ROAM
                        voice_override = True
                        set_mode_visual(led, mode)
                        chirp(buzzer, n=2)
                        print("[MODE] ROAM (voice override)")
                    elif cmd == "STOP" and mode != MODE_STOP:
                        mode = MODE_STOP
                        voice_override = True
                        set_mode_visual(led, mode)
                        chirp(buzzer, n=3)
                        print("[MODE] STOP (voice override)")

            # Battery safety
            power_raw = car.adc.read_adc(2)
            power = (
                power_raw * (3 if car.adc.pcb_version == 1 else 2)
                if power_raw is not None
                else None
            )
            if power is not None and power < LOW_BATTERY_THRESHOLD:
                print(f"[WARN] Low battery: {power:.2f} V - stopping motors.")
                car.set_motors(0, 0, 0, 0)
                buzzer.set_state(True)
                led.ledIndex(0xFF, 255, 0, 0)
                time.sleep(2)
                buzzer.set_state(False)
                break

            vs = vision.get_state()
            ts = tracker.get_state()

            if ts is not None and ts.seen:
                if ts.cx_norm < -0.12:
                    last_seen_dir = "L"
                elif ts.cx_norm > 0.12:
                    last_seen_dir = "R"

            # Sonar distances
            if mode == MODE_FOLLOW:
                forward_target = car.get_distance_current_direction(ensure_tilt_center=True, settle=0.0)
            else:
                forward_target = car.get_forward_distance()

            # Periodic forward safety probe even while head is tracking you
            if (now - last_probe_ts) >= FORWARD_PROBE_PERIOD:
                forward_ahead_cache = forward_sonar_probe(car)
                last_probe_ts = now

            # Immediate human safety
            if getattr(vs, "person_count", 0) > 0 and getattr(vs, "person_close", False):
                car.set_motors(0, 0, 0, 0)
                buzzer.set_state(True)
                led.ledIndex(0xFF, 0, 0, 255)
                time.sleep(0.25)
                buzzer.set_state(False)
                set_mode_visual(led, mode)

            # -------------------------
            # Server "brain" step (optional)
            # -------------------------
            if bridge.enabled and (now - last_brain_ts) >= BRAIN_STEP_INTERVAL:
                snap = {
                    "id": "pi",
                    "ts": now,
                    "mode": mode,
                    "battery_v": power,
                    "sonar_target_cm": forward_target,
                    "sonar_ahead_cm": forward_ahead_cache,
                    "stuck": False,  # updated later if we trigger escape
                    "vision": {
                        "person_count": int(getattr(vs, "person_count", 0) or 0),
                        "person_close": bool(getattr(vs, "person_close", False)),
                        "collision_risk": float(getattr(vs, "collision_risk", 0.0) or 0.0),
                        "free_dir": getattr(vs, "free_dir", None),
                        "motion_score": float(getattr(vs, "motion_score", 0.0) or 0.0),
                        "head_tilt_deg": float(getattr(vs, "head_tilt_deg", 0.0) or 0.0),
                        "face_count": int(getattr(vs, "face_count", 0) or 0),
                    },
                    "track": {
                        "seen": bool(ts.seen) if ts else False,
                        "cx_norm": float(ts.cx_norm) if (ts and ts.seen) else 0.0,
                    },
                }
                out = bridge.step(snap, timeout=0.65)
                last_brain_ts = now
                if out:
                    last_brain_out = out
                    _apply_brain_directives(led, buzzer, out, talker)

                    out_mode = out.get("mode", None)
                    if isinstance(out_mode, str) and out_mode in (MODE_ROAM, MODE_FOLLOW, MODE_STOP):
                        if (not voice_override) or BRAIN_CAN_OVERRIDE_VOICE:
                            if out_mode != mode:
                                mode = out_mode
                                if not isinstance(out.get("led", None), (list, tuple)):
                                    set_mode_visual(led, mode)

            motor_cmd = (0, 0, 0, 0)

            if mode == MODE_STOP:
                motor_cmd = (0, 0, 0, 0)
                follow_search.stop(car)

            elif mode == MODE_FOLLOW:
                if target_seen(vs, ts):
                    last_person_seen = now
                    if follow_search.searching:
                        follow_search.stop(car)

                # Head pan tracking (rate-limited + speed-limited)
                if ts is not None and ts.seen and not follow_search.searching:
                    desired_pan = 90.0 + float(ts.cx_norm) * FOLLOW_PAN_GAIN_DEG
                    desired_pan = max(FOLLOW_PAN_MIN, min(FOLLOW_PAN_MAX, desired_pan))

                    smoothed_pan = (1.0 - FOLLOW_PAN_SMOOTH) * smoothed_pan + FOLLOW_PAN_SMOOTH * desired_pan

                    if last_pan_cmd_ts == 0.0:
                        last_pan_cmd_ts = now

                    min_dt = 1.0 / max(1.0, FOLLOW_PAN_UPDATE_HZ)
                    if (now - last_pan_cmd_ts) >= min_dt:
                        cur_pan = float(getattr(car, "current_pan", 90))
                        dt = max(0.0, now - last_pan_cmd_ts)

                        step = FOLLOW_PAN_MAX_DEG_PER_SEC * dt
                        step = max(1.0, min(FOLLOW_PAN_MAX_STEP_DEG, step))

                        target = float(smoothed_pan)
                        new_pan = _move_toward(cur_pan, target, step)

                        if abs(new_pan - cur_pan) >= FOLLOW_PAN_DEADBAND:
                            car.set_head_pose(
                                pan=int(round(new_pan)),
                                tilt=car.TILT_CENTER,
                                settle=FOLLOW_PAN_SETTLE,
                            )
                            last_pan_cmd_ts = now

                # LOST -> search
                if (now - last_person_seen) > FOLLOW_LOST_TIMEOUT:
                    if not follow_search.searching:
                        follow_search.start(now, car, bias_dir=last_seen_dir)
                    motor_cmd = follow_search.tick(now, car)
                else:
                    motor_cmd = follow_drive_cmd(forward_target)

                    steer = 0.0
                    if ts is not None and ts.seen:
                        steer = float(ts.cx_norm)

                    motor_cmd = apply_steering(motor_cmd, steer)

                    if motor_cmd == (-650, -650, -650, -650):
                        car.set_motors(*motor_cmd)
                        time.sleep(0.12)
                        motor_cmd = (0, 0, 0, 0)

            else:  # MODE_ROAM
                follow_search.stop(car)
                motor_cmd, next_pet_pause_at, last_curious_scan = roam_step(
                    car, forward_target, now, next_pet_pause_at, last_curious_scan
                )

                if motor_cmd == (0, 0, 0, 0) and now >= (next_pet_pause_at - 9.0):
                    time.sleep(random.uniform(PET_PAUSE_MIN, PET_PAUSE_MAX))

            # Safety overrides (true forward distance)
            did_override = apply_safety_overrides(car, vs, forward_ahead_cache)

            if not did_override:
                if float(getattr(vs, "collision_risk", 0.0) or 0.0) > VISION_RISK_SLOW:
                    fl, bl, fr, br = motor_cmd
                    if fl > 0 and bl > 0 and fr > 0 and br > 0:
                        motor_cmd = (650, 650, 650, 650)
                car.set_motors(*motor_cmd)

            # Stuck detection + escape
            moving_forward = car.is_commanding_forward()

            if car.detect_stuck(forward_ahead_cache, moving_forward):
                print("[WARN] Ultrasonic-based stuck! Executing escape.")
                buzzer.set_state(True)
                escape_stuck_basic(car)
                buzzer.set_state(False)
                visual_stuck_start = None
                stuck_this_tick = True

            # vision-based "not moving" stuck
            motion_score = float(getattr(vs, "motion_score", 0.0) or 0.0)
            if moving_forward:
                if motion_score < VISUAL_STUCK_MOTION_THRESH:
                    if visual_stuck_start is None:
                        visual_stuck_start = time.time()
                    elif (time.time() - visual_stuck_start) > VISUAL_STUCK_TIMEOUT:
                        print("[WARN] Vision-based stuck! Executing escape.")
                        buzzer.set_state(True)
                        escape_stuck_basic(car)
                        buzzer.set_state(False)
                        visual_stuck_start = None
                        stuck_this_tick = True
                else:
                    visual_stuck_start = None
            else:
                visual_stuck_start = None

            # If we got stuck, tell the server brain quickly (best-effort)
            if stuck_this_tick and bridge.enabled:
                snap2 = {
                    "id": "pi",
                    "ts": time.time(),
                    "mode": mode,
                    "battery_v": power,
                    "sonar_target_cm": forward_target,
                    "sonar_ahead_cm": forward_ahead_cache,
                    "stuck": True,
                    "vision": {
                        "person_count": int(getattr(vs, "person_count", 0) or 0),
                        "person_close": bool(getattr(vs, "person_close", False)),
                        "collision_risk": float(getattr(vs, "collision_risk", 0.0) or 0.0),
                        "free_dir": getattr(vs, "free_dir", None),
                        "motion_score": float(getattr(vs, "motion_score", 0.0) or 0.0),
                        "head_tilt_deg": float(getattr(vs, "head_tilt_deg", 0.0) or 0.0),
                        "face_count": int(getattr(vs, "face_count", 0) or 0),
                    },
                    "track": {
                        "seen": bool(ts.seen) if ts else False,
                        "cx_norm": float(ts.cx_norm) if (ts and ts.seen) else 0.0,
                    },
                }
                out2 = bridge.step(snap2, timeout=0.65)
                if out2:
                    last_brain_out = out2
                    _apply_brain_directives(led, buzzer, out2, talker)

            buzzer.set_state(bool(forward_ahead_cache is not None and float(forward_ahead_cache) < 15))

            if now - last_status_print > STATUS_PRINT_INTERVAL:
                pan = getattr(car, "current_pan", 90)
                pv = f"{power:.2f}" if power is not None else "NA"

                brain_mode = None
                brain_notes = None
                if isinstance(last_brain_out, dict) and last_brain_out.get("ok"):
                    brain_mode = last_brain_out.get("mode", None)
                    brain_notes = last_brain_out.get("notes", None)

                print(
                    f"[INFO] Bat={pv}V  "
                    f"US_target={forward_target}  US_ahead={forward_ahead_cache}  pan={pan}  "
                    f"persons={getattr(vs,'person_count',0)} close={getattr(vs,'person_close',False)} "
                    f"tilt={getattr(vs,'head_tilt_deg',0)}  "
                    f"risk={float(getattr(vs,'collision_risk',0.0) or 0.0):.2f} free={getattr(vs,'free_dir',None)} "
                    f"motion={motion_score:.2f} faces={getattr(vs,'face_count',0)} "
                    f"mode={mode} searching={follow_search.searching} "
                    f"trk_seen={(ts.seen if ts else False)} trk_cx={(ts.cx_norm if ts else 0):.2f} "
                    f"last_dir={last_seen_dir} brain_mode={brain_mode} brain_notes={brain_notes}"
                )
                last_status_print = now

            if SHOW_VIEW:
                jpg = cam.get_frame(timeout=0.001)
                if jpg:
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        if ts is not None and ts.seen and ts.bbox is not None:
                            x, y, w, h = ts.bbox
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(
                                img,
                                f"cx={ts.cx_norm:.2f}",
                                (x, max(15, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1,
                                cv2.LINE_AA,
                            )

                        cv2.putText(
                            img,
                            f"mode={mode} searching={follow_search.searching} "
                            f"risk={float(getattr(vs,'collision_risk',0.0) or 0.0):.2f} "
                            f"ahead={forward_ahead_cache} target={forward_target} pan={getattr(car,'current_pan',90)}",
                            (8, 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.imshow("Car Camera", img)
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            raise KeyboardInterrupt

            time.sleep(FOLLOW_SEARCH_STEP_DT)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received, stopping autonomous mode...")

    finally:
        try:
            if voice is not None:
                voice.stop()
        except Exception:
            pass

        try:
            if SHOW_VIEW:
                cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            tracker.stop()
        except Exception:
            pass

        try:
            vision.stop()
        except Exception:
            pass

        try:
            cam.stop_stream()
            cam.close()
        except Exception:
            pass

        buzzer.set_state(False)
        try:
            led.colorBlink(0)
        except Exception:
            pass

        try:
            car.close()
        except Exception:
            pass

        print("[INFO] Autonomous mode stopped, resources cleaned up.")


if __name__ == "__main__":
    main()
