#!/usr/bin/env python3
"""
ollama_brain.py
---------------
OllamaBrain — a LocalPetBrain subclass that augments rule-based behaviour
with on-device LLM reasoning via Ollama.

Brain hierarchy
    OllamaBrain  (this file)
        └─ LocalPetBrain  (pet_brain_local.py)  — safety reflexes always run

Two background workers:

    _LLMWorker   (llama3.2:1b)
        Receives a sensor/emotion snapshot every ~2 s.
        Returns a structured action decision overriding _pick_action.
        Falls back to parent rule-based pick silently on any error or timeout.

    _SceneWorker (moondream:1.8b, optional)
        Runs only when PET_USE_CAMERA=1 and a camera frame is available.
        Produces a plain-text scene description stored as self.scene_caption.
        Used as extra context in the LLM prompt — never blocks the main loop.

Env vars
    OLLAMA_URL          e.g. http://localhost:11434  (no trailing slash)
    OLLAMA_LLM_MODEL    default: llama3.2:1b
    OLLAMA_VIS_MODEL    default: moondream:1.8b
    OLLAMA_TIMEOUT_S    default: 8.0
    OLLAMA_LLM_DT_S     how often to request a new LLM decision, default: 2.5
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
import urllib.request
from queue import Empty, Queue
from typing import Optional

from pet_brain_local import (
    BORED, CURIOUS, HAPPY, PLAYFUL, SCARED, SLEEP, TIRED,
    LocalPetBrain,
    _DEFAULT_SPEED, _TIRED_SPEED, _PLAY_SPEED, _TURN_BIAS,
    _HEAD_CENTER, _HEAD_LEFT, _HEAD_RIGHT,
    _ALL_SEQUENCES,
)

log = logging.getLogger("autonomous3")

# ── Constants ─────────────────────────────────────────────────────────────────

_OLLAMA_LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "llama3.2:1b").strip()
_OLLAMA_VIS_MODEL = os.environ.get("OLLAMA_VIS_MODEL", "moondream:1.8b").strip()
_OLLAMA_TIMEOUT   = float(os.environ.get("OLLAMA_TIMEOUT_S", "8").strip() or "8")
_OLLAMA_LLM_DT    = float(os.environ.get("OLLAMA_LLM_DT_S", "2.5").strip() or "2.5")

_ALLOWED_ACTIONS = {
    "forward", "reverse", "turn_left", "turn_right",
    "spin_left", "spin_right", "wander", "stop",
    "sequence_zoomies", "sequence_wiggle", "sequence_spin",
}

# The LLM prompt template — intentionally terse so 1b models keep up
_SYSTEM_PROMPT = """\
You are the decision-making module of a small autonomous pet robot (Freenove 4WD Smart Car).
Your role is to return a single JSON object — nothing else.

Schema (all keys required):
{
  "action":   one of: forward | reverse | turn_left | turn_right | spin_left | spin_right | wander | stop | sequence_zoomies | sequence_wiggle | sequence_spin,
  "duration": float seconds (0.3 – 3.0),
  "speed":    integer PWM 800–1400 (normal 1100),
  "reason":   one short sentence
}

Constraints:
- NEVER use forward/wander if ahead_cm < 50.
- Use stop or spin if ahead_cm < 30.
- Prefer wander most of the time.
- Respect the robot's current emotion — e.g. SCARED → reverse/spin, BORED → stop/slow.
- sequence_* actions are play moves — only pick them when emotion is HAPPY or PLAYFUL.
"""

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


# ── LLM decision worker ───────────────────────────────────────────────────────

class _LLMWorker:
    """
    Runs in a background daemon thread.
    Caller pushes a snapshot dict via .request(); worker stores result in .latest.
    """

    def __init__(self, ollama_url: str):
        self._url = f"{ollama_url}/api/generate"
        self._q: Queue = Queue(maxsize=1)
        self.latest: Optional[dict] = None
        self.latest_ts: float = 0.0
        self.busy: bool = False
        self.consecutive_failures: int = 0

        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        log.info("[OLLAMA] LLM worker started (%s)", _OLLAMA_LLM_MODEL)

    def request(self, snapshot: dict) -> None:
        if self.busy:
            return
        # keep only the most recent request
        try:
            while True:
                self._q.get_nowait()
        except Empty:
            pass
        try:
            self._q.put_nowait(snapshot)
        except Exception:
            pass

    def _build_prompt(self, s: dict) -> str:
        lines = [
            f"emotion={s.get('emotion', '?')}",
            f"energy={s.get('energy', 0):.2f}",
            f"battery_pct={s.get('battery_pct', 1.0):.0%}",
            f"ahead_cm={s.get('ahead_cm', 'unknown')}",
            f"stuck={s.get('stuck', False)}",
            f"action_prev={s.get('action', 'wander')}",
        ]
        rc = s.get("scan_right_cm")
        cc = s.get("scan_center_cm")
        lc = s.get("scan_left_cm")
        if isinstance(rc, (int, float)) and isinstance(cc, (int, float)) and isinstance(lc, (int, float)):
            lines.append(f"scan_right={rc:.0f} center={cc:.0f} left={lc:.0f}")
        if s.get("scene_caption"):
            lines.append(f"scene=\"{s['scene_caption']}\"")
        return "Robot state:\n" + "\n".join(lines) + "\n\nDecide the next action."

    def _run(self):
        while True:
            snapshot = self._q.get()
            self.busy = True
            t0 = time.monotonic()
            try:
                prompt = self._build_prompt(snapshot)
                payload = {
                    "model": _OLLAMA_LLM_MODEL,
                    "system": _SYSTEM_PROMPT,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.4,
                        "num_predict": 120,
                    },
                }
                data = _post_json(self._url, payload, timeout=_OLLAMA_TIMEOUT)
                dt = time.monotonic() - t0

                raw = data.get("response", "").strip()
                # Extract first {...} block robustly
                start = raw.find("{")
                end   = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    decision = json.loads(raw[start:end])
                    action = decision.get("action", "wander").strip()
                    if action not in _ALLOWED_ACTIONS:
                        action = "wander"
                    self.latest = {
                        "action":   action,
                        "duration": float(decision.get("duration", 1.0)),
                        "speed":    int(decision.get("speed", _DEFAULT_SPEED)),
                        "reason":   str(decision.get("reason", ""))[:120],
                    }
                    self.latest_ts = time.time()
                    self.consecutive_failures = 0
                    log.info(
                        "[OLLAMA] decision in %.2fs: action=%s dur=%.1fs spd=%d  \"%s\"",
                        dt,
                        self.latest["action"],
                        self.latest["duration"],
                        self.latest["speed"],
                        self.latest["reason"],
                    )
                else:
                    log.warning("[OLLAMA] no JSON in response (%.2fs): %.80s", dt, raw)
                    self.consecutive_failures += 1

            except Exception as exc:
                log.warning("[OLLAMA] LLM error: %s", exc)
                self.consecutive_failures += 1
            finally:
                self.busy = False


# ── Vision / scene worker ─────────────────────────────────────────────────────

class _SceneWorker:
    """
    Background worker that sends camera frames to moondream:1.8b
    and stores a plain-text description in .caption.

    Runs only when a camera frame source is provided.
    Moondream is unreliable on synthetic/static frames — treat captions as optional.
    """

    def __init__(self, ollama_url: str):
        self._url = f"{ollama_url}/api/generate"
        self._q: Queue = Queue(maxsize=1)
        self.caption: str = ""
        self.caption_ts: float = 0.0
        self._running: bool = True

        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        log.info("[OLLAMA] Scene worker started (%s)", _OLLAMA_VIS_MODEL)

    def request(self, image_b64: str) -> None:
        """Push a base-64 JPEG frame for captioning (drops old if busy)."""
        try:
            while True:
                self._q.get_nowait()
        except Empty:
            pass
        try:
            self._q.put_nowait(image_b64)
        except Exception:
            pass

    def _run(self):
        while self._running:
            try:
                image_b64 = self._q.get(timeout=5.0)
            except Empty:
                continue
            try:
                payload = {
                    "model": _OLLAMA_VIS_MODEL,
                    "prompt": "Describe what you see in one short sentence.",
                    "images": [image_b64],
                    "stream": False,
                    "options": {"num_predict": 60, "temperature": 0.2},
                }
                data = _post_json(self._url, payload, timeout=_OLLAMA_TIMEOUT + 4)
                caption = data.get("response", "").strip()
                if caption:
                    self.caption    = caption[:200]
                    self.caption_ts = time.time()
                    log.debug("[OLLAMA] scene: %s", self.caption)
            except Exception as exc:
                log.debug("[OLLAMA] scene error: %s", exc)


# ── OllamaBrain ───────────────────────────────────────────────────────────────

class OllamaBrain(LocalPetBrain):
    """
    Extends LocalPetBrain with Ollama-based decision making.

    The rule-based emotion state machine, energy dynamics, safety reflexes,
    and all hardware calls are inherited unchanged.

    This subclass:
      • Overrides _pick_action() to apply LLM decisions when fresh.
      • Fires an LLM request every ~OLLAMA_LLM_DT_S seconds.
      • Falls back to parent _pick_action() silently when LLM is busy / cold.
      • Optionally requests scene captions from moondream (if camera available).
    """

    # How many seconds before an LLM decision is considered stale
    _LLM_STALE_S = _OLLAMA_LLM_DT * 3.5

    # Fall back to rule-based after N consecutive LLM failures
    _LLM_FALLBACK_THRESHOLD = 8

    def __init__(self, sensors, ollama_url: str = "http://localhost:11434"):
        super().__init__(sensors)
        self._ollama_url  = ollama_url.rstrip("/")
        self._llm         = _LLMWorker(self._ollama_url)
        self._scene: Optional[_SceneWorker] = None
        self.scene_caption: str = ""

        self._last_llm_request_ts: float  = 0.0
        self._using_llm: bool             = True  # set False when LLM keeps failing

    # ── Camera (optional scene worker) ────────────────────────────────────────

    def _maybe_start_scene_worker(self):
        if self._scene is None:
            self._scene = _SceneWorker(self._ollama_url)

    def push_camera_frame(self, image_b64: str):
        """
        Called externally (e.g. from main loop) when a camera frame is ready.
        Kicks off an async moondream caption request.
        """
        self._maybe_start_scene_worker()
        self._scene.request(image_b64)

    def _refresh_scene_caption(self):
        if self._scene is None:
            return
        c = self._scene.caption
        if c and (time.time() - self._scene.caption_ts) < 30.0:
            self.scene_caption = c

    # ── LLM request throttle ──────────────────────────────────────────────────

    def _maybe_request_llm(self, now: float, ahead_cm):
        if self._llm.busy:
            return
        if (now - self._last_llm_request_ts) < _OLLAMA_LLM_DT:
            return
        self._last_llm_request_ts = now

        self._refresh_scene_caption()

        snap = {
            "emotion":      self.emotion,
            "energy":       round(self.energy, 3),
            "battery_pct":  round(self._battery_pct, 3),
            "ahead_cm":     (round(float(ahead_cm), 1) if isinstance(ahead_cm, (int, float)) else None),
            "stuck":        self.stuck_recent,
            "action":       self.action,
        }

        s = self.sensors.s
        for attr in ("scan_right_cm", "scan_center_cm", "scan_left_cm"):
            v = getattr(s, attr, None)
            if isinstance(v, (int, float)):
                snap[attr] = round(float(v), 1)

        if self.scene_caption:
            snap["scene_caption"] = self.scene_caption

        self._llm.request(snap)

    # ── Override _pick_action ─────────────────────────────────────────────────

    def _pick_action(self, car, buzzer, now: float, ahead_cm):
        """
        Try to apply the latest LLM decision.
        Fall back to parent (rule-based) if:
          - LLM result is missing or stale
          - action is unsafe given current sonar reading
          - LLM has been failing consistently
        """
        # Check if LLM is degraded
        if self._llm.consecutive_failures >= self._LLM_FALLBACK_THRESHOLD:
            if self._using_llm:
                log.warning(
                    "[OLLAMA] %d consecutive failures — falling back to rule-based pick",
                    self._llm.consecutive_failures,
                )
                self._using_llm = False
        elif not self._using_llm and self._llm.consecutive_failures == 0:
            log.info("[OLLAMA] LLM recovered — resuming LLM decisions")
            self._using_llm = True

        decision = self._llm.latest
        decision_age = (time.time() - self._llm.latest_ts) if self._llm.latest_ts else 1e9

        use_llm = (
            self._using_llm
            and decision is not None
            and decision_age < self._LLM_STALE_S
        )

        if not use_llm:
            # Rule-based fallback — same as plain LocalPetBrain
            super()._pick_action(car, buzzer, now, ahead_cm)
            return

        # Consume the decision
        self._llm.latest = None

        action   = decision["action"]
        dur      = max(0.3, min(3.0, float(decision["duration"])))
        speed    = max(800, min(1400, int(decision["speed"])))

        # Obstacle safety gate — LLM may not know the latest sonar reading.
        # Thresholds match FORWARD_HARD_STOP_CM (30) and ROAM_OBS_TRIGGER_CM (50).
        if isinstance(ahead_cm, (int, float)):
            d = float(ahead_cm)
            if d < 30.0:
                action = random.choice(["spin_left", "spin_right", "reverse"])
                dur    = 0.4
            elif d < 50.0 and action in {"forward", "wander"}:
                action = random.choice(["turn_left", "turn_right", "stop"])
                dur    = 0.5

        # SLEEP / low-energy veto — LLM shouldn't be sending action plans during sleep
        if self.emotion == SLEEP:
            action = "stop"
            dur    = 1.5

        # Map sequence pseudo-actions → real sequence mechanics
        if action.startswith("sequence_"):
            seq_name = action[len("sequence_"):]
            seq_map = {"zoomies": 0, "wiggle": 1, "spin": 2}
            idx = seq_map.get(seq_name, random.randrange(len(_ALL_SEQUENCES)))
            seq = _ALL_SEQUENCES[idx]()
            self._runner.load(seq, now)
            self.action      = "sequence"
            self.speed_pwm   = _PLAY_SPEED
            self.action_until = now + sum(s.dur for s in seq) + 0.2
            self.mode        = "PLAY"
            return

        self.action      = action
        self.speed_pwm   = speed
        self.action_until = now + dur

        mode_map = {
            "stop":       "IDLE",
            "forward":    "ROAM",
            "reverse":    "ROAM",
            "wander":     "ROAM",
            "turn_left":  "ROAM",
            "turn_right": "ROAM",
            "spin_left":  "ROAM",
            "spin_right": "ROAM",
        }
        self.mode = mode_map.get(action, "ROAM")

    # ── tick override — add LLM request cadence ───────────────────────────────

    def tick(self, car, led, buzzer, now: float, dt: float):
        """
        Extends LocalPetBrain.tick() by inserting LLM request before action pick.
        All safety reflexes from the parent are still executed first.
        """
        # Kick off a new LLM request on the background thread (non-blocking)
        ahead_cm = self.sensors.s.ahead_cm
        self._maybe_request_llm(now, ahead_cm)

        # Parent handles: sensor reads, safety reflexes, energy, emotion, action, LED, buzzer, motors
        super().tick(car, led, buzzer, now, dt)
