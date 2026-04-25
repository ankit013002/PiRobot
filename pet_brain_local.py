#!/usr/bin/env python3
"""
pet_brain_local.py
------------------
Rule-based pet brain that runs entirely on the Pi — no network required.
Used as fallback when PET_SERVER_URL is not configured.

Emotional state machine:

  CURIOUS  <->  HAPPY  <->  PLAYFUL
      |            |
    BORED        TIRED  -->  SLEEP  --> HAPPY (on wake)
                  ^
               SCARED (timed, triggered by repeated close obstacles)

Each emotion drives: motion style, LED color, buzzer chirp pattern.

Drop-in replacement for ServerPetBrain — same .tick() signature.
"""

import time
import random
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Deque

log = logging.getLogger("autonomous3")

# ── Motion constants (keep in sync with autonomous3.py) ──────────────────────
_DEFAULT_SPEED    = 1100
_TIRED_SPEED      =  680
_PLAY_SPEED       = 1350
_TURN_BIAS        =  250
_HEAD_CENTER      =   90
_HEAD_LEFT        =  150
_HEAD_RIGHT       =   30

# ── Emotional states ──────────────────────────────────────────────────────────
CURIOUS = "CURIOUS"
HAPPY   = "HAPPY"
TIRED   = "TIRED"
SCARED  = "SCARED"
BORED   = "BORED"
PLAYFUL = "PLAYFUL"
SLEEP   = "SLEEP"

# ── LED colors per emotion (R, G, B) ─────────────────────────────────────────
LED_COLORS: dict = {
    CURIOUS: (  0, 190, 220),   # cyan        — calm interest
    HAPPY:   (255, 200,  40),   # warm yellow — joy
    TIRED:   (  0,  30, 110),   # dim blue    — low energy
    SCARED:  (255,   0,   0),   # red         — fear
    BORED:   (140,  80,   0),   # amber       — dull
    PLAYFUL: (180,   0, 255),   # purple      — excitement
    SLEEP:   (  0,   0,  20),   # very dim blue — sleeping
}

# ── Chirp patterns per emotion: (count, on_s, off_s) ─────────────────────────
CHIRP_PATTERNS: dict = {
    CURIOUS: (1, 0.07, 0.05),
    HAPPY:   (3, 0.05, 0.04),
    TIRED:   (1, 0.18, 0.12),
    SCARED:  (4, 0.04, 0.03),
    BORED:   (0, 0.00, 0.00),
    PLAYFUL: (2, 0.06, 0.04),
    SLEEP:   (0, 0.00, 0.00),
}


# ── Minimal StepRunner + play sequences (standalone, no autonomous3 import) ───

@dataclass
class _Step:
    cmd: Tuple[int, int, int, int]
    dur: float
    head_pan: Optional[int] = None
    head_tilt: Optional[int] = None


class _StepRunner:
    def __init__(self):
        self.steps: Deque[_Step] = deque()
        self._step_end: float = 0.0

    def load(self, steps: list, now: float):
        self.steps = deque(steps)
        self._step_end = now

    def active(self) -> bool:
        return len(self.steps) > 0

    def tick(self, car, now: float) -> bool:
        if not self.steps:
            return False
        if now >= self._step_end:
            st = self.steps.popleft()
            try:
                if st.head_pan is not None or st.head_tilt is not None:
                    car.set_head_pose(pan=st.head_pan, tilt=st.head_tilt, settle=0.01)
            except Exception:
                pass
            try:
                car.set_motors(*st.cmd)
            except Exception:
                pass
            self._step_end = now + max(0.02, st.dur)
        return True


def _seq_zoomies() -> list:
    C, L, R = _HEAD_CENTER, _HEAD_LEFT, _HEAD_RIGHT
    return [
        _Step((   0,    0,    0,    0), 0.15, head_pan=C),
        _Step((1200, 1200, 1200, 1200), 0.35, head_pan=R),
        _Step((   0,    0,    0,    0), 0.12, head_pan=L),
        _Step((1200, 1200, 1200, 1200), 0.30, head_pan=C),
        _Step((1200, 1200,-1200,-1200), 0.45, head_pan=R),
        _Step((   0,    0,    0,    0), 0.10, head_pan=L),
        _Step((1000, 1000, 1000, 1000), 0.28, head_pan=C),
        _Step((   0,    0,    0,    0), 0.15, head_pan=C),
    ]


def _seq_wiggle() -> list:
    C, L, R = _HEAD_CENTER, _HEAD_LEFT, _HEAD_RIGHT
    return [
        _Step((   0,    0,    0,    0), 0.12, head_pan=R),
        _Step(( 900,  900, -900, -900), 0.20, head_pan=L),
        _Step((-900, -900,  900,  900), 0.20, head_pan=R),
        _Step(( 900,  900, -900, -900), 0.20, head_pan=L),
        _Step((   0,    0,    0,    0), 0.10, head_pan=C),
        _Step(( 950,  950,  950,  950), 0.22, head_pan=R),
        _Step((   0,    0,    0,    0), 0.10, head_pan=L),
        _Step(( 950,  950,  950,  950), 0.22, head_pan=C),
        _Step((   0,    0,    0,    0), 0.18, head_pan=C),
    ]


def _seq_spin() -> list:
    C, L, R = _HEAD_CENTER, _HEAD_LEFT, _HEAD_RIGHT
    return [
        _Step((    0,    0,    0,    0), 0.18, head_pan=C),
        _Step(( 1200, 1200,-1200,-1200), 0.70, head_pan=R),
        _Step((    0,    0,    0,    0), 0.12, head_pan=L),
        _Step((-1200,-1200, 1200, 1200), 0.55, head_pan=L),
        _Step((    0,    0,    0,    0), 0.20, head_pan=C),
    ]


_ALL_SEQUENCES = [_seq_zoomies, _seq_wiggle, _seq_spin]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _do_chirp(buzzer, count: int, on: float, off: float):
    for _ in range(count):
        try:
            buzzer.set_state(True)
            time.sleep(on)
            buzzer.set_state(False)
            time.sleep(off)
        except Exception:
            pass


def _set_led(led, emotion: str):
    r, g, b = LED_COLORS.get(emotion, (0, 255, 0))
    try:
        led.ledIndex(0xFF, r, g, b)
    except Exception:
        pass


# ── LocalPetBrain ─────────────────────────────────────────────────────────────

class LocalPetBrain:
    """
    Rule-based pet brain. Drop-in replacement for ServerPetBrain.
    Accepts the same SensorHub duck-typed object as ServerPetBrain.

    Interface used by autonomous3.py main loop:
        brain.tick(car, led, buzzer, now, dt)
        brain.stuck_recent = bool(...)
    """

    # Energy thresholds
    _ENERGY_PLAY_MIN  = 0.65
    _ENERGY_TIRED_MAX = 0.38
    _ENERGY_SLEEP_MAX = 0.12
    _ENERGY_WAKE_MIN  = 0.55

    # Distance thresholds (cm)
    _SCARED_CM = 28.0   # obstacle this close increments fear counter
    _ALERT_CM  = 55.0   # obstacle this close shifts toward CURIOUS

    # Timing (seconds)
    _BORED_AFTER_S     = 20.0
    _PLAY_COOLDOWN_S   = 30.0
    _HEAD_SCAN_DT_S    =  5.0   # head sweep interval when idle
    _SCARED_DURATION_S =  3.5   # fear state duration after trigger
    _CHIRP_COOLDOWN_S  = 10.0   # min gap between same-emotion chirps

    def __init__(self, sensors):
        # sensors is a SensorHub (duck-typed — no import needed)
        self.sensors = sensors

        self.emotion: str   = CURIOUS
        self.energy: float  = 0.85

        # Attributes read/written by autonomous3.py main loop
        self.stuck_recent: bool = False
        self.mode:         str  = "ROAM"
        self.action:       str  = "wander"
        self.action_until: float = time.time() + 1.0
        self.speed_pwm:    int  = _DEFAULT_SPEED

        self._runner = _StepRunner()

        self._wander_drift_ts: float = 0.0
        self._wander_bias:     int   = 0

        self._last_play_ts:   float          = 0.0
        self._last_head_scan: float          = 0.0
        self._last_chirp:     float          = 0.0
        self._scared_until:   float          = 0.0
        self._idle_since:     Optional[float] = None
        self._sleep_entered_ts: float        = 0.0

        self._obstacle_count:  int   = 0
        self._obstacle_win_ts: float = 0.0

        self._battery_pct: float = 1.0  # cached, updated each tick

    # ── Battery awareness ─────────────────────────────────────────────────────

    def _battery_tier(self) -> int:
        """
        0 = full  (> 70%)   — normal behaviour
        1 = medium (40-70%) — reduced activity
        2 = low   (20-40%)  — mostly resting
        3 = critical (< 20%)— sleep almost all the time
        """
        p = self._battery_pct
        if p > 0.70: return 0
        if p > 0.40: return 1
        if p > 0.20: return 2
        return 3

    def _min_sleep_s(self) -> float:
        """Minimum sleep duration before the robot is allowed to wake up."""
        return (45.0, 120.0, 300.0, 600.0)[self._battery_tier()]

    # ── Energy ────────────────────────────────────────────────────────────────

    def _tick_energy(self, dt: float):
        rates = {
            SLEEP:   +0.022,
            TIRED:   +0.003,
            BORED:   -0.003,
            CURIOUS: -0.005,
            HAPPY:   -0.007,
            PLAYFUL: -0.015,
            SCARED:  -0.010,
        }
        delta = rates.get(self.emotion, -0.005) * dt
        self.energy = max(0.0, min(1.0, self.energy + delta))

    # ── Emotion state machine ─────────────────────────────────────────────────

    def _update_emotion(self, now: float, ahead_cm: Optional[float]):
        tier = self._battery_tier()

        # Sleep cycle has priority: must sleep for at least _min_sleep_s() before waking
        if self.emotion == SLEEP:
            slept = now - self._sleep_entered_ts
            if self.energy >= self._ENERGY_WAKE_MIN and slept >= self._min_sleep_s():
                self._enter(HAPPY, now)
            return

        # Battery-driven forced rest: at tier 3 (critical) the robot sleeps almost all the time
        if tier == 3:
            self._enter(SLEEP, now)
            return

        if self.energy < self._ENERGY_SLEEP_MAX:
            self._enter(SLEEP, now)
            return

        # Timed fear state
        if now < self._scared_until:
            self.emotion = SCARED
            return

        # Energy floor → tired
        if self.energy < self._ENERGY_TIRED_MAX:
            if self.emotion != TIRED:
                self._enter(TIRED, now)
            return

        # At tier 2 (low battery): skip PLAYFUL/HAPPY, default to BORED/TIRED
        if tier >= 2:
            if self.emotion not in (BORED, TIRED):
                self._enter(BORED, now)
            return

        # At tier 1 (medium): PLAYFUL suppressed, longer cooldown applied dynamically
        play_cooldown = self._PLAY_COOLDOWN_S * (1 + tier * 2)  # 30s / 90s / never
        if (tier == 0
                and self.energy >= self._ENERGY_PLAY_MIN
                and (now - self._last_play_ts) > play_cooldown
                and not self.stuck_recent):
            self._enter(PLAYFUL, now)
            return

        # Something nearby → heightened curiosity
        if ahead_cm is not None and float(ahead_cm) < self._ALERT_CM:
            if self.emotion not in (CURIOUS, SCARED):
                self._enter(CURIOUS, now)
            return

        # Boredom threshold shortens with battery tier
        bored_after = self._BORED_AFTER_S / max(1, tier + 1)  # 20s / 10s / 6s
        if self._idle_since is not None and (now - self._idle_since) > bored_after:
            if self.emotion != BORED:
                self._enter(BORED, now)
            return

        # Default: curious (or stay happy/curious)
        if self.emotion not in (CURIOUS, HAPPY):
            self._enter(CURIOUS, now)

    def _enter(self, emotion: str, now: float):
        if emotion == self.emotion:
            return
        log.info("[PET] %s → %s  energy=%.2f  battery=%.0f%%",
                 self.emotion, emotion, self.energy, self._battery_pct * 100)
        self.emotion = emotion
        self._last_chirp = 0.0  # chirp on next tick for new emotion
        if emotion == PLAYFUL:
            self._last_play_ts = now
        if emotion in (BORED, SLEEP):
            self._idle_since = now
        if emotion == SLEEP:
            self._sleep_entered_ts = now

    # ── Action selection ──────────────────────────────────────────────────────

    def _pick_action(self, car, buzzer, now: float, ahead_cm: Optional[float]):
        e = self.emotion

        if e == SLEEP:
            self.mode = "SLEEP"
            self.action = "stop"
            self.action_until = now + 1.5
            return

        if e == TIRED:
            self.mode = "IDLE"
            if random.random() < 0.35:
                self.action = "stop"
                self.action_until = now + random.uniform(1.0, 2.5)
            else:
                self.action = "wander"
                self.speed_pwm = _TIRED_SPEED
                self.action_until = now + random.uniform(0.8, 1.8)
            return

        if e == SCARED:
            self.mode = "ROAM"
            self.action = random.choice(["spin_left", "spin_right", "reverse"])
            self.speed_pwm = _DEFAULT_SPEED
            self.action_until = now + random.uniform(0.3, 0.55)
            return

        if e == BORED:
            self.mode = "IDLE"
            if now >= self._last_head_scan + self._HEAD_SCAN_DT_S:
                self._do_head_scan(car, now)
            self.action = "stop"
            self.action_until = now + random.uniform(0.8, 1.6)
            # Occasional restless nudge forward so it doesn't freeze entirely
            if random.random() < 0.12:
                self.action = "forward"
                self.speed_pwm = _TIRED_SPEED
                self.action_until = now + 0.5
            return

        if e == PLAYFUL:
            self.mode = "PLAY"
            seq = random.choice(_ALL_SEQUENCES)()
            self._runner.load(seq, now)
            self.action = "sequence"
            self.speed_pwm = _PLAY_SPEED
            self.action_until = now + sum(s.dur for s in seq) + 0.2
            # Transition to HAPPY immediately so energy drains at the HAPPY rate
            # while the sequence plays out (PLAYFUL drains too fast for a long seq)
            self._enter(HAPPY, now)
            return

        if e == HAPPY:
            self.mode = "ROAM"
            self.action = "wander"
            self.speed_pwm = _DEFAULT_SPEED + 120
            self.action_until = now + random.uniform(0.5, 1.2)
            return

        # CURIOUS (default)
        self.mode = "ROAM"
        if ahead_cm is not None and float(ahead_cm) < self._ALERT_CM:
            # Something ahead — look around it rather than charging in
            self.action = random.choice(["turn_left", "turn_right"])
            self.speed_pwm = _DEFAULT_SPEED
            self.action_until = now + random.uniform(0.3, 0.55)
        else:
            self.action = "wander"
            self.speed_pwm = _DEFAULT_SPEED
            self.action_until = now + random.uniform(0.7, 1.5)

    def _do_head_scan(self, car, now: float):
        """Curious look-around: sweep head left then right, park forward."""
        self._last_head_scan = now
        try:
            car.set_head_pose(pan=_HEAD_LEFT,  settle=0.07)
            time.sleep(0.15)
            car.set_head_pose(pan=_HEAD_RIGHT, settle=0.07)
            time.sleep(0.15)
            car.park_head_for_drive()
        except Exception:
            pass

    # ── Apply action ──────────────────────────────────────────────────────────

    def _apply(self, car, now: float):
        if self.action == "sequence":
            self._runner.tick(car, now)
            if not self._runner.active() and now >= self.action_until:
                try:
                    car.set_motors(0, 0, 0, 0)
                except Exception:
                    pass
            return

        p = self.speed_pwm
        drive_actions = {"forward", "wander", "turn_left", "turn_right", "spin_left", "spin_right"}

        if self.action in drive_actions:
            try:
                car.park_head_for_drive()
            except Exception:
                pass

        try:
            if self.action == "stop":
                car.set_motors(0, 0, 0, 0)
            elif self.action == "forward":
                car.set_motors(p, p, p, p)
            elif self.action == "reverse":
                try:
                    car.park_head_for_reverse()
                except Exception:
                    pass
                car.set_motors(-p, -p, -p, -p)
            elif self.action == "turn_left":
                car.set_motors(p - _TURN_BIAS, p - _TURN_BIAS, p, p)
            elif self.action == "turn_right":
                car.set_motors(p, p, p - _TURN_BIAS, p - _TURN_BIAS)
            elif self.action == "spin_left":
                car.set_motors(-p, -p, p, p)
            elif self.action == "spin_right":
                car.set_motors(p, p, -p, -p)
            elif self.action == "wander":
                if now >= self._wander_drift_ts:
                    self._wander_drift_ts = now + random.uniform(0.3, 0.9)
                    self._wander_bias = random.choice([-1, 0, 0, 1])
                b = self._wander_bias
                if b < 0:
                    car.set_motors(p - _TURN_BIAS, p - _TURN_BIAS, p, p)
                elif b > 0:
                    car.set_motors(p, p, p - _TURN_BIAS, p - _TURN_BIAS)
                else:
                    car.set_motors(p, p, p, p)
            else:
                car.set_motors(0, 0, 0, 0)
        except Exception:
            pass

    # ── Main tick (same signature as ServerPetBrain.tick) ────────────────────

    def tick(self, car, led, buzzer, now: float, dt: float):
        # Read sensors via SensorHub
        battery_v = self.sensors.update_battery(now)
        ahead_cm  = self.sensors.update_forward_sonar(now)
        self.sensors.update_ir(now)
        self.sensors.update_light(now)
        self.sensors.update_pose(now)

        # Cache battery percentage (used by tier logic throughout this tick)
        pct = getattr(self.sensors, "battery_pct", lambda: None)()
        self._battery_pct = pct if pct is not None else self._battery_pct

        # Hard shutdown threshold
        if battery_v is not None and battery_v < 5.0:
            self.emotion = SLEEP
            try:
                car.set_motors(0, 0, 0, 0)
            except Exception:
                pass
            _set_led(led, SLEEP)
            return

        # Track repeated close obstacles to trigger fear
        if ahead_cm is not None and float(ahead_cm) < self._SCARED_CM:
            if (now - self._obstacle_win_ts) > 6.0:
                self._obstacle_count = 0
                self._obstacle_win_ts = now
            self._obstacle_count += 1
            if self._obstacle_count >= 3:
                self._scared_until = now + self._SCARED_DURATION_S
                log.info(
                    "[PET] fear triggered — obstacle count=%d, ahead=%.1f cm",
                    self._obstacle_count, float(ahead_cm),
                )

        # Idle tracking (for boredom)
        if self.action == "stop":
            if self._idle_since is None:
                self._idle_since = now
        else:
            self._idle_since = None

        # Update energy and pick emotional state
        self._tick_energy(dt)
        self._update_emotion(now, ahead_cm)

        # Choose new action when current one has expired
        if now >= self.action_until:
            self._pick_action(car, buzzer, now, ahead_cm)

        # Expressive outputs — battery tier overrides LED color when resting
        tier = self._battery_tier()
        if self.emotion == SLEEP and tier >= 1:
            # Dim LED: green=good, amber=low, red=critical
            _batt_colors = {1: (0, 20, 0), 2: (30, 12, 0), 3: (40, 0, 0)}
            r, g, b = _batt_colors[tier]
            try:
                led.ledIndex(0xFF, r, g, b)
            except Exception:
                pass
        else:
            _set_led(led, self.emotion)

        if now >= self._last_chirp + self._CHIRP_COOLDOWN_S:
            count, on, off = CHIRP_PATTERNS.get(self.emotion, (0, 0.0, 0.0))
            if count > 0:
                _do_chirp(buzzer, count, on, off)
                self._last_chirp = now

        # Drive motors
        self._apply(car, now)
