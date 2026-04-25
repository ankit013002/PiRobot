#!/usr/bin/env python3
from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Optional, Deque, Tuple
from collections import deque

from car import Car
from led import Led
from buzzer import Buzzer

# ============================================================
# PET CONFIG (tweak these to taste)
# ============================================================
LOOP_DT = 0.05  # 20 Hz main loop

LOW_BATTERY_THRESHOLD = 5.0
STATUS_PRINT_INTERVAL = 1.2

# Sonar cadence (avoid hammering servo/sonar)
SONAR_UPDATE_DT = 0.18

# Roam tuning
ROAM_FWD_PWM = 1100
ROAM_OBS_TRIGGER_CM = 45.0
FORWARD_HARD_STOP_CM = 18.0

# Stuck detection
STUCK_MIN_DELTA_CM = 2.0
STUCK_TIMEOUT_S = 1.0

# Pet “life” timing
ROAM_MIN_S, ROAM_MAX_S = 10.0, 22.0
IDLE_MIN_S, IDLE_MAX_S = 4.0, 10.0
PLAY_MIN_S, PLAY_MAX_S = 3.5, 7.5
SLEEP_MIN_S, SLEEP_MAX_S = 18.0, 45.0

# Energy model (0..1)
ENERGY_DRAIN_ROAM = 0.0045   # per second
ENERGY_DRAIN_PLAY = 0.0100
ENERGY_GAIN_SLEEP = 0.0200
ENERGY_GAIN_IDLE  = 0.0040

# Head motion
HEAD_CENTER = 90
HEAD_LEFT = 150
HEAD_RIGHT = 30

# ============================================================
# LED helper
# ============================================================
MODE_ROAM = "ROAM"
MODE_IDLE = "IDLE"
MODE_PLAY = "PLAY"
MODE_SLEEP = "SLEEP"


def set_mode_led(led: Led, mode: str):
    # Simple colors (tweak if you like)
    # ROAM: green, IDLE: warm white/yellow, PLAY: purple, SLEEP: dim blue
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
        pass


def chirp(buzzer: Buzzer, n: int = 1, on: float = 0.06, off: float = 0.05):
    for _ in range(n):
        try:
            buzzer.set_state(True)
            time.sleep(on)
            buzzer.set_state(False)
            time.sleep(off)
        except Exception:
            pass


# ============================================================
# Pet play sequences (non-blocking step runner)
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

    def active(self) -> bool:
        return len(self.steps) > 0

    def tick(self, car: Car, now: float) -> bool:
        if not self.steps:
            return False

        if now >= self.step_end:
            st = self.steps.popleft()
            # apply head pose first (safe)
            try:
                if st.head_pan is not None or st.head_tilt is not None:
                    car.set_head_pose(pan=st.head_pan, tilt=st.head_tilt, settle=0.01)
            except Exception:
                pass

            try:
                car.set_motors(*st.cmd)
            except Exception:
                pass

            self.step_end = now + max(0.02, float(st.dur))

        return True


def play_sequence_zoomies() -> list[Step]:
    # quick forward bursts + spins + head wag
    return [
        Step((0, 0, 0, 0), 0.15, head_pan=HEAD_CENTER),
        Step((1200, 1200, 1200, 1200), 0.35, head_pan=HEAD_RIGHT),
        Step((0, 0, 0, 0), 0.12, head_pan=HEAD_LEFT),
        Step((1200, 1200, 1200, 1200), 0.30, head_pan=HEAD_CENTER),
        Step((1200, 1200, -1200, -1200), 0.45, head_pan=HEAD_RIGHT),  # spin
        Step((0, 0, 0, 0), 0.10, head_pan=HEAD_LEFT),
        Step((1000, 1000, 1000, 1000), 0.28, head_pan=HEAD_CENTER),
        Step((0, 0, 0, 0), 0.15, head_pan=HEAD_CENTER),
    ]


def play_sequence_wiggle() -> list[Step]:
    # playful wiggle in place + tiny hops
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


# ============================================================
# Pet brain
# ============================================================
class PetBrain:
    def __init__(self):
        self.mode = MODE_ROAM
        self.mode_until = time.time() + random.uniform(ROAM_MIN_S, ROAM_MAX_S)

        self.energy = 0.85  # start lively
        self.runner = StepRunner()

        self._last_sonar_ts = 0.0
        self._forward_cm_cache = None

        self._last_status_ts = 0.0

        # idle head scan
        self._next_idle_look = 0.0
        self._idle_look_dir = 1  # 1 -> right, -1 -> left

    def _choose_next_mode(self) -> str:
        # Weighted by energy
        e = self.energy
        if e < 0.22:
            return MODE_SLEEP
        if e < 0.40:
            return random.choices([MODE_IDLE, MODE_SLEEP, MODE_ROAM], weights=[4, 4, 2])[0]
        if e < 0.70:
            return random.choices([MODE_ROAM, MODE_IDLE, MODE_PLAY], weights=[5, 3, 2])[0]
        return random.choices([MODE_ROAM, MODE_PLAY, MODE_IDLE], weights=[5, 3, 2])[0]

    def _set_mode(self, mode: str, now: float, buzzer: Buzzer):
        self.mode = mode
        if mode == MODE_ROAM:
            self.mode_until = now + random.uniform(ROAM_MIN_S, ROAM_MAX_S)
        elif mode == MODE_IDLE:
            self.mode_until = now + random.uniform(IDLE_MIN_S, IDLE_MAX_S)
        elif mode == MODE_PLAY:
            self.mode_until = now + random.uniform(PLAY_MIN_S, PLAY_MAX_S)
            chirp(buzzer, n=1)
        elif mode == MODE_SLEEP:
            self.mode_until = now + random.uniform(SLEEP_MIN_S, SLEEP_MAX_S)
        else:
            self.mode_until = now + 8.0

        # cancel any active sequence when switching away from PLAY
        if mode != MODE_PLAY:
            self.runner.load([], now)

    def _update_energy(self, now: float, dt: float):
        if self.mode == MODE_ROAM:
            self.energy -= ENERGY_DRAIN_ROAM * dt
        elif self.mode == MODE_PLAY:
            self.energy -= ENERGY_DRAIN_PLAY * dt
        elif self.mode == MODE_SLEEP:
            self.energy += ENERGY_GAIN_SLEEP * dt
        elif self.mode == MODE_IDLE:
            self.energy += ENERGY_GAIN_IDLE * dt

        self.energy = max(0.0, min(1.0, self.energy))

    def _sonar_forward(self, car: Car, now: float):
        if (now - self._last_sonar_ts) < SONAR_UPDATE_DT:
            return self._forward_cm_cache

        self._last_sonar_ts = now
        try:
            self._forward_cm_cache = car.get_forward_distance()
        except Exception:
            self._forward_cm_cache = None
        return self._forward_cm_cache

    def _play_tick(self, car: Car, now: float):
        # If no sequence loaded, pick one
        if not self.runner.active():
            seq = random.choice([play_sequence_zoomies, play_sequence_wiggle, play_sequence_spin_showoff])()
            self.runner.load(seq, now)

        # Safety: if too close ahead, abort play and avoid
        d = self._sonar_forward(car, now)
        if d is not None and float(d) < FORWARD_HARD_STOP_CM:
            try:
                car.set_motors(0, 0, 0, 0)
            except Exception:
                pass
            # use your memory avoid (blocks briefly, but OK)
            try:
                car.scan_and_avoid_with_memory()
            except Exception:
                pass
            self.runner.load([], now)
            return

        self.runner.tick(car, now)

    def _idle_tick(self, car: Car, now: float):
        # just chill, occasionally look around
        try:
            car.set_motors(0, 0, 0, 0)
        except Exception:
            pass

        if now >= self._next_idle_look:
            self._next_idle_look = now + random.uniform(0.6, 1.4)
            # ping-pong head
            pan = getattr(car, "current_pan", 90)
            target = HEAD_RIGHT if self._idle_look_dir > 0 else HEAD_LEFT
            if abs(pan - target) < 8:
                self._idle_look_dir *= -1
                target = HEAD_RIGHT if self._idle_look_dir > 0 else HEAD_LEFT
            try:
                car.set_head_pose(pan=target, tilt=car.TILT_CENTER, settle=0.01)
            except Exception:
                pass

    def _sleep_tick(self, car: Car, now: float):
        # settle: stop + head down; tiny “dream twitch” sometimes
        try:
            car.set_motors(0, 0, 0, 0)
        except Exception:
            pass
        try:
            car.park_head_for_reverse()  # uses your “down” tilt; compact
        except Exception:
            pass

        # occasional tiny twitch
        if random.random() < 0.015:
            try:
                car.set_head_pose(pan=random.choice([80, 90, 100]), tilt=car.TILT_DOWN, settle=0.01)
            except Exception:
                pass

    def _roam_tick(self, car: Car, now: float):
        d = self._sonar_forward(car, now)

        # Hard stop safety
        if d is not None and float(d) < FORWARD_HARD_STOP_CM:
            try:
                car.set_motors(0, 0, 0, 0)
            except Exception:
                pass
            try:
                car.scan_and_avoid_with_memory()
            except Exception:
                pass
            return

        # Obstacle -> memory avoid
        if d is not None and float(d) < ROAM_OBS_TRIGGER_CM:
            try:
                car.scan_and_avoid_with_memory()
            except Exception:
                pass
            return

        # Normal forward “wander”
        # small random “drift” by momentary differential bias
        if random.random() < 0.03:
            bias = random.choice([-1, 1])
            if bias < 0:
                cmd = (ROAM_FWD_PWM - 250, ROAM_FWD_PWM - 250, ROAM_FWD_PWM, ROAM_FWD_PWM)
            else:
                cmd = (ROAM_FWD_PWM, ROAM_FWD_PWM, ROAM_FWD_PWM - 250, ROAM_FWD_PWM - 250)
        else:
            cmd = (ROAM_FWD_PWM, ROAM_FWD_PWM, ROAM_FWD_PWM, ROAM_FWD_PWM)

        try:
            car.set_motors(*cmd)
        except Exception:
            pass

    def tick(
        self,
        car: Car,
        led: Led,
        buzzer: Buzzer,
        now: float,
        dt: float,
        battery_v: Optional[float],
    ):
        # Low battery -> sleep immediately
        if battery_v is not None and battery_v < LOW_BATTERY_THRESHOLD:
            if self.mode != MODE_SLEEP:
                self._set_mode(MODE_SLEEP, now, buzzer)

        # Mode timeout -> switch
        if now >= self.mode_until:
            nxt = self._choose_next_mode()
            self._set_mode(nxt, now, buzzer)

        # Update LED
        set_mode_led(led, self.mode)

        # Run behavior
        if self.mode == MODE_PLAY:
            self._play_tick(car, now)
        elif self.mode == MODE_IDLE:
            self._idle_tick(car, now)
        elif self.mode == MODE_SLEEP:
            self._sleep_tick(car, now)
        else:
            self._roam_tick(car, now)

        # Energy update
        self._update_energy(now, dt)

        # Status print
        if (now - self._last_status_ts) > STATUS_PRINT_INTERVAL:
            pv = f"{battery_v:.2f}" if battery_v is not None else "NA"
            d = self._forward_cm_cache
            print(f"[PET] mode={self.mode:<5} energy={self.energy:.2f} bat={pv}V ahead={d}")
            self._last_status_ts = now


# ============================================================
# MAIN
# ============================================================
def main():
    print("Starting autonomous2 pet mode (NO SERVER)...", flush=True)

    car = Car()
    led = Led()
    buzzer = Buzzer()

    brain = PetBrain()

    # little boot chirp
    chirp(buzzer, n=2)

    last_ts = time.time()
    last_status = 0.0

    try:
        # start in roam
        set_mode_led(led, MODE_ROAM)

        while True:
            now = time.time()
            dt = max(0.0, now - last_ts)
            last_ts = now

            # Battery read
            power_raw = None
            try:
                power_raw = car.adc.read_adc(2)
            except Exception:
                power_raw = None

            power_v = None
            if power_raw is not None:
                try:
                    power_v = power_raw * (3 if car.adc.pcb_version == 1 else 2)
                except Exception:
                    power_v = None

            # Brain tick (drives motors/head)
            brain.tick(car, led, buzzer, now, dt, power_v)

            # Stuck detection + escape (only if we’re trying to move forward)
            try:
                moving_forward = car.is_commanding_forward()
                ahead = brain._forward_cm_cache
                if car.detect_stuck(
                    current_distance=ahead,
                    moving_forward=moving_forward,
                    min_delta=STUCK_MIN_DELTA_CM,
                    timeout=STUCK_TIMEOUT_S,
                ):
                    print("[WARN] Stuck detected -> escape", flush=True)
                    chirp(buzzer, n=1)
                    try:
                        car.escape_stuck_with_memory()
                    except Exception:
                        # fallback: basic reverse + turn
                        car.set_motors(0, 0, 0, 0)
                        time.sleep(0.05)
                        car.park_head_for_reverse()
                        car.set_motors(-1300, -1300, -1300, -1300)
                        time.sleep(0.35)
                        car.set_motors(0, 0, 0, 0)
                        time.sleep(0.05)
                        car.park_head_for_drive()
                        car.set_motors(1200, 1200, -1200, -1200)
                        time.sleep(0.65)
                        car.set_motors(0, 0, 0, 0)
            except Exception:
                pass

            time.sleep(LOOP_DT)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received, stopping pet mode...", flush=True)

    finally:
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
        print("[INFO] autonomous2 stopped, cleaned up.", flush=True)


if __name__ == "__main__":
    main()
