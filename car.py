import time
import math
import threading
from collections import deque

from motor import MotorController
from servo import Servo
from infrared import Infrared
from adc import ADC
from ultrasonic import Ultrasonic


# ── Maths helpers ─────────────────────────────────────────────────────────────

def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0

def _deg2rad(d: float) -> float:
    return d * math.pi / 180.0


# ── Dead-reckoning pose ───────────────────────────────────────────────────────

class PoseEstimator:
    """
    Approximate pose from motor commands (no encoders).
    Calibration: 25 cm/s at PWM 600; 220 °/s at PWM 1500.
    """

    def __init__(self,
                 cm_per_sec_at_600:  float = 25.0,
                 deg_per_sec_at_1500: float = 220.0):
        self.x_cm   = 0.0
        self.y_cm   = 0.0
        self.th_deg = 0.0
        self._cms   = cm_per_sec_at_600
        self._dps   = deg_per_sec_at_1500

    def integrate(self, cmd: tuple, dt: float) -> None:
        if dt <= 0:
            return
        fl, bl, fr, br = cmd
        mags = [abs(fl), abs(bl), abs(fr), abs(br)]
        avg  = sum(mags) / 4.0
        if avg < 50:
            return

        same_sign = all(v >= 0 for v in cmd) or all(v <= 0 for v in cmd)
        forward_like = same_sign and (max(mags) - min(mags)) < 250

        left_avg  = (fl + bl) / 2.0
        right_avg = (fr + br) / 2.0
        rotate_like = (abs(abs(left_avg) - abs(right_avg)) < 300
                       and left_avg * right_avg < 0)

        if forward_like:
            sign = 1.0 if sum(cmd) > 0 else -1.0
            dist = self._cms * (avg / 600.0) * sign * dt
            th   = _deg2rad(self.th_deg)
            self.x_cm += dist * math.cos(th)
            self.y_cm += dist * math.sin(th)
        elif rotate_like:
            direction = -1.0 if left_avg > 0 and right_avg < 0 else 1.0
            omega = self._dps * (avg / 1500.0) * direction
            self.th_deg = _wrap_deg(self.th_deg + omega * dt)


# ── Occupancy memory ──────────────────────────────────────────────────────────

class OccupancyMemory:
    """
    Lightweight log-odds occupancy grid and visit counter on a coarse dict-based grid.
    10 cm cells keep RAM usage negligible on Raspberry Pi.
    """

    LO_OCC  =  0.85
    LO_FREE = -0.35
    LO_MIN  = -4.0
    LO_MAX  =  4.0

    def __init__(self, cell_cm: float = 10.0):
        self.cell_cm = cell_cm
        self.logodds: dict = {}
        self.visited: dict = {}

    def _cell(self, x_cm: float, y_cm: float) -> tuple:
        return (int(round(x_cm / self.cell_cm)), int(round(y_cm / self.cell_cm)))

    def mark_visited(self, x_cm: float, y_cm: float) -> None:
        c = self._cell(x_cm, y_cm)
        self.visited[c] = self.visited.get(c, 0) + 1

    def visited_count_at(self, x_cm: float, y_cm: float) -> int:
        return self.visited.get(self._cell(x_cm, y_cm), 0)

    def occ_prob_at(self, x_cm: float, y_cm: float) -> float:
        lo = self.logodds.get(self._cell(x_cm, y_cm), 0.0)
        return 1.0 / (1.0 + math.exp(-lo))

    def _add_lo(self, cell: tuple, delta: float) -> None:
        lo = self.logodds.get(cell, 0.0) + delta
        self.logodds[cell] = max(self.LO_MIN, min(self.LO_MAX, lo))

    def update_from_scan(self, pose: PoseEstimator,
                         servo_angles_deg, distances_cm,
                         max_range_cm: float = 180.0) -> None:
        """
        Ray-cast scan results into the grid.
        servo_angles_deg: [30, 90, 150] → right, center, left.
        90° = forward; distance None = ignore.
        """
        rx, ry, rth = pose.x_cm, pose.y_cm, pose.th_deg
        self.mark_visited(rx, ry)

        for ang, dist in zip(servo_angles_deg, distances_cm):
            if dist is None:
                continue
            dist    = max(0.0, min(float(dist), max_range_cm))
            bearing = _deg2rad(_wrap_deg(rth + (ang - 90.0)))

            # Mark free space along the beam
            step = self.cell_cm * 0.7
            t    = 0.0
            while t < dist:
                self._add_lo(self._cell(rx + t * math.cos(bearing),
                                        ry + t * math.sin(bearing)), self.LO_FREE)
                t += step

            # Mark obstacle at endpoint
            if self.cell_cm < dist < max_range_cm * 0.95:
                self._add_lo(self._cell(rx + dist * math.cos(bearing),
                                        ry + dist * math.sin(bearing)), self.LO_OCC)


# ── Memory navigator ──────────────────────────────────────────────────────────

class MemoryNavigator:
    """
    Choose a turn direction using clearance, visit history, and occupancy.
    Detects L/R oscillation and local looping and escapes with stronger manoeuvres.
    """

    def __init__(self, pose: PoseEstimator, mem: OccupancyMemory):
        self.pose    = pose
        self.mem     = mem
        self.actions: deque = deque(maxlen=14)
        self.cells:   deque = deque(maxlen=20)

    def _record(self, act: str) -> None:
        self.actions.append(act)
        self.cells.append(self.mem._cell(self.pose.x_cm, self.pose.y_cm))

    def oscillating(self) -> bool:
        if len(self.actions) < 8:
            return False
        tail = ''.join(list(self.actions)[-8:])
        return tail in ("LRLRLRLR", "RLRLRLRL", "LRLRRLRL", "RLRLLRLR")

    def stuck_in_same_area(self) -> bool:
        if len(self.cells) < 12:
            return False
        return len(set(list(self.cells)[-12:])) <= 2

    def score_direction(self, heading_offset_deg: float, clearance_cm: float) -> float:
        th = _deg2rad(_wrap_deg(self.pose.th_deg + heading_offset_deg))
        px = self.pose.x_cm + 35.0 * math.cos(th)
        py = self.pose.y_cm + 35.0 * math.sin(th)

        visited = self.mem.visited_count_at(px, py)
        occ_p   = self.mem.occ_prob_at(px, py)

        score  = min(clearance_cm, 160.0)
        score -= min(visited, 6) * 18.0
        if occ_p > 0.62:
            score -= 80.0
        uncertainty = 1.0 - min(1.0, abs(occ_p - 0.5) * 2.0)
        score += uncertainty * 22.0
        return score

    def choose_action(self, left_cm, center_cm, right_cm) -> str:
        l = 180.0 if left_cm   is None else float(left_cm)
        c = 180.0 if center_cm is None else float(center_cm)
        r = 180.0 if right_cm  is None else float(right_cm)

        if c < 25 and l < 28 and r < 28:
            return "U"

        if self.oscillating() or self.stuck_in_same_area():
            return "S"

        left_score  = self.score_direction(+60.0, l)
        right_score = self.score_direction(-60.0, r)

        if self.actions:
            if self.actions[-1] == "L":
                left_score  -= 12.0
            elif self.actions[-1] == "R":
                right_score -= 12.0

        return "L" if left_score >= right_score else "R"


# ── Car ───────────────────────────────────────────────────────────────────────

class Car:
    """
    Top-level robot controller.

    Motor convention (positive = forward):
        set_motors(front_left, back_left, front_right, back_right)

    Head: pan channel 0 (0=right, 90=forward, 150=left),
          tilt channel 1 (85=up, 120=neutral, 155=down).
    """

    SPEED_SCALE     = 0.65
    MIN_EFFECTIVE   = 260    # PWM below this stalls motors
    TILT_CENTER     = 120
    TILT_UP         =  85
    TILT_DOWN       = 155

    def __init__(self):
        self.servo    = Servo()
        self.sonic    = Ultrasonic()
        self.motor    = MotorController()
        self.infrared = Infrared()
        self.adc      = ADC()

        self.head_lock    = threading.Lock()
        self.current_pan  = 90
        self.current_tilt = self.TILT_CENTER

        self.pose = PoseEstimator()
        self.mem  = OccupancyMemory(cell_cm=10.0)
        self.nav  = MemoryNavigator(self.pose, self.mem)

        self._last_cmd:    tuple = (0, 0, 0, 0)
        self._last_cmd_ts: float = time.time()

        # Stuck detection state
        self._last_fwd_dist: float | None = None
        self._stuck_since:   float | None = None

        # Ultrasonic scan rotation state (legacy mode_ultrasonic)
        self._sonic_angle = 30
        self._sonic_dir   = 1
        self._sonic_dists = [30, 30, 30]
        self._mode_ts     = time.time()

    # ── Head ─────────────────────────────────────────────────────────────────

    def set_head_pose(self, pan: int = None, tilt: int = None,
                      settle: float = 0.02) -> None:
        """Thread-safe head positioning."""
        with self.head_lock:
            if tilt is not None:
                self.current_tilt = int(tilt)
                self.servo.set_servo_pwm('1', self.current_tilt)
            if pan is not None:
                self.current_pan = int(pan)
                self.servo.set_servo_pwm('0', self.current_pan)
        if settle:
            time.sleep(settle)

    def park_head_for_drive(self) -> None:
        self.set_head_pose(pan=90, tilt=self.TILT_CENTER, settle=0.02)

    def park_head_for_reverse(self) -> None:
        self.set_head_pose(pan=90, tilt=self.TILT_DOWN, settle=0.03)

    # ── Motor wrapper ─────────────────────────────────────────────────────────

    def _scale(self, v: int) -> int:
        v = int(round(v * self.SPEED_SCALE))
        v = max(-4095, min(4095, v))
        if v != 0 and abs(v) < self.MIN_EFFECTIVE:
            v = self.MIN_EFFECTIVE if v > 0 else -self.MIN_EFFECTIVE
        return v

    def set_motors(self, fl: int, bl: int, fr: int, br: int) -> None:
        """Apply global speed scale, deadband, update odometry, then drive."""
        now = time.time()
        dt  = now - self._last_cmd_ts
        try:
            self.pose.integrate(self._last_cmd, dt)
            self.mem.mark_visited(self.pose.x_cm, self.pose.y_cm)
        except Exception:
            pass

        sfl, sbl, sfr, sbr = self._scale(fl), self._scale(bl), self._scale(fr), self._scale(br)
        self.motor.set_motor_model(sfl, sbl, sfr, sbr)
        self._last_cmd    = (sfl, sbl, sfr, sbr)
        self._last_cmd_ts = now

    def is_commanding_forward(self, min_pwm: int = 200) -> bool:
        return all(v > min_pwm for v in self._last_cmd)

    # ── Distance reading ──────────────────────────────────────────────────────

    def get_forward_distance(self) -> float | None:
        """Park head forward, read ultrasonic, return distance in cm."""
        with self.head_lock:
            self.servo.set_servo_pwm('1', self.TILT_CENTER)
            self.servo.set_servo_pwm('0', 90)
            time.sleep(0.05)
            return self.sonic.get_distance()

    def get_distance_current_direction(self, ensure_tilt_center: bool = True,
                                       settle: float = 0.01) -> float | None:
        with self.head_lock:
            if ensure_tilt_center:
                self.servo.set_servo_pwm('1', self.TILT_CENTER)
                self.current_tilt = self.TILT_CENTER
                if settle:
                    time.sleep(settle)
            return self.sonic.get_distance()

    def scan_distances(self, angles=(30, 90, 150), settle=0.08,
                       samples=1) -> tuple[list, list]:
        """Sweep head to each angle and return (angles, distances_cm)."""
        angles_list = list(angles)
        distances   = []
        with self.head_lock:
            self.servo.set_servo_pwm('1', self.TILT_CENTER)
            time.sleep(0.04)
            for ang in angles_list:
                self.servo.set_servo_pwm('0', ang)
                time.sleep(settle)
                if samples <= 1:
                    d = self.sonic.get_distance()
                else:
                    vals = [v for _ in range(samples)
                            if (v := self.sonic.get_distance()) is not None]
                    d = sum(vals) / len(vals) if vals else None
                distances.append(d)
        return angles_list, distances

    def pose_string(self) -> str:
        return f"x={self.pose.x_cm:.0f} y={self.pose.y_cm:.0f} th={self.pose.th_deg:.0f}°"

    # ── Obstacle avoidance ────────────────────────────────────────────────────

    def _run_avoidance(self, right_cm, center_cm, left_cm) -> None:
        """Reactive avoidance without memory (used as fallback only)."""
        r = right_cm  or 180
        c = center_cm or 180
        l = left_cm   or 180
        if (r < 50 and c < 50 and l < 50) or c < 50:
            self.set_motors(-1450, -1450, -1450, -1450)
            time.sleep(0.1)
            if r < l:
                self.set_motors(1450, 1450, -1450, -1450)
            else:
                self.set_motors(-1450, -1450, 1450, 1450)
        elif r < 30 and c < 30:
            self.set_motors(1500, 1500, -1500, -1500)
        elif l < 30 and c < 30:
            self.set_motors(-1500, -1500, 1500, 1500)
        elif r < 20:
            self.set_motors(1500 if r < 10 else 2000, 1500 if r < 10 else 2000, -1000, -1000)
        elif l < 20:
            self.set_motors(-1000, -1000, 1500 if l < 10 else 2000, 1500 if l < 10 else 2000)
        else:
            self.set_motors(600, 600, 600, 600)

    def scan_and_avoid_with_memory(self) -> None:
        """Stop immediately, scan left/center/right, then execute a memory-guided evasion."""
        self.set_motors(0, 0, 0, 0)  # halt before the ~0.4s scan so we don't coast closer
        angles, dists = self.scan_distances(samples=2)
        right_cm, center_cm, left_cm = dists[0], dists[1], dists[2]
        self.mem.update_from_scan(self.pose, angles, dists)

        act = self.nav.choose_action(left_cm, center_cm, right_cm)

        TURN_R = ( 1500,  1500, -1500, -1500)
        TURN_L = (-1500, -1500,  1500,  1500)
        NUDGE  = 850
        NUDGE_T = 0.28
        CLEAR  = 35.0

        if act == "L":
            self.nav._record("L")
            self.set_motors(*TURN_L); time.sleep(0.28)
            self.set_motors(0, 0, 0, 0); time.sleep(0.03)
            if left_cm is None or float(left_cm) > CLEAR:
                self.set_motors(NUDGE, NUDGE, NUDGE, NUDGE); time.sleep(NUDGE_T)
                self.set_motors(0, 0, 0, 0)

        elif act == "R":
            self.nav._record("R")
            self.set_motors(*TURN_R); time.sleep(0.28)
            self.set_motors(0, 0, 0, 0); time.sleep(0.03)
            if right_cm is None or float(right_cm) > CLEAR:
                self.set_motors(NUDGE, NUDGE, NUDGE, NUDGE); time.sleep(NUDGE_T)
                self.set_motors(0, 0, 0, 0)

        elif act == "U":
            self.nav._record("U")
            ls  = 180 if left_cm  is None else float(left_cm)
            rs  = 180 if right_cm is None else float(right_cm)
            cmd = TURN_L if ls >= rs else TURN_R
            if center_cm is not None and float(center_cm) < 18:
                self.park_head_for_reverse()
                self.set_motors(-1200, -1200, -1200, -1200); time.sleep(0.25)
                self.set_motors(0, 0, 0, 0); time.sleep(0.03)
            self.park_head_for_drive()
            self.set_motors(*cmd); time.sleep(0.90)
            self.set_motors(0, 0, 0, 0); time.sleep(0.05)
            self.set_motors(850, 850, 850, 850); time.sleep(0.50)
            self.set_motors(0, 0, 0, 0)

        elif act == "B":
            self.nav._record("B")
            self.set_motors(-1200, -1200, -1200, -1200); time.sleep(0.60)
            last = self.nav.actions[-1] if self.nav.actions else None
            cmd  = TURN_R if last == "L" else TURN_L
            self.set_motors(*cmd); time.sleep(0.85)
            self.set_motors(0, 0, 0, 0); time.sleep(0.05)
            self.set_motors(750, 750, 750, 750); time.sleep(0.45)
            self.set_motors(0, 0, 0, 0)

        elif act == "S":
            self.nav._record("S")
            last = self.nav.actions[-2] if len(self.nav.actions) >= 2 else None
            cmd  = TURN_R if last == "L" else TURN_L
            self.park_head_for_drive()
            self.set_motors(*cmd); time.sleep(0.95)
            self.set_motors(0, 0, 0, 0); time.sleep(0.05)
            self.set_motors(850, 850, 850, 850); time.sleep(0.55)
            self.set_motors(0, 0, 0, 0)

        else:
            self._run_avoidance(right_cm, center_cm, left_cm)

    # ── Stuck detection & escape ──────────────────────────────────────────────

    def detect_stuck(self, current_distance: float | None,
                     moving_forward: bool,
                     min_delta: float = 2.0,
                     timeout: float   = 1.0) -> bool:
        now = time.time()
        if not moving_forward or current_distance is None:
            self._stuck_since    = None
            self._last_fwd_dist  = current_distance
            return False
        if self._last_fwd_dist is None:
            self._last_fwd_dist = current_distance
            return False
        if abs(current_distance - self._last_fwd_dist) < min_delta:
            if self._stuck_since is None:
                self._stuck_since = now
            elif now - self._stuck_since > timeout:
                return True
        else:
            self._stuck_since = None
        self._last_fwd_dist = current_distance
        return False

    def escape_stuck_with_memory(self) -> None:
        angles, dists = self.scan_distances(samples=2)
        self.mem.update_from_scan(self.pose, angles, dists)
        ls = 180 if dists[2] is None else dists[2]
        rs = 180 if dists[0] is None else dists[0]

        self.park_head_for_reverse()
        self.set_motors(-1300, -1300, -1300, -1300); time.sleep(0.40)
        self.set_motors(0, 0, 0, 0); time.sleep(0.03)
        self.park_head_for_drive()

        cmd = (-1500, -1500, 1500, 1500) if ls >= rs else (1500, 1500, -1500, -1500)
        self.set_motors(*cmd); time.sleep(0.75)
        self.set_motors(800, 800, 800, 800); time.sleep(0.45)
        self.set_motors(0, 0, 0, 0)

    # ── Autonomous sub-modes (used by main.py remote control server) ──────────

    def mode_ultrasonic(self) -> None:
        self.servo.set_servo_pwm('0', self._sonic_angle)
        if   self._sonic_angle == 30:  self._sonic_dists[0] = self.sonic.get_distance()
        elif self._sonic_angle == 90:  self._sonic_dists[1] = self.sonic.get_distance()
        elif self._sonic_angle == 150: self._sonic_dists[2] = self.sonic.get_distance()

        self._run_avoidance(*self._sonic_dists)

        if   self._sonic_angle <= 30:  self._sonic_dir = 1
        elif self._sonic_angle >= 150: self._sonic_dir = 0
        self._sonic_angle += 60 if self._sonic_dir == 1 else -60

    def mode_infrared(self) -> None:
        if (time.time() - self._mode_ts) < 0.2:
            return
        self._mode_ts = time.time()
        v = self.infrared.read_all_infrared()
        cmds = {
            2: (800, 800, 800, 800),
            4: (-1500, -1500, 2500, 2500),
            6: (-2000, -2000, 4000, 4000),
            1: (2500, 2500, -1500, -1500),
            3: (4000, 4000, -2000, -2000),
            7: (0, 0, 0, 0),
        }
        if v in cmds:
            self.set_motors(*cmds[v])

    def mode_light(self) -> None:
        if (time.time() - self._mode_ts) < 0.2:
            return
        self._mode_ts = time.time()
        L = self.adc.read_adc(0)
        R = self.adc.read_adc(1)
        if L < 2.99 and R < 2.99:
            self.set_motors(600, 600, 600, 600)
        elif abs(L - R) < 0.15:
            self.set_motors(0, 0, 0, 0)
        elif L > R:
            self.set_motors(-1200, -1200, 1400, 1400)
        else:
            self.set_motors(1400, 1400, -1200, -1200)

    def mode_rotate(self, angle: int, stop_event: threading.Event = None) -> None:
        """
        Continuous rotation command, used by remote-control server.
        Pass a threading.Event to stop_event for clean shutdown.
        """
        bat_v        = max(self.adc.read_adc(2) * (3 if self.adc.pcb_version == 1 else 2), 0.1)
        bat_factor   = 7.5 / bat_v
        step_delay   = 5 * 3 * bat_factor / 1000  # 3 = time_compensate constant

        a = float(angle)
        while stop_event is None or not stop_event.is_set():
            W  = 2000
            VY = int(2000 * math.cos(math.radians(a)))
            VX = -int(2000 * math.sin(math.radians(a)))
            self.set_motors(VY + VX - W, VY - VX - W, VY - VX + W, VY + VX + W)
            time.sleep(step_delay)
            a -= 5

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self.set_motors(0, 0, 0, 0)
        self.sonic.close()
        self.motor.close()
        self.infrared.close()
        self.adc.close_i2c()
