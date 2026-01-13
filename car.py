from ultrasonic import Ultrasonic
from motor import Ordinary_Car
from servo import Servo
from infrared import Infrared
from adc import ADC
import time
import math
from collections import deque

# -----------------------------
# "Virtual map" + memory helpers
# -----------------------------

def _wrap_deg(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    return a

def _deg2rad(d: float) -> float:
    return d * math.pi / 180.0

class PoseEstimator:
    """
    Dead-reckoning from motor commands (no encoders).
    It's approximate but good enough to avoid immediate re-trapping.
    """
    def __init__(self,
                 cm_per_sec_at_600: float = 25.0,
                 deg_per_sec_at_1500: float = 220.0):
        self.x_cm = 0.0
        self.y_cm = 0.0
        self.th_deg = 0.0  # heading; 0 means +X in our virtual frame
        self.cm_per_sec_at_600 = cm_per_sec_at_600
        self.deg_per_sec_at_1500 = deg_per_sec_at_1500

    def integrate(self, cmd, dt: float):
        if dt <= 0:
            return

        fl, bl, fr, br = cmd

        # detect "forward-ish": all wheels same sign, similar magnitude
        mags = [abs(fl), abs(bl), abs(fr), abs(br)]
        avg = sum(mags) / 4.0
        if avg < 50:
            return

        same_sign = (fl >= 0 and bl >= 0 and fr >= 0 and br >= 0) or (fl <= 0 and bl <= 0 and fr <= 0 and br <= 0)
        forward_like = same_sign and (max(mags) - min(mags) < 250)

        # detect "rotate-ish": left ~ -right
        left_avg = (fl + bl) / 2.0
        right_avg = (fr + br) / 2.0
        rotate_like = (abs(abs(left_avg) - abs(right_avg)) < 300) and (left_avg * right_avg < 0)

        if forward_like:
            # scale speed
            sign = 1.0 if (fl + bl + fr + br) > 0 else -1.0
            v = self.cm_per_sec_at_600 * (avg / 600.0) * sign
            dist = v * dt
            th = _deg2rad(self.th_deg)
            self.x_cm += dist * math.cos(th)
            self.y_cm += dist * math.sin(th)

        elif rotate_like:
            # If left wheels forward and right wheels backward => typically clockwise (right turn)
            # We'll treat CCW as +th, so clockwise is negative.
            # If your car turns the opposite way, flip the sign logic below.
            direction = -1.0 if left_avg > 0 and right_avg < 0 else 1.0
            omega = self.deg_per_sec_at_1500 * (avg / 1500.0) * direction
            self.th_deg = _wrap_deg(self.th_deg + omega * dt)

class OccupancyMemory:
    """
    Very lightweight occupancy + visited memory on a coarse grid.
    Stored as dicts so it won't blow up RAM.
    """
    def __init__(self, cell_cm: float = 10.0):
        self.cell_cm = cell_cm
        self.logodds = {}   # (ix, iy) -> logodds
        self.visited = {}   # (ix, iy) -> count

        # log-odds tuning
        self.LO_OCC = 0.85
        self.LO_FREE = -0.35
        self.LO_MIN = -4.0
        self.LO_MAX = 4.0

    def _cell(self, x_cm: float, y_cm: float):
        return (int(round(x_cm / self.cell_cm)), int(round(y_cm / self.cell_cm)))

    def mark_visited(self, x_cm: float, y_cm: float):
        c = self._cell(x_cm, y_cm)
        self.visited[c] = self.visited.get(c, 0) + 1

    def visited_count_at(self, x_cm: float, y_cm: float) -> int:
        return self.visited.get(self._cell(x_cm, y_cm), 0)

    def occ_prob_at(self, x_cm: float, y_cm: float) -> float:
        lo = self.logodds.get(self._cell(x_cm, y_cm), 0.0)
        # sigmoid
        return 1.0 / (1.0 + math.exp(-lo))

    def _add_lo(self, cell, delta):
        lo = self.logodds.get(cell, 0.0) + delta
        lo = max(self.LO_MIN, min(self.LO_MAX, lo))
        self.logodds[cell] = lo

    def update_from_scan(self, pose: PoseEstimator, servo_angles_deg, distances_cm, max_range_cm: float = 180.0):
        """
        servo_angles_deg: e.g. [30, 90, 150]
        distances_cm: same length, values can be None
        Assumes servo 90 is forward; 30 is right; 150 is left.
        """
        rx, ry, rth = pose.x_cm, pose.y_cm, pose.th_deg
        self.mark_visited(rx, ry)

        for ang, dist in zip(servo_angles_deg, distances_cm):
            if dist is None:
                continue

            dist = max(0.0, min(float(dist), max_range_cm))
            bearing = _wrap_deg(rth + (ang - 90.0))
            br = _deg2rad(bearing)

            # Raycast: mark freespace along the beam
            step = self.cell_cm * 0.7
            t = 0.0
            while t < dist:
                px = rx + t * math.cos(br)
                py = ry + t * math.sin(br)
                self._add_lo(self._cell(px, py), self.LO_FREE)
                t += step

            # Mark obstacle at end if it's not "far away"
            if dist < max_range_cm * 0.95 and dist > self.cell_cm:
                ox = rx + dist * math.cos(br)
                oy = ry + dist * math.sin(br)
                self._add_lo(self._cell(ox, oy), self.LO_OCC)

class MemoryNavigator:
    """
    Chooses turns using:
    - current scan clearance
    - visited penalty (don't go where you just were)
    - oscillation detection (L/R ping-pong)
    - optional occupancy check
    """
    def __init__(self, pose: PoseEstimator, mem: OccupancyMemory):
        self.pose = pose
        self.mem = mem
        self.actions = deque(maxlen=14)   # 'L','R','B','U','F'
        self.cells = deque(maxlen=20)     # recent visited cells (coarse)
        self.last_decision_ts = time.time()

    def _record(self, act: str):
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
        # if most of the last 12 samples are in <=2 cells, you're basically looping in a corner
        unique = len(set(list(self.cells)[-12:]))
        return unique <= 2

    def score_direction(self, heading_offset_deg: float, clearance_cm: float):
        """
        heading_offset_deg: -60 for right, 0 forward, +60 left (relative to current heading)
        """
        # predict a point 35cm in that direction
        th = _deg2rad(_wrap_deg(self.pose.th_deg + heading_offset_deg))
        px = self.pose.x_cm + 35.0 * math.cos(th)
        py = self.pose.y_cm + 35.0 * math.sin(th)

        visited = self.mem.visited_count_at(px, py)
        occ_p = self.mem.occ_prob_at(px, py)

        # core score: prefer more clearance
        s = min(clearance_cm, 160.0)

        # avoid re-visiting
        s -= min(visited, 6) * 18.0

        # avoid likely-occupied
        if occ_p > 0.62:
            s -= 80.0

        return s

    def choose_action(self, left_cm, center_cm, right_cm):
        # handle None as "far"
        l = 180.0 if left_cm is None else float(left_cm)
        c = 180.0 if center_cm is None else float(center_cm)
        r = 180.0 if right_cm is None else float(right_cm)

        # dead-end / corner: everything close
        if c < 25 and l < 28 and r < 28:
            return "U"  # U-turn escape

        # oscillation or local looping: do stronger escape
        if self.oscillating() or self.stuck_in_same_area():
            return "B"  # back up + strong turn

        # Otherwise choose best turn direction (left/right) with memory scoring.
        # Note: servo angles: 150=left(+60), 90=center(0), 30=right(-60)
        left_score = self.score_direction(+60.0, l)
        right_score = self.score_direction(-60.0, r)

        # slight preference to not repeat same turn forever
        if len(self.actions) > 0:
            if self.actions[-1] == "L":
                left_score -= 12.0
            elif self.actions[-1] == "R":
                right_score -= 12.0

        if left_score >= right_score:
            return "L"
        else:
            return "R"

# -----------------------------
# Car class
# -----------------------------

class Car:
    def __init__(self):
        self.servo = None
        self.sonic = None
        self.motor = None
        self.infrared = None
        self.adc = None

        self.car_record_time = time.time()
        self.car_sonic_servo_angle = 30
        self.car_sonic_servo_dir = 1
        self.car_sonic_distance = [30, 30, 30]
        self.time_compensate = 3

        self.last_forward_distance = None
        self.stuck_start_time = None

        # --- Memory / map ---
        self.pose = PoseEstimator(
            cm_per_sec_at_600=25.0,     # <<< calibrate if you want
            deg_per_sec_at_1500=220.0   # <<< calibrate if you want
        )
        self.mem = OccupancyMemory(cell_cm=10.0)
        self.nav = MemoryNavigator(self.pose, self.mem)

        # Track motor command for odometry integration
        self._last_cmd = (0, 0, 0, 0)
        self._last_cmd_ts = time.time()

        self.start()

    def pose_string(self):
        return f"x={self.pose.x_cm:.0f}cm y={self.pose.y_cm:.0f}cm th={self.pose.th_deg:.0f}deg"

    def start(self):
        if self.servo is None:
            self.servo = Servo()
        if self.sonic is None:
            self.sonic = Ultrasonic()
        if self.motor is None:
            self.motor = Ordinary_Car()
        if self.infrared is None:
            self.infrared = Infrared()
        if self.adc is None:
            self.adc = ADC()

    def close(self):
        self.set_motors(0, 0, 0, 0)
        self.sonic.close()
        self.motor.close()
        self.infrared.close()
        self.adc.close_i2c()
        self.servo = None
        self.sonic = None
        self.motor = None
        self.infrared = None
        self.adc = None

    # -----------------------------
    # Motor wrapper (CRITICAL for memory/map)
    # -----------------------------
    def set_motors(self, fl, bl, fr, br):
        now = time.time()
        dt = now - self._last_cmd_ts

        # integrate last motion before changing command
        try:
            self.pose.integrate(self._last_cmd, dt)
            self.mem.mark_visited(self.pose.x_cm, self.pose.y_cm)
        except Exception:
            pass

        self.motor.set_motor_model(fl, bl, fr, br)
        self._last_cmd = (fl, bl, fr, br)
        self._last_cmd_ts = now

    # -----------------------------
    # Base behavior (kept, but uses set_motors)
    # -----------------------------
    def run_motor_ultrasonic(self, distance):
        # distance = [right, center, left] in YOUR original code style
        # We'll keep behavior but call set_motors so odometry updates.
        if (distance[0] < 50 and distance[1] < 50 and distance[2] < 50) or distance[1] < 50:
            self.set_motors(-1450, -1450, -1450, -1450)
            time.sleep(0.1)
            if distance[0] < distance[2]:
                self.set_motors(1450, 1450, -1450, -1450)
            else:
                self.set_motors(-1450, -1450, 1450, 1450)
        elif distance[0] < 30 and distance[1] < 30:
            self.set_motors(1500, 1500, -1500, -1500)
        elif distance[2] < 30 and distance[1] < 30:
            self.set_motors(-1500, -1500, 1500, 1500)
        elif distance[0] < 20:
            self.set_motors(2000, 2000, -500, -500)
            if distance[0] < 10:
                self.set_motors(1500, 1500, -1000, -1000)
        elif distance[2] < 20:
            self.set_motors(-500, -500, 2000, 2000)
            if distance[2] < 10:
                self.set_motors(-1500, -1500, 1500, 1500)
        else:
            self.set_motors(600, 600, 600, 600)

    def get_forward_distance(self):
        self.servo.set_servo_pwm('0', 90)
        time.sleep(0.05)
        return self.sonic.get_distance()

    # -----------------------------
    # Scan helpers
    # -----------------------------
    def scan_distances(self, angles=(30, 90, 150), settle=0.08, samples=1):
        """
        Returns (angles_list, distances_list) where angles are servo angles.
        30 = right, 90 = forward, 150 = left
        """
        distances = []
        angles_list = list(angles)

        for ang in angles_list:
            self.servo.set_servo_pwm('0', ang)
            time.sleep(settle)

            if samples <= 1:
                d = self.sonic.get_distance()
            else:
                vals = []
                for _ in range(samples):
                    dd = self.sonic.get_distance()
                    if dd is not None:
                        vals.append(dd)
                    time.sleep(0.02)
                d = (sum(vals) / len(vals)) if vals else None

            distances.append(d)

        return angles_list, distances

    # -----------------------------
    # NEW: memory-based avoid
    # -----------------------------
    def scan_and_avoid_with_memory(self):
        angles, dists = self.scan_distances(samples=2)

        # angles: [30, 90, 150] => right, center, left
        right_cm, center_cm, left_cm = dists[0], dists[1], dists[2]

        # update "virtual map"
        self.mem.update_from_scan(self.pose, angles, dists)

        act = self.nav.choose_action(left_cm=left_cm, center_cm=center_cm, right_cm=right_cm)

        # Motor commands for turns
        # If your car turns the wrong direction, swap these two.
        TURN_RIGHT = (1500, 1500, -1500, -1500)
        TURN_LEFT  = (-1500, -1500, 1500, 1500)

        if act == "L":
            self.nav._record("L")
            self.set_motors(*TURN_LEFT)
            time.sleep(0.32)
            self.set_motors(0, 0, 0, 0)
            time.sleep(0.05)

        elif act == "R":
            self.nav._record("R")
            self.set_motors(*TURN_RIGHT)
            time.sleep(0.32)
            self.set_motors(0, 0, 0, 0)
            time.sleep(0.05)

        elif act == "U":
            # dead-end: reverse + stronger turn (escape corners)
            self.nav._record("U")
            self.set_motors(-1200, -1200, -1200, -1200)
            time.sleep(0.55)
            # pick direction based on "more open" + less visited
            # (simple: compare left/right scan)
            ls = 180 if left_cm is None else left_cm
            rs = 180 if right_cm is None else right_cm
            cmd = TURN_LEFT if ls >= rs else TURN_RIGHT
            self.set_motors(*cmd)
            time.sleep(0.65)
            self.set_motors(0, 0, 0, 0)
            time.sleep(0.05)

            # push forward a bit to actually exit the pocket
            self.set_motors(700, 700, 700, 700)
            time.sleep(0.35)
            self.set_motors(0, 0, 0, 0)

        elif act == "B":
            # oscillation escape: back up + 120-ish degree turn
            self.nav._record("B")
            self.set_motors(-1200, -1200, -1200, -1200)
            time.sleep(0.60)

            # Do a longer turn opposite of what we've been doing
            last = self.nav.actions[-1] if self.nav.actions else None
            cmd = TURN_LEFT
            if last == "L":
                cmd = TURN_RIGHT
            elif last == "R":
                cmd = TURN_LEFT

            self.set_motors(*cmd)
            time.sleep(0.85)
            self.set_motors(0, 0, 0, 0)
            time.sleep(0.05)

            self.set_motors(750, 750, 750, 750)
            time.sleep(0.45)
            self.set_motors(0, 0, 0, 0)

        else:
            # fallback: old logic
            self.run_motor_ultrasonic([right_cm or 180, center_cm or 180, left_cm or 180])

    # -----------------------------
    # Stuck detection (your original)
    # -----------------------------
    def detect_stuck(self, current_distance, moving_forward,
                    min_delta=2.0, timeout=1.0):
        now = time.time()

        if not moving_forward or current_distance is None:
            self.stuck_start_time = None
            self.last_forward_distance = current_distance
            return False

        if self.last_forward_distance is None:
            self.last_forward_distance = current_distance
            return False

        distance_change = abs(current_distance - self.last_forward_distance)

        if distance_change < min_delta:
            if self.stuck_start_time is None:
                self.stuck_start_time = now
            elif now - self.stuck_start_time > timeout:
                return True
        else:
            self.stuck_start_time = None

        self.last_forward_distance = current_distance
        return False

    # -----------------------------
    # NEW: memory-based stuck escape
    # -----------------------------
    def escape_stuck_with_memory(self):
        # quick scan before escaping to bias direction
        angles, dists = self.scan_distances(samples=2)
        self.mem.update_from_scan(self.pose, angles, dists)

        right_cm, center_cm, left_cm = dists[0], dists[1], dists[2]
        ls = 180 if left_cm is None else left_cm
        rs = 180 if right_cm is None else right_cm

        TURN_RIGHT = (1500, 1500, -1500, -1500)
        TURN_LEFT  = (-1500, -1500, 1500, 1500)

        # reverse longer
        self.set_motors(-1300, -1300, -1300, -1300)
        time.sleep(0.70)

        # choose turn toward "more open" and less visited
        cmd = TURN_LEFT if ls >= rs else TURN_RIGHT
        self.set_motors(*cmd)
        time.sleep(0.75)

        # nudge forward so we don't re-stick immediately
        self.set_motors(800, 800, 800, 800)
        time.sleep(0.45)

        self.set_motors(0, 0, 0, 0)
        time.sleep(0.05)

    # -----------------------------
    # (Kept) your original scan-and-avoid (no memory)
    # -----------------------------
    def scan_and_avoid(self):
        distances = [0, 0, 0]
        angles = [30, 90, 150]
        for i, angle in enumerate(angles):
            self.servo.set_servo_pwm('0', angle)
            time.sleep(0.08)
            distances[i] = self.sonic.get_distance()
        self.run_motor_ultrasonic(distances)

    # -----------------------------
    # Other modes (kept, but make sure they use set_motors if you want mapping there too)
    # -----------------------------
    def mode_ultrasonic(self):
        print("Ultrasonic Mode Running")

        self.servo.set_servo_pwm('0', self.car_sonic_servo_angle)

        if self.car_sonic_servo_angle == 30:
            self.car_sonic_distance[0] = self.sonic.get_distance()
        elif self.car_sonic_servo_angle == 90:
            self.car_sonic_distance[1] = self.sonic.get_distance()
        elif self.car_sonic_servo_angle == 150:
            self.car_sonic_distance[2] = self.sonic.get_distance()

        self.run_motor_ultrasonic(self.car_sonic_distance)

        if self.car_sonic_servo_angle <= 30:
            self.car_sonic_servo_dir = 1
        elif self.car_sonic_servo_angle >= 150:
            self.car_sonic_servo_dir = 0

        if self.car_sonic_servo_dir == 1:
            self.car_sonic_servo_angle += 60
        else:
            self.car_sonic_servo_angle -= 60

    def mode_infrared(self):
        if (time.time() - self.car_record_time) > 0.2:
            self.car_record_time = time.time()
            infrared_value = self.infrared.read_all_infrared()
            if infrared_value == 2:
                self.set_motors(800, 800, 800, 800)
            elif infrared_value == 4:
                self.set_motors(-1500, -1500, 2500, 2500)
            elif infrared_value == 6:
                self.set_motors(-2000, -2000, 4000, 4000)
            elif infrared_value == 1:
                self.set_motors(2500, 2500, -1500, -1500)
            elif infrared_value == 3:
                self.set_motors(4000, 4000, -2000, -2000)
            elif infrared_value == 7:
                self.set_motors(0, 0, 0, 0)

    def mode_light(self):
        if (time.time() - self.car_record_time) > 0.2:
            self.car_record_time = time.time()
            self.set_motors(0, 0, 0, 0)
            L = self.adc.read_adc(0)
            R = self.adc.read_adc(1)
            if L < 2.99 and R < 2.99:
                self.set_motors(600, 600, 600, 600)
            elif abs(L - R) < 0.15:
                self.set_motors(0, 0, 0, 0)
            elif L > 3 or R > 3:
                if L > R:
                    self.set_motors(-1200, -1200, 1400, 1400)
                elif R > L:
                    self.set_motors(1400, 1400, -1200, -1200)

    def mode_rotate(self, n):
        angle = n
        bat_compensate = 7.5 / (self.adc.read_adc(2) * (3 if self.adc.pcb_version == 1 else 2))
        while True:
            W = 2000
            VY = int(2000 * math.cos(math.radians(angle)))
            VX = -int(2000 * math.sin(math.radians(angle)))
            FR = VY - VX + W
            FL = VY + VX - W
            BL = VY - VX - W
            BR = VY + VX + W
            print("rotating")
            self.set_motors(FL, BL, FR, BR)
            time.sleep(5 * self.time_compensate * bat_compensate / 1000)
            angle -= 5
