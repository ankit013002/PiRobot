#!/usr/bin/env python3
import time
import os
import cv2
import numpy as np

from car import Car
from led import Led
from buzzer import Buzzer
from camera import Camera
from vision_perception import VisionPerception
from mapping import render_occupancy_map, save_map_png

SHOW_VIEW = os.environ.get("DISPLAY") is not None  # auto-disable if headless

SHOW_MAP = SHOW_VIEW  # only show if display exists
MAP_REFRESH = 0.25
MAP_SAVE_INTERVAL = 2.0

LOW_BATTERY_THRESHOLD = 5.0
STATUS_PRINT_INTERVAL = 1.0

# Vision tuning
VISION_RISK_TURN = 0.80
VISION_RISK_SLOW = 0.65
VISUAL_STUCK_MOTION_THRESH = 0.35   # tune in your environment
VISUAL_STUCK_TIMEOUT = 0.9          # seconds


def main():
    print("Starting FULL autonomous mode (ultrasonic + memory + vision)...")

    car = Car()
    led = Led()
    buzzer = Buzzer()

    # Camera + perception
    cam = Camera(stream_size=(320, 240))     # smaller stream = faster
    cam.start_stream()                       # starts JPEG frames into StreamingOutput

    if SHOW_VIEW:
        cv2.namedWindow("Car Camera", cv2.WINDOW_NORMAL)

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
        tilt_angles=(car.TILT_UP, car.TILT_CENTER, car.TILT_DOWN, car.TILT_CENTER),
        head_sweep_period=1.2,
    )
    vision.start()

    last_status_print = 0.0

    # Visual stuck tracking
    visual_stuck_start = None

    # ✅ init these ONCE (before loop)
    last_map_ts = 0.0
    last_map_save = 0.0
    last_wide_scan = 0.0

    # Create map window once (if you have a display)
    if SHOW_MAP:
        cv2.namedWindow("Occupancy Map", cv2.WINDOW_NORMAL)

    try:
        led.ledIndex(0xFF, 0, 255, 0)

        while True:
            now = time.time()

            # Battery safety
            power_raw = car.adc.read_adc(2)
            power = power_raw * (3 if car.adc.pcb_version == 1 else 2) if power_raw is not None else None
            if power is not None and power < LOW_BATTERY_THRESHOLD:
                print(f"[WARN] Low battery: {power:.2f} V - stopping motors.")
                car.set_motors(0, 0, 0, 0)
                buzzer.set_state(True)
                led.ledIndex(0xFF, 255, 0, 0)
                time.sleep(2)
                buzzer.set_state(False)
                break

            # Read sensors
            forward = car.get_forward_distance()  # ultrasonic forward
            vs = vision.get_state()

            # Human detection behavior (safe default)
            if vs.person_count > 0:
                if vs.person_close:
                    car.set_motors(0, 0, 0, 0)
                    buzzer.set_state(True)
                    led.ledIndex(0xFF, 0, 0, 255)
                    time.sleep(0.25)
                    buzzer.set_state(False)
                    led.ledIndex(0xFF, 0, 255, 0)

            # Face behavior
            if vs.face_count > 0 and vs.collision_risk > 0.70:
                car.set_motors(0, 0, 0, 0)
                buzzer.set_state(True)
                led.ledIndex(0xFF, 0, 0, 255)
                time.sleep(0.25)
                buzzer.set_state(False)
                led.ledIndex(0xFF, 0, 255, 0)

            # ---- Update memory from forward ----
            if forward is not None:
                car.mem.update_from_scan(car.pose, [90], [forward])

                # occasional wide scan when things are open
                if forward > 60 and (now - last_wide_scan) > 3.5:
                    angs, dists = car.scan_distances(
                        angles=(30, 60, 90, 120, 150),
                        settle=0.04,
                        samples=1
                    )
                    car.mem.update_from_scan(car.pose, angs, dists)
                    last_wide_scan = now

            # ---- Primary drive / avoidance decision ----
            if forward is not None and forward < 20:
                car.set_motors(0, 0, 0, 0)
                car.scan_and_avoid_with_memory()
            else:
                # Ultrasonic seems clear -> use vision risk to decide speed/turn
                if vs.collision_risk > VISION_RISK_TURN:
                    car.set_motors(0, 0, 0, 0)
                    if vs.free_dir == "L":
                        car.set_motors(-1500, -1500, 1500, 1500)
                    elif vs.free_dir == "R":
                        car.set_motors(1500, 1500, -1500, -1500)
                    else:
                        car.scan_and_avoid_with_memory()
                    time.sleep(0.25)
                    car.set_motors(0, 0, 0, 0)

                elif vs.collision_risk > VISION_RISK_SLOW:
                    car.set_motors(650, 650, 650, 650)
                else:
                    car.set_motors(1100, 1100, 1100, 1100)

            moving_forward = car.is_commanding_forward()

            # ---- Stuck detection (fusion) ----
            if car.detect_stuck(forward, moving_forward):
                print("[WARN] Ultrasonic-based stuck! Executing escape.")
                buzzer.set_state(True)
                car.escape_stuck_with_memory()
                buzzer.set_state(False)
                visual_stuck_start = None

            if moving_forward:
                if vs.motion_score < VISUAL_STUCK_MOTION_THRESH:
                    if visual_stuck_start is None:
                        visual_stuck_start = time.time()
                    elif (time.time() - visual_stuck_start) > VISUAL_STUCK_TIMEOUT:
                        print("[WARN] Vision-based stuck! Executing escape.")
                        buzzer.set_state(True)
                        car.escape_stuck_with_memory()
                        buzzer.set_state(False)
                        visual_stuck_start = None
                else:
                    visual_stuck_start = None
            else:
                visual_stuck_start = None

            # ✅ Reuse forward for buzzer (don’t re-trigger sonar here)
            buzzer.set_state(bool(forward is not None and forward < 15))

            # ---- Map display/save (independent, NO else) ----
            if SHOW_MAP and (now - last_map_ts) > MAP_REFRESH:
                map_img = render_occupancy_map(car.mem, car.pose, scale=6, pad=8, draw_grid=False)
                cv2.imshow("Occupancy Map", map_img)
                cv2.waitKey(1)
                last_map_ts = now

            if (now - last_map_save) > MAP_SAVE_INTERVAL:
                map_img = render_occupancy_map(car.mem, car.pose, scale=6, pad=8, draw_grid=False)
                save_map_png("maps/map_latest.png", map_img)
                last_map_save = now

            if now - last_status_print > STATUS_PRINT_INTERVAL:
                pose = car.pose_string()
                print(
                    f"[INFO] Bat={power:.2f}V  US_fwd={forward}  "
                    f"persons={vs.person_count} close={vs.person_close} tilt={vs.head_tilt_deg}  "
                    f"risk={vs.collision_risk:.2f} free={vs.free_dir} motion={vs.motion_score:.2f} "
                    f"faces={vs.face_count}"
                )
                last_status_print = now

            # ---- Live camera window ----
            if SHOW_VIEW:
                jpg = cam.get_frame(timeout=0.001)
                if jpg:
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        cv2.putText(
                            img,
                            f"risk={vs.collision_risk:.2f} free={vs.free_dir} motion={vs.motion_score:.2f} faces={vs.face_count}",
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

            time.sleep(0.05)


    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received, stopping autonomous mode...")

    finally:
        try:
            if SHOW_VIEW:
                cv2.destroyAllWindows()
        except:
            pass
        try:
            vision.stop()
        except:
            pass
        try:
            cam.stop_stream()
            cam.close()
        except:
            pass

        buzzer.set_state(False)
        led.colorBlink(0)
        car.close()
        print("[INFO] Full autonomous mode stopped, resources cleaned up.")


if __name__ == "__main__":
    main()
