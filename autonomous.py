#!/usr/bin/env python3
import time
from car import Car
from led import Led
from buzzer import Buzzer

# Be more lenient so it will actually move.
# You can tweak this later once you're confident in the voltage reading.
LOW_BATTERY_THRESHOLD = 5.0   # volts
STATUS_PRINT_INTERVAL = 1.0   # seconds

def main():
    print("Starting autonomous mode (ultrasonic obstacle avoidance)...")
    car = Car()
    led = Led()
    buzzer = Buzzer()

    last_status_print = 0.0

    try:
        # Turn LEDs green to show “autonomous mode”
        led.ledIndex(0xFF, 0, 255, 0)

        while True:
            # --- Battery safety (lenient) ---
            power_raw = car.adc.read_adc(2)
            power = power_raw * (3 if car.adc.pcb_version == 1 else 2) if power_raw is not None else None

            print("Power:", power)

            if power is not None and power < LOW_BATTERY_THRESHOLD:
                print(f"[WARN] Low battery: {power:.2f} V – stopping motors.")
                car.motor.set_motor_model(0, 0, 0, 0)
                buzzer.set_state(True)
                led.ledIndex(0xFF, 255, 0, 0)  # red
                time.sleep(2)
                buzzer.set_state(False)
                break

            # --- Core autonomy: obstacle avoidance ---
            forward = car.get_forward_distance()
            moving_forward = False

            if forward is not None and forward < 25:
                car.motor.set_motor_model(0, 0, 0, 0)
                car.scan_and_avoid()
            else:
                car.motor.set_motor_model(600, 600, 600, 600)
                moving_forward = True

            # --- STUCK DETECTION ---
            if car.detect_stuck(forward, moving_forward):
                print("[WARN] Car stuck! Executing escape maneuver.")
                buzzer.set_state(True)
                car.escape_stuck()
                buzzer.set_state(False)



            # Read distance for logging / buzzer behavior
            distance = car.sonic.get_distance()
            if distance is not None and distance < 15:
                buzzer.set_state(True)
            else:
                buzzer.set_state(False)

            # Periodic status print
            now = time.time()
            if now - last_status_print > STATUS_PRINT_INTERVAL:
                if power is not None:
                    print(f"[INFO] Battery: {power:.2f} V, Distance: {distance}")
                else:
                    print(f"[INFO] Battery: <None>, Distance: {distance}")
                last_status_print = now

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received, stopping autonomous mode...")

    finally:
        # Clean shutdown
        buzzer.set_state(False)
        led.colorBlink(0)     # turn off LEDs
        car.close()
        print("[INFO] Autonomous mode stopped, resources cleaned up.")

if __name__ == "__main__":
    main()
