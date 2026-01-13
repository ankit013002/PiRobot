from pca9685 import PCA9685
import json
import os

def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))

def _load_servo_trim(path: str = "params.json") -> dict:
    """
    Loads optional per-servo trim from params.json:
      "Servo_Trim": { "0": 0, "1": -8 }
    """
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            data = json.load(f)
        trim = data.get("Servo_Trim", {})
        if isinstance(trim, dict):
            # ensure ints
            out = {}
            for k, v in trim.items():
                try:
                    out[str(k)] = int(v)
                except:
                    pass
            return out
    except Exception as e:
        print(f"[Servo] Warning: could not load Servo_Trim from {path}: {e}")
    return {}

class Servo:
    def __init__(self):
        self.pwm_frequency = 50
        self.initial_pulse = 1500
        self.pwm_channel_map = {
            '0': 8,
            '1': 9,
            '2': 10,
            '3': 11,
            '4': 12,
            '5': 13,
            '6': 14,
            '7': 15
        }

        # Per-servo trim in *degrees* (added to commanded angle)
        self.trim_deg = {ch: 0 for ch in self.pwm_channel_map.keys()}
        self.trim_deg.update(_load_servo_trim("params.json"))

        self.pwm_servo = PCA9685(0x40, debug=True)
        self.pwm_servo.set_pwm_freq(self.pwm_frequency)
        # Center only the “head” servos using set_servo_pwm so trim applies.
        for ch in ("0", "1"):
            try:
                self.set_servo_pwm(ch, 120)
            except Exception as e:
                print(f"[Servo] init center failed for {ch}: {e}")


    def set_servo_pwm(self, channel: str, angle: int, error: int = 0) -> None:
        """
        channel: '0'..'7'
        angle:   0..180 (logical angle)
        error:   legacy global offset used in original kit code
        trim:    per-channel degrees from params.json "Servo_Trim"
        """
        channel = str(channel)
        if channel not in self.pwm_channel_map:
            raise ValueError(
                f"Invalid channel: {channel}. Valid channels are {list(self.pwm_channel_map.keys())}."
            )

        angle = int(angle)

        # Apply per-channel trim FIRST (so you can fix “head slanted down”)
        angle += int(self.trim_deg.get(channel, 0))

        # Keep existing legacy error behavior (your original code)
        angle += int(error)

        # Clamp to servo-safe range
        angle = _clamp(angle, 0, 180)

        # Original mapping (kit-specific)
        pulse = (
            2500 - int(angle / 0.09)
            if channel == '0'
            else 500 + int(angle / 0.09)
        )

        self.pwm_servo.set_servo_pulse(self.pwm_channel_map[channel], pulse)

# Main program logic follows:
if __name__ == '__main__':
    print("Now servos will rotate to 90 degree.")
    print("Please keep the program running when installing the servos.")
    print("After that, you can press ctrl-C to end the program.")
    pwm_servo = Servo()
    try:
        while True:
            pwm_servo.set_servo_pwm('0', 90)
            pwm_servo.set_servo_pwm('1', 90)
    except KeyboardInterrupt:
        print("\nEnd the program")
