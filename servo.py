import json
import os
from pca9685 import PCA9685


# Logical channel index → PCA9685 physical channel (servos on channels 8–15)
_CHANNEL_MAP: dict[str, int] = {str(i): 8 + i for i in range(8)}


def _load_trim(path: str = 'params.json') -> dict[str, int]:
    """Load per-servo trim offsets (degrees) from params.json."""
    try:
        if os.path.exists(path):
            with open(path) as f:
                raw = json.load(f).get('Servo_Trim', {})
            return {str(k): int(v) for k, v in raw.items() if str(v).lstrip('-').isdigit()}
    except Exception:
        pass
    return {}


class Servo:
    """
    Pan/tilt servo driver.

    Logical channels 0–7 map to PCA9685 channels 8–15.
    Channel 0 (pan)  pulse formula: 2500 − angle/0.09  (inverted direction)
    Channel 1+ (tilt) pulse formula: 500  + angle/0.09
    """

    def __init__(self):
        self._trim = _load_trim()
        self._pwm  = PCA9685(0x40)
        self._pwm.set_pwm_freq(50)
        # Centre the head servos on startup
        for ch in ('0', '1'):
            try:
                self.set_servo_pwm(ch, 120)
            except Exception as e:
                print(f"[Servo] init warning ch{ch}: {e}")

    def set_servo_pwm(self, channel: str, angle: int) -> None:
        """
        Move servo on logical channel to angle (0–180°).
        Per-channel trim from params.json is applied automatically.
        """
        ch = str(channel)
        if ch not in _CHANNEL_MAP:
            raise ValueError(f"Invalid servo channel '{ch}'. Valid: {list(_CHANNEL_MAP)}")

        angle = max(0, min(180, int(angle) + self._trim.get(ch, 0)))
        pulse = 2500 - int(angle / 0.09) if ch == '0' else 500 + int(angle / 0.09)
        self._pwm.set_servo_pulse(_CHANNEL_MAP[ch], pulse)


if __name__ == '__main__':
    print("Centering servos at 90°. Press Ctrl-C to exit.")
    s = Servo()
    try:
        while True:
            s.set_servo_pwm('0', 90)
            s.set_servo_pwm('1', 90)
    except KeyboardInterrupt:
        print("Done.")
