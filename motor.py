import time
from pca9685 import PCA9685


# PCA9685 channel pairs for each wheel: (forward_ch, reverse_ch)
# Duties are inverted before dispatch to match physical motor wiring.
_WHEEL_CHANNELS = (
    (1, 0),  # front-left
    (2, 3),  # back-left
    (7, 6),  # front-right
    (5, 4),  # back-right
)


class MotorController:
    """4-wheel DC motor driver via PCA9685 PWM."""

    def __init__(self):
        self.pwm = PCA9685(0x40)
        self.pwm.set_pwm_freq(50)

    def set_motor_model(self, fl: int, bl: int, fr: int, br: int) -> None:
        """
        Drive all four wheels.  Positive = forward, negative = reverse.
        Values are clamped to ±4095 (12-bit PWM range).
        """
        for raw, (fwd, rev) in zip((fl, bl, fr, br), _WHEEL_CHANNELS):
            duty = max(-4095, min(4095, int(raw)))
            duty = -duty  # hardware polarity inversion
            if duty > 0:
                self.pwm.set_motor_pwm(rev, 0)
                self.pwm.set_motor_pwm(fwd, duty)
            elif duty < 0:
                self.pwm.set_motor_pwm(fwd, 0)
                self.pwm.set_motor_pwm(rev, -duty)
            else:
                self.pwm.set_motor_pwm(fwd, 4095)
                self.pwm.set_motor_pwm(rev, 4095)

    def close(self) -> None:
        self.set_motor_model(0, 0, 0, 0)
        self.pwm.close()


if __name__ == '__main__':
    m = MotorController()
    try:
        for label, args in [
            ("Forward",  ( 2000,  2000,  2000,  2000)),
            ("Reverse",  (-2000, -2000, -2000, -2000)),
            ("Spin-L",   (-2000, -2000,  2000,  2000)),
            ("Spin-R",   ( 2000,  2000, -2000, -2000)),
        ]:
            print(label)
            m.set_motor_model(*args)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        m.close()
