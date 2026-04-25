import time
import math
import smbus


class PCA9685:
    """16-channel 12-bit PWM driver over I2C (NXP PCA9685)."""

    _MODE1     = 0x00
    _PRESCALE  = 0xFE
    _LED0_ON_L = 0x06

    def __init__(self, address: int = 0x40):
        self.bus     = smbus.SMBus(1)
        self.address = address
        self._write(self._MODE1, 0x00)

    def _write(self, reg: int, value: int) -> None:
        self.bus.write_byte_data(self.address, reg, value)

    def _read(self, reg: int) -> int:
        return self.bus.read_byte_data(self.address, reg)

    def set_pwm_freq(self, freq: float) -> None:
        """Set the PWM carrier frequency in Hz (typ. 50 Hz for servos)."""
        prescale = round(25_000_000.0 / (4096.0 * float(freq))) - 1
        old_mode = self._read(self._MODE1)
        self._write(self._MODE1, (old_mode & 0x7F) | 0x10)   # sleep
        self._write(self._PRESCALE, prescale)
        self._write(self._MODE1, old_mode)
        time.sleep(0.005)
        self._write(self._MODE1, old_mode | 0x80)             # restart

    def set_pwm(self, channel: int, on: int, off: int) -> None:
        base = self._LED0_ON_L + 4 * channel
        self._write(base,     on  & 0xFF)
        self._write(base + 1, on  >> 8)
        self._write(base + 2, off & 0xFF)
        self._write(base + 3, off >> 8)

    def set_motor_pwm(self, channel: int, duty: int) -> None:
        self.set_pwm(channel, 0, duty)

    def set_servo_pulse(self, channel: int, pulse: float) -> None:
        """Set servo pulse width in microseconds (PWM freq must be 50 Hz)."""
        self.set_pwm(channel, 0, int(pulse * 4096 / 20000))

    def close(self) -> None:
        self.bus.close()
