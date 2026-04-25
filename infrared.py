import time
from gpiozero import LineSensor


class Infrared:
    _IR_PINS = {1: 14, 2: 15, 3: 23}

    def __init__(self):
        self._sensors = {ch: LineSensor(pin) for ch, pin in self._IR_PINS.items()}

    def read_one_infrared(self, channel: int) -> int:
        if channel not in self._sensors:
            raise ValueError(f"Invalid channel {channel}. Valid: {list(self._IR_PINS)}")
        return 1 if self._sensors[channel].value else 0

    def read_all_infrared(self) -> int:
        """Returns a 3-bit int: bit2=ch1, bit1=ch2, bit0=ch3."""
        return (self.read_one_infrared(1) << 2 |
                self.read_one_infrared(2) << 1 |
                self.read_one_infrared(3))

    def close(self) -> None:
        for s in self._sensors.values():
            s.close()


if __name__ == "__main__":
    ir = Infrared()
    try:
        while True:
            print(f"IR: {ir.read_all_infrared():03b}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        ir.close()
