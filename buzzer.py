import time
from gpiozero import OutputDevice


class Buzzer:
    _PIN = 17

    def __init__(self):
        self._pin = OutputDevice(self._PIN)

    def set_state(self, state: bool) -> None:
        self._pin.on() if state else self._pin.off()

    def close(self) -> None:
        self._pin.close()


if __name__ == "__main__":
    b = Buzzer()
    try:
        for _ in range(3):
            b.set_state(True)
            time.sleep(0.1)
            b.set_state(False)
            time.sleep(0.1)
    finally:
        b.close()
