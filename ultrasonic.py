import time
import warnings
from gpiozero import DistanceSensor, PWMSoftwareFallback, DistanceSensorNoEcho


class Ultrasonic:
    def __init__(self, trigger_pin: int = 27, echo_pin: int = 22, max_distance: float = 3.0):
        warnings.filterwarnings("ignore", category=DistanceSensorNoEcho)
        warnings.filterwarnings("ignore", category=PWMSoftwareFallback)
        self._sensor = DistanceSensor(echo=echo_pin, trigger=trigger_pin, max_distance=max_distance)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def get_distance(self) -> float | None:
        """Return distance in centimetres, or None on error."""
        try:
            return round(self._sensor.distance * 100, 1)
        except RuntimeWarning as e:
            print(f"[Ultrasonic] {e}")
            return None

    def close(self) -> None:
        self._sensor.close()


if __name__ == "__main__":
    with Ultrasonic() as us:
        try:
            while True:
                d = us.get_distance()
                if d is not None:
                    print(f"{d} cm")
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
