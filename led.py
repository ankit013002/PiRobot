import time
from parameter import ParameterManager
from rpi_ledpixel import Freenove_RPI_WS281X
from spi_ledpixel import Freenove_SPI_LedPixel


class Led:
    """NeoPixel LED strip controller (8 pixels)."""

    def __init__(self):
        params  = ParameterManager()
        connect = params.get_connect_version()
        pi      = params.get_pi_version()

        if connect == 1 and pi == 1:
            self.strip = Freenove_RPI_WS281X(8, 255, 'RGB')
            self._ok   = True
        elif connect == 2 and pi in (1, 2):
            self.strip = Freenove_SPI_LedPixel(8, 255, 'GRB')
            self._ok   = True
        else:
            print(f"[LED] Unsupported combination Connect={connect} Pi={pi}. LEDs disabled.")
            self.strip = None
            self._ok   = False

        self._start = time.time()
        self._next  = 0.0

        # Animation state
        self._wheel_val        = 100
        self._chase_idx        = 0
        self._wipe_idx         = 0
        self._breathe_bright   = 0

    def _guard(self) -> bool:
        return self._ok

    # ── Per-frame animations (call in a tight loop) ───────────────────────────

    def colorBlink(self, state: int = 1, wait_ms: float = 300) -> None:
        """Cycle through a simple 4-step color sequence. state=0 to clear."""
        if not self._guard():
            return
        if not state:
            self.strip.set_all_led_color(0, 0, 0)
            self.strip.show()
            return
        now = time.time()
        if now - self._start > wait_ms / 1000.0:
            self._start = now
            palette = [(255, 0, 0), (0, 0, 0), (0, 255, 0), (0, 0, 0)]
            color   = palette[self._wipe_idx % 4]
            for i in range(self.strip.get_led_count()):
                self.strip.set_led_rgb_data(i, color)
            self.strip.show()
            self._wipe_idx += 1

    def following(self, wait_ms: float = 50) -> None:
        """Single rainbow pixel chasing around the strip."""
        if not self._guard():
            return
        now = time.time()
        if now - self._start > wait_ms / 1000.0:
            self._start = now
            for i in range(self.strip.get_led_count()):
                self.strip.set_led_rgb_data(i, [0, 0, 0])
            self.strip.set_led_rgb_data(self._chase_idx, self._wheel(self._wheel_val & 255))
            self.strip.show()
            self._chase_idx   = (self._chase_idx + 1) % self.strip.get_led_count()
            self._wheel_val   = (self._wheel_val + 5) % 256

    def rainbowbreathing(self, wait_ms: float = 10) -> None:
        """Whole strip breathes through rainbow colors."""
        if not self._guard():
            return
        now = time.time()
        if now - self._start > wait_ms / 1000.0:
            self._start = now
            base  = self._wheel(self._wheel_val & 255)
            phase = self._breathe_bright % 200
            b     = phase if phase <= 100 else 200 - phase
            color = [int(c * b / 100) for c in base]
            for i in range(self.strip.get_led_count()):
                self.strip.set_led_rgb_data(i, color)
            self.strip.show()
            self._breathe_bright += 1
            if self._breathe_bright >= 200:
                self._breathe_bright = 0
                self._wheel_val = (self._wheel_val + 32) % 256

    def rainbowCycle(self, wait_ms: float = 20) -> None:
        """Rainbow distributed evenly across all pixels."""
        if not self._guard():
            return
        now = time.time()
        if now - self._start > wait_ms / 1000.0:
            self._start = now
            n = self.strip.get_led_count()
            for i in range(n):
                self.strip.set_led_rgb_data(
                    i, self._wheel((int(i * 256 / n) + self._wheel_val) & 255))
            self.strip.show()
            self._wheel_val = (self._wheel_val + 1) % 256

    # ── Direct control ────────────────────────────────────────────────────────

    def ledIndex(self, index: int, R: int, G: int, B: int) -> None:
        """
        Set pixels selected by a bitmask.
        index=0xFF sets all 8 pixels to (R, G, B) in a single SPI write.
        """
        if not self._guard():
            return
        color = (R, G, B)
        mask  = index
        for i in range(self.strip.get_led_count()):
            if mask & 0x01:
                self.strip.set_led_rgb_data(i, color)
            mask >>= 1
        self.strip.show()  # single flush after setting all pixels

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _wheel(pos: int) -> tuple:
        """Map 0–255 to an (R, G, B) rainbow color."""
        if pos < 85:
            return (pos * 3, 255 - pos * 3, 0)
        elif pos < 170:
            pos -= 85
            return (255 - pos * 3, 0, pos * 3)
        else:
            pos -= 170
            return (0, pos * 3, 255 - pos * 3)


if __name__ == '__main__':
    led = Led()
    animations = [
        ("colorBlink",      lambda: led.colorBlink(1)),
        ("following",       lambda: led.following(50)),
        ("rainbowbreathing",lambda: led.rainbowbreathing(10)),
        ("rainbowCycle",    lambda: led.rainbowCycle(20)),
    ]
    try:
        for name, fn in animations:
            print(name)
            t0 = time.time()
            while time.time() - t0 < 5:
                fn()
        led.colorBlink(0)
    except KeyboardInterrupt:
        led.colorBlink(0)
