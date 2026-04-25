import smbus
import time
from parameter import ParameterManager


class ADC:
    """ADS7830 8-channel, 8-bit ADC over I2C (address 0x48)."""

    _I2C_ADDR   = 0x48
    _CMD_BASE   = 0x84  # single-ended, internal ref, ADC on

    def __init__(self):
        params = ParameterManager()
        self.pcb_version = params.get_pcb_version()
        # PCB v1: 3.3 V reference; PCB v2: 5.2 V reference
        self._v_ref = 3.3 if self.pcb_version == 1 else 5.2
        self._bus = smbus.SMBus(1)

    def read_adc(self, channel: int) -> float:
        """Return voltage on the given channel (0–7), rounded to 2 dp."""
        cmd = self._CMD_BASE | (((channel << 2) | (channel >> 1)) & 0x07) << 4
        self._bus.write_byte(self._I2C_ADDR, cmd)
        raw = self._read_stable()
        return round(raw / 255.0 * self._v_ref, 2)

    def _read_stable(self, max_tries: int = 20) -> int:
        """Read two consecutive identical bytes to filter ADC noise."""
        v1 = self._bus.read_byte(self._I2C_ADDR)
        for _ in range(max_tries):
            v2 = self._bus.read_byte(self._I2C_ADDR)
            if v1 == v2:
                return v1
            v1 = v2
        return v1  # best effort

    def close_i2c(self) -> None:
        self._bus.close()


if __name__ == '__main__':
    adc = ADC()
    try:
        while True:
            l = adc.read_adc(0)
            r = adc.read_adc(1)
            batt = adc.read_adc(2) * (3 if adc.pcb_version == 1 else 2)
            print(f"Light L={l:.2f}V  R={r:.2f}V  Battery={batt:.2f}V")
            time.sleep(1)
    except KeyboardInterrupt:
        adc.close_i2c()
