import os
import sys
import json
import subprocess


_PARAM_FILE = 'params.json'
_REQUIRED   = {'Connect_Version': [1, 2], 'Pcb_Version': [1, 2], 'Pi_Version': [1, 2]}


def _detect_pi_version() -> int:
    try:
        result = subprocess.run(
            ['cat', '/sys/firmware/devicetree/base/model'],
            capture_output=True, text=True
        )
        if result.returncode == 0 and 'Raspberry Pi 5' in result.stdout:
            return 2
    except Exception:
        pass
    return 1


def _load() -> dict:
    try:
        with open(_PARAM_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save(params: dict) -> None:
    with open(_PARAM_FILE, 'w') as f:
        json.dump(params, f, indent=4)


def _valid(params: dict) -> bool:
    return all(params.get(k) in v for k, v in _REQUIRED.items())


def _prompt_int(prompt: str, valid: list) -> int:
    while True:
        try:
            v = int(input(prompt))
            if v in valid:
                return v
            print(f"  Enter one of {valid}.")
        except ValueError:
            print("  Enter a number.")


class ParameterManager:
    """
    Manages hardware version config in params.json.
    In headless environments (no TTY), missing or invalid params fall back
    to safe defaults so the robot can boot without hanging on input().
    """

    def __init__(self):
        params = _load()
        if not _valid(params):
            params = self._initialise(params)

    def _initialise(self, existing: dict) -> dict:
        pi = _detect_pi_version()

        if sys.stdin.isatty():
            print(f"[Config] Hardware config missing or invalid.")
            connect = _prompt_int("  Connect Version (1=GPIO, 2=SPI LEDs): ", [1, 2])
            pcb     = _prompt_int("  PCB Version (1=3.3 V ADC, 2=5.2 V ADC): ", [1, 2])
        else:
            connect = existing.get('Connect_Version', 2)
            pcb     = existing.get('Pcb_Version', 1)
            if connect not in (1, 2):
                connect = 2
            if pcb not in (1, 2):
                pcb = 1
            print(f"[Config] Using defaults Connect={connect}, PCB={pcb}, Pi={pi}")

        params = {'Connect_Version': connect, 'Pcb_Version': pcb, 'Pi_Version': pi}
        _save(params)
        return params

    # ── Accessors ────────────────────────────────────────────────────────────

    def get_connect_version(self) -> int:
        return _load().get('Connect_Version', 2)

    def get_pcb_version(self) -> int:
        return _load().get('Pcb_Version', 1)

    def get_pi_version(self) -> int:
        return _load().get('Pi_Version', 1)

    def get_raspberry_pi_version(self) -> int:
        return _detect_pi_version()

    def set_param(self, name: str, value) -> None:
        params = _load()
        params[name] = value
        _save(params)

    def get_param(self, name: str):
        return _load().get(name)


if __name__ == '__main__':
    m = ParameterManager()
    print(f"Connect={m.get_connect_version()}  PCB={m.get_pcb_version()}  Pi={m.get_pi_version()}")
