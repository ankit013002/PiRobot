# pet_server_bridge.py
import os, time, json, uuid, requests, subprocess, base64

class PetServerBridge:
    def __init__(self):
        self.base = os.environ.get("PET_SERVER_URL", "").rstrip("/")
        self.key = os.environ.get("PI_API_KEY", "")
        self.enabled = bool(self.base)
        self.last_ok = 0.0
        self.tts_path = os.environ.get("PET_TTS_PATH", "/pet/tts")
        self.aplay_device = os.environ.get("PET_APLAY_DEVICE", "") or os.environ.get("ASSIST_APLAY_DEVICE", "")

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["X-API-Key"] = self.key
        return h

    def step(self, snapshot: dict, timeout=0.65):
        if not self.enabled:
            return None
        snapshot = dict(snapshot)
        snapshot["id"] = snapshot.get("id") or uuid.uuid4().hex[:10]
        snapshot["ts"] = snapshot.get("ts") or time.time()

        try:
            r = requests.post(
                self.base + "/pet/step",
                headers=self._headers(),
                data=json.dumps(snapshot),
                timeout=timeout,
            )
            if r.ok:
                out = r.json()
                if out.get("ok"):
                    self.last_ok = time.time()
                return out
            return None
        except Exception:
            return None

    def tts(self, text: str, timeout=12.0) -> bytes | None:
        if not self.enabled:
            return None
        text = (text or "").strip()
        if not text:
            return None

        url = self.base + self.tts_path
        try:
            r = requests.post(url, headers=self._headers(), json={"text": text}, timeout=timeout)
            if not r.ok:
                return None

            ct = (r.headers.get("Content-Type") or "").lower()
            if "application/json" in ct:
                data = r.json()
                b64 = data.get("wav_b64") or data.get("audio_b64")
                return base64.b64decode(b64) if b64 else None

            return r.content
        except Exception:
            return None

    def play_wav_bytes(self, wav_bytes: bytes):
        try:
            args = ["aplay", "-q", "-t", "wav"]
            if self.aplay_device:
                args += ["-D", self.aplay_device]
            args += ["-"]
            subprocess.run(args, input=wav_bytes, check=False)
        except Exception:
            pass
