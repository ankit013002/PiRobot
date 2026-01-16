# voice_command.py
import json
import threading
from queue import Queue, Empty

import sounddevice as sd
from vosk import Model, KaldiRecognizer


class VoiceCommandListener:
    """
    Offline Vosk listener that:
      - queues ALL final transcripts (poll_text)
      - optionally queues recognized commands FOLLOW/ROAM/STOP (poll_cmd)
    """

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        device: int | None = None,
        blocksize: int = 8000,
        command_words=("FOLLOW", "ROAM", "STOP"),
        print_heard: bool = True,
    ):
        self.model_path = model_path
        self.sample_rate = int(sample_rate)
        self.device = device
        self.blocksize = int(blocksize)
        self.command_words = {w.upper() for w in command_words}
        self.print_heard = bool(print_heard)

        self._model = Model(model_path)
        self._rec = KaldiRecognizer(self._model, self.sample_rate)

        self._text_q: Queue[str] = Queue()
        self._cmd_q: Queue[str] = Queue()

        self._running = False
        self._stream = None
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self._running:
                return
            self._running = True

            def _callback(indata, frames, time_info, status):
                if not self._running:
                    return
                if status:
                    # not fatal; device underruns etc
                    pass

                data = bytes(indata)
                if self._rec.AcceptWaveform(data):
                    try:
                        res = json.loads(self._rec.Result())
                        text = (res.get("text") or "").strip()
                    except Exception:
                        text = ""

                    if text:
                        if self.print_heard:
                            print(f"[VOICE] Heard: {text}", flush=True)

                        # queue raw transcript
                        self._text_q.put(text)

                        # queue command if it matches exactly
                        up = text.strip().upper()
                        if up in self.command_words:
                            self._cmd_q.put(up)

            # RawInputStream gives bytes directly (best for Vosk)
            self._stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                device=self.device,
                dtype="int16",
                channels=1,
                callback=_callback,
            )
            self._stream.start()

            dev = self.device if self.device is not None else "default"
            print(f"[VOICE] Listening... sr={self.sample_rate} block={self.blocksize} dev={dev}", flush=True)

    def stop(self):
        with self._lock:
            self._running = False
            try:
                if self._stream is not None:
                    self._stream.stop()
                    self._stream.close()
            except Exception:
                pass
            self._stream = None

    def poll_text(self) -> str | None:
        try:
            return self._text_q.get_nowait()
        except Empty:
            return None

    def poll_cmd(self) -> str | None:
        try:
            return self._cmd_q.get_nowait()
        except Empty:
            return None

    # Optional convenience: returns either a command (preferred) or raw text
    def poll(self) -> str | None:
        c = self.poll_cmd()
        if c is not None:
            return c
        return self.poll_text()
