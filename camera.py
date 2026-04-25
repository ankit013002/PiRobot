import io
import time
from threading import Condition

from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder, JpegEncoder
from picamera2.outputs import FileOutput
from libcamera import Transform


class _StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame: bytes | None = None
        self.condition = Condition()

    def write(self, buf: bytes) -> int:
        with self.condition:
            self.frame = buf
            self.condition.notify_all()
        return len(buf)


class Camera:
    def __init__(self, preview_size: tuple = (640, 480),
                 hflip: bool = False, vflip: bool = False,
                 stream_size: tuple = (400, 300)):
        self.camera = Picamera2()
        transform = Transform(hflip=int(hflip), vflip=int(vflip))

        preview_cfg = self.camera.create_preview_configuration(
            main={"size": preview_size}, transform=transform)
        self.camera.configure(preview_cfg)

        self._stream_cfg = self.camera.create_video_configuration(
            main={"size": stream_size}, transform=transform)
        self._output = _StreamingOutput()
        self.streaming = False

    def start_image(self) -> None:
        self.camera.start_preview(Preview.QTGL)
        self.camera.start()

    def save_image(self, filename: str) -> dict | None:
        try:
            return self.camera.capture_file(filename)
        except Exception as e:
            print(f"[Camera] capture error: {e}")
            return None

    def start_stream(self) -> None:
        if self.streaming:
            return
        if self.camera.started:
            self.camera.stop()
        self.camera.configure(self._stream_cfg)
        self.camera.start_recording(JpegEncoder(), FileOutput(self._output))
        self.streaming = True

    def stop_stream(self) -> None:
        if not self.streaming:
            return
        try:
            self.camera.stop_recording()
        except Exception as e:
            print(f"[Camera] stop_recording error: {e}")
        finally:
            self.streaming = False

    def get_frame(self, timeout: float = 1.0) -> bytes | None:
        with self._output.condition:
            if not self._output.condition.wait(timeout=timeout):
                return None
            return self._output.frame

    def close(self) -> None:
        self.stop_stream()
        self.camera.close()


if __name__ == "__main__":
    cam = Camera()
    cam.start_image()
    time.sleep(10)
    cam.save_image("image.jpg")
    cam.close()
