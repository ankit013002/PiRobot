import sys
import argparse
import struct
import time
import math
import signal
import threading
import multiprocessing
from threading import Thread
from typing import Optional

from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import QTimer

from server_ui import Ui_server_ui
from server import Server
from message import MessageParser
import command as CMD
from led import Led
from camera import Camera
from car import Car
from buzzer import Buzzer


class RobotServer(QMainWindow, Ui_server_ui):
    """
    GUI server window for the Freenove 4WD Smart Car.

    Manages five concurrent workers:
      cmd_thread   — receives and dispatches TCP commands
      video_thread — streams JPEG frames to video client
      car_thread   — runs autonomous sub-modes
      led_process  — runs LED animations in a separate process
    """

    def __init__(self):
        # Use an existing QApplication if one was already created (headless case)
        self.app = QApplication.instance() or QApplication(sys.argv)
        super().__init__()
        self.setupUi(self)

        self._init_hardware()
        self._init_threads()

        self.Button_Server.clicked.connect(self._toggle_server)
        self.app.lastWindowClosed.connect(self.close_application)
        signal.signal(signal.SIGINT, self._signal_handler)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._check_signals)
        self._timer.start(100)

        if self._auto_start:
            self._toggle_server()

    def _init_hardware(self) -> None:
        self.tcp_server = Server()
        self.led        = Led()
        self.car        = Car()
        self.buzzer     = Buzzer()
        self.camera     = Camera(stream_size=(400, 300))
        self._cmd_parse = MessageParser()
        self._led_parse = MessageParser()
        self._auto_start = True

    def _init_threads(self) -> None:
        self._queue_cmd = multiprocessing.Queue()
        self._queue_led = multiprocessing.Queue()

        self._cmd_thread:   Optional[Thread]              = None
        self._video_thread: Optional[Thread]              = None
        self._car_thread:   Optional[Thread]              = None
        self._led_proc:     Optional[multiprocessing.Process] = None

        self._cmd_running   = False
        self._video_running = False
        self._car_running   = False
        self._led_running   = False

        self._car_mode    = 1   # 1=manual 2=light 3=infrared 4=ultrasonic
        self._led_mode    = 0

        # Rotation thread management
        self._rotate_thread: Optional[Thread]           = None
        self._rotate_stop   = threading.Event()
        self._rotation_active = False

        # Telemetry rate-limiters
        self._t_sonic = time.time()
        self._t_light = time.time()
        self._t_line  = time.time()

    # ── Server on/off ─────────────────────────────────────────────────────────

    def _toggle_server(self) -> None:
        if self.label.text() == "Server Off":
            self.label.setText("Server On")
            self.Button_Server.setText("Off")
            self.tcp_server.start_tcp_servers()
            self._set_cmd_thread(True)
            self._set_video_thread(True)
            self._set_car_thread(True)
            self._set_led_process(True)
        else:
            self.label.setText("Server Off")
            self.Button_Server.setText("On")
            self.tcp_server.stop_tcp_servers()
            self._set_cmd_thread(False)
            self._set_video_thread(False)
            self._set_car_thread(False)
            self._set_led_process(False)
            self.tcp_server = Server()

    # ── Telemetry senders ─────────────────────────────────────────────────────

    def _send_sonic(self) -> None:
        if time.time() - self._t_sonic < 0.5:
            return
        self._t_sonic = time.time()
        if not self.tcp_server.get_command_server_busy():
            d = self.car.sonic.get_distance()
            self.tcp_server.send_data_to_command_client(
                f"{CMD.CMD_MODE}#3#{d:.2f}\n")

    def _send_light(self) -> None:
        if time.time() - self._t_light < 0.3:
            return
        self._t_light = time.time()
        if not self.tcp_server.get_command_server_busy():
            l = self.car.adc.read_adc(0)
            r = self.car.adc.read_adc(1)
            self.tcp_server.send_data_to_command_client(
                f"{CMD.CMD_MODE}#2#{l:.2f}#{r:.2f}\n")

    def _send_line(self) -> None:
        if time.time() - self._t_line < 0.3:
            return
        self._t_line = time.time()
        if not self.tcp_server.get_command_server_busy():
            v1 = self.car.infrared.read_one_infrared(1)
            v2 = self.car.infrared.read_one_infrared(2)
            v3 = self.car.infrared.read_one_infrared(3)
            self.tcp_server.send_data_to_command_client(
                f"{CMD.CMD_MODE}#4#{v1:.2f}#{v2:.2f}#{v3:.2f}\n")

    def _send_power(self) -> None:
        if not self.tcp_server.get_command_server_busy():
            batt = self.car.adc.read_adc(2) * (3 if self.car.adc.pcb_version == 1 else 2)
            self.tcp_server.send_data_to_command_client(
                f"{CMD.CMD_POWER}#{batt}\n")

    # ── Command thread ────────────────────────────────────────────────────────

    def _set_cmd_thread(self, run: bool, join_t: float = 0.3) -> None:
        alive = self._cmd_thread is not None and self._cmd_thread.is_alive()
        if run == alive:
            return
        if run:
            self._cmd_running = True
            self._cmd_thread  = Thread(target=self._cmd_loop, daemon=True)
            self._cmd_thread.start()
        else:
            self._cmd_running = False
            if self._cmd_thread:
                self._cmd_thread.join(join_t)
                self._cmd_thread = None

    def _cmd_loop(self) -> None:
        while self._cmd_running:
            q = self.tcp_server.read_data_from_command_server()
            if q.qsize() > 0:
                _, raw = q.get()
                for msg in raw.strip().split("\n"):
                    self._queue_cmd.put(msg)

            while not self._queue_cmd.empty():
                msg = self._queue_cmd.get()
                self._dispatch(msg)

            if self._queue_cmd.empty():
                time.sleep(0.001)

    def _dispatch(self, msg: str) -> None:
        p = self._cmd_parse
        if not p.parse(msg):
            return

        cmd = p.command

        # LED commands go to the LED process
        if cmd in (CMD.CMD_LED, CMD.CMD_LED_MOD):
            self._queue_led.put(msg)
            return

        if cmd == CMD.CMD_SONIC:
            self._send_sonic()
        elif cmd == CMD.CMD_LIGHT:
            self._send_light()
        elif cmd == CMD.CMD_LINE:
            self._send_line()
        elif cmd == CMD.CMD_POWER:
            self._send_power()
        elif cmd == CMD.CMD_BUZZER:
            self.buzzer.set_state(bool(p.params[0]))
        elif cmd == CMD.CMD_SERVO:
            try:
                self.car.servo.set_servo_pwm(str(p.params[0]), int(p.params[1]))
            except Exception as e:
                print(f"[SERVO] {e}")
        elif cmd == CMD.CMD_MOTOR:
            self._car_mode = 1
            try:
                fl, bl, fr, br = (int(round(v * 0.8)) for v in p.params[:4])
                self.car.motor.set_motor_model(fl, bl, fr, br)
            except Exception:
                pass
        elif cmd == CMD.CMD_M_MOTOR:
            self._car_mode = 1
            try:
                angle1, speed1, angle2, speed2 = p.params[:4]
                LX = -int(speed1 * math.sin(math.radians(angle1)))
                LY =  int(speed1 * math.cos(math.radians(angle1)))
                RX =  int(speed2 * math.sin(math.radians(angle2)))
                FL = LY + LX - RX
                BL = LY - LX - RX
                FR = LY - LX + RX
                BR = LY + LX + RX
                self.car.motor.set_motor_model(FL, BL, FR, BR)
            except Exception:
                pass
        elif cmd == CMD.CMD_CAR_ROTATE:
            self._handle_rotate(p.params)
        elif cmd == CMD.CMD_MODE:
            mode_map = {0: 1, 1: 2, 2: 3, 3: 4}
            self._car_mode = mode_map.get(p.params[0], 1)
            if p.params[0] == 0:
                self.car.motor.set_motor_model(0, 0, 0, 0)

    def _handle_rotate(self, duty: list) -> None:
        self._car_mode = 1
        try:
            if duty[3] == 0:
                # Stop rotation and apply a standard mecanum command
                self._rotate_stop.set()
                if self._rotate_thread and self._rotate_thread.is_alive():
                    self._rotate_thread.join(0.4)
                self._rotate_thread   = None
                self._rotation_active = False
                self._rotate_stop.clear()

                angle1, speed1, angle2, speed2 = duty[:4]
                LX = -int(speed1 * math.sin(math.radians(angle1)))
                LY =  int(speed1 * math.cos(math.radians(angle1)))
                RX =  int(speed2 * math.sin(math.radians(angle2)))
                self.car.motor.set_motor_model(
                    LY + LX - RX, LY - LX - RX, LY - LX + RX, LY + LX + RX)

            elif not self._rotation_active:
                self._rotate_stop.clear()
                self._rotation_active = True
                self._rotate_thread = Thread(
                    target=self.car.mode_rotate,
                    args=(duty[2], self._rotate_stop),
                    daemon=True,
                )
                self._rotate_thread.start()
        except Exception as e:
            print(f"[ROTATE] {e}")

    # ── Car task thread ───────────────────────────────────────────────────────

    def _set_car_thread(self, run: bool, join_t: float = 0.3) -> None:
        alive = self._car_thread is not None and self._car_thread.is_alive()
        if run == alive:
            return
        if run:
            self._car_running = True
            self._car_thread  = Thread(target=self._car_loop, daemon=True)
            self._car_thread.start()
        else:
            self._car_running = False
            if self._car_thread:
                self._car_thread.join(join_t)
                self._car_thread = None

    def _car_loop(self) -> None:
        while self._car_running:
            if   self._car_mode == 2: self.car.mode_light();     self._send_light()
            elif self._car_mode == 3: self.car.mode_infrared()
            elif self._car_mode == 4: self.car.mode_ultrasonic(); self._send_sonic()
            time.sleep(0.01)

    # ── Video thread ──────────────────────────────────────────────────────────

    def _set_video_thread(self, run: bool, join_t: float = 0.3) -> None:
        alive = self._video_thread is not None and self._video_thread.is_alive()
        if run == alive:
            return
        if run:
            self._video_running = True
            self._video_thread  = Thread(target=self._video_loop, daemon=True)
            self._video_thread.start()
        else:
            self._video_running = False
            if self._video_thread:
                self._video_thread.join(join_t)
                self._video_thread = None

    def _video_loop(self) -> None:
        while self._video_running:
            if not self.tcp_server.is_video_server_connected():
                time.sleep(0.1)
                continue
            self.camera.start_stream()
            while self.tcp_server.is_video_server_connected():
                frame = self.camera.get_frame()
                if frame is None:
                    continue
                try:
                    self.tcp_server.send_data_to_video_client(struct.pack('<I', len(frame)))
                    self.tcp_server.send_data_to_video_client(frame)
                except Exception:
                    break
            self.camera.stop_stream()

    # ── LED process ───────────────────────────────────────────────────────────

    def _set_led_process(self, run: bool, join_t: float = 0.3) -> None:
        alive = self._led_proc is not None and self._led_proc.is_alive()
        if run == alive:
            return
        if run:
            self._led_running = True
            self._led_proc    = multiprocessing.Process(
                target=_led_process_fn, args=(self._queue_led,), daemon=True)
            self._led_proc.start()
        else:
            self._led_running = False
            if self._led_proc:
                try:
                    self._led_proc.terminate()
                    self._led_proc.join(join_t)
                except Exception as e:
                    print(f"[LED] terminate error: {e}")
                self._led_proc = None

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def close_application(self) -> None:
        self._auto_start = False
        self._set_cmd_thread(False)
        self._set_video_thread(False)
        self._set_car_thread(False)
        self._set_led_process(False)

        # Stop any running rotation
        self._rotate_stop.set()
        if self._rotate_thread and self._rotate_thread.is_alive():
            self._rotate_thread.join(0.4)

        if self.tcp_server:
            self.tcp_server.stop_tcp_servers()
        try:
            self.led.colorBlink(0)
            self.camera.stop_stream()
            self.camera.close()
            self.car.close()
        except Exception:
            pass
        self.app.quit()
        sys.exit(0)

    def _signal_handler(self, sig, frame) -> None:
        print("Ctrl+C — stopping.")
        self.close_application()

    def _check_signals(self) -> None:
        if self.app.hasPendingEvents():
            self.app.processEvents()


# ── LED process (runs in separate process so it doesn't block the main loop) ──

def _led_process_fn(queue_led: multiprocessing.Queue) -> None:
    """
    Runs LED animations in a dedicated process.
    Receives CMD_LED / CMD_LED_MOD messages via queue.
    """
    led      = Led()
    parser   = MessageParser()
    led_mode = 0

    try:
        while True:
            while not queue_led.empty():
                parser.parse(queue_led.get())
                if parser.command == CMD.CMD_LED and led_mode == 1:
                    try:
                        idx, R, G, B = parser.params[:4]
                        led.ledIndex(idx, R, G, B)
                    except Exception:
                        pass
                elif parser.command == CMD.CMD_LED_MOD:
                    led_mode = parser.params[0] if parser.params else 0

            if   led_mode == 0: led.colorBlink(0)
            elif led_mode == 1: pass
            elif led_mode == 2: led.following()
            elif led_mode == 3: led.colorBlink(1)
            elif led_mode == 4: led.rainbowbreathing()
            elif led_mode == 5: led.rainbowCycle()

    except KeyboardInterrupt:
        led.colorBlink(0)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Freenove 4WD Smart Car Server')
    parser.add_argument('-t', '--terminal', action='store_true',
                        help='Run headless (no GUI window)')
    parser.add_argument('-n', '--no-gui',   action='store_true',
                        help='Alias for --terminal')
    args = parser.parse_args()

    window = RobotServer()

    if args.terminal or args.no_gui:
        signal.signal(signal.SIGINT, window._signal_handler)
        try:
            sys.exit(window.app.exec_())
        except KeyboardInterrupt:
            window.close_application()
    else:
        window.show()
        sys.exit(window.app.exec_())
