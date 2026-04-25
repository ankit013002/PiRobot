import socket
import fcntl
import struct
import threading
from tcp_server import TCPServer


class Server:
    def __init__(self):
        self.ip_address = self._get_interface_ip()
        self.command_server = TCPServer()
        self.video_server = TCPServer()
        self._cmd_busy_lock = threading.Lock()
        self._cmd_busy = False

    @staticmethod
    def _get_interface_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            return socket.inet_ntoa(
                fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", b"wlan0"[:15]))[20:24]
            )
        except Exception as e:
            print(f"[Server] could not resolve wlan0 IP: {e}")
            return "127.0.0.1"

    def start_tcp_servers(self, command_port: int = 5000, video_port: int = 8000,
                          max_clients: int = 1, listen_count: int = 1) -> None:
        try:
            self.command_server.start(self.ip_address, command_port, max_clients, listen_count)
            self.video_server.start(self.ip_address, video_port, max_clients, listen_count)
        except Exception as e:
            print(f"[Server] start error: {e}")

    def stop_tcp_servers(self) -> None:
        try:
            self.command_server.close()
            self.video_server.close()
        except Exception as e:
            print(f"[Server] stop error: {e}")

    # ── Command channel ───────────────────────────────────────────────────────

    def get_command_server_busy(self) -> bool:
        with self._cmd_busy_lock:
            return self._cmd_busy

    def send_data_to_command_client(self, data: bytes | str, ip_address: tuple | None = None) -> None:
        with self._cmd_busy_lock:
            self._cmd_busy = True
        try:
            if ip_address is not None:
                self.command_server.send_to_client(ip_address, data)
            else:
                self.command_server.send_to_all_client(data)
        except Exception as e:
            print(f"[Server] command send error: {e}")
        finally:
            with self._cmd_busy_lock:
                self._cmd_busy = False

    def read_data_from_command_server(self) -> "queue.Queue":
        return self.command_server.message_queue

    def is_command_server_connected(self) -> bool:
        return self.command_server.active_connections > 0

    # ── Video channel ─────────────────────────────────────────────────────────

    def send_data_to_video_client(self, data: bytes | str, ip_address: tuple | None = None) -> None:
        try:
            if ip_address is not None:
                self.video_server.send_to_client(ip_address, data)
            else:
                self.video_server.send_to_all_client(data)
        except Exception as e:
            print(f"[Server] video send error: {e}")

    def is_video_server_connected(self) -> bool:
        return self.video_server.active_connections > 0


if __name__ == "__main__":
    import time
    server = Server()
    server.start_tcp_servers(5000, 8000)
    try:
        while True:
            q = server.read_data_from_command_server()
            if q.qsize() > 0:
                addr, msg = q.get()
                print(addr, msg)
                server.send_data_to_command_client(msg, addr)
            time.sleep(0.01)
    except KeyboardInterrupt:
        server.stop_tcp_servers()
