import socket
import select
import threading
import queue


class TCPServer:
    def __init__(self):
        self._lock = threading.Lock()
        self.server_socket = None
        self.client_sockets: dict[socket.socket, tuple] = {}
        self.message_queue: queue.Queue = queue.Queue()
        self.max_clients = 1
        self.active_connections = 0
        self.accept_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.stop_pipe_r, self.stop_pipe_w = socket.socketpair()
        self.stop_pipe_r.setblocking(False)
        self.stop_pipe_w.setblocking(False)

    def start(self, ip: str, port: int, max_clients: int = 1, listen_count: int = 1) -> None:
        self.max_clients = max_clients
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((ip, port))
        self.server_socket.listen(listen_count)
        self.server_socket.setblocking(False)
        print(f"Server listening on {ip}:{port}")
        self.accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.accept_thread.start()

    def _accept_loop(self) -> None:
        while not self.stop_event.is_set():
            with self._lock:
                watch = list(self.client_sockets.keys())
            readable, _, exceptional = select.select(
                [self.server_socket, self.stop_pipe_r] + watch, [], watch
            )

            for s in exceptional:
                self._remove_client(s)

            for s in readable:
                if s is self.server_socket:
                    self._accept_new()
                elif s is self.stop_pipe_r:
                    self.stop_event.set()
                    return
                else:
                    self._recv_from(s)

    def _accept_new(self) -> None:
        try:
            client_socket, client_address = self.server_socket.accept()
            with self._lock:
                if self.active_connections < self.max_clients:
                    client_socket.setblocking(False)
                    self.client_sockets[client_socket] = client_address
                    self.active_connections += 1
                    print(f"Connected: {client_address} ({self.active_connections} active)")
                else:
                    client_socket.close()
                    print(f"Rejected {client_address}: max {self.max_clients} clients reached")
        except OSError:
            pass

    def _recv_from(self, s: socket.socket) -> None:
        try:
            data = s.recv(1024)
            if data:
                with self._lock:
                    addr = self.client_sockets.get(s)
                if addr:
                    self.message_queue.put((addr, data.decode("utf-8")))
            else:
                self._remove_client(s)
        except OSError as e:
            if e.errno in (9, 32):
                self._remove_client(s)
            else:
                print(f"[TCPServer] unexpected recv error: {e}")

    def _remove_client(self, s: socket.socket) -> None:
        with self._lock:
            addr = self.client_sockets.pop(s, None)
            if addr is not None:
                self.active_connections -= 1
                print(f"Disconnected: {addr}")
        try:
            s.close()
        except OSError:
            pass

    def send_to_all_client(self, message: bytes | str) -> None:
        payload = message.encode("utf-8") if isinstance(message, str) else message
        with self._lock:
            targets = list(self.client_sockets.keys())
        for s in targets:
            try:
                s.sendall(payload)
            except socket.error as e:
                print(f"[TCPServer] send error: {e}")
                self._remove_client(s)

    def send_to_client(self, client_address: tuple, message: bytes | str) -> None:
        payload = message.encode("utf-8") if isinstance(message, str) else message
        with self._lock:
            target = next((s for s, a in self.client_sockets.items() if a == client_address), None)
        if target is None:
            return
        try:
            target.sendall(payload)
        except socket.error as e:
            print(f"[TCPServer] send error to {client_address}: {e}")
            self._remove_client(target)

    def get_client_ips(self) -> list[str]:
        with self._lock:
            return [addr[0] for addr in self.client_sockets.values()]

    def close(self) -> None:
        try:
            self.stop_pipe_w.send(b"\x00")
        except OSError:
            pass
        if self.accept_thread is not None:
            self.accept_thread.join(timeout=1.0)
        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except OSError:
                pass
        with self._lock:
            for s in list(self.client_sockets):
                try:
                    s.close()
                except OSError:
                    pass
            self.client_sockets.clear()
            self.active_connections = 0
        print("Server stopped.")


if __name__ == "__main__":
    import fcntl
    import struct

    def _get_interface_ip() -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return socket.inet_ntoa(
            fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", b"wlan0"[:15]))[20:24]
        )

    srv = TCPServer()
    srv.start(_get_interface_ip(), 12345)
    try:
        while True:
            while not srv.message_queue.empty():
                addr, msg = srv.message_queue.get()
                print(f"{addr}: {msg}")
                srv.send_to_client(addr, msg)
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
