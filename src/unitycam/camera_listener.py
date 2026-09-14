import socket
from typing import Optional, Tuple

from src.unitycam.camera_protocol import UnityCameraPacket, read_unity_camera_packet


class CamListener:
    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 5005,
        backlog: int = 1,
    ):
        self.host = host
        self.port = port
        self.backlog = backlog
        self.server_sock: Optional[socket.socket] = None
        self.conn: Optional[socket.socket] = None
        self.addr: Optional[Tuple[str, int]] = None

    def start(self):
        if self.server_sock is not None:
            return
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(self.backlog)

    def accept(self) -> Tuple[str, int]:
        if self.server_sock is None:
            self.start()
        self.conn, self.addr = self.server_sock.accept()
        return self.addr

    def recv_packet(self) -> UnityCameraPacket:
        if self.conn is None:
            raise RuntimeError('No active client connection. Call accept() first.')
        return read_unity_camera_packet(self.conn)

    def close_client(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            self.addr = None

    def close(self):
        self.close_client()
        if self.server_sock is not None:
            self.server_sock.close()
            self.server_sock = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()