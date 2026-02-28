from __future__ import annotations

import socket
import struct


def _pad4(data: bytes) -> bytes:
    pad = (4 - (len(data) % 4)) % 4
    return data + (b"\x00" * pad)


def _osc_string(text: str) -> bytes:
    return _pad4(text.encode("utf-8") + b"\x00")


def build_osc_message(address: str, value: int | float | bool) -> bytes:
    if not address.startswith("/"):
        address = "/" + address
    if isinstance(value, bool):
        type_tag = ",T" if value else ",F"
        args = b""
    elif isinstance(value, int):
        type_tag = ",i"
        args = struct.pack(">i", value)
    else:
        type_tag = ",f"
        args = struct.pack(">f", float(value))
    return _osc_string(address) + _osc_string(type_tag) + args


class OscClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 9000) -> None:
        self.host = host
        self.port = int(port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_message(self, address: str, value: int | float | bool) -> bool:
        try:
            payload = build_osc_message(address, value)
            sent = self.sock.sendto(payload, (self.host, self.port))
            return sent > 0
        except OSError:
            return False

    def send_button(self, address: str, pressed: bool) -> bool:
        return self.send_message(address, 1 if pressed else 0)

    def send_axis(self, address: str, value: float) -> bool:
        value = max(-1.0, min(1.0, float(value)))
        return self.send_message(address, value)
