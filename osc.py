from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
import time

LISTEN_IP = "127.0.0.1"   # 本机所有网卡都接收；只本机可改成 127.0.0.1
LISTEN_PORT = 9000      # 改成 VRChat Outgoing 端口

def any_handler(address, *args):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {address} {args}")

dispatcher = Dispatcher()
dispatcher.set_default_handler(any_handler)  # 捕获所有地址

server = ThreadingOSCUDPServer((LISTEN_IP, LISTEN_PORT), dispatcher)
print(f"Listening OSC on udp://{LISTEN_IP}:{LISTEN_PORT} ...")
server.serve_forever()