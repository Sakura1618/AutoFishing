# pip install pynput
from pynput import mouse
import time

down_ms = None
last_release_ms = None   # 上一次左键释放的时间戳（毫秒）

def now_ms() -> int:
    return time.time_ns() // 1_000_000  # 当前毫秒级时间戳

def on_click(x, y, button, pressed):
    global down_ms, last_release_ms
    if button != mouse.Button.left:
        return

    if pressed:
        current = now_ms()
        # 如果之前有释放记录，则计算并输出点击间隔（释放到按下）
        if last_release_ms is not None:
            interval = current - last_release_ms
            print(f"Click interval (release to press): {interval} ms")
            last_release_ms = None   # 已使用，清空
        down_ms = current
    else:
        if down_ms is None:
            return
        duration = now_ms() - down_ms
        down_ms = None
        # 记录本次释放的时间，用于下一次点击间隔的计算
        last_release_ms = now_ms()
        print(f"Duration: {duration} ms")

if __name__ == "__main__":
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()