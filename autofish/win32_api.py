from __future__ import annotations

import ctypes
import time
from ctypes import wintypes


WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101

VK_W = 0x57
VK_S = 0x53
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004


def make_lparam(x: int, y: int) -> int:
    return (y << 16) | (x & 0xFFFF)


class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", INPUTUNION)]


def list_visible_windows() -> list[tuple[int, str]]:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    windows: list[tuple[int, str]] = []

    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    @EnumWindowsProc
    def _enum_proc(hwnd, lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip()
        if title:
            windows.append((int(hwnd), title))
        return True

    user32.EnumWindows(_enum_proc, 0)
    return windows


def get_window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    rect = RECT()
    ok = user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect))
    if not ok:
        return None
    return rect.left, rect.top, rect.right, rect.bottom


class Win32InputSink:
    def __init__(self, hwnd: int, click_x: int = 50, click_y: int = 50) -> None:
        self.hwnd = hwnd
        self.click_x = click_x
        self.click_y = click_y
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._left_held = False

    def _focus_target_window(self) -> bool:
        if not self.hwnd:
            return False
        self.user32.ShowWindow(wintypes.HWND(self.hwnd), 5)  # SW_SHOW
        return bool(self.user32.SetForegroundWindow(wintypes.HWND(self.hwnd)))

    def _move_cursor_to_window_center(self) -> bool:
        rect = get_window_rect(self.hwnd)
        if rect is None:
            return False
        left, top, right, bottom = rect
        if right <= left or bottom <= top:
            return False
        cx = int((left + right) / 2)
        cy = int((top + bottom) / 2)
        return bool(self.user32.SetCursorPos(cx, cy))

    def _send_mouse_flag(self, flag: int) -> bool:
        extra = ctypes.c_ulong(0)
        inp = INPUT(
            type=INPUT_MOUSE,
            union=INPUTUNION(
                mi=MOUSEINPUT(
                    dx=0,
                    dy=0,
                    mouseData=0,
                    dwFlags=flag,
                    time=0,
                    dwExtraInfo=ctypes.pointer(extra),
                )
            ),
        )
        sent = self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        return sent == 1

    def _send_key(self, vk_code: int, keyup: bool) -> bool:
        extra = ctypes.c_ulong(0)
        inp = INPUT(
            type=INPUT_KEYBOARD,
            union=INPUTUNION(
                ki=KEYBDINPUT(
                    wVk=vk_code,
                    wScan=0,
                    dwFlags=KEYEVENTF_KEYUP if keyup else 0,
                    time=0,
                    dwExtraInfo=ctypes.pointer(extra),
                )
            ),
        )
        sent = self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        return sent == 1

    def click_left_message(self) -> bool:
        if not self.hwnd:
            return False
        lparam = make_lparam(self.click_x, self.click_y)
        down_ok = self.user32.PostMessageW(wintypes.HWND(self.hwnd), WM_LBUTTONDOWN, 1, lparam)
        up_ok = self.user32.PostMessageW(wintypes.HWND(self.hwnd), WM_LBUTTONUP, 0, lparam)
        return bool(down_ok and up_ok)

    def click_left_sendinput(self) -> bool:
        self._focus_target_window()
        self._move_cursor_to_window_center()
        ok1 = self._send_mouse_flag(MOUSEEVENTF_LEFTDOWN)
        time.sleep(0.01)
        ok2 = self._send_mouse_flag(MOUSEEVENTF_LEFTUP)
        return bool(ok1 and ok2)

    def key_hold_message(self, vk_code: int, duration_s: float) -> bool:
        if not self.hwnd:
            return False
        down_ok = self.user32.PostMessageW(wintypes.HWND(self.hwnd), WM_KEYDOWN, vk_code, 0)
        time.sleep(max(0.0, duration_s))
        up_ok = self.user32.PostMessageW(wintypes.HWND(self.hwnd), WM_KEYUP, vk_code, 0)
        return bool(down_ok and up_ok)

    def key_hold_sendinput(self, vk_code: int, duration_s: float) -> bool:
        self._focus_target_window()
        ok1 = self._send_key(vk_code, keyup=False)
        time.sleep(max(0.0, duration_s))
        ok2 = self._send_key(vk_code, keyup=True)
        return bool(ok1 and ok2)

    def set_left_hold_message(self, hold: bool) -> bool:
        if not self.hwnd:
            return False
        if hold == self._left_held:
            return True
        lparam = make_lparam(self.click_x, self.click_y)
        if hold:
            ok = self.user32.PostMessageW(wintypes.HWND(self.hwnd), WM_LBUTTONDOWN, 1, lparam)
        else:
            ok = self.user32.PostMessageW(wintypes.HWND(self.hwnd), WM_LBUTTONUP, 0, lparam)
        if ok:
            self._left_held = hold
        return bool(ok)

    def set_left_hold_sendinput(self, hold: bool) -> bool:
        if hold == self._left_held:
            return True
        self._focus_target_window()
        self._move_cursor_to_window_center()
        ok = self._send_mouse_flag(MOUSEEVENTF_LEFTDOWN if hold else MOUSEEVENTF_LEFTUP)
        if ok:
            self._left_held = hold
        return ok

    def release_all(self) -> None:
        if self._left_held:
            self.set_left_hold_sendinput(False)
