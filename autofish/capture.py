from __future__ import annotations

from dataclasses import dataclass, field

import ctypes
import threading
from typing import Optional

import numpy as np

from .win32_api import get_client_rect_screen, get_window_rect


PW_RENDERFULLCONTENT = 0x00000002
DIB_RGB_COLORS = 0
BI_RGB = 0


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32),
        ("biWidth", ctypes.c_int32),
        ("biHeight", ctypes.c_int32),
        ("biPlanes", ctypes.c_uint16),
        ("biBitCount", ctypes.c_uint16),
        ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_int32),
        ("biYPelsPerMeter", ctypes.c_int32),
        ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", ctypes.c_uint32 * 3)]


@dataclass(slots=True)
class WindowCapture:
    hwnd: int
    window_name: str | None = None
    _wgc_started: bool = field(default=False, init=False)
    _wgc_capture: Optional[object] = field(default=None, init=False)
    _wgc_control: Optional[object] = field(default=None, init=False)
    _latest_frame: Optional[np.ndarray] = field(default=None, init=False)
    _frame_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self._try_start_wgc()

    def close(self) -> None:
        if self._wgc_control is not None:
            try:
                self._wgc_control.stop()
            except Exception:
                pass
        self._wgc_control = None
        self._wgc_capture = None
        self._wgc_started = False

    def grab(self):
        if self._wgc_started:
            frame = self._grab_wgc_frame()
            if frame is not None:
                rect = get_client_rect_screen(self.hwnd)
                if rect is None:
                    return None
                left, top, _, _ = rect
                return frame, (left, top)

        cap = self._grab_printwindow_client()
        return cap

    def _try_start_wgc(self) -> None:
        try:
            from windows_capture import InternalCaptureControl, WindowsCapture
        except Exception:
            return
        if not self.window_name:
            return

        cap = WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            monitor_index=None,
            window_name=self.window_name,
        )

        @cap.event
        def on_frame_arrived(frame, capture_control: InternalCaptureControl):
            arr = frame.frame_buffer[:, :, :3]
            with self._frame_lock:
                self._latest_frame = np.ascontiguousarray(arr.copy())

        @cap.event
        def on_closed():
            self._wgc_started = False

        try:
            self._wgc_control = cap.start_free_threaded()
            self._wgc_capture = cap
            self._wgc_started = True
        except Exception:
            self._wgc_capture = None
            self._wgc_control = None
            self._wgc_started = False

    def _grab_wgc_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def _grab_printwindow_client(self):
        full_rect = get_window_rect(self.hwnd)
        client_rect = get_client_rect_screen(self.hwnd)
        if full_rect is None or client_rect is None:
            return None
        wl, wt, wr, wb = full_rect
        left, top, right, bottom = client_rect
        full_w = wr - wl
        full_h = wb - wt
        if full_w <= 0 or full_h <= 0:
            return None
        frame_full = self._print_window_bgr(self.hwnd, full_w, full_h)
        if frame_full is None:
            return None
        ox = left - wl
        oy = top - wt
        cw = right - left
        ch = bottom - top
        if ox < 0 or oy < 0 or ox + cw > frame_full.shape[1] or oy + ch > frame_full.shape[0]:
            return None
        frame = frame_full[oy : oy + ch, ox : ox + cw].copy()
        return frame, (left, top)

    @staticmethod
    def _print_window_bgr(hwnd: int, width: int, height: int) -> np.ndarray | None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        hwnd_dc = user32.GetWindowDC(ctypes.c_void_p(hwnd))
        if not hwnd_dc:
            return None
        mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
        if not mem_dc:
            user32.ReleaseDC(ctypes.c_void_p(hwnd), hwnd_dc)
            return None
        bmp = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
        if not bmp:
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(ctypes.c_void_p(hwnd), hwnd_dc)
            return None
        old = gdi32.SelectObject(mem_dc, bmp)
        ok = user32.PrintWindow(ctypes.c_void_p(hwnd), mem_dc, PW_RENDERFULLCONTENT)
        if not ok:
            ok = user32.PrintWindow(ctypes.c_void_p(hwnd), mem_dc, 0)
        if not ok:
            gdi32.SelectObject(mem_dc, old)
            gdi32.DeleteObject(bmp)
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(ctypes.c_void_p(hwnd), hwnd_dc)
            return None
        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = width
        bmi.bmiHeader.biHeight = -height
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB
        size = width * height * 4
        buf = (ctypes.c_ubyte * size)()
        rows = gdi32.GetDIBits(mem_dc, bmp, 0, height, ctypes.byref(buf), ctypes.byref(bmi), DIB_RGB_COLORS)
        gdi32.SelectObject(mem_dc, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(ctypes.c_void_p(hwnd), hwnd_dc)
        if rows != height:
            return None
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
        return arr[:, :, :3].copy()

