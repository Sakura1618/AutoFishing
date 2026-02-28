from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .win32_api import get_window_rect


@dataclass(slots=True)
class WindowCapture:
    hwnd: int

    def grab(self) -> np.ndarray | None:
        rect = get_window_rect(self.hwnd)
        if rect is None:
            return None
        left, top, right, bottom = rect
        if right <= left or bottom <= top:
            return None
        try:
            from PIL import ImageGrab

            img = ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True)
            arr = np.array(img)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

