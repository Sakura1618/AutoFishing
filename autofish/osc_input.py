from __future__ import annotations

import time

from .config import AutoFishConfig
from .osc_api import OscClient
from .win32_api import VK_S, VK_W


class OscInputSink:
    def __init__(self, cfg: AutoFishConfig, win32_sink=None) -> None:
        self.cfg = cfg
        self.client = OscClient(cfg.osc_host, cfg.osc_port)
        self.win32_sink = win32_sink
        self._left_held = False

    def _pulse_button(self, address: str, duration_s: float = 0.05) -> bool:
        ok1 = self.client.send_button(address, True)
        time.sleep(max(0.0, duration_s))
        ok2 = self.client.send_button(address, False)
        return bool(ok1 and ok2)

    def _pulse_axis(self, address: str, value: float, duration_s: float = 0.05) -> bool:
        ok1 = self.client.send_axis(address, value)
        time.sleep(max(0.0, duration_s))
        ok2 = self.client.send_axis(address, 0.0)
        return bool(ok1 and ok2)

    def click_left_message(self) -> bool:
        ok_button = self._pulse_button(self.cfg.osc_click_button, 0.03)
        ok_axis = self._pulse_axis(self.cfg.osc_click_axis, 1.0, 0.03)
        return bool(ok_button or ok_axis)

    def click_left_sendinput(self) -> bool:
        return self.click_left_message()

    def key_hold_message(self, vk_code: int, duration_s: float) -> bool:
        if vk_code == VK_S:
            return self._pulse_axis(self.cfg.osc_vertical_axis, -1.0, duration_s)
        if vk_code == VK_W:
            return self._pulse_axis(self.cfg.osc_vertical_axis, 1.0, duration_s)
        return False

    def key_hold_sendinput(self, vk_code: int, duration_s: float) -> bool:
        return self.key_hold_message(vk_code, duration_s)

    def set_left_hold_message(self, hold: bool) -> bool:
        if self.win32_sink is not None:
            ok = bool(self.win32_sink.set_left_hold_sendinput(hold))
            if ok:
                self._left_held = hold
            return ok
        if hold == self._left_held:
            return True
        ok_button = self.client.send_button(self.cfg.osc_click_button, hold)
        ok_axis = self.client.send_axis(self.cfg.osc_click_axis, 1.0 if hold else 0.0)
        ok = bool(ok_button or ok_axis)
        if ok:
            self._left_held = hold
        return ok

    def set_left_hold_sendinput(self, hold: bool) -> bool:
        return self.set_left_hold_message(hold)

    def release_all(self) -> None:
        if self.win32_sink is not None:
            try:
                self.win32_sink.release_all()
            except Exception:
                pass
        if self._left_held:
            self.client.send_button(self.cfg.osc_click_button, False)
            self.client.send_axis(self.cfg.osc_click_axis, 0.0)
            self._left_held = False
        self.client.send_axis(self.cfg.osc_vertical_axis, 0.0)
