from __future__ import annotations

import time

from .config import AutoFishConfig
from .osc_api import OscClient
from .win32_api import VK_S, VK_W


class OscInputSink:
    def __init__(self, cfg: AutoFishConfig) -> None:
        self.cfg = cfg
        self.client = OscClient(cfg.osc_host, cfg.osc_port)
        self._left_held = False

    def _pulse(self, param_name: str, duration_s: float = 0.05) -> bool:
        ok1 = self.client.send_parameter(param_name, True)
        time.sleep(max(0.0, duration_s))
        ok2 = self.client.send_parameter(param_name, False)
        return bool(ok1 and ok2)

    def click_left_message(self) -> bool:
        return self._pulse(self.cfg.osc_param_click, 0.03)

    def click_left_sendinput(self) -> bool:
        return self._pulse(self.cfg.osc_param_click, 0.03)

    def key_hold_message(self, vk_code: int, duration_s: float) -> bool:
        if vk_code == VK_S:
            return self._pulse(self.cfg.osc_param_back, duration_s)
        if vk_code == VK_W:
            return self._pulse(self.cfg.osc_param_forward, duration_s)
        return False

    def key_hold_sendinput(self, vk_code: int, duration_s: float) -> bool:
        return self.key_hold_message(vk_code, duration_s)

    def set_left_hold_message(self, hold: bool) -> bool:
        if hold == self._left_held:
            return True
        ok = self.client.send_parameter(self.cfg.osc_param_hold, hold)
        if ok:
            self._left_held = hold
        return ok

    def set_left_hold_sendinput(self, hold: bool) -> bool:
        return self.set_left_hold_message(hold)

    def release_all(self) -> None:
        if self._left_held:
            self.client.send_parameter(self.cfg.osc_param_hold, False)
            self._left_held = False
        self.client.send_parameter(self.cfg.osc_param_click, False)
        self.client.send_parameter(self.cfg.osc_param_back, False)
        self.client.send_parameter(self.cfg.osc_param_forward, False)

