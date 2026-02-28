from __future__ import annotations

from enum import Enum


class InputMode(str, Enum):
    MESSAGE = "message"
    SENDINPUT = "sendinput"


class SmartInputController:
    def __init__(self, sink, retry_limit: int = 12) -> None:
        self.sink = sink
        self.retry_limit = retry_limit
        self.mode = InputMode.MESSAGE
        self._fail_count = 0

    def _on_result(self, ok: bool) -> None:
        if ok:
            self._fail_count = 0
            return
        self._fail_count += 1
        if self.mode == InputMode.MESSAGE and self._fail_count > self.retry_limit:
            self.mode = InputMode.SENDINPUT
            self._fail_count = 0

    def click_left(self) -> bool:
        if self.mode == InputMode.MESSAGE:
            ok = bool(self.sink.click_left_message())
            self._on_result(ok)
            if ok:
                return True
            if self.mode == InputMode.SENDINPUT:
                ok2 = bool(self.sink.click_left_sendinput())
                self._on_result(ok2)
                return ok2
            return False

        ok = bool(self.sink.click_left_sendinput())
        self._on_result(ok)
        return ok

    def hold_key_for(self, vk_code: int, duration_s: float) -> bool:
        if self.mode == InputMode.MESSAGE:
            ok = bool(self.sink.key_hold_message(vk_code, duration_s))
            self._on_result(ok)
            if ok:
                return True
            if self.mode == InputMode.SENDINPUT:
                ok2 = bool(self.sink.key_hold_sendinput(vk_code, duration_s))
                self._on_result(ok2)
                return ok2
            return False
        ok = bool(self.sink.key_hold_sendinput(vk_code, duration_s))
        self._on_result(ok)
        return ok

    def set_left_hold(self, hold: bool) -> bool:
        if self.mode == InputMode.MESSAGE:
            ok = bool(self.sink.set_left_hold_message(hold))
            self._on_result(ok)
            if ok:
                return True
            if self.mode == InputMode.SENDINPUT:
                ok2 = bool(self.sink.set_left_hold_sendinput(hold))
                self._on_result(ok2)
                return ok2
            return False
        ok = bool(self.sink.set_left_hold_sendinput(hold))
        self._on_result(ok)
        return ok

    def release_all(self) -> None:
        self.sink.release_all()
