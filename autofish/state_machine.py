from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AutoFishState(str, Enum):
    CAST = "cast"
    WAIT_BITE = "wait_bite"
    MINIGAME = "minigame"


@dataclass(slots=True)
class TickOutput:
    click_cast: bool = False
    click_hook: bool = False
    hold_back_s: float = 0.0
    click_collect: bool = False
    hold_forward_s: float = 0.0


class FishingStateMachine:
    def __init__(self, cast_wait_s: float, move_back_s: float, move_forward_s: float, success_disappear_ms: int) -> None:
        self.cast_wait_s = cast_wait_s
        self.move_back_s = move_back_s
        self.move_forward_s = move_forward_s
        self.success_disappear_ms = success_disappear_ms
        self.state = AutoFishState.CAST
        self._cast_started_ms: int | None = None
        self._bar_missing_since_ms: int | None = None
        self._bite_latched = False

    def reset(self) -> None:
        self.state = AutoFishState.CAST
        self._cast_started_ms = None
        self._bar_missing_since_ms = None
        self._bite_latched = False

    def tick(self, now_ms: int, has_bite: bool, has_bar: bool) -> TickOutput:
        out = TickOutput()
        if self.state == AutoFishState.CAST:
            if self._cast_started_ms is None:
                self._cast_started_ms = now_ms
                out.click_cast = True
                return out
            if now_ms - self._cast_started_ms >= int(self.cast_wait_s * 1000):
                out.hold_back_s = self.move_back_s
                self.state = AutoFishState.WAIT_BITE
            return out

        if self.state == AutoFishState.WAIT_BITE:
            if has_bite:
                if not self._bite_latched:
                    out.click_hook = True
                self._bite_latched = True
            if self._bite_latched and has_bar:
                self.state = AutoFishState.MINIGAME
                self._bar_missing_since_ms = None
            return out

        if self.state == AutoFishState.MINIGAME:
            if has_bar:
                self._bar_missing_since_ms = None
                return out
            if self._bar_missing_since_ms is None:
                self._bar_missing_since_ms = now_ms
                return out
            if now_ms - self._bar_missing_since_ms >= self.success_disappear_ms:
                out.click_collect = True
                out.hold_forward_s = self.move_forward_s
                self.reset()
            return out

        return out
