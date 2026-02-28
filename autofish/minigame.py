from __future__ import annotations

from enum import Enum


class HoldAction(str, Enum):
    HOLD = "hold"
    RELEASE = "release"
    KEEP = "keep"


class MinigameController:
    def __init__(self, dead_zone_px: float = 4.0) -> None:
        self.dead_zone_px = dead_zone_px
        self._last = HoldAction.RELEASE

    def decide(self, fish_y: float, zone_center_y: float) -> HoldAction:
        delta = fish_y - zone_center_y
        if delta < -self.dead_zone_px:
            self._last = HoldAction.HOLD
            return HoldAction.HOLD
        if delta > self.dead_zone_px:
            self._last = HoldAction.RELEASE
            return HoldAction.RELEASE
        return HoldAction.KEEP

