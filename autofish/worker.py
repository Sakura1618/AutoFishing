from __future__ import annotations

import threading
import time
from typing import Callable

from .config import AutoFishConfig
from .minigame import HoldAction, MinigameController
from .state_machine import AutoFishState, FishingStateMachine
from .win32_api import VK_S, VK_W


class AutoFishWorker:
    def __init__(
        self,
        cfg: AutoFishConfig,
        detector,
        capture,
        input_ctl,
        log_cb: Callable[[str], None] | None = None,
        status_cb: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.detector = detector
        self.capture = capture
        self.input_ctl = input_ctl
        self.log_cb = log_cb or (lambda _: None)
        self.status_cb = status_cb or (lambda _: None)
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._sm = FishingStateMachine(
            cast_wait_s=cfg.cast_wait_s,
            move_back_s=cfg.move_back_s,
            move_forward_s=cfg.move_forward_s,
            success_disappear_ms=cfg.success_disappear_ms,
        )
        self._mini = MinigameController()
        self._tick_count = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.log_cb("worker started")

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if hasattr(self.input_ctl, "release_all"):
            self.input_ctl.release_all()
        self.log_cb("worker stopped")

    def _run(self) -> None:
        frame_interval = 1.0 / max(1, self.cfg.loop_fps)
        while not self._stop_evt.is_set():
            self._tick_count += 1
            t0 = time.time()
            frame = self.capture.grab()
            det = self.detector.detect(frame)
            if self._tick_count % max(1, self.cfg.loop_fps * 2) == 0:
                if frame is None:
                    self.log_cb("diag: capture frame is None")
                else:
                    self.log_cb(
                        f"diag: has_bite={bool(det.get('has_bite'))}, "
                        f"has_bar={bool(det.get('has_bar'))}, fish_y={det.get('fish_y')}, zone_y={det.get('zone_y')}"
                    )
            now_ms = int(time.time() * 1000)
            out = self._sm.tick(now_ms=now_ms, has_bite=bool(det.get("has_bite")), has_bar=bool(det.get("has_bar")))
            self.status_cb(self._sm.state.value)

            if out.click_cast:
                self.input_ctl.click_left()
                self.log_cb("cast click")
            if out.hold_back_s > 0:
                if hasattr(self.input_ctl, "hold_key_for"):
                    self.input_ctl.hold_key_for(VK_S, out.hold_back_s)
                self.log_cb("move back")
            if out.click_collect:
                self.input_ctl.click_left()
                self.log_cb("collect click")
            if out.hold_forward_s > 0:
                if hasattr(self.input_ctl, "hold_key_for"):
                    self.input_ctl.hold_key_for(VK_W, out.hold_forward_s)
                self.log_cb("move forward")

            if self._sm.state == AutoFishState.MINIGAME:
                fish_y = det.get("fish_y")
                zone_y = det.get("zone_y")
                if fish_y is not None and zone_y is not None:
                    action = self._mini.decide(fish_y=float(fish_y), zone_center_y=float(zone_y))
                    if action == HoldAction.HOLD:
                        if hasattr(self.input_ctl, "set_left_hold"):
                            self.input_ctl.set_left_hold(True)
                    elif action == HoldAction.RELEASE:
                        if hasattr(self.input_ctl, "set_left_hold"):
                            self.input_ctl.set_left_hold(False)

            elapsed = time.time() - t0
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
