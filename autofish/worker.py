from __future__ import annotations

import threading
import time
from typing import Callable

import cv2

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
        preview_cb: Callable[[object, object], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.detector = detector
        self.capture = capture
        self.input_ctl = input_ctl
        self.log_cb = log_cb or (lambda _: None)
        self.status_cb = status_cb or (lambda _: None)
        self.preview_cb = preview_cb or (lambda _a, _b: None)
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
        self._last_det = {"has_bite": False, "has_bar": False, "bar_bbox": None, "fish_y": None, "zone_y": None, "boxes": []}
        self._last_infer_ts = 0.0
        self._bite_hits = 0
        self._bar_hits = 0

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
        if hasattr(self.capture, "close"):
            try:
                self.capture.close()
            except Exception:
                pass
        if hasattr(self.input_ctl, "release_all"):
            self.input_ctl.release_all()
        self.log_cb("worker stopped")

    def _run(self) -> None:
        frame_interval = 1.0 / max(1, self.cfg.loop_fps)
        while not self._stop_evt.is_set():
            self._tick_count += 1
            t0 = time.time()
            cap = self.capture.grab()
            frame, frame_origin = self._normalize_capture_result(cap)
            now = time.time()
            infer_interval = 1.0 / max(1, self.cfg.infer_fps)
            if now - self._last_infer_ts >= infer_interval:
                try:
                    det = self.detector.detect(frame, imgsz=self.cfg.imgsz)
                except TypeError:
                    det = self.detector.detect(frame)
                self._last_det = det
                self._last_infer_ts = now
            else:
                det = self._last_det
            det["origin"] = frame_origin
            det["boxes_screen"] = [
                {
                    "cls": b["cls"],
                    "conf": b["conf"],
                    "bbox": (
                        b["bbox"][0] + frame_origin[0],
                        b["bbox"][1] + frame_origin[1],
                        b["bbox"][2] + frame_origin[0],
                        b["bbox"][3] + frame_origin[1],
                    ),
                }
                for b in det.get("boxes", [])
            ]
            if det.get("has_bite"):
                self._bite_hits += 1
            if det.get("has_bar"):
                self._bar_hits += 1
            if self._tick_count % max(1, self.cfg.loop_fps * 2) == 0:
                if frame is None:
                    self.log_cb("diag: capture frame is None")
                else:
                    self.log_cb(
                        f"diag: has_bite={bool(det.get('has_bite'))}, "
                        f"has_bar={bool(det.get('has_bar'))}, fish_y={det.get('fish_y')}, zone_y={det.get('zone_y')}, "
                        f"origin={frame_origin}"
                    )
                    self.log_cb(
                        f"diag: yolo_hits_2s bite={self._bite_hits}, bar={self._bar_hits}, "
                        f"conf0={self.cfg.conf_yolo0:.2f}, conf1={self.cfg.conf_yolo1:.2f}, imgsz={self.cfg.imgsz}"
                    )
                    self._bite_hits = 0
                    self._bar_hits = 0
            now_ms = int(time.time() * 1000)
            out = self._sm.tick(now_ms=now_ms, has_bite=bool(det.get("has_bite")), has_bar=bool(det.get("has_bar")))
            self.status_cb(self._sm.state.value)

            if out.click_cast:
                self.input_ctl.click_left()
                self.log_cb("cast click")
            if out.click_hook:
                self.input_ctl.click_left()
                self.log_cb("hook click")
            if out.hold_back_s > 0:
                if hasattr(self.input_ctl, "hold_key_for"):
                    self.input_ctl.hold_key_for(VK_S, out.hold_back_s)
                self.log_cb("move back")
            if out.click_collect:
                self.input_ctl.click_left()
                self.log_cb("collect click")
                self.status_cb("success")
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

            yolo_preview, roi_preview = self._build_previews(frame, det)
            self.preview_cb(yolo_preview, roi_preview)

            elapsed = time.time() - t0
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    def _build_previews(self, frame, det):
        if frame is None:
            return None, None
        yolo = frame.copy()
        for b in det.get("boxes", []):
            x1, y1, x2, y2 = b["bbox"]
            cls_id = b["cls"]
            conf = b["conf"]
            color = (0, 255, 255) if cls_id == 0 else (0, 180, 0)
            cv2.rectangle(yolo, (x1, y1), (x2, y2), color, 2)
            cv2.putText(yolo, f"{cls_id}:{conf:.2f}", (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        bar = det.get("bar_bbox")
        roi = None
        if bar is not None:
            x1, y1, x2, y2 = bar
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2].copy()
                fish_y = det.get("fish_y")
                zone_y = det.get("zone_y")
                if fish_y is not None:
                    fy = int(float(fish_y) - y1)
                    cv2.line(roi, (0, fy), (roi.shape[1] - 1, fy), (0, 0, 255), 2)
                if zone_y is not None:
                    zy = int(float(zone_y) - y1)
                    cv2.line(roi, (0, zy), (roi.shape[1] - 1, zy), (255, 255, 255), 2)
        return yolo, roi

    @staticmethod
    def _normalize_capture_result(cap):
        if cap is None:
            return None, (0, 0)
        if isinstance(cap, tuple) and len(cap) == 2:
            return cap[0], cap[1]
        return cap, (0, 0)
