from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import AutoFishConfig
from .minigame import (
    FishTemplateMatcher,
    HoldAction,
    MinigameController,
    detect_fish_by_color_peak,
    detect_fish_by_motion_peak,
    detect_fish_by_width_peak,
    detect_white_zone_band,
)
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
        fps_cb: Callable[[float, float], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.detector = detector
        self.capture = capture
        self.input_ctl = input_ctl
        self.log_cb = log_cb or (lambda _: None)
        self.status_cb = status_cb or (lambda _: None)
        self.preview_cb = preview_cb or (lambda _a, _b: None)
        self.fps_cb = fps_cb or (lambda _a, _b: None)
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self.screen_height = 1080
        self._screen_bottom_margin_px = 50.0
        self._sm = FishingStateMachine(
            cast_wait_s=cfg.cast_wait_s,
            move_back_s=cfg.move_back_s,
            move_forward_s=cfg.move_forward_s,
            success_disappear_ms=cfg.success_disappear_ms,
        )
        self._mini = MinigameController(hold_decreases_y=True)
        app_dir = Path(__file__).resolve().parents[1]
        self._template_file = app_dir / "img" / "fish.png"
        try:
            if self._template_file.exists():
                self._matcher = FishTemplateMatcher.from_template_file(
                    self._template_file,
                    threshold=0.43,
                    scales=(0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00),
                    lost_hold_ms=300,
                    local_expand=2.1,
                    local_track_ms=300,
                    smooth_alpha=0.40,
                )
            else:
                self._matcher = None
        except Exception:
            self._matcher = None
        self._tick_count = 0
        self._last_det = {
            "has_bite": False,
            "has_bar": False,
            "bar_bbox": None,
            "fish_y": None,
            "zone_y": None,
            "zone_top": None,
            "zone_bottom": None,
            "boxes": [],
        }
        self._last_infer_ts = 0.0
        self._bite_hits = 0
        self._bar_hits = 0
        self._loop_id = 0
        self._yolo0_id = 0
        self._yolo1_id = 0
        self._await_next_yolo1 = False
        self._roi_anchor_bbox = None
        self._roi_anchor_last_seen_ms = 0
        self._roi_lock_delay_ms = int(cfg.roi_lock_delay_ms)
        self._roi_lock_candidate_bbox = None
        self._roi_lock_candidate_ms = 0
        self._mini_score = 0.0
        self._mini_template = ""
        self._mini_scale: float | None = None
        self._last_hold_action: HoldAction | None = None
        self._loop_stat_count = 0
        self._infer_stat_count = 0
        self._stat_last_ts = time.time()
        self._last_sm_state = self._sm.state
        self._mini_ready = False
        self._mini_enter_ms = 0
        self._mini_prev_zone_y: float | None = None
        self._mini_drop_start_y: float | None = None
        self._mini_drop_need_px = float(cfg.mini_drop_need_px)
        self._mini_wait_max_ms = int(cfg.mini_wait_max_ms)
        self._mini_signal_timeout_ms = int(cfg.mini_signal_timeout_ms)
        self._last_mini_signal_ms = 0
        self._mini_last_ctrl_ms = 0
        self._mini_last_fish_y: float | None = None
        self._mini_fish_vel_ema = 0.0
        self._mini_vel_alpha = float(cfg.mini_vel_alpha)
        self._mini_predict_ms = int(cfg.mini_predict_ms)
        self._mini_err_prev: float | None = None
        self._mini_mode = "idle"
        self._mini_brake_until_ms = 0
        self._mini_dead_px = float(cfg.mini_dead_px)
        self._mini_far_px = float(cfg.mini_far_px)
        self._mini_edge_guard_px = float(cfg.mini_edge_guard_px)
        self._mini_hold_interval_track_ms = int(cfg.mini_hold_interval_track_ms)
        self._mini_hold_interval_catch_ms = int(cfg.mini_hold_interval_catch_ms)
        self._mini_hold_last_ms = 0
        self._mini_hold_active = False
        self._mini_hold_until_ms = 0
        self._mini_release_until_ms = 0
        self._mini_release_lock_ms = 80
        self._mini_track_px_ref = float(cfg.mini_track_px_ref)
        self._mini_up_full_ms = float(cfg.mini_up_full_ms)
        self._mini_hold_min_ms = int(cfg.mini_hold_min_ms)
        self._mini_hold_max_ms = int(cfg.mini_hold_max_ms)
        self._mini_brake_ms = int(cfg.mini_brake_ms)
        self._left_hold_active = False
        self._left_hold_since_ms = 0
        self._max_hold_ms = 260
        self._hold_cooldown_until_ms = 0
        self._rel_dead_px = 2.0
        self._tap_last_ms = 0
        self._tap_active = False
        self._tap_started_ms = 0
        self._tap_hold_ms_current = 20
        self._tap_hold_ms_light = 14
        self._tap_hold_ms_mid = 20
        self._tap_hold_ms_heavy = 36
        self._tap_interval_ms_light = 115
        self._tap_interval_ms_mid = 90
        self._tap_interval_ms_heavy = 70
        self._bottom_rescue_margin_px = 2.5
        self._bottom_rescue_interval_ms = 75
        self._bottom_rescue_hold_ms = 26
        self._bottom_rescue_active = False
        self._bottom_rescue_started_ms = 0
        self._bottom_rescue_last_ms = 0
        self._roi_smooth_alpha = 0.2
        self._roi_jump_px = 6
        self._signal_alpha = 0.35
        self._signal_jump_px = 15.0
        self._smooth_values: dict[str, float] = {}
        self._roi_prev_gray: np.ndarray | None = None
        self._fish_prev_gray: np.ndarray | None = None
        self._roi_motion_dx = 0.0
        self._roi_motion_dy = 0.0
        self._roi_motion_alpha = 0.45
        self._roi_motion_max_px = 10.0
        self._roi_motion_min_resp = 0.12
        self._roi_stab_dx = 0.0
        self._roi_stab_dy = 0.0
        self._signal_hist: dict[str, deque[float]] = {
            "fish_y": deque(maxlen=3),
            "zone_top": deque(maxlen=3),
            "zone_bottom": deque(maxlen=3),
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.log_cb("worker started")
        self.log_cb(f"minigame axis: hold_decreases_y={self._mini.hold_decreases_y}")
        if self._matcher is None:
            self.log_cb("warning: fish template missing, expected ./img/fish.png")

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
            self._loop_stat_count += 1
            t0 = time.time()
            cap = self.capture.grab()
            frame, frame_origin = self._normalize_capture_result(cap)
            if frame is not None:
                try:
                    self.screen_height = int(frame.shape[0])
                except Exception:
                    pass
            now = time.time()
            infer_interval = 1.0 / max(1, self.cfg.infer_fps)
            if now - self._last_infer_ts >= infer_interval:
                try:
                    det = self.detector.detect(frame, imgsz=self.cfg.imgsz)
                except TypeError:
                    det = self.detector.detect(frame)
                self._last_det = det
                self._last_infer_ts = now
                self._infer_stat_count += 1
            else:
                det = self._last_det
            now_ms = int(time.time() * 1000)
            det = dict(det)
            det["origin"] = frame_origin
            raw_boxes = det.get("boxes", [])
            annotated_boxes = self._annotate_boxes(raw_boxes)
            det["boxes"] = annotated_boxes
            det["boxes_screen"] = [
                {
                    "cls": b["cls"],
                    "conf": b["conf"],
                    "id": b.get("id"),
                    "bbox": (
                        b["bbox"][0] + frame_origin[0],
                        b["bbox"][1] + frame_origin[1],
                        b["bbox"][2] + frame_origin[0],
                        b["bbox"][3] + frame_origin[1],
                    ),
                }
                for b in annotated_boxes
            ]
            selected_bar = self._select_bar_bbox(annotated_boxes, now_ms=now_ms)
            fish_y, zone_y, zone_top, zone_bottom = self._analyze_minigame_roi(frame, selected_bar, now_ms)
            fish_y, zone_y, zone_top, zone_bottom = self._stabilize_measurements(fish_y, zone_y, zone_top, zone_bottom)
            det["bar_bbox"] = selected_bar
            det["has_bar"] = selected_bar is not None
            det["fish_y"] = fish_y
            det["zone_y"] = zone_y
            det["zone_top"] = zone_top
            det["zone_bottom"] = zone_bottom
            if det.get("has_bite"):
                self._bite_hits += 1
            if det.get("has_bar"):
                self._bar_hits += 1
            if self._tick_count % max(1, self.cfg.loop_fps * 2) == 0:
                self._bite_hits = 0
                self._bar_hits = 0
            out = self._sm.tick(now_ms=now_ms, has_bite=bool(det.get("has_bite")), has_bar=bool(det.get("has_bar")))
            self._handle_state_transition(now_ms=now_ms)
            self.status_cb(self._sm.state.value)

            if out.click_cast:
                self._loop_id += 1
                self._yolo0_id = 0
                self._yolo1_id = 0
                self._await_next_yolo1 = False
                self._roi_anchor_bbox = None
                self._roi_anchor_last_seen_ms = 0
                self._roi_lock_candidate_bbox = None
                self._roi_lock_candidate_ms = 0
                self._smooth_values.clear()
                for hist in self._signal_hist.values():
                    hist.clear()
                self._roi_prev_gray = None
                self._fish_prev_gray = None
                self._roi_motion_dx = 0.0
                self._roi_motion_dy = 0.0
                self._roi_stab_dx = 0.0
                self._roi_stab_dy = 0.0
                self._last_mini_signal_ms = 0
                self._mini_last_ctrl_ms = 0
                self._mini_last_fish_y = None
                self._mini_fish_vel_ema = 0.0
                self._mini_err_prev = None
                self._mini_mode = "idle"
                self._mini_brake_until_ms = 0
                self._mini_hold_last_ms = 0
                self._mini_hold_active = False
                self._mini_hold_until_ms = 0
                self._mini_release_until_ms = 0
                self._left_hold_active = False
                self._left_hold_since_ms = 0
                self._hold_cooldown_until_ms = 0
                self._tap_last_ms = 0
                self._tap_active = False
                self._tap_started_ms = 0
                self._bottom_rescue_active = False
                self._bottom_rescue_started_ms = 0
                self._bottom_rescue_last_ms = 0
                self.input_ctl.click_left()
                self.log_cb(f"cast click (loop={self._loop_id})")
            if out.click_hook:
                self.input_ctl.click_left()
                self._await_next_yolo1 = True
                self.log_cb(f"hook click (loop={self._loop_id})")
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
                ready = self._update_minigame_ready(zone_y=None if zone_y is None else float(zone_y), now_ms=now_ms)
                if not ready:
                    self._apply_left_hold(False, now_ms=now_ms)
                    self._mini_mode = "warmup"
                    self._mini_hold_active = False
                    self._last_hold_action = None
                elif fish_y is not None and zone_y is not None:
                    zone_y_adj, fish_y_adj = self._update_zone_y(float(zone_y), float(fish_y))
                    det["zone_y"] = zone_y_adj
                    det["fish_y"] = fish_y_adj
                    self._last_mini_signal_ms = now_ms
                    self._run_minigame_controller(
                        fish_y=float(fish_y_adj),
                        zone_y=float(zone_y_adj),
                        zone_top=None if det.get("zone_top") is None else float(det.get("zone_top")),
                        zone_bottom=None if det.get("zone_bottom") is None else float(det.get("zone_bottom")),
                        bar_bbox=det.get("bar_bbox"),
                        now_ms=now_ms,
                    )
                    self._last_hold_action = HoldAction.HOLD if self._left_hold_active else HoldAction.RELEASE
                elif now_ms - self._last_mini_signal_ms >= self._mini_signal_timeout_ms:
                    self._apply_left_hold(False, now_ms=now_ms)
                    self._mini_mode = "signal_lost"
                    self._mini_hold_active = False
                    self._last_hold_action = None

            yolo_preview, roi_preview = self._build_previews(frame, det)
            self.preview_cb(yolo_preview, roi_preview)

            stat_now = time.time()
            dt = stat_now - self._stat_last_ts
            if dt >= 1.0:
                loop_actual = self._loop_stat_count / dt
                infer_actual = self._infer_stat_count / dt
                self.fps_cb(loop_actual, infer_actual)
                self._loop_stat_count = 0
                self._infer_stat_count = 0
                self._stat_last_ts = stat_now

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
            bid = b.get("id", "")
            cv2.putText(yolo, f"{bid} {cls_id}:{conf:.2f}", (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        bar = det.get("bar_bbox")
        roi = None
        if bar is not None:
            x1, y1, x2, y2 = bar
            cv2.rectangle(yolo, (x1, y1), (x2, y2), (255, 120, 0), 2)
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            if x2 > x1 and y2 > y1:
                roi_strip = frame[y1:y2, x1:x2].copy()
                fish_y = det.get("fish_y")
                zone_y = det.get("zone_y")
                if fish_y is not None:
                    fy = int(float(fish_y) - y1)
                    cv2.line(roi_strip, (0, fy), (roi_strip.shape[1] - 1, fy), (0, 0, 255), 2)
                if zone_y is not None:
                    zy = int(float(zone_y) - y1)
                    cv2.line(roi_strip, (0, zy), (roi_strip.shape[1] - 1, zy), (255, 255, 255), 2)
                zone_top = det.get("zone_top")
                if zone_top is not None:
                    zt = int(float(zone_top) - y1)
                    cv2.line(roi_strip, (0, zt), (roi_strip.shape[1] - 1, zt), (255, 220, 80), 1)
                zone_bottom = det.get("zone_bottom")
                if zone_bottom is not None:
                    zb = int(float(zone_bottom) - y1)
                    cv2.line(roi_strip, (0, zb), (roi_strip.shape[1] - 1, zb), (255, 220, 80), 1)

                panel_w = 280
                panel_h = max(int(roi_strip.shape[0]), 220)
                roi = np.full((panel_h, int(roi_strip.shape[1]) + panel_w, 3), 255, dtype=np.uint8)
                roi[0 : roi_strip.shape[0], 0 : roi_strip.shape[1]] = roi_strip
                cv2.line(roi, (roi_strip.shape[1], 0), (roi_strip.shape[1], panel_h - 1), (220, 220, 220), 1)

                action_text = self._last_hold_action.value.upper() if self._last_hold_action is not None else "NONE"
                action_color = (0, 220, 0) if action_text == "HOLD" else (0, 120, 255) if action_text == "RELEASE" else (200, 200, 200)
                text_x = int(roi_strip.shape[1]) + 10
                text_y = 24
                line_h = 24
                cv2.putText(
                    roi,
                    f"score:{self._mini_score:.2f} {self._mini_template}",
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (10, 90, 190),
                    1,
                )
                text_y += line_h
                cv2.putText(roi, f"action:{action_text}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, action_color, 2)
                text_y += line_h
                cv2.putText(
                    roi,
                    f"ctl:{self._mini.last_control:+.1f}",
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.54,
                    (140, 100, 0),
                    1,
                )
                rel_y = det.get("rel_y")
                if rel_y is not None:
                    text_y += line_h
                    cv2.putText(
                        roi,
                        f"rel_y:{float(rel_y):+.1f}",
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.54,
                        (0, 120, 60),
                        1,
                    )
                text_y += line_h
                cv2.putText(
                    roi,
                    f"mode:{self._mini_mode}",
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.54,
                    (40, 120, 40),
                    1,
                )
                if self._sm.state == AutoFishState.MINIGAME and not self._mini_ready:
                    text_y += line_h
                    cv2.putText(
                        roi,
                        "ready:WAIT_DROP",
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.54,
                        (0, 100, 180),
                        1,
                    )
                text_y += line_h
                cv2.putText(
                    roi,
                    f"stab:dx={self._roi_stab_dx:+.1f} dy={self._roi_stab_dy:+.1f}",
                    (text_x, min(panel_h - 8, text_y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (90, 90, 90),
                    1,
                )
        return yolo, roi

    @staticmethod
    def _normalize_capture_result(cap):
        if cap is None:
            return None, (0, 0)
        if isinstance(cap, tuple) and len(cap) == 2:
            return cap[0], cap[1]
        return cap, (0, 0)

    def _annotate_boxes(self, boxes):
        out = []
        for b in boxes:
            cls_id = int(b.get("cls", -1))
            if cls_id == 0:
                self._yolo0_id += 1
                bid = f"L{self._loop_id}-0-{self._yolo0_id}"
            elif cls_id == 1:
                self._yolo1_id += 1
                bid = f"L{self._loop_id}-1-{self._yolo1_id}"
            else:
                bid = f"L{self._loop_id}-x"
            bb = dict(b)
            bb["id"] = bid
            out.append(bb)
        return out

    def _select_bar_bbox(self, boxes, now_ms: int | None = None):
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        # class-1 before hook click should not be used as ROI.
        class1 = [b for b in boxes if int(b.get("cls", -1)) == 1]
        if self._sm.state == AutoFishState.WAIT_BITE:
            if self._await_next_yolo1:
                if class1:
                    cand = max(class1, key=lambda b: float(b.get("conf", 0.0)))
                    if self._roi_lock_candidate_bbox is None:
                        self._roi_lock_candidate_bbox = cand["bbox"]
                        self._roi_lock_candidate_ms = now_ms
                    else:
                        self._roi_lock_candidate_bbox = cand["bbox"]
                if self._roi_lock_candidate_bbox is None:
                    return None
                if now_ms - self._roi_lock_candidate_ms < self._roi_lock_delay_ms:
                    return None
                self._update_roi_anchor(self._roi_lock_candidate_bbox, now_ms=now_ms)
                self._roi_lock_candidate_bbox = None
                self._await_next_yolo1 = False
                return self._roi_anchor_bbox
            return None
        if self._sm.state == AutoFishState.MINIGAME:
            if not class1:
                # Keep using locked ROI for a while when yolo:1 flickers.
                if self._roi_anchor_bbox is not None and now_ms - self._roi_anchor_last_seen_ms <= 1500:
                    return self._roi_anchor_bbox
                return None
            if self._roi_anchor_bbox is None:
                self._update_roi_anchor(class1[0]["bbox"], now_ms=now_ms)
                return self._roi_anchor_bbox
            best = max(class1, key=lambda b: self._bbox_iou(self._roi_anchor_bbox, b["bbox"]))
            self._update_roi_anchor(best["bbox"], now_ms=now_ms)
            return self._roi_anchor_bbox
        return None

    def _update_roi_anchor(self, new_bbox, now_ms: int) -> None:
        if self._roi_anchor_bbox is None:
            self._roi_anchor_bbox = new_bbox
            self._roi_anchor_last_seen_ms = now_ms
            return
        ox1, oy1, ox2, oy2 = self._roi_anchor_bbox
        nx1, ny1, nx2, ny2 = new_bbox
        old_cx = (ox1 + ox2) * 0.5
        old_cy = (oy1 + oy2) * 0.5
        new_cx = (nx1 + nx2) * 0.5
        new_cy = (ny1 + ny2) * 0.5
        if abs(new_cx - old_cx) <= self._roi_jump_px and abs(new_cy - old_cy) <= self._roi_jump_px:
            self._roi_anchor_last_seen_ms = now_ms
            return
        a = self._roi_smooth_alpha
        sx1 = int(round(ox1 * (1.0 - a) + nx1 * a))
        sy1 = int(round(oy1 * (1.0 - a) + ny1 * a))
        sx2 = int(round(ox2 * (1.0 - a) + nx2 * a))
        sy2 = int(round(oy2 * (1.0 - a) + ny2 * a))
        self._roi_anchor_bbox = (sx1, sy1, sx2, sy2)
        self._roi_anchor_last_seen_ms = now_ms

    @staticmethod
    def _bbox_iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter
        return float(inter / max(1, union))

    def _analyze_minigame_roi(self, frame, bar_bbox, now_ms: int):
        self._mini_score = 0.0
        self._mini_template = ""
        self._mini_scale = None
        if frame is None or bar_bbox is None:
            self._roi_prev_gray = None
            self._fish_prev_gray = None
            self._roi_motion_dx = 0.0
            self._roi_motion_dy = 0.0
            self._roi_stab_dx = 0.0
            self._roi_stab_dy = 0.0
            return None, None, None, None
        x1, y1, x2, y2 = bar_bbox
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None, None, None, None
        roi_raw = frame[y1:y2, x1:x2]
        roi, stab_dx, stab_dy = self._stabilize_roi_image(roi_raw)
        self._roi_stab_dx = stab_dx
        self._roi_stab_dy = stab_dy
        band = detect_white_zone_band(roi)
        hit = self._matcher.locate(roi, now_ms=now_ms) if self._matcher is not None else None
        fish_y = None
        if hit is not None:
            fish_y = y1 + hit.fish_y + stab_dy
            self._mini_score = hit.score
            self._mini_template = hit.template_name
            self._mini_scale = float(hit.scale)
            # Keep motion reference updated even when template hits.
            self._fish_prev_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            # Non-template fish detection (no fallback-dark).
            # Priority: width-peak -> color-peak -> motion-peak.
            # Motion needs previous stable ROI gray frame.
            diff = None
            gray_u8 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if self._fish_prev_gray is not None and self._fish_prev_gray.shape == gray_u8.shape:
                diff = cv2.absdiff(gray_u8, self._fish_prev_gray)
            self._fish_prev_gray = gray_u8

            cand = detect_fish_by_width_peak(roi, band)
            if cand is None:
                cand = detect_fish_by_color_peak(roi, band)
            if cand is None:
                cand = detect_fish_by_motion_peak(roi, band, diff)

            if cand is not None:
                fish_y = y1 + float(cand.fish_y) + stab_dy
                self._mini_score = float(cand.score)
                self._mini_template = cand.method
                self._mini_scale = None
        if band is None:
            return fish_y, None, None, None
        zone_y = y1 + band.center + stab_dy
        zone_top = y1 + float(band.top) + stab_dy
        zone_bottom = y1 + float(band.bottom) + stab_dy
        return fish_y, zone_y, zone_top, zone_bottom

    def _stabilize_roi_image(self, roi_bgr):
        if roi_bgr is None or roi_bgr.size == 0:
            self._roi_prev_gray = None
            self._fish_prev_gray = None
            self._roi_motion_dx = 0.0
            self._roi_motion_dy = 0.0
            return roi_bgr, 0.0, 0.0
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._roi_prev_gray is None or self._roi_prev_gray.shape != gray.shape:
            self._roi_prev_gray = gray
            self._roi_motion_dx = 0.0
            self._roi_motion_dy = 0.0
            return roi_bgr, 0.0, 0.0
        (dx, dy), resp = cv2.phaseCorrelate(self._roi_prev_gray, gray)
        self._roi_prev_gray = gray
        if resp is None or float(resp) < self._roi_motion_min_resp:
            dx, dy = 0.0, 0.0
        dx = float(max(-self._roi_motion_max_px, min(self._roi_motion_max_px, dx)))
        dy = float(max(-self._roi_motion_max_px, min(self._roi_motion_max_px, dy)))
        a = self._roi_motion_alpha
        self._roi_motion_dx = self._roi_motion_dx * (1.0 - a) + dx * a
        self._roi_motion_dy = self._roi_motion_dy * (1.0 - a) + dy * a
        m = np.float32([[1.0, 0.0, -self._roi_motion_dx], [0.0, 1.0, -self._roi_motion_dy]])
        stable = cv2.warpAffine(
            roi_bgr,
            m,
            (roi_bgr.shape[1], roi_bgr.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return stable, self._roi_motion_dx, self._roi_motion_dy

    def _handle_state_transition(self, now_ms: int) -> None:
        if self._sm.state == self._last_sm_state:
            return
        if self._sm.state == AutoFishState.MINIGAME:
            self._mini_ready = False
            self._mini_enter_ms = now_ms
            self._mini_prev_zone_y = None
            self._mini_drop_start_y = None
            self._apply_left_hold(False, now_ms=now_ms)
            self.log_cb("minigame warmup: waiting for bar drop")
        elif self._last_sm_state == AutoFishState.MINIGAME:
            self._mini_ready = False
            self._mini_prev_zone_y = None
            self._mini_drop_start_y = None
            self._mini_last_ctrl_ms = 0
            self._mini_last_fish_y = None
            self._mini_fish_vel_ema = 0.0
            self._mini_err_prev = None
            self._mini_mode = "idle"
            self._mini_brake_until_ms = 0
            self._mini_hold_last_ms = 0
            self._mini_hold_active = False
            self._mini_hold_until_ms = 0
            self._mini_release_until_ms = 0
            self._apply_left_hold(False, now_ms=now_ms)
            self._reset_relative_tap()
            self._reset_bottom_rescue()
        self._last_sm_state = self._sm.state

    def _update_minigame_ready(self, zone_y: float | None, now_ms: int) -> bool:
        if self._mini_ready:
            return True
        if now_ms - self._mini_enter_ms >= self._mini_wait_max_ms:
            self._mini_ready = True
            self.log_cb("minigame warmup: timeout, control enabled")
            return True
        if zone_y is None:
            return False
        y = float(zone_y)
        if self._mini_drop_start_y is None:
            self._mini_drop_start_y = y
            self._mini_prev_zone_y = y
            return False
        self._mini_prev_zone_y = y
        if y - float(self._mini_drop_start_y) >= self._mini_drop_need_px:
            self._mini_ready = True
            self.log_cb("minigame warmup: bar drop detected, control enabled")
            return True
        return False

    def _update_zone_y(self, zone_y: float | None, fish_y: float | None):
        """Update zone/fish y and prevent bottom-stuck tracking."""
        bottom = float(max(0.0, float(self.screen_height) - self._screen_bottom_margin_px))
        if zone_y is None:
            z = None
        else:
            z = float(zone_y)
            if z > bottom:
                z = bottom
            z = min(z, bottom)

        if fish_y is None:
            f = None
        else:
            f = float(fish_y)
            if z is not None and z >= bottom:
                f = bottom
            if f > bottom:
                f = bottom
        return z, f

    def minigame_hold(
        self,
        ms: int,
        score: float,
        template: str,
        scale,
        fish_y: float | None,
        zone_y: float | None,
    ) -> None:
        zone_y, fish_y = self._update_zone_y(zone_y, fish_y)
        scale_text = "None" if scale is None else f"{float(scale):.2f}"
        fish_text = "None" if fish_y is None else f"{float(fish_y):.1f}"
        zone_text = "None" if zone_y is None else f"{float(zone_y):.1f}"
        self.log_cb(
            f"minigame_hold ms={int(ms)} score={float(score):.3f} "
            f"template={template} scale={scale_text} fish_y={fish_text} zone_y={zone_text}"
        )

    @staticmethod
    def _calc_hold_ms_from_error(
        abs_err_px: float,
        track_px: float,
        up_full_ms: float = 700.0,
        hold_min_ms: int = 150,
        hold_max_ms: int = 350,
    ) -> int:
        tpx = max(1.0, float(track_px))
        lo = float(min(hold_min_ms, hold_max_ms))
        hi = float(max(hold_min_ms, hold_max_ms))
        ms = lo + (float(up_full_ms) * max(0.0, float(abs_err_px)) / tpx)
        return int(max(lo, min(hi, round(ms))))

    def _run_minigame_controller(
        self,
        fish_y: float,
        zone_y: float,
        zone_top: float | None,
        zone_bottom: float | None,
        bar_bbox,
        now_ms: int,
    ) -> None:
        dt = max(1.0, float(now_ms - self._mini_last_ctrl_ms)) if self._mini_last_ctrl_ms > 0 else 16.0
        if self._mini_last_fish_y is not None:
            inst_v = (fish_y - self._mini_last_fish_y) / dt
            self._mini_fish_vel_ema = self._mini_fish_vel_ema * (1.0 - self._mini_vel_alpha) + inst_v * self._mini_vel_alpha
        self._mini_last_fish_y = fish_y
        self._mini_last_ctrl_ms = now_ms
        # Adaptive prediction: reduce lead when close to target to avoid overshoot.
        err_raw = fish_y - zone_y
        close_k = max(0.15, min(1.0, abs(err_raw) / max(1.0, float(self._mini_far_px))))
        eff_predict_ms = float(self._mini_predict_ms) * close_k
        pred_off = self._mini_fish_vel_ema * eff_predict_ms
        pred_cap = max(3.0, self._mini_far_px * 0.45)
        pred_off = max(-pred_cap, min(pred_cap, pred_off))
        fish_pred = fish_y + pred_off
        err = fish_pred - zone_y
        self._mini.last_control = float(err)
        self._mini.last_pred_fish_y = float(fish_pred)

        want_hold = False
        mode = "track"
        urgent_up = False
        urgent_down = False

        # Edge guard has highest priority.
        if zone_top is not None and fish_y - zone_top <= self._mini_edge_guard_px:
            want_hold = True
            mode = "edge_up"
            urgent_up = True
        elif zone_bottom is not None and zone_bottom - fish_y <= self._mini_edge_guard_px:
            want_hold = False
            mode = "edge_down"
            urgent_down = True
        else:
            # Overshoot brake window when sign flips near target.
            if self._mini_err_prev is not None and (self._mini_err_prev * err < 0.0) and abs(self._mini_err_prev) > self._mini_dead_px:
                self._mini_brake_until_ms = now_ms + self._mini_brake_ms
            if now_ms < self._mini_brake_until_ms:
                want_hold = err < -self._mini_dead_px * 0.4
                mode = "brake"
            elif abs(err) >= self._mini_far_px:
                want_hold = err < 0.0
                if want_hold:
                    urgent_up = True
                else:
                    urgent_down = True
                mode = "catchup"
            elif err < -self._mini_dead_px:
                want_hold = True
                mode = "track_up"
            elif err > self._mini_dead_px * 0.65:
                want_hold = False
                mode = "track_down"
            else:
                want_hold = False
                mode = "coast"
        self._mini_err_prev = err
        self._mini_mode = mode

        # End pulse when time is up.
        if self._mini_hold_active and now_ms >= self._mini_hold_until_ms:
            self._apply_left_hold(False, now_ms=now_ms)
            self._mini_hold_active = False
            self._mini_release_until_ms = now_ms + self._mini_release_lock_ms

        if self._mini_hold_active:
            # Early release branch: when direction turns against current pulse.
            if urgent_down or (mode in {"brake", "track_down"} and err > self._mini_dead_px * 0.2):
                self._apply_left_hold(False, now_ms=now_ms)
                self._mini_hold_active = False
                self._mini_release_until_ms = now_ms + self._mini_release_lock_ms
            else:
                return

        if not want_hold:
            self._apply_left_hold(False, now_ms=now_ms)
            return

        interval_ms = self._mini_hold_interval_catch_ms if mode == "catchup" else self._mini_hold_interval_track_ms
        urgent_press = urgent_up or err < -self._mini_far_px * 1.15
        if not urgent_press and now_ms - self._mini_hold_last_ms < interval_ms:
            return
        if not urgent_press and now_ms < self._mini_release_until_ms:
            return

        track_px = self._mini_track_px_ref
        if bar_bbox is not None:
            try:
                track_px = max(40.0, float(bar_bbox[3] - bar_bbox[1]))
            except Exception:
                track_px = self._mini_track_px_ref
        up_full_ms = self._mini_up_full_ms * (0.90 if mode == "catchup" else 0.75)
        hold_ms = self._calc_hold_ms_from_error(
            abs_err_px=abs(err),
            track_px=track_px,
            up_full_ms=up_full_ms,
            hold_min_ms=self._mini_hold_min_ms,
            hold_max_ms=self._mini_hold_max_ms,
        )
        self._apply_left_hold(True, now_ms=now_ms)
        self._mini_hold_active = True
        self._mini_hold_until_ms = now_ms + hold_ms
        self._mini_hold_last_ms = now_ms
        self.minigame_hold(
            ms=hold_ms,
            score=self._mini_score,
            template=self._mini_template,
            scale=self._mini_scale,
            fish_y=fish_y,
            zone_y=zone_y,
        )

    def _apply_left_hold(self, want_hold: bool, now_ms: int) -> None:
        if want_hold and now_ms < self._hold_cooldown_until_ms:
            want_hold = False
        if want_hold and self._left_hold_active:
            if now_ms - self._left_hold_since_ms >= self._max_hold_ms:
                want_hold = False
                self._hold_cooldown_until_ms = now_ms + 60
                self.log_cb("minigame guard: force release from long hold")
        if want_hold == self._left_hold_active:
            return
        if hasattr(self.input_ctl, "set_left_hold"):
            self.input_ctl.set_left_hold(want_hold)
        self._left_hold_active = want_hold
        if want_hold:
            self._left_hold_since_ms = now_ms

    def _apply_relative_tap(self, rel_y: float, now_ms: int) -> None:
        if rel_y < -self._rel_dead_px:
            hold_ms = self._tap_hold_ms_heavy
            interval_ms = self._tap_interval_ms_heavy
        elif rel_y > self._rel_dead_px:
            hold_ms = self._tap_hold_ms_light
            interval_ms = self._tap_interval_ms_light
        else:
            hold_ms = self._tap_hold_ms_mid
            interval_ms = self._tap_interval_ms_mid

        self._tap_hold_ms_current = hold_ms
        if self._tap_active:
            if now_ms - self._tap_started_ms >= self._tap_hold_ms_current:
                self._apply_left_hold(False, now_ms=now_ms)
                self._tap_active = False
                self._tap_last_ms = now_ms
            return
        if now_ms - self._tap_last_ms >= interval_ms:
            self._apply_left_hold(True, now_ms=now_ms)
            self._tap_active = True
            self._tap_started_ms = now_ms

    def _reset_relative_tap(self) -> None:
        self._tap_active = False

    @staticmethod
    def _zone_near_bottom(zone_bottom: float | None, bar_bbox) -> bool:
        if zone_bottom is None or bar_bbox is None:
            return False
        try:
            bottom_limit = float(bar_bbox[3])
            zb = float(zone_bottom)
        except Exception:
            return False
        return (bottom_limit - zb) <= 2.5

    def _apply_bottom_rescue(self, now_ms: int) -> None:
        if self._bottom_rescue_active:
            if now_ms - self._bottom_rescue_started_ms >= self._bottom_rescue_hold_ms:
                self._apply_left_hold(False, now_ms=now_ms)
                self._bottom_rescue_active = False
                self._bottom_rescue_last_ms = now_ms
            return
        if now_ms - self._bottom_rescue_last_ms >= self._bottom_rescue_interval_ms:
            self._apply_left_hold(True, now_ms=now_ms)
            self._bottom_rescue_active = True
            self._bottom_rescue_started_ms = now_ms

    def _reset_bottom_rescue(self) -> None:
        self._bottom_rescue_active = False

    def _stabilize_measurements(self, fish_y, zone_y, zone_top, zone_bottom):
        fish_s = self._smooth_signal("fish_y", fish_y)
        top_s = self._smooth_signal("zone_top", zone_top)
        bot_s = self._smooth_signal("zone_bottom", zone_bottom)
        if top_s is not None and bot_s is not None:
            if bot_s < top_s:
                top_s, bot_s = bot_s, top_s
            center_s = (top_s + bot_s) * 0.5
        else:
            center_s = self._smooth_signal("zone_center", zone_y)
        return fish_s, center_s, top_s, bot_s

    def _smooth_signal(self, key: str, value: float | None) -> float | None:
        if value is None:
            return None
        v = float(value)
        prev = self._smooth_values.get(key)
        if prev is not None and abs(v - prev) > self._signal_jump_px:
            v = prev + (self._signal_jump_px if v > prev else -self._signal_jump_px)
        if prev is None:
            smoothed = v
        else:
            smoothed = prev * (1.0 - self._signal_alpha) + v * self._signal_alpha
        self._smooth_values[key] = smoothed
        hist = self._signal_hist.get(key)
        if hist is None:
            return smoothed
        hist.append(smoothed)
        if len(hist) < 3:
            return smoothed
        vals = sorted(hist)
        return vals[1]
