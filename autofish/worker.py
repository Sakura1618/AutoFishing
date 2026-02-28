from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

import cv2
from pathlib import Path

from .config import AutoFishConfig
from .minigame import FishTemplateMatcher, HoldAction, MinigameController, detect_dark_blob_center, detect_white_zone_band
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
        self._sm = FishingStateMachine(
            cast_wait_s=cfg.cast_wait_s,
            move_back_s=cfg.move_back_s,
            move_forward_s=cfg.move_forward_s,
            success_disappear_ms=cfg.success_disappear_ms,
        )
        self._mini = MinigameController(hold_decreases_y=False)
        try:
            self._matcher = FishTemplateMatcher.from_template_dir(
                Path.cwd() / "img",
                threshold=0.55,
                scales=(0.9, 1.0, 1.1),
                lost_hold_ms=300,
                local_expand=2.1,
                local_track_ms=300,
                smooth_alpha=0.40,
            )
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
        self._mini_score = 0.0
        self._mini_template = ""
        self._last_hold_action: HoldAction | None = None
        self._loop_stat_count = 0
        self._infer_stat_count = 0
        self._stat_last_ts = time.time()
        self._last_sm_state = self._sm.state
        self._mini_ready = False
        self._mini_enter_ms = 0
        self._mini_prev_zone_y: float | None = None
        self._mini_drop_start_y: float | None = None
        self._mini_drop_need_px = 3.0
        self._mini_wait_max_ms = 1200
        self._mini_signal_timeout_ms = 100
        self._last_mini_signal_ms = 0
        self._left_hold_active = False
        self._left_hold_since_ms = 0
        self._max_hold_ms = 260
        self._hold_cooldown_until_ms = 0
        self._keep_tap_interval_ms = 90
        self._keep_tap_hold_ms = 22
        self._keep_tap_last_ms = 0
        self._keep_tap_active = False
        self._keep_tap_started_ms = 0
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
            selected_bar = self._select_bar_bbox(annotated_boxes)
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
                if frame is None:
                    self.log_cb("diag: capture frame is None")
                else:
                    self.log_cb(
                        f"diag: has_bite={bool(det.get('has_bite'))}, "
                        f"has_bar={bool(det.get('has_bar'))}, fish_y={det.get('fish_y')}, zone_y={det.get('zone_y')}, "
                        f"origin={frame_origin}"
                    )
                    self.log_cb(f"diag: mini score={self._mini_score:.3f}, template={self._mini_template}")
                    self.log_cb(
                        f"diag: yolo_hits_2s bite={self._bite_hits}, bar={self._bar_hits}, "
                        f"conf0={self.cfg.conf_yolo0:.2f}, conf1={self.cfg.conf_yolo1:.2f}, imgsz={self.cfg.imgsz}"
                    )
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
                self._smooth_values.clear()
                for hist in self._signal_hist.values():
                    hist.clear()
                self._last_mini_signal_ms = 0
                self._left_hold_active = False
                self._left_hold_since_ms = 0
                self._hold_cooldown_until_ms = 0
                self._keep_tap_last_ms = 0
                self._keep_tap_active = False
                self._keep_tap_started_ms = 0
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
                    self._reset_keep_tap()
                    self._reset_bottom_rescue()
                    self._last_hold_action = None
                elif fish_y is not None and zone_y is not None:
                    self._reset_bottom_rescue()
                    self._last_mini_signal_ms = now_ms
                    conservative = self._mini_template == "fallback-dark" or self._mini_score < 0.58
                    action = self._mini.decide(
                        fish_y=float(fish_y),
                        zone_center_y=float(zone_y),
                        zone_top_y=det.get("zone_top"),
                        zone_bottom_y=det.get("zone_bottom"),
                        conservative_mode=conservative,
                        now_ms=now_ms,
                    )
                    if action == HoldAction.HOLD:
                        self._reset_keep_tap()
                        self._apply_left_hold(True, now_ms=now_ms)
                        if self._last_hold_action != action:
                            self.log_cb("minigame action: HOLD")
                    elif action == HoldAction.RELEASE:
                        self._reset_keep_tap()
                        self._apply_left_hold(False, now_ms=now_ms)
                        if self._last_hold_action != action:
                            self.log_cb("minigame action: RELEASE")
                    else:
                        if self._fish_inside_white_zone(float(fish_y), det.get("zone_top"), det.get("zone_bottom")):
                            self._apply_keep_tap(now_ms=now_ms)
                        else:
                            self._reset_keep_tap()
                    self._last_hold_action = action
                elif now_ms - self._last_mini_signal_ms >= self._mini_signal_timeout_ms:
                    if self._zone_near_bottom(det.get("zone_bottom"), det.get("bar_bbox")):
                        self._apply_bottom_rescue(now_ms=now_ms)
                        self._last_hold_action = HoldAction.HOLD if self._bottom_rescue_active else HoldAction.RELEASE
                    else:
                        self._apply_left_hold(False, now_ms=now_ms)
                        self._reset_bottom_rescue()
                        self._last_hold_action = None
                    self._reset_keep_tap()

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
                roi = frame[y1:y2, x1:x2].copy()
                fish_y = det.get("fish_y")
                zone_y = det.get("zone_y")
                if fish_y is not None:
                    fy = int(float(fish_y) - y1)
                    cv2.line(roi, (0, fy), (roi.shape[1] - 1, fy), (0, 0, 255), 2)
                if zone_y is not None:
                    zy = int(float(zone_y) - y1)
                    cv2.line(roi, (0, zy), (roi.shape[1] - 1, zy), (255, 255, 255), 2)
                zone_top = det.get("zone_top")
                if zone_top is not None:
                    zt = int(float(zone_top) - y1)
                    cv2.line(roi, (0, zt), (roi.shape[1] - 1, zt), (255, 220, 80), 1)
                zone_bottom = det.get("zone_bottom")
                if zone_bottom is not None:
                    zb = int(float(zone_bottom) - y1)
                    cv2.line(roi, (0, zb), (roi.shape[1] - 1, zb), (255, 220, 80), 1)
                cv2.putText(
                    roi,
                    f"score:{self._mini_score:.2f} {self._mini_template}",
                    (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 220, 0),
                    1,
                )
                action_text = self._last_hold_action.value.upper() if self._last_hold_action is not None else "NONE"
                action_color = (0, 220, 0) if action_text == "HOLD" else (0, 120, 255) if action_text == "RELEASE" else (200, 200, 200)
                cv2.putText(
                    roi,
                    f"action:{action_text}",
                    (4, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    action_color,
                    2,
                )
                cv2.putText(
                    roi,
                    f"ctl:{self._mini.last_control:+.1f}",
                    (4, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (120, 220, 255),
                    1,
                )
                cv2.putText(
                    roi,
                    f"mode:{self._mini.last_mode}",
                    (4, 76),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (120, 255, 160),
                    1,
                )
                if self._sm.state == AutoFishState.MINIGAME and not self._mini_ready:
                    cv2.putText(
                        roi,
                        "ready:WAIT_DROP",
                        (4, 94),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (80, 200, 255),
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

    def _select_bar_bbox(self, boxes):
        # class-1 before hook click should not be used as ROI.
        class1 = [b for b in boxes if int(b.get("cls", -1)) == 1]
        if self._sm.state == AutoFishState.WAIT_BITE:
            if self._await_next_yolo1 and class1:
                now_ms = int(time.time() * 1000)
                self._update_roi_anchor(class1[0]["bbox"], now_ms=now_ms)
                self._await_next_yolo1 = False
                return self._roi_anchor_bbox
            return None
        if self._sm.state == AutoFishState.MINIGAME:
            now_ms = int(time.time() * 1000)
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
        if frame is None or bar_bbox is None:
            return None, None, None, None
        x1, y1, x2, y2 = bar_bbox
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None, None, None, None
        roi = frame[y1:y2, x1:x2]
        if self._matcher is None:
            return None, None, None, None
        band = detect_white_zone_band(roi)
        hit = self._matcher.locate(roi, now_ms=now_ms)
        fish_y = None
        if hit is not None:
            fish_y = y1 + hit.fish_y
            self._mini_score = hit.score
            self._mini_template = hit.template_name
        else:
            # Fallback: detect darkest connected blob center, less noisy than min-pixel.
            prefer_local_y = None
            if band is not None:
                prefer_local_y = float(band.center)
            fb_y = detect_dark_blob_center(roi, prefer_y=prefer_local_y)
            if fb_y is not None:
                fish_y = y1 + float(fb_y)
                self._mini_score = 0.0
                self._mini_template = "fallback-dark"
        if band is None:
            return fish_y, None, None, None
        zone_y = y1 + band.center
        zone_top = y1 + float(band.top)
        zone_bottom = y1 + float(band.bottom)
        return fish_y, zone_y, zone_top, zone_bottom

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
            self._apply_left_hold(False, now_ms=now_ms)
            self._reset_keep_tap()
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

    @staticmethod
    def _fish_inside_white_zone(fish_y: float, zone_top: float | None, zone_bottom: float | None) -> bool:
        if zone_top is None or zone_bottom is None:
            return False
        top = float(zone_top) + 1.0
        bottom = float(zone_bottom) - 1.0
        return top <= fish_y <= bottom

    def _apply_keep_tap(self, now_ms: int) -> None:
        if self._keep_tap_active:
            if now_ms - self._keep_tap_started_ms >= self._keep_tap_hold_ms:
                self._apply_left_hold(False, now_ms=now_ms)
                self._keep_tap_active = False
                self._keep_tap_last_ms = now_ms
            return
        if now_ms - self._keep_tap_last_ms >= self._keep_tap_interval_ms:
            self._apply_left_hold(True, now_ms=now_ms)
            self._keep_tap_active = True
            self._keep_tap_started_ms = now_ms

    def _reset_keep_tap(self) -> None:
        self._keep_tap_active = False

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
