from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


class HoldAction(str, Enum):
    HOLD = "hold"
    RELEASE = "release"
    KEEP = "keep"


class TrackMode(str, Enum):
    TRACK = "track"
    CATCHUP = "catchup"
    BRAKE = "brake"


@dataclass(slots=True)
class MatchHit:
    fish_y: float
    score: float
    bbox: tuple[int, int, int, int]
    template_name: str


@dataclass(slots=True)
class TemplatePack:
    name: str
    edge: np.ndarray


@dataclass(slots=True)
class WhiteZoneBand:
    top: int
    bottom: int
    center: float
    strength: float


class MinigameController:
    def __init__(
        self,
        dead_zone_px: float = 4.0,
        kp: float = 1.0,
        kd: float = 0.75,
        far_px: float = 24.0,
        near_px: float = 12.0,
        predict_ms: int = 120,
        fish_vel_alpha: float = 0.35,
        zone_vel_alpha: float = 0.35,
        brake_window_px: float = 20.0,
        brake_base_px: float = 3.0,
        brake_speed_gain: float = 0.08,
        target_bias_px: float = 0.0,
        edge_guard_px: float = 3.5,
        duration_scale_px: float = 28.0,
        hold_time_factor: float = 0.65,
        min_hold_ms: int = 45,
        max_hold_ms: int = 180,
        min_release_ms: int = 40,
        max_release_ms: int = 160,
    ) -> None:
        self.dead_zone_px = dead_zone_px
        self.kp = kp
        self.kd = kd
        self.far_px = max(0.0, far_px)
        self.near_px = max(0.0, min(near_px, far_px))
        self.predict_s = max(0.0, predict_ms / 1000.0)
        self.fish_vel_alpha = min(1.0, max(0.0, fish_vel_alpha))
        self.zone_vel_alpha = min(1.0, max(0.0, zone_vel_alpha))
        self.brake_window_px = max(0.0, brake_window_px)
        self.brake_base_px = max(0.0, brake_base_px)
        self.brake_speed_gain = max(0.0, brake_speed_gain)
        self.target_bias_px = float(target_bias_px)
        self.edge_guard_px = max(0.0, edge_guard_px)
        self.duration_scale_px = max(1.0, duration_scale_px)
        self.hold_time_factor = min(1.0, max(0.2, hold_time_factor))
        self.min_hold_ms = min_hold_ms
        self.max_hold_ms = max(max_hold_ms, min_hold_ms)
        self.min_release_ms = min_release_ms
        self.max_release_ms = max(max_release_ms, min_release_ms)
        self._last = HoldAction.RELEASE
        self._pressed = False
        self._last_error: float | None = None
        self._last_ts_ms: int | None = None
        self._last_fish_y: float | None = None
        self._last_zone_y: float | None = None
        self._fish_vel_ema: float = 0.0
        self._zone_vel_ema: float = 0.0
        self._mode = TrackMode.TRACK
        self._switch_allowed_at_ms: int = 0
        self.last_control: float = 0.0
        self.last_mode: str = self._mode.value
        self.last_pred_fish_y: float | None = None

    def decide(
        self,
        fish_y: float,
        zone_center_y: float,
        now_ms: int | None = None,
        zone_top_y: float | None = None,
        zone_bottom_y: float | None = None,
    ) -> HoldAction:
        if now_ms is None:
            now_ms = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())
        first_sample = self._last_ts_ms is None
        dt_s = 0.0
        if self._last_ts_ms is not None:
            dt_s = max(0.001, (now_ms - self._last_ts_ms) / 1000.0)
            fish_v = (fish_y - float(self._last_fish_y)) / dt_s if self._last_fish_y is not None else 0.0
            zone_v = (zone_center_y - float(self._last_zone_y)) / dt_s if self._last_zone_y is not None else 0.0
            self._fish_vel_ema = (1.0 - self.fish_vel_alpha) * self._fish_vel_ema + self.fish_vel_alpha * fish_v
            self._zone_vel_ema = (1.0 - self.zone_vel_alpha) * self._zone_vel_ema + self.zone_vel_alpha * zone_v
        fish_pred = fish_y + self._fish_vel_ema * self.predict_s
        edge_dead_zone = self.dead_zone_px
        zone_target = zone_center_y + self.target_bias_px
        error = zone_target - fish_pred
        force_action: HoldAction | None = None
        if zone_top_y is not None and zone_bottom_y is not None and zone_bottom_y > zone_top_y:
            zone_target = min(max(zone_target, zone_top_y), zone_bottom_y)
            height = max(1.0, zone_bottom_y - zone_top_y)
            edge_dead_zone = max(self.dead_zone_px, height * 0.2)
            error = zone_target - fish_pred
            dist_to_top = fish_pred - zone_top_y
            dist_to_bottom = zone_bottom_y - fish_pred
            if dist_to_top <= self.edge_guard_px:
                force_action = HoldAction.RELEASE
            elif dist_to_bottom <= self.edge_guard_px:
                force_action = HoldAction.HOLD
        d_error = 0.0 if self._last_error is None else (error - self._last_error)
        control = self.kp * error + self.kd * d_error
        self.last_control = float(control)
        self.last_pred_fish_y = float(fish_pred)

        if abs(error) >= self.far_px:
            self._mode = TrackMode.CATCHUP
        elif self._mode == TrackMode.CATCHUP and abs(error) > self.near_px:
            self._mode = TrackMode.CATCHUP
        else:
            self._mode = TrackMode.TRACK

        want_hold = self._pressed
        if force_action is not None:
            want_hold = force_action == HoldAction.HOLD
            self._mode = TrackMode.BRAKE
        elif self._mode == TrackMode.CATCHUP:
            want_hold = error > 0
        else:
            if self._pressed:
                want_hold = not self._should_pre_brake_hold(error)
                if error < -edge_dead_zone:
                    want_hold = False
            else:
                want_hold = self._should_pre_brake_release(error)
                if error > edge_dead_zone:
                    want_hold = True

        if self._mode == TrackMode.TRACK and want_hold != self._pressed:
            self._mode = TrackMode.BRAKE

        self.last_mode = self._mode.value
        self._last_error = float(error)
        self._last_ts_ms = now_ms
        self._last_fish_y = float(fish_y)
        self._last_zone_y = float(zone_center_y)

        if first_sample:
            self._pressed = want_hold
            self._switch_allowed_at_ms = now_ms + self._next_lock_ms(control=control, for_hold=want_hold)
            self._last = HoldAction.HOLD if want_hold else HoldAction.RELEASE
            return self._last

        if force_action is not None and want_hold != self._pressed:
            # Emergency edge guard: switch immediately and enforce minimal hold/release time.
            self._pressed = want_hold
            self._switch_allowed_at_ms = now_ms + self._next_lock_ms(control=control, for_hold=want_hold)
            self._last = HoldAction.HOLD if want_hold else HoldAction.RELEASE
            return self._last

        if want_hold != self._pressed and now_ms >= self._switch_allowed_at_ms:
            self._pressed = want_hold
            self._switch_allowed_at_ms = now_ms + self._next_lock_ms(control=control, for_hold=want_hold)
            self._last = HoldAction.HOLD if want_hold else HoldAction.RELEASE
            return self._last
        return HoldAction.KEEP

    def _next_lock_ms(self, control: float, for_hold: bool) -> int:
        strength = min(1.0, abs(control) / self.duration_scale_px)
        if for_hold:
            base = self.min_hold_ms + (self.max_hold_ms - self.min_hold_ms) * strength
            return int(round(base * self.hold_time_factor))
        return int(round(self.min_release_ms + (self.max_release_ms - self.min_release_ms) * strength))

    def _should_pre_brake_hold(self, error: float) -> bool:
        if abs(error) > self.brake_window_px:
            return False
        up_speed = max(0.0, -self._zone_vel_ema)
        threshold = self.brake_base_px + self.brake_speed_gain * up_speed
        return error <= threshold

    def _should_pre_brake_release(self, error: float) -> bool:
        if abs(error) > self.brake_window_px:
            return False
        down_speed = max(0.0, self._zone_vel_ema)
        threshold = self.brake_base_px + self.brake_speed_gain * down_speed
        return error >= -threshold


class FishTemplateMatcher:
    def __init__(
        self,
        templates: Sequence[TemplatePack],
        scales: Sequence[float] = (0.9, 1.0, 1.1),
        threshold: float = 0.55,
        lost_hold_ms: int = 300,
    ) -> None:
        self.templates = list(templates)
        self.scales = list(scales)
        self.threshold = threshold
        self.lost_hold_ms = lost_hold_ms
        self._last_hit: MatchHit | None = None
        self._last_hit_ms: int = 0

    @classmethod
    def from_template_dir(
        cls,
        template_dir: Path,
        scales: Sequence[float] = (0.9, 1.0, 1.1),
        threshold: float = 0.55,
        lost_hold_ms: int = 300,
    ) -> "FishTemplateMatcher":
        packs: list[TemplatePack] = []
        files = sorted(template_dir.glob("*.png")) + sorted(template_dir.glob("*.jpg")) + sorted(template_dir.glob("*.jpeg"))
        files = sorted(files, key=lambda p: (0 if p.name.lower() == "fish.png" else 1, p.name.lower()))
        for p in files:
            edge = _template_edge_from_file(p)
            if edge is None:
                continue
            if edge.shape[0] < 6 or edge.shape[1] < 6:
                continue
            packs.append(TemplatePack(name=p.name, edge=edge))
        if not packs:
            raise RuntimeError(f"No valid fish templates loaded from: {template_dir}")
        return cls(packs, scales=scales, threshold=threshold, lost_hold_ms=lost_hold_ms)

    def locate(self, roi_bgr: np.ndarray, now_ms: int) -> MatchHit | None:
        if roi_bgr is None or roi_bgr.size == 0:
            return self._hold_last(now_ms)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        edge_roi = cv2.Canny(gray, 50, 120)
        best: MatchHit | None = None
        for pack in self.templates:
            for s in self.scales:
                t = _resize_edge(pack.edge, s)
                th, tw = t.shape[:2]
                rh, rw = edge_roi.shape[:2]
                if th < 4 or tw < 4 or th >= rh or tw >= rw:
                    continue
                result = cv2.matchTemplate(edge_roi, t, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                x, y = max_loc
                hit = MatchHit(
                    fish_y=float(y + th / 2.0),
                    score=float(max_val),
                    bbox=(x, y, x + tw, y + th),
                    template_name=pack.name,
                )
                if best is None or hit.score > best.score:
                    best = hit
        if best is not None and best.score >= self.threshold:
            self._last_hit = best
            self._last_hit_ms = now_ms
            return best
        return self._hold_last(now_ms)

    def _hold_last(self, now_ms: int) -> MatchHit | None:
        if self._last_hit is None:
            return None
        if now_ms - self._last_hit_ms <= self.lost_hold_ms:
            return self._last_hit
        return None


def estimate_white_zone_center(roi_bgr: np.ndarray) -> float | None:
    band = detect_white_zone_band(roi_bgr)
    return None if band is None else band.center


def detect_white_zone_band(roi_bgr: np.ndarray) -> WhiteZoneBand | None:
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8))
    h, w = mask.shape[:2]
    if h < 5 or w < 5:
        return None
    row_ratio = (mask > 0).mean(axis=1)
    rows = np.where(row_ratio >= 0.18)[0]
    if rows.size == 0:
        return None
    runs: list[tuple[int, int]] = []
    start = int(rows[0])
    prev = int(rows[0])
    for y in rows[1:]:
        y = int(y)
        if y == prev + 1:
            prev = y
            continue
        runs.append((start, prev))
        start = y
        prev = y
    runs.append((start, prev))
    top, bottom = max(runs, key=lambda r: (r[1] - r[0] + 1))
    if bottom - top + 1 < 3:
        return None
    strength = float(row_ratio[top : bottom + 1].mean())
    return WhiteZoneBand(top=top, bottom=bottom, center=(top + bottom) / 2.0, strength=strength)


def _resize_edge(edge: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return edge
    h, w = edge.shape[:2]
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(edge, (nw, nh), interpolation=cv2.INTER_AREA)


def _extract_fish_shape_crop(img_bgr: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 50, 120)
    cnts, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    best = None
    best_area = 0
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        # filter out long rod-like contour and borders
        if area < 40:
            continue
        if ch > h * 0.75 and cw < w * 0.35:
            continue
        if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
            continue
        if area > best_area:
            best_area = area
            best = (x, y, cw, ch)
    if best is None:
        return None
    x, y, cw, ch = best
    pad = 2
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + cw + pad)
    y2 = min(h, y + ch + pad)
    return gray[y1:y2, x1:x2]


def _template_edge_from_file(path: Path) -> np.ndarray | None:
    # Prefer transparent fish template if available (fish.png with alpha channel).
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        _, mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
        edge = cv2.Canny(mask, 50, 120)
        if np.count_nonzero(edge) > 10:
            return edge
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    crop = _extract_fish_shape_crop(bgr)
    if crop is None:
        return None
    return cv2.Canny(crop, 50, 120)
