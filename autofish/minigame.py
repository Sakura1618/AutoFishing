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
        for p in sorted(template_dir.glob("*.png")) + sorted(template_dir.glob("*.jpg")) + sorted(template_dir.glob("*.jpeg")):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            crop = _extract_fish_shape_crop(img)
            if crop is None:
                continue
            edge = cv2.Canny(crop, 50, 120)
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
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)
    ys = np.where(mask > 0)[0]
    if ys.size == 0:
        return None
    return float(np.mean(ys))


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

