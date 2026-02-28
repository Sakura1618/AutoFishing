from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class VisionResult:
    has_bite: bool
    has_bar: bool
    bar_bbox: tuple[int, int, int, int] | None
    fish_y: float | None
    zone_y: float | None


class YoloVision:
    def __init__(self, model_path: str, conf_yolo0: float = 0.5, conf_yolo1: float = 0.5) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf_yolo0 = conf_yolo0
        self.conf_yolo1 = conf_yolo1

    def detect(self, frame: np.ndarray | None, imgsz: int = 640) -> dict[str, Any]:
        if frame is None:
            return {"has_bite": False, "has_bar": False, "bar_bbox": None, "fish_y": None, "zone_y": None, "boxes": []}
        result = self.model.predict(frame, verbose=False, conf=min(self.conf_yolo0, self.conf_yolo1), max_det=20, imgsz=imgsz)[0]
        has_bite = False
        bar_bbox: tuple[int, int, int, int] | None = None
        boxes: list[dict[str, Any]] = []
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            if cls_id == 0 and conf < self.conf_yolo0:
                continue
            if cls_id == 1 and conf < self.conf_yolo1:
                continue
            boxes.append({"cls": cls_id, "conf": conf, "bbox": (x1, y1, x2, y2)})
            if cls_id == 0 and conf >= self.conf_yolo0:
                has_bite = True
            if cls_id == 1 and conf >= self.conf_yolo1:
                bar_bbox = (x1, y1, x2, y2)
        fish_y, zone_y = estimate_fish_and_zone(frame, bar_bbox)
        return {
            "has_bite": has_bite,
            "has_bar": bar_bbox is not None,
            "bar_bbox": bar_bbox,
            "fish_y": fish_y,
            "zone_y": zone_y,
            "boxes": boxes,
        }


def estimate_fish_and_zone(frame: np.ndarray, bar_bbox: tuple[int, int, int, int] | None) -> tuple[float | None, float | None]:
    if frame is None or bar_bbox is None:
        return None, None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bar_bbox
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None, None
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    coords = np.where(white_mask > 0)[0]
    zone_y = (float(np.mean(coords)) + y1) if coords.size else (y1 + y2) / 2.0

    # fish candidate: darkest region in bar (heuristic)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    min_val, _, min_loc, _ = cv2.minMaxLoc(blur)
    fish_y = float(min_loc[1] + y1) if min_val < 150 else None
    return fish_y, zone_y
