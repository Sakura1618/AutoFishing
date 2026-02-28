from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AutoFishConfig:
    conf_yolo0: float = 0.5
    conf_yolo1: float = 0.5
    roi_expand: float = 0.2
    success_disappear_ms: int = 500
    cast_wait_s: float = 0.5
    move_back_s: float = 0.5
    move_forward_s: float = 0.5
    loop_fps: int = 20
    infer_fps: int = 10
    imgsz: int = 640
    input_retry_limit: int = 12
    osc_host: str = "127.0.0.1"
    osc_port: int = 9000
    osc_click_button: str = "/input/UseRight"
    osc_click_axis: str = "/input/UseAxisRight"
    osc_vertical_axis: str = "/input/Vertical"


def resolve_model_path(model_arg: str, app_dir: Path) -> Path:
    model = Path(model_arg)
    if model.is_absolute():
        return model
    local_model = app_dir / model_arg
    if local_model.exists():
        return local_model
    return model


def pick_default_model_path(app_dir: Path) -> Path:
    candidates = [
        app_dir / "yolo_train" / "runs" / "autofishing_exp2" / "weights" / "best.pt",
        app_dir / "yolo_train" / "runs" / "autofishing_exp" / "weights" / "best.pt",
        app_dir / "yolo_train" / "yolo11n.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[-1]
