from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AutoFishConfig:
    conf_yolo0: float = 0.75
    conf_yolo1: float = 0.75
    roi_expand: float = 0.2
    success_disappear_ms: int = 500
    cast_wait_s: float = 1.0
    move_back_s: float = 0.5
    move_forward_s: float = 0.5
    loop_fps: int = 60
    infer_fps: int = 60
    imgsz: int = 640
    input_retry_limit: int = 12
    roi_lock_delay_ms: int = 500
    mini_drop_need_px: float = 3.0
    mini_wait_max_ms: int = 1200
    mini_signal_timeout_ms: int = 100
    mini_vel_alpha: float = 0.35
    mini_predict_ms: int = 140
    mini_dead_px: float = 3.0
    mini_far_px: float = 22.0
    mini_edge_guard_px: float = 2.5
    mini_brake_ms: int = 140
    mini_hold_interval_track_ms: int = 120
    mini_hold_interval_catch_ms: int = 70
    mini_track_px_ref: float = 90.0
    mini_up_full_ms: float = 700.0
    mini_hold_min_ms: int = 150
    mini_hold_max_ms: int = 350
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
