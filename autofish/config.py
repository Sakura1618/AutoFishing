from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AutoFishConfig:
    conf_yolo0: float = 0.5
    conf_yolo1: float = 0.5
    roi_expand: float = 0.2
    success_disappear_ms: int = 500
    cast_wait_s: float = 1.0
    move_back_s: float = 0.5
    move_forward_s: float = 0.5
    loop_fps: int = 20
    input_retry_limit: int = 12


def resolve_model_path(model_arg: str, app_dir: Path) -> Path:
    model = Path(model_arg)
    if model.is_absolute():
        return model
    local_model = app_dir / model_arg
    if local_model.exists():
        return local_model
    return model

