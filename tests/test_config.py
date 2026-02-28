from pathlib import Path

from autofish.config import AutoFishConfig, pick_default_model_path, resolve_model_path


def test_default_config_values():
    cfg = AutoFishConfig()
    assert cfg.conf_yolo0 == 0.75
    assert cfg.conf_yolo1 == 0.75
    assert cfg.cast_wait_s == 1.0
    assert cfg.move_back_s == 0.5
    assert cfg.move_forward_s == 0.5
    assert cfg.success_disappear_ms == 500
    assert cfg.infer_fps == 60
    assert cfg.loop_fps == 60
    assert cfg.mini_up_full_ms == 700.0
    assert cfg.mini_hold_min_ms == 150
    assert cfg.mini_hold_max_ms == 350


def test_resolve_model_path_prefers_explicit(tmp_path: Path):
    model = tmp_path / "custom.pt"
    model.write_bytes(b"test")
    got = resolve_model_path(str(model), tmp_path)
    assert got == model


def test_pick_default_model_path_prefers_trained_best(tmp_path: Path):
    app = tmp_path
    trained = app / "yolo_train" / "runs" / "autofishing_exp2" / "weights"
    trained.mkdir(parents=True)
    best = trained / "best.pt"
    best.write_bytes(b"x")
    fallback = app / "yolo_train" / "yolo11n.pt"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    fallback.write_bytes(b"y")
    assert pick_default_model_path(app) == best


def test_pick_default_model_path_falls_back_to_yolo11n(tmp_path: Path):
    app = tmp_path
    fallback = app / "yolo_train" / "yolo11n.pt"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    fallback.write_bytes(b"y")
    assert pick_default_model_path(app) == fallback
