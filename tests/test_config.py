from pathlib import Path

from autofish.config import AutoFishConfig, pick_default_model_path, resolve_model_path


def test_default_config_values():
    cfg = AutoFishConfig()
    assert cfg.cast_wait_s == 1.0
    assert cfg.move_back_s == 0.5
    assert cfg.move_forward_s == 0.5
    assert cfg.success_disappear_ms == 500


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
