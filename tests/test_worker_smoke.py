import cv2
import numpy as np

from autofish.config import AutoFishConfig
from autofish.state_machine import AutoFishState
from autofish.worker import AutoFishWorker


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return {"has_bite": False, "has_bar": False, "bar_bbox": None}


class FakeCapture:
    def grab(self):
        return None


class FakeInput:
    def __init__(self) -> None:
        self.hold_states = []

    def click_left(self):
        return True

    def set_left_hold(self, hold: bool):
        self.hold_states.append(bool(hold))


def test_worker_starts_and_stops_cleanly():
    logs = []
    worker = AutoFishWorker(
        cfg=AutoFishConfig(loop_fps=10),
        detector=FakeDetector(),
        capture=FakeCapture(),
        input_ctl=FakeInput(),
        log_cb=logs.append,
    )
    worker.start()
    worker.stop()
    assert any("started" in msg for msg in logs)
    assert any("stopped" in msg for msg in logs)


def test_worker_uses_normal_y_axis_for_minigame():
    worker = AutoFishWorker(
        cfg=AutoFishConfig(loop_fps=10),
        detector=FakeDetector(),
        capture=FakeCapture(),
        input_ctl=FakeInput(),
        log_cb=lambda _x: None,
    )
    assert worker._mini.hold_decreases_y is True


def test_worker_prefers_fish_png_template_if_exists():
    worker = AutoFishWorker(
        cfg=AutoFishConfig(loop_fps=10),
        detector=FakeDetector(),
        capture=FakeCapture(),
        input_ctl=FakeInput(),
        log_cb=lambda _x: None,
    )
    assert worker._template_file.name == "fish.png"


def test_worker_normalize_capture_result_tuple():
    frame, origin = AutoFishWorker._normalize_capture_result((object(), (123, 456)))
    assert origin == (123, 456)
    assert frame is not None


def test_bbox_iou_for_overlap():
    iou = AutoFishWorker._bbox_iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert iou > 0


def test_keep_roi_anchor_when_class1_flickers():
    worker = AutoFishWorker(
        cfg=AutoFishConfig(loop_fps=10),
        detector=FakeDetector(),
        capture=FakeCapture(),
        input_ctl=FakeInput(),
        log_cb=lambda _x: None,
    )
    worker._sm.state = AutoFishState.MINIGAME
    worker._roi_anchor_bbox = (10, 20, 30, 60)
    worker._roi_anchor_last_seen_ms = 10_000
    # simulate within hold window (<=1500ms)
    import time
    now_ms = int(time.time() * 1000)
    worker._roi_anchor_last_seen_ms = now_ms - 500
    got = worker._select_bar_bbox([])
    assert got == (10, 20, 30, 60)


def test_roi_anchor_smooths_large_jitter():
    worker = AutoFishWorker(
        cfg=AutoFishConfig(loop_fps=10),
        detector=FakeDetector(),
        capture=FakeCapture(),
        input_ctl=FakeInput(),
        log_cb=lambda _x: None,
    )
    worker._roi_anchor_bbox = (100, 100, 140, 180)
    worker._update_roi_anchor((140, 130, 180, 210), now_ms=1000)
    # With smoothing alpha=0.2 this should move only part-way.
    assert worker._roi_anchor_bbox == (108, 106, 148, 186)


def test_signal_smoothing_clamps_spike():
    worker = AutoFishWorker(
        cfg=AutoFishConfig(loop_fps=10),
        detector=FakeDetector(),
        capture=FakeCapture(),
        input_ctl=FakeInput(),
        log_cb=lambda _x: None,
    )
    s1 = worker._smooth_signal("fish_y", 100.0)
    s2 = worker._smooth_signal("fish_y", 200.0)
    assert s1 is not None
    assert s2 is not None
    # Spike should be clamped and smoothed, not jump to 200.
    assert s2 < 130.0


def test_minigame_ready_after_bar_drop():
    worker = AutoFishWorker(
        cfg=AutoFishConfig(loop_fps=10),
        detector=FakeDetector(),
        capture=FakeCapture(),
        input_ctl=FakeInput(),
        log_cb=lambda _x: None,
    )
    worker._mini_enter_ms = 1000
    worker._mini_ready = False
    assert worker._update_minigame_ready(zone_y=100.0, now_ms=1010) is False
    assert worker._update_minigame_ready(zone_y=102.0, now_ms=1030) is False
    assert worker._update_minigame_ready(zone_y=104.2, now_ms=1060) is True


def test_minigame_ready_timeout_fallback():
    worker = AutoFishWorker(
        cfg=AutoFishConfig(loop_fps=10),
        detector=FakeDetector(),
        capture=FakeCapture(),
        input_ctl=FakeInput(),
        log_cb=lambda _x: None,
    )
    worker._mini_enter_ms = 1000
    worker._mini_ready = False
    assert worker._update_minigame_ready(zone_y=None, now_ms=2205) is True


def test_apply_left_hold_forces_release_after_max_hold():
    fake_in = FakeInput()
    worker = AutoFishWorker(
        cfg=AutoFishConfig(loop_fps=10),
        detector=FakeDetector(),
        capture=FakeCapture(),
        input_ctl=fake_in,
        log_cb=lambda _x: None,
    )
    worker._max_hold_ms = 50
    worker._apply_left_hold(True, now_ms=1000)
    worker._apply_left_hold(True, now_ms=1060)
    assert fake_in.hold_states[0] is True
    assert fake_in.hold_states[-1] is False


def test_apply_left_hold_respects_cooldown():
    fake_in = FakeInput()
    worker = AutoFishWorker(
        cfg=AutoFishConfig(loop_fps=10),
        detector=FakeDetector(),
        capture=FakeCapture(),
        input_ctl=fake_in,
        log_cb=lambda _x: None,
    )
    worker._hold_cooldown_until_ms = 2000
    worker._apply_left_hold(True, now_ms=1500)
    assert fake_in.hold_states == []


def test_relative_tap_generates_short_hold_pulse():
    fake_in = FakeInput()
    worker = AutoFishWorker(
        cfg=AutoFishConfig(loop_fps=10),
        detector=FakeDetector(),
        capture=FakeCapture(),
        input_ctl=fake_in,
        log_cb=lambda _x: None,
    )
    worker._tap_interval_ms_mid = 50
    worker._tap_hold_ms_mid = 20
    worker._apply_relative_tap(rel_y=0.0, now_ms=1000)
    worker._apply_relative_tap(rel_y=0.0, now_ms=1025)
    assert fake_in.hold_states[0] is True
    assert fake_in.hold_states[-1] is False


def test_relative_tap_uses_heavier_hold_when_fish_above_center():
    fake_in = FakeInput()
    worker = AutoFishWorker(
        cfg=AutoFishConfig(loop_fps=10),
        detector=FakeDetector(),
        capture=FakeCapture(),
        input_ctl=fake_in,
        log_cb=lambda _x: None,
    )
    worker._tap_interval_ms_heavy = 1
    worker._apply_relative_tap(rel_y=-5.0, now_ms=1000)
    assert worker._tap_hold_ms_current == worker._tap_hold_ms_heavy


def test_zone_near_bottom_check():
    assert AutoFishWorker._zone_near_bottom(198.2, (0, 100, 20, 200)) is True
    assert AutoFishWorker._zone_near_bottom(194.0, (0, 100, 20, 200)) is False


def test_bottom_rescue_generates_short_hold():
    fake_in = FakeInput()
    worker = AutoFishWorker(
        cfg=AutoFishConfig(loop_fps=10),
        detector=FakeDetector(),
        capture=FakeCapture(),
        input_ctl=fake_in,
        log_cb=lambda _x: None,
    )
    worker._bottom_rescue_interval_ms = 50
    worker._bottom_rescue_hold_ms = 20
    worker._apply_bottom_rescue(now_ms=1000)
    worker._apply_bottom_rescue(now_ms=1025)
    assert fake_in.hold_states[0] is True
    assert fake_in.hold_states[-1] is False


def test_wait_bite_delays_roi_lock_for_500ms_after_first_yolo1():
    worker = AutoFishWorker(
        cfg=AutoFishConfig(loop_fps=10),
        detector=FakeDetector(),
        capture=FakeCapture(),
        input_ctl=FakeInput(),
        log_cb=lambda _x: None,
    )
    worker._sm.state = AutoFishState.WAIT_BITE
    worker._await_next_yolo1 = True
    boxes = [{"cls": 1, "conf": 0.9, "bbox": (10, 20, 30, 60)}]
    assert worker._select_bar_bbox(boxes, now_ms=1000) is None
    assert worker._select_bar_bbox(boxes, now_ms=1499) is None
    got = worker._select_bar_bbox(boxes, now_ms=1500)
    assert got == (10, 20, 30, 60)
    assert worker._await_next_yolo1 is False


def test_roi_image_stabilization_reduces_vertical_jitter():
    worker = AutoFishWorker(
        cfg=AutoFishConfig(loop_fps=10),
        detector=FakeDetector(),
        capture=FakeCapture(),
        input_ctl=FakeInput(),
        log_cb=lambda _x: None,
    )
    worker._roi_motion_alpha = 1.0
    base = np.zeros((90, 40, 3), dtype=np.uint8)
    cv2.rectangle(base, (8, 24), (30, 36), (220, 220, 220), -1)
    cv2.circle(base, (22, 64), 5, (255, 255, 255), -1)
    m = np.float32([[1, 0, 0], [0, 1, 4]])
    shifted = cv2.warpAffine(base, m, (base.shape[1], base.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    worker._stabilize_roi_image(base)
    stabilized, _, dy = worker._stabilize_roi_image(shifted)
    assert abs(dy) >= 2.0
    raw_diff = float(np.mean(np.abs(shifted.astype(np.float32) - base.astype(np.float32))))
    stab_diff = float(np.mean(np.abs(stabilized.astype(np.float32) - base.astype(np.float32))))
    assert stab_diff < raw_diff


def test_roi_preview_has_fixed_white_info_panel():
    worker = AutoFishWorker(
        cfg=AutoFishConfig(loop_fps=10),
        detector=FakeDetector(),
        capture=FakeCapture(),
        input_ctl=FakeInput(),
        log_cb=lambda _x: None,
    )
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    det = {
        "boxes": [],
        "bar_bbox": (40, 20, 56, 100),
        "fish_y": 64.0,
        "zone_y": 58.0,
        "zone_top": 52.0,
        "zone_bottom": 66.0,
    }
    _, roi_preview = worker._build_previews(frame, det)
    assert roi_preview is not None
    strip_w = det["bar_bbox"][2] - det["bar_bbox"][0]
    assert roi_preview.shape[1] > strip_w + 200
    panel = roi_preview[:, strip_w + 5 :, :]
    assert float(panel.mean()) > 200.0


def test_update_zone_y_clamps_to_bottom_margin():
    worker = AutoFishWorker(
        cfg=AutoFishConfig(loop_fps=10),
        detector=FakeDetector(),
        capture=FakeCapture(),
        input_ctl=FakeInput(),
        log_cb=lambda _x: None,
    )
    worker.screen_height = 200
    zone, fish = worker._update_zone_y(zone_y=190.0, fish_y=198.0)
    assert zone == 150.0
    assert fish == 150.0


def test_minigame_hold_logs_expected_fields():
    logs = []
    worker = AutoFishWorker(
        cfg=AutoFishConfig(loop_fps=10),
        detector=FakeDetector(),
        capture=FakeCapture(),
        input_ctl=FakeInput(),
        log_cb=logs.append,
    )
    worker.screen_height = 600
    worker.minigame_hold(ms=220, score=0.517, template="fish.png", scale=0.75, fish_y=510.2, zone_y=580.6)
    assert logs
    s = logs[-1]
    assert "minigame_hold" in s
    assert "ms=220" in s
    assert "score=0.517" in s
    assert "template=fish.png" in s
    assert "scale=0.75" in s
    assert "fish_y=" in s
    assert "zone_y=" in s


def test_hold_ms_from_error_is_clamped_150_to_350():
    assert AutoFishWorker._calc_hold_ms_from_error(abs_err_px=0.0, track_px=100.0) == 150
    mid = AutoFishWorker._calc_hold_ms_from_error(abs_err_px=20.0, track_px=100.0)
    assert 150 <= mid <= 350
    assert AutoFishWorker._calc_hold_ms_from_error(abs_err_px=500.0, track_px=100.0) == 350


def test_hold_ms_uses_700ms_full_travel_ratio():
    ms = AutoFishWorker._calc_hold_ms_from_error(abs_err_px=35.0, track_px=70.0, up_full_ms=700.0)
    # 150 + 700 * (35/70) = 500 -> clamp to 350
    assert ms == 350
