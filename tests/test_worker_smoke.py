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

