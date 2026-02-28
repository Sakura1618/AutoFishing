from autofish.config import AutoFishConfig
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
    def click_left(self):
        return True


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
