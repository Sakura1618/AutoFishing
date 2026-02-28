import numpy as np

from autofish.minigame import HoldAction, MinigameController, detect_dark_blob_center, detect_white_zone_band


def test_hold_when_fish_above_center():
    ctrl = MinigameController(dead_zone_px=4, min_hold_ms=0, min_release_ms=0)
    action = ctrl.decide(fish_y=90.0, zone_center_y=100.0, now_ms=0)
    assert action == HoldAction.HOLD


def test_release_when_fish_below_center():
    ctrl = MinigameController(dead_zone_px=4, min_hold_ms=0, min_release_ms=0)
    action = ctrl.decide(fish_y=112.0, zone_center_y=100.0, now_ms=0)
    assert action == HoldAction.RELEASE


def test_keep_inside_dead_zone():
    ctrl = MinigameController(dead_zone_px=5, min_hold_ms=0, min_release_ms=0)
    ctrl.decide(fish_y=90.0, zone_center_y=100.0, now_ms=0)
    action = ctrl.decide(fish_y=96.0, zone_center_y=100.0, now_ms=16)
    assert action == HoldAction.KEEP


def test_edge_guard_forces_action():
    ctrl = MinigameController(edge_guard_px=3.5, min_hold_ms=0, min_release_ms=0)
    a1 = ctrl.decide(fish_y=91.0, zone_center_y=100.0, zone_top_y=90.0, zone_bottom_y=110.0, now_ms=0)
    a2 = ctrl.decide(fish_y=109.0, zone_center_y=100.0, zone_top_y=90.0, zone_bottom_y=110.0, now_ms=200)
    assert a1 == HoldAction.RELEASE
    assert a2 == HoldAction.HOLD


def test_hold_direction_can_flip():
    ctrl = MinigameController(hold_decreases_y=False, min_hold_ms=0, min_release_ms=0)
    action = ctrl.decide(fish_y=115.0, zone_center_y=100.0, now_ms=0)
    assert action == HoldAction.HOLD


def test_detect_white_zone_band_edges():
    roi = np.zeros((80, 30, 3), dtype=np.uint8)
    roi[24:38, :, :] = 255
    band = detect_white_zone_band(roi)
    assert band is not None
    assert 23 <= band.top <= 25
    assert 36 <= band.bottom <= 38


def test_detect_dark_blob_center():
    roi = np.full((90, 40, 3), 210, dtype=np.uint8)
    roi[58:70, 12:28, :] = 20
    cy = detect_dark_blob_center(roi, prefer_y=64.0)
    assert cy is not None
    assert 60.0 <= cy <= 68.0
