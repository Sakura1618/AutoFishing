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
    roi = np.full((90, 36, 3), (90, 170, 90), dtype=np.uint8)  # green-ish bg
    roi[26:62, 9:27, :] = 245  # white zone
    roi[26:28, 8:28, :] = (90, 255, 90)   # green border top
    roi[60:62, 8:28, :] = (90, 255, 90)   # green border bottom
    band = detect_white_zone_band(roi)
    assert band is not None
    assert 26 <= band.top <= 30
    assert 58 <= band.bottom <= 62


def test_detect_white_zone_band_works_when_white_is_occluded():
    roi = np.full((96, 40, 3), (88, 168, 90), dtype=np.uint8)
    roi[24:74, 10:30, :] = 245
    roi[24:26, 9:31, :] = (80, 255, 80)   # green border top
    roi[72:74, 9:31, :] = (80, 255, 80)   # green border bottom
    # fish occlusion splits white area in the middle
    roi[44:56, 12:28, :] = (170, 40, 170)
    band = detect_white_zone_band(roi)
    assert band is not None
    assert 24 <= band.top <= 28
    assert 70 <= band.bottom <= 74


def test_detect_dark_blob_center():
    roi = np.full((90, 40, 3), 210, dtype=np.uint8)
    roi[58:70, 12:28, :] = 20
    cy = detect_dark_blob_center(roi, prefer_y=64.0)
    assert cy is not None
    assert 60.0 <= cy <= 68.0


def test_detect_dark_blob_center_respects_band_limit():
    roi = np.full((100, 44, 3), 210, dtype=np.uint8)
    roi[15:24, 12:28, :] = 15
    roi[62:72, 14:30, :] = 15
    cy = detect_dark_blob_center(roi, prefer_y=66.0, band_top=56.0, band_bottom=80.0)
    assert cy is not None
    assert 62.0 <= cy <= 72.0
