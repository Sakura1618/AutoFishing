import numpy as np

from autofish.minigame import HoldAction, MinigameController, detect_white_zone_band


def test_hold_when_fish_is_above_zone():
    ctrl = MinigameController(dead_zone_px=4)
    action = ctrl.decide(fish_y=90.0, zone_center_y=100.0, now_ms=0)
    assert action == HoldAction.HOLD


def test_release_when_fish_is_below_zone():
    ctrl = MinigameController(dead_zone_px=4)
    action = ctrl.decide(fish_y=115.0, zone_center_y=100.0, now_ms=0)
    assert action == HoldAction.RELEASE


def test_keep_inside_dead_zone():
    ctrl = MinigameController(dead_zone_px=5)
    ctrl.decide(fish_y=90.0, zone_center_y=100.0, now_ms=0)  # enter hold
    action = ctrl.decide(fish_y=96.0, zone_center_y=100.0, now_ms=16)
    assert action == HoldAction.KEEP


def test_does_not_flip_immediately_on_reverse_signal():
    ctrl = MinigameController(
        dead_zone_px=4,
        min_hold_ms=120,
        max_hold_ms=120,
        min_release_ms=120,
        max_release_ms=120,
    )
    a1 = ctrl.decide(fish_y=80.0, zone_center_y=100.0, now_ms=0)
    a2 = ctrl.decide(fish_y=130.0, zone_center_y=100.0, now_ms=50)
    a3 = ctrl.decide(fish_y=130.0, zone_center_y=100.0, now_ms=130)
    assert a1 == HoldAction.HOLD
    assert a2 == HoldAction.KEEP
    assert a3 == HoldAction.RELEASE


def test_prediction_can_pre_brake_before_crossing_zone():
    ctrl = MinigameController(
        dead_zone_px=4,
        far_px=10_000,
        min_hold_ms=0,
        max_hold_ms=0,
        min_release_ms=0,
        max_release_ms=0,
        predict_ms=200,
    )
    a1 = ctrl.decide(fish_y=80.0, zone_center_y=100.0, now_ms=0)
    # Fish still above zone, but moving down fast. Prediction should pre-brake to RELEASE.
    a2 = ctrl.decide(fish_y=95.0, zone_center_y=100.0, now_ms=16)
    assert a1 == HoldAction.HOLD
    assert a2 == HoldAction.RELEASE


def test_far_error_enters_catchup_and_switches_direction():
    ctrl = MinigameController(
        dead_zone_px=4,
        far_px=25,
        near_px=12,
        min_hold_ms=0,
        max_hold_ms=0,
        min_release_ms=0,
        max_release_ms=0,
    )
    a1 = ctrl.decide(fish_y=140.0, zone_center_y=100.0, now_ms=0)
    a2 = ctrl.decide(fish_y=60.0, zone_center_y=100.0, now_ms=16)
    assert a1 == HoldAction.RELEASE
    assert a2 == HoldAction.HOLD


def test_detect_white_zone_band_edges():
    roi = np.zeros((80, 30, 3), dtype=np.uint8)
    roi[24:38, :, :] = 255
    band = detect_white_zone_band(roi)
    assert band is not None
    assert 23 <= band.top <= 25
    assert 36 <= band.bottom <= 38
    assert 29.0 <= band.center <= 31.5


def test_controller_uses_zone_edges_as_guardrails():
    ctrl = MinigameController(
        dead_zone_px=4,
        far_px=10_000,
        min_hold_ms=0,
        max_hold_ms=0,
        min_release_ms=0,
        max_release_ms=0,
    )
    a1 = ctrl.decide(fish_y=89.0, zone_center_y=100.0, zone_top_y=90.0, zone_bottom_y=110.0, now_ms=0)
    a2 = ctrl.decide(fish_y=111.0, zone_center_y=100.0, zone_top_y=90.0, zone_bottom_y=110.0, now_ms=16)
    assert a1 == HoldAction.RELEASE
    assert a2 == HoldAction.HOLD


def test_edge_guard_forces_release_near_top():
    ctrl = MinigameController(
        dead_zone_px=4,
        far_px=10_000,
        edge_guard_px=4.0,
        min_hold_ms=0,
        max_hold_ms=0,
        min_release_ms=80,
        max_release_ms=80,
    )
    a1 = ctrl.decide(fish_y=95.0, zone_center_y=100.0, zone_top_y=90.0, zone_bottom_y=110.0, now_ms=0)
    # Fish near top edge should trigger emergency RELEASE despite center error preferring HOLD.
    a2 = ctrl.decide(fish_y=91.0, zone_center_y=100.0, zone_top_y=90.0, zone_bottom_y=110.0, now_ms=16)
    assert a1 == HoldAction.HOLD
    assert a2 == HoldAction.RELEASE


def test_target_bias_shifts_center_decision():
    ctrl = MinigameController(
        dead_zone_px=4,
        target_bias_px=6.0,
        far_px=10_000,
        min_hold_ms=0,
        max_hold_ms=0,
        min_release_ms=0,
        max_release_ms=0,
    )
    # Without bias this is exactly centered, with positive bias target moves down so should HOLD.
    a1 = ctrl.decide(fish_y=100.0, zone_center_y=100.0, zone_top_y=90.0, zone_bottom_y=110.0, now_ms=0)
    assert a1 == HoldAction.HOLD
