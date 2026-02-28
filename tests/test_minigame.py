from autofish.minigame import HoldAction, MinigameController


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
