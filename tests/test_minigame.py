from autofish.minigame import HoldAction, MinigameController


def test_hold_when_fish_is_above_zone():
    ctrl = MinigameController(dead_zone_px=4)
    action = ctrl.decide(fish_y=90.0, zone_center_y=100.0)
    assert action == HoldAction.HOLD


def test_release_when_fish_is_below_zone():
    ctrl = MinigameController(dead_zone_px=4)
    action = ctrl.decide(fish_y=115.0, zone_center_y=100.0)
    assert action == HoldAction.RELEASE


def test_keep_inside_dead_zone():
    ctrl = MinigameController(dead_zone_px=5)
    ctrl.decide(fish_y=90.0, zone_center_y=100.0)  # enter hold
    action = ctrl.decide(fish_y=96.0, zone_center_y=100.0)
    assert action == HoldAction.KEEP

