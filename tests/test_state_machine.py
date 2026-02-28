from autofish.state_machine import AutoFishState, FishingStateMachine


def test_transitions_from_cast_to_wait_bite():
    sm = FishingStateMachine(cast_wait_s=1.0, move_back_s=0.5, move_forward_s=0.5, success_disappear_ms=500)
    out = sm.tick(now_ms=0, has_bite=False, has_bar=False)
    assert out.click_cast is False
    assert sm.state == AutoFishState.CAST

    out = sm.tick(now_ms=1001, has_bite=False, has_bar=False)
    assert out.click_cast is True
    assert sm.state == AutoFishState.CAST

    out = sm.tick(now_ms=2002, has_bite=False, has_bar=False)
    assert out.hold_back_s == 0.5
    assert sm.state == AutoFishState.WAIT_BITE


def test_transitions_minigame_to_success_to_cast():
    sm = FishingStateMachine(cast_wait_s=1.0, move_back_s=0.5, move_forward_s=0.5, success_disappear_ms=500)
    sm.state = AutoFishState.WAIT_BITE

    sm.tick(now_ms=2000, has_bite=True, has_bar=True)
    assert sm.state == AutoFishState.MINIGAME

    sm.tick(now_ms=2300, has_bite=True, has_bar=False)
    assert sm.state == AutoFishState.MINIGAME

    out = sm.tick(now_ms=2801, has_bite=True, has_bar=False)
    assert sm.state == AutoFishState.MINIGAME
    assert out.click_collect is True
    assert out.hold_forward_s == 0.0

    out = sm.tick(now_ms=3790, has_bite=True, has_bar=False)
    assert sm.state == AutoFishState.MINIGAME
    assert out.hold_forward_s == 0.0

    out = sm.tick(now_ms=3802, has_bite=True, has_bar=False)
    assert sm.state == AutoFishState.CAST
    assert out.hold_forward_s == 0.5


def test_wait_stage_latches_bite_then_accepts_bar_later():
    sm = FishingStateMachine(cast_wait_s=1.0, move_back_s=0.5, move_forward_s=0.5, success_disappear_ms=500)
    sm.state = AutoFishState.WAIT_BITE
    out = sm.tick(now_ms=1000, has_bite=True, has_bar=False)
    assert out.click_hook is True
    assert sm.state == AutoFishState.WAIT_BITE
    sm.tick(now_ms=1100, has_bite=False, has_bar=True)
    assert sm.state == AutoFishState.MINIGAME
