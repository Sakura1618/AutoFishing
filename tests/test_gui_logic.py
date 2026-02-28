from autofish.gui_logic import choose_vrchat_candidates, state_to_cn


def test_choose_vrchat_candidates_case_insensitive():
    windows = [
        (101, "Notepad"),
        (202, "VRChat"),
        (303, "vrchat [Build & Test]"),
    ]
    got = choose_vrchat_candidates(windows)
    assert got == [(202, "VRChat"), (303, "vrchat [Build & Test]")]


def test_state_to_cn_mapping():
    assert state_to_cn("cast") == "抛竿中"
    assert state_to_cn("wait_bite") == "等待上钩"
    assert state_to_cn("minigame") == "小游戏中"
    assert state_to_cn("success") == "收杆复位"

