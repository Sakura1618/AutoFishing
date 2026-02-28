from autofish.config import AutoFishConfig
from autofish.osc_input import OscInputSink


class FakeWin32Sink:
    def __init__(self) -> None:
        self.calls = []

    def set_left_hold_sendinput(self, hold: bool) -> bool:
        self.calls.append(("hold", hold))
        return True

    def release_all(self) -> None:
        self.calls.append(("release_all", True))


def test_minigame_hold_prefers_win32_sendinput():
    sink = FakeWin32Sink()
    osc = OscInputSink(AutoFishConfig(), win32_sink=sink)
    assert osc.set_left_hold_message(True) is True
    assert sink.calls[-1] == ("hold", True)

