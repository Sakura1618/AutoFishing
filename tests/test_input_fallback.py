from autofish.input_controller import InputMode, SmartInputController


class FakeSink:
    def __init__(self, message_ok: bool, sendinput_ok: bool = True) -> None:
        self.message_ok = message_ok
        self.sendinput_ok = sendinput_ok
        self.message_calls = 0
        self.sendinput_calls = 0

    def click_left_message(self) -> bool:
        self.message_calls += 1
        return self.message_ok

    def click_left_sendinput(self) -> bool:
        self.sendinput_calls += 1
        return self.sendinput_ok


def test_switches_to_sendinput_after_retry_limit():
    sink = FakeSink(message_ok=False, sendinput_ok=True)
    ctl = SmartInputController(sink=sink, retry_limit=3)
    for _ in range(4):
        ctl.click_left()
    assert ctl.mode == InputMode.SENDINPUT
    assert sink.sendinput_calls >= 1


def test_stays_on_message_mode_when_successful():
    sink = FakeSink(message_ok=True, sendinput_ok=True)
    ctl = SmartInputController(sink=sink, retry_limit=3)
    ctl.click_left()
    assert ctl.mode == InputMode.MESSAGE
    assert sink.message_calls == 1
    assert sink.sendinput_calls == 0

