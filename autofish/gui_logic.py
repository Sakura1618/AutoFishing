from __future__ import annotations


def choose_vrchat_candidates(windows: list[tuple[int, str]]) -> list[tuple[int, str]]:
    return [(hwnd, title) for hwnd, title in windows if "vrchat" in title.lower()]


def state_to_cn(state: str) -> str:
    mapping = {
        "cast": "抛竿中",
        "wait_bite": "等待上钩",
        "minigame": "小游戏中",
        "success": "收杆复位",
    }
    return mapping.get(state, state)

