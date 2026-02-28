# VRChat Auto Fishing GUI Design

**Date:** 2026-02-28  
**Scope:** Build a Windows-native auto-fishing app with GUI for VRChat, using YOLO + OpenCV and background-window control fallback.

## Goals

- Implement the fishing loop defined in `自动钓鱼.txt`.
- Provide a desktop GUI to configure and run automation.
- Prefer background control via window messages; fall back to foreground `SendInput` when needed.
- Show runtime preview/logging and explicit current control mode.

## Non-Goals

- Kernel/driver-level input injection.
- Anti-cheat bypass.
- Multi-game abstractions.

## Architecture

- `GUI Layer (Tkinter)`: Start/Stop, model path, thresholds, target window picker, runtime status and logs, preview panel.
- `Automation Layer`: Finite state machine with phases: `CAST -> WAIT_BITE -> MINIGAME -> SUCCESS -> CAST`.
- `Vision Layer`:
  - YOLO detects `yolo:0` (bite) and `yolo:1` (bar region).
  - OpenCV in dynamic ROI estimates fish vs white-zone and emits hold/release command.
- `Capture Layer`:
  - Preferred: window capture from target `HWND` (`PrintWindow`/`BitBlt` path).
  - Fallback: desktop capture + crop by target window rectangle.
- `Input Layer`:
  - Preferred: `PostMessage/SendMessage` to game window (background intent).
  - Fallback: `SendInput` global input, with GUI warning.

## State Machine

- `CAST`
  - left click
  - sleep 1.0s
  - hold `S` for 0.5s
  - transition to `WAIT_BITE`
- `WAIT_BITE`
  - wait for YOLO class `0`
  - acquire `yolo:1` bbox and build ROI
  - transition to `MINIGAME`
- `MINIGAME`
  - track fish vs white zone in ROI
  - hold/release left button to keep fish centered
  - if `yolo:1` absent for >= 500ms -> `SUCCESS`
- `SUCCESS`
  - left click to collect
  - hold `W` for 0.5s
  - transition to `CAST`

## Configurable Parameters

- `conf_yolo0`, `conf_yolo1`
- `roi_expand`
- `success_disappear_ms`
- `cast_wait_s`, `move_back_s`, `move_forward_s`
- `loop_fps`
- `input_retry_limit`

## Failure Handling

- If message-based input appears ineffective for N retries, auto-switch to `SendInput`.
- If ROI tracking fails, return to global YOLO localization.
- On stop/error: release all held buttons/keys and reset to idle.

## Testing Strategy

- Unit tests for state transitions and minigame control policy.
- Smoke test for app bootstrap and config validation.
- Manual validation against VRChat window behavior for background/foreground mode transitions.

