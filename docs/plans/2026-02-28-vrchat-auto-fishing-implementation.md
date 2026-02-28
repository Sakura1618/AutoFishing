# VRChat Auto Fishing GUI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Windows-native GUI automation app for VRChat fishing using YOLO + OpenCV with background-control preference and safe fallback.

**Architecture:** The app has a Tkinter GUI, an automation worker thread, Win32 capture/input adapters, and a deterministic state machine that executes fishing phases. Vision combines YOLO detections with ROI-based OpenCV logic for minigame control. Input attempts HWND-targeted messages first, then falls back to global `SendInput`.

**Tech Stack:** Python 3.11+, Tkinter, ultralytics, OpenCV, NumPy, ctypes/Win32 APIs, pytest.

---

### Task 1: Scaffold Package and Runtime Entrypoint

**Files:**
- Create: `yolo_train/autofish/__init__.py`
- Create: `yolo_train/autofish/config.py`
- Create: `yolo_train/autofish/main.py`
- Create: `yolo_train/autofish_gui.py`
- Test: `yolo_train/tests/test_config.py`

**Step 1: Write the failing test**
- Add tests for default config values and path validation behavior.

**Step 2: Run test to verify it fails**
- Run: `python -m pytest yolo_train/tests/test_config.py -q`
- Expected: FAIL because config module not implemented.

**Step 3: Write minimal implementation**
- Add dataclass config and basic validation helpers.

**Step 4: Run test to verify it passes**
- Run: `python -m pytest yolo_train/tests/test_config.py -q`
- Expected: PASS.

### Task 2: Implement State Machine Core

**Files:**
- Create: `yolo_train/autofish/state_machine.py`
- Test: `yolo_train/tests/test_state_machine.py`

**Step 1: Write the failing test**
- Cover transitions `CAST -> WAIT_BITE -> MINIGAME -> SUCCESS -> CAST`.
- Cover `yolo:1` disappearance timeout.

**Step 2: Run test to verify it fails**
- Run: `python -m pytest yolo_train/tests/test_state_machine.py -q`
- Expected: FAIL.

**Step 3: Write minimal implementation**
- Implement deterministic tick-based state machine independent from GUI.

**Step 4: Run test to verify it passes**
- Run: `python -m pytest yolo_train/tests/test_state_machine.py -q`
- Expected: PASS.

### Task 3: Implement Minigame ROI Control Policy

**Files:**
- Create: `yolo_train/autofish/minigame.py`
- Test: `yolo_train/tests/test_minigame.py`

**Step 1: Write the failing test**
- Add tests for hold/release decisions with hysteresis around center gap.

**Step 2: Run test to verify it fails**
- Run: `python -m pytest yolo_train/tests/test_minigame.py -q`
- Expected: FAIL.

**Step 3: Write minimal implementation**
- Implement controller that outputs `hold`, `release`, or `keep`.

**Step 4: Run test to verify it passes**
- Run: `python -m pytest yolo_train/tests/test_minigame.py -q`
- Expected: PASS.

### Task 4: Implement Win32 Capture/Input Adapters

**Files:**
- Create: `yolo_train/autofish/win32_api.py`
- Create: `yolo_train/autofish/capture.py`
- Create: `yolo_train/autofish/input_controller.py`
- Test: `yolo_train/tests/test_input_fallback.py`

**Step 1: Write the failing test**
- Add fallback behavior tests using fake adapter that simulates message failure.

**Step 2: Run test to verify it fails**
- Run: `python -m pytest yolo_train/tests/test_input_fallback.py -q`
- Expected: FAIL.

**Step 3: Write minimal implementation**
- Implement message-first input strategy with retry counter and fallback mode flag.

**Step 4: Run test to verify it passes**
- Run: `python -m pytest yolo_train/tests/test_input_fallback.py -q`
- Expected: PASS.

### Task 5: Implement YOLO/OpenCV Worker and GUI Integration

**Files:**
- Create: `yolo_train/autofish/vision.py`
- Create: `yolo_train/autofish/worker.py`
- Create: `yolo_train/autofish/gui.py`
- Modify: `yolo_train/autofish/main.py`
- Modify: `yolo_train/autofish_gui.py`

**Step 1: Write the failing test**
- Add smoke test for worker bootstrap with mocked YOLO model.

**Step 2: Run test to verify it fails**
- Run: `python -m pytest yolo_train/tests/test_worker_smoke.py -q`
- Expected: FAIL.

**Step 3: Write minimal implementation**
- Build worker loop and GUI wiring for start/stop, logging, status updates.

**Step 4: Run test to verify it passes**
- Run: `python -m pytest yolo_train/tests/test_worker_smoke.py -q`
- Expected: PASS.

### Task 6: Verification and Run Script

**Files:**
- Create: `yolo_train/run_autofish.ps1`
- Create: `yolo_train/README_AUTOFISH.md`

**Step 1: Verification run**
- Run full test suite:
  - `python -m pytest yolo_train/tests -q`
- Expected: PASS.

**Step 2: Manual smoke command**
- `python yolo_train/autofish_gui.py`
- Expected: GUI launches and can start/stop in dry conditions.

**Step 3: Document usage**
- Include VRChat window selection, model path, and fallback mode meanings.

