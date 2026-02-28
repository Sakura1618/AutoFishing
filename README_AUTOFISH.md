# Auto Fishing GUI (VRChat)

## Location

- App entry: `D:\Repositories\AutoFishing\autofish_gui.py`
- Package: `D:\Repositories\AutoFishing\autofish`
- Tests: `D:\Repositories\AutoFishing\tests`

## Run

```powershell
powershell -ExecutionPolicy Bypass -File "D:\Repositories\AutoFishing\run_autofish.ps1"
```

## Workflow

1. Select target window from list (`VRChat`).
2. Confirm model path (`.pt`) in GUI.
3. Confirm OSC host/port (`127.0.0.1:9000` by default).
3. Click `Start`.
4. Click `Stop` for safe release.

## Notes

- Input is sent through VRChat OSC Input Controller (`/input/*`), not avatar parameters.
- Default mappings:
  - Click / Hold: `/input/UseRight` + `/input/UseAxisRight`
  - Move back/forward: `/input/Vertical` (`-1` / `+1`)
- `yolo_train` remains dedicated to manual model training only.
- Default fishing sequence follows `自动钓鱼.txt`.
