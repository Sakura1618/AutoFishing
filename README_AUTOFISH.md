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
3. Click `Start`.
4. Click `Stop` for safe release.

## Notes

- Input mode defaults to HWND message mode; fallback to global input is automatic when message input keeps failing.
- `yolo_train` remains dedicated to manual model training only.
- Default fishing sequence follows `自动钓鱼.txt`.

