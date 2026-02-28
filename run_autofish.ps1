$ErrorActionPreference = "Stop"

$pythonCandidates = @(
    "C:\Users\28952\AppData\Local\Python\pythoncore-3.11-64\python.exe",
    "C:\Users\28952\AppData\Local\Python\pythoncore-3.12-64\python.exe",
    "C:\Users\28952\AppData\Local\Python\pythoncore-3.14-64\python.exe"
)

$pythonExe = $pythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $pythonExe) {
    throw "Python not found. Checked: $($pythonCandidates -join ', ')"
}

& $pythonExe -m pip install --upgrade ultralytics opencv-python pillow numpy
& $pythonExe "D:\Repositories\AutoFishing\autofish_gui.py"

