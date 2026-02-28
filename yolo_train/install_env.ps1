param(
    [ValidateSet("cu128", "cu130", "cpu")]
    [string]$TorchChannel = "cu130"
)

$ErrorActionPreference = "Stop"

$pythonCandidates = @(
    "C:\Users\28952\AppData\Local\Python\pythoncore-3.11-64\python.exe",
    "C:\Users\28952\AppData\Local\Python\pythoncore-3.12-64\python.exe",
    "C:\Users\28952\AppData\Local\Python\pythoncore-3.14-64\python.exe"
)

$pythonExe = $pythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $pythonExe) {
    throw "No supported Python found. Checked: $($pythonCandidates -join ', ')"
}

Write-Host "Using Python: $pythonExe"

& $pythonExe -m pip install --upgrade pip setuptools wheel

if ($TorchChannel -eq "cpu") {
    & $pythonExe -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
} else {
    & $pythonExe -m pip install --upgrade torch torchvision torchaudio --index-url ("https://download.pytorch.org/whl/" + $TorchChannel)
}

& $pythonExe -m pip install --upgrade ultralytics

Write-Host ""
Write-Host "Verification:"
& $pythonExe -c "import sys, torch, ultralytics; print('python=', sys.version); print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_version=', torch.version.cuda); print('gpu_count=', torch.cuda.device_count()); print('ultralytics=', ultralytics.__version__)"

Write-Host ""
Write-Host "Environment setup complete."
