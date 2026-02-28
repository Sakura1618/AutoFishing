param(
    [string]$DataYaml = "",
    [string]$Model = "yolo11n.pt",
    [int]$Epochs = 100,
    [int]$Imgsz = 640,
    [int]$Batch = 16,
    [string]$Project = "",
    [string]$Name = "autofishing_exp",
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($DataYaml)) {
    $DataYaml = Join-Path $PSScriptRoot "yolo_dataset_v2\data.yaml"
}

if ([string]::IsNullOrWhiteSpace($Project)) {
    $Project = Join-Path $PSScriptRoot "runs"
}

if (-not [System.IO.Path]::IsPathRooted($Model)) {
    $localModel = Join-Path $PSScriptRoot $Model
    if (Test-Path $localModel) {
        $Model = $localModel
    }
}

$pythonCandidates = @(
    "C:\Users\28952\AppData\Local\Python\pythoncore-3.11-64\python.exe",
    "C:\Users\28952\AppData\Local\Python\pythoncore-3.12-64\python.exe",
    "C:\Users\28952\AppData\Local\Python\pythoncore-3.14-64\python.exe"
)

$pythonExe = $pythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $pythonExe) {
    throw "No supported Python found. Checked: $($pythonCandidates -join ', ')"
}

if (-not (Test-Path $DataYaml)) {
    throw "Data yaml not found: $DataYaml"
}

New-Item -ItemType Directory -Force -Path $Project | Out-Null

if ($Device -eq "auto") {
    $cudaCheck = @"
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
"@
    $autoDevice = ($cudaCheck | & $pythonExe -).Trim()
    $Device = if ($autoDevice -eq "cuda") { "0" } else { "cpu" }
}

Write-Host "Using Python: $pythonExe"
Write-Host "Data: $DataYaml"
Write-Host "Model: $Model"
Write-Host "Epochs: $Epochs, Imgsz: $Imgsz, Batch: $Batch, Device: $Device"
Write-Host ""

& $pythonExe -c "from ultralytics import YOLO; import sys; YOLO(sys.argv[1]).train(data=sys.argv[2], epochs=int(sys.argv[3]), imgsz=int(sys.argv[4]), batch=int(sys.argv[5]), project=sys.argv[6], name=sys.argv[7], device=sys.argv[8])" `
    "$Model" `
    "$DataYaml" `
    "$Epochs" `
    "$Imgsz" `
    "$Batch" `
    "$Project" `
    "$Name" `
    "$Device"
if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Training finished. Check results in: $Project\$Name"
