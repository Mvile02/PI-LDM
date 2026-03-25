<#
.SYNOPSIS
Triggers the PI-LDM offline training.
#>

$ErrorActionPreference = "Stop"

$PYTHONPATH = "$PSScriptRoot\..\.."
$env:PYTHONPATH = $PYTHONPATH

Write-Host "Starting PI-LDM Training Loop..."
Write-Host "Using PYTHONPATH: $env:PYTHONPATH"

# Run the training script using the local virtual environment
$python_exe = "$PSScriptRoot\..\..\venv\Scripts\python.exe"
& $python_exe "$PSScriptRoot\..\src\train.py"
