<#
.SYNOPSIS
Triggers the PI-LDM offline training.
#>

$ErrorActionPreference = "Stop"

$PYTHONPATH = "$PSScriptRoot\..\.."
$env:PYTHONPATH = $PYTHONPATH

Write-Host "Starting PI-LDM Sampling/Evaluation..."
Write-Host "Using PYTHONPATH: $env:PYTHONPATH"

# Run the inference script
python "$PSScriptRoot\..\src\sample.py"
