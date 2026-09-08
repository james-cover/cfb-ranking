$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

& .\.venv\Scripts\python.exe -m cfb_rankings.cli build-features
& .\.venv\Scripts\python.exe -m cfb_rankings.cli train
& .\.venv\Scripts\python.exe -m cfb_rankings.cli predict

Write-Host "Both model families retrained and predictions regenerated."
