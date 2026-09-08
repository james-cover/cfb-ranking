$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

& .\.venv\Scripts\python.exe -m cfb_rankings.cli refresh
& .\.venv\Scripts\python.exe -m cfb_rankings.cli audit
& .\.venv\Scripts\python.exe -m cfb_rankings.cli build-features
& .\.venv\Scripts\python.exe -m cfb_rankings.cli predict

Write-Host "College-football rankings updated successfully."

