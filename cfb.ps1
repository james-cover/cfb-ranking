$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$RuntimeRoot = Join-Path $env:LOCALAPPDATA "CFBRankingRuntime\py314"
$ReadyFile = Join-Path $RuntimeRoot "READY"

if (-not (Test-Path $ReadyFile)) {
    throw "Local runtime is not ready. Run .\setup-offline.ps1 once first."
}

& python (Join-Path $ProjectRoot "run.py") @args
exit $LASTEXITCODE

