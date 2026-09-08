$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$Wheelhouse = Join-Path $ProjectRoot "wheelhouse"
$Requirements = Join-Path $ProjectRoot "requirements-runtime.txt"
$RuntimeRoot = Join-Path $env:LOCALAPPDATA "CFBRankingRuntime\py314"
$Packages = Join-Path $RuntimeRoot "packages"
$ReadyFile = Join-Path $RuntimeRoot "READY"

$PythonVersion = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) {
    throw "Python could not be started."
}
if ($PythonVersion.Trim() -ne "3.14") {
    throw "This offline bundle targets Python 3.14, but python resolves to $PythonVersion."
}

New-Item -ItemType Directory -Force -Path $Packages | Out-Null

Write-Host "Installing the bundled Windows packages into your user profile..."
& python -m pip install `
    --disable-pip-version-check `
    --no-index `
    --only-binary=:all: `
    --upgrade `
    --target $Packages `
    --find-links $Wheelhouse `
    -r $Requirements

if ($LASTEXITCODE -ne 0) {
    throw "The bundled dependency installation failed."
}

$env:PYTHONPATH = "$Packages;$ProjectRoot\src"
& python -c "import numpy, pandas, sklearn, xgboost, lightgbm, streamlit; print('Runtime verified successfully.')"
if ($LASTEXITCODE -ne 0) {
    throw "Packages were copied, but the runtime verification failed."
}

Set-Content -Path $ReadyFile -Value "Python $PythonVersion runtime verified" -Encoding UTF8
Write-Host "Setup complete. Next command: .\cfb.ps1 bootstrap --start-year 2014"

