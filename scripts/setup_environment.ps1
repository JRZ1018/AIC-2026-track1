param(
    [string]$VenvPath = ".venv",
    [string]$PythonLauncher = "py"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedVenv = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $VenvPath))
$venvPython = Join-Path $resolvedVenv "Scripts\python.exe"
$requirements = Join-Path $repoRoot "requirements-lock.txt"

if (-not (Test-Path -LiteralPath $venvPython)) {
    $launcher = Get-Command $PythonLauncher -ErrorAction Stop
    Write-Host "Creating Python 3.12 environment at $resolvedVenv"
    & $launcher.Source -3.12 -m venv $resolvedVenv
}

Write-Host "Installing the locked environment (network access required)"
& $venvPython -m pip install --upgrade pip==25.0.1 setuptools==78.1.0
& $venvPython -m pip install --requirement $requirements
& $venvPython -m pip install --editable $repoRoot --no-deps

Write-Host "Running environment and project checks"
& $venvPython -m pytest
& $venvPython (Join-Path $repoRoot "scripts\check_governance.py")
& (Join-Path $repoRoot "scripts\check_environment.ps1") -PythonExecutable $venvPython

Write-Host "Environment ready: $venvPython"
