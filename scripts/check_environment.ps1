$ErrorActionPreference = "Continue"

Write-Host "=== OS ==="
Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, OsArchitecture

Write-Host "`n=== Python launchers ==="
foreach ($name in @("python", "py", "pip")) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        Write-Host "${name}: NOT FOUND"
    } else {
        Write-Host "${name}: $($cmd.Source)"
    }
}

$projectPython = "D:\aicomp\python312\python.exe"
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (Test-Path -LiteralPath $projectPython) {
    $pythonExecutable = $projectPython
} elseif ($null -ne $pythonCmd) {
    $pythonExecutable = $pythonCmd.Source
} else {
    $pythonExecutable = $null
}

if ($null -ne $pythonExecutable) {
    Write-Host "`n=== Python ==="
    Write-Host "selected interpreter: $pythonExecutable"
    & $pythonExecutable --version
    & $pythonExecutable -m pip --version

    Write-Host "`n=== ML Python packages ==="
    $probe = @'
import importlib

packages = [
    "torch", "torchvision", "numpy", "PIL", "pandas", "sklearn",
    "scipy", "yaml", "tqdm", "tensorboard", "matplotlib", "pytest"
]
for name in packages:
    try:
        module = importlib.import_module(name)
        print(f"{name}: {getattr(module, '__version__', 'installed')}")
    except Exception as exc:
        print(f"{name}: NOT AVAILABLE ({type(exc).__name__}: {exc})")

try:
    import torch
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.backends.cudnn.version:", torch.backends.cudnn.version())
    print("torch.cuda.device_count:", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(
            f"gpu[{index}]: {props.name}; "
            f"vram={props.total_memory / 1024**3:.1f} GiB; "
            f"capability={props.major}.{props.minor}"
        )
except Exception as exc:
    print("PyTorch CUDA probe failed:", repr(exc))
'@
    # PowerShell can alter embedded quotation marks passed to native programs via
    # `python -c`. Write a short temporary script instead so this diagnostic is
    # reliable on both Windows PowerShell and PowerShell 7.
    $probeFile = [System.IO.Path]::ChangeExtension([System.IO.Path]::GetTempFileName(), ".py")
    try {
        Set-Content -LiteralPath $probeFile -Value $probe -Encoding utf8
        & $pythonExecutable $probeFile
    } finally {
        Remove-Item -LiteralPath $probeFile -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "`n=== NVIDIA driver and GPU ==="
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($null -eq $nvidiaSmi) {
    Write-Host "nvidia-smi: NOT FOUND"
} else {
    & $nvidiaSmi.Source
}

Write-Host "`n=== CUDA Toolkit (optional for prebuilt PyTorch wheels) ==="
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if ($null -eq $nvcc) {
    Write-Host "nvcc: NOT FOUND (this is acceptable unless custom CUDA extensions are compiled)"
} else {
    & $nvcc.Source --version
}
