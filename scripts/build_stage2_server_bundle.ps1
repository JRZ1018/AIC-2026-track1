param(
    [Parameter(Mandatory = $true)]
    [string]$Destination
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$destinationPath = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $destinationPath | Out-Null

$codeArchive = Join-Path $destinationPath "stage2-code.tar"
$assetsArchive = Join-Path $destinationPath "stage2-assets.tar"

git -C $repo archive --format=tar --output=$codeArchive HEAD
if ($LASTEXITCODE -ne 0) {
    throw "git archive failed"
}

$assetPaths = @(
    "artifacts/weights/ViT-B-32.pt",
    "artifacts/audits/stage2",
    "artifacts/experiments/20260921-stage2-h1-linear-fixed-seed20260921/best.pt",
    "artifacts/experiments/20260921-stage2-peft-ac-last4-mw02-magmix75-16ep-seed20260921/last.pt"
)

tar -C $repo -cf $assetsArchive @assetPaths
if ($LASTEXITCODE -ne 0) {
    throw "asset archive failed"
}

Get-FileHash -Algorithm SHA256 -LiteralPath $codeArchive, $assetsArchive |
    Select-Object Path, Hash
