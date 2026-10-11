param(
    [string]$LocalRoot = 'D:\aicomp\AICtrack1',
    [string]$RemoteHost = '39.96.78.164',
    [string]$RemoteRoot = '/data/AICtrack1',
    [string]$KeyPath = 'D:\aicomp\stage2-server-transfer\aic_competition_ed25519',
    [int]$TestFilesPerChunk = 1000,
    [int]$MaxRetries = 4
)

$ErrorActionPreference = 'Stop'
$tar = 'C:\Windows\System32\tar.exe'
$ssh = 'C:\Windows\System32\OpenSSH\ssh.exe'
$markerRoot = '/data/.aic-stage2-transfer-markers'

function Invoke-Remote([string]$command) {
    & $ssh -i $KeyPath -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL `
        -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o LogLevel=ERROR `
        "root@$RemoteHost" $command
    if ($LASTEXITCODE -ne 0) { throw "remote command failed ($LASTEXITCODE): $command" }
}

function Transfer-RelativePaths([string[]]$relativePaths, [string]$markerName, [string]$displayName) {
    $marker = "$markerRoot/$markerName.ok"
    try {
        & $ssh -i $KeyPath -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL `
            -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o LogLevel=ERROR `
            "root@$RemoteHost" "test -f $marker" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "SKIP $displayName"
            return
        }
    } catch { }

    $listPath = Join-Path $env:TEMP ("aic-stage2-list-" + [guid]::NewGuid().ToString('N') + '.txt')
    try {
        [System.IO.File]::WriteAllLines($listPath, $relativePaths)
        $success = $false
        for ($attempt = 1; $attempt -le $MaxRetries -and -not $success; $attempt++) {
            Write-Host ("SEND {0} (attempt {1}/{2})" -f $displayName, $attempt, $MaxRetries)
            $cmd = "`"$tar`" -C `"$LocalRoot`" -cf - -T `"$listPath`" | `"$ssh`" -i `"$KeyPath`" -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o LogLevel=ERROR root@$RemoteHost tar -C $RemoteRoot -xf -"
            & cmd.exe /d /s /c $cmd
            if ($LASTEXITCODE -eq 0) {
                Invoke-Remote "mkdir -p $markerRoot; touch $marker"
                $success = $true
                Write-Host "DONE $displayName"
            } else {
                Write-Warning ("transfer failed for {0} (exit {1})" -f $displayName, $LASTEXITCODE)
                Start-Sleep -Seconds 2
            }
        }
        if (-not $success) { throw "transfer failed after retries: $displayName" }
    } finally {
        Remove-Item -LiteralPath $listPath -Force -ErrorAction SilentlyContinue
    }
}

Invoke-Remote "mkdir -p $markerRoot"

$trainRoot = Join-Path $LocalRoot 'DataSetStage2\train'
$trainDirs = Get-ChildItem -LiteralPath $trainRoot -Directory | Sort-Object Name
$trainGroupMaxBytes = 450MB
$trainGroup = [System.Collections.Generic.List[string]]::new()
$trainGroupBytes = [int64]0
$trainGroupNumber = 0
foreach ($dir in $trainDirs) {
    $stats = Get-ChildItem -LiteralPath $dir.FullName -File | Measure-Object Length -Sum
    $dirBytes = [int64]$stats.Sum
    if ($trainGroup.Count -gt 0 -and ($trainGroupBytes + $dirBytes) -gt $trainGroupMaxBytes) {
        Transfer-RelativePaths $trainGroup.ToArray() ("train-group-{0:D4}" -f $trainGroupNumber) ("train group $trainGroupNumber")
        $trainGroupNumber++
        $trainGroup = [System.Collections.Generic.List[string]]::new()
        $trainGroupBytes = 0
    }
    [void]$trainGroup.Add("DataSetStage2/train/$($dir.Name)")
    $trainGroupBytes += $dirBytes
}
if ($trainGroup.Count -gt 0) {
    Transfer-RelativePaths $trainGroup.ToArray() ("train-group-{0:D4}" -f $trainGroupNumber) ("train group $trainGroupNumber")
}

$testRoot = Join-Path $LocalRoot 'DataSetStage2\test'
$testFiles = Get-ChildItem -LiteralPath $testRoot -File | Sort-Object Name
for ($i = 0; $i -lt $testFiles.Count; $i += $TestFilesPerChunk) {
    $end = [Math]::Min($i + $TestFilesPerChunk, $testFiles.Count)
    $chunk = $testFiles[$i..($end - 1)] | ForEach-Object { "DataSetStage2/test/$($_.Name)" }
    $number = [int]($i / $TestFilesPerChunk)
    Transfer-RelativePaths $chunk ("test-{0:D4}" -f $number) ("test chunk $number ($i-$($end-1))")
}

Write-Host 'Stage2 dataset transfer complete.'
