# Wait for K-seed3 to finish, then run the K full retrain automatically.
# Logs everything to artifacts/queues/20260917-k-full/.
Set-Location D:\aicomp\AICtrack1
$logDir = "artifacts\queues\20260917-k-full"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$py = ".\.venv\Scripts\python.exe"
$seed3 = "artifacts\experiments\20260917-k-t2-last4-lr175e6-cwh-mw10-12ep-seed3"
$fullOut = "artifacts\experiments\20260917-k-final-h1init-t2-last4-lr175e6-cwh-mw10-12ep-full-seed1"
$fullCfg = "configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_12ep_full_seed1_k.yaml"

Write-Host "WAIT_SEED3_START=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
while (-not (Test-Path "$seed3\metrics.json")) {
    $procs = @(Get-Process python -ErrorAction SilentlyContinue)
    if ($procs.Count -eq 0) { Write-Host "SEED3_NOT_RUNNING_AND_NO_METRICS $(Get-Date -Format 'HH:mm:ss')"; exit 1 }
    Start-Sleep -Seconds 30
}
Write-Host "SEED3_DONE=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

Write-Host "FULL_START=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
if (Test-Path "$fullOut\metrics.json") {
    Write-Host "FULL_ALREADY_DONE"
} elseif (-not (Test-Path $fullOut)) {
    & $py scripts\train_peft.py --config $fullCfg 2>&1 | Tee-Object -FilePath "$logDir\K-full.log"
} elseif (Test-Path "$fullOut\last.pt") {
    & $py scripts\train_peft.py --config $fullCfg --resume "$fullOut\last.pt" 2>&1 | Tee-Object -FilePath "$logDir\K-full-resume.log"
} else {
    Write-Host "FULL_NON_EMPTY_NON_RESUMABLE"; exit 1
}
Write-Host "FULL_EXIT=$LASTEXITCODE  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "FULL_METRICS=$(Test-Path "$fullOut\metrics.json")"
Write-Host "CHAIN_ALL_DONE=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
