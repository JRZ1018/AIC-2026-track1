param(
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$jobs = @(
    @("p3_stage1_overnight_t2_b2_lr1e6_8ep_seed1_b.yaml", "20260915-b-overnight-t2-b2-lr1e6-8ep-seed1"),
    @("p3_stage1_overnight_t2_b2_lr2e6_8ep_seed1_b.yaml", "20260915-b-overnight-t2-b2-lr2e6-8ep-seed1"),
    @("p3_stage1_overnight_t2_b2_lr3e6_8ep_seed1_b.yaml", "20260915-b-overnight-t2-b2-lr3e6-8ep-seed1"),
    @("p3_stage1_overnight_t2_b3_lr1e6_8ep_seed1_b.yaml", "20260915-b-overnight-t2-b3-lr1e6-8ep-seed1"),
    @("p3_stage1_overnight_t2_b3_lr2e6_8ep_seed1_b.yaml", "20260915-b-overnight-t2-b3-lr2e6-8ep-seed1"),
    @("p3_stage1_overnight_t2_b3_lr3e6_8ep_seed1_b.yaml", "20260915-b-overnight-t2-b3-lr3e6-8ep-seed1"),
    @("p3_stage1_overnight_t2_b4_lr1e6_8ep_seed1_b.yaml", "20260915-b-overnight-t2-b4-lr1e6-8ep-seed1"),
    @("p3_stage1_overnight_t2_b4_lr2e6_8ep_seed1_b.yaml", "20260915-b-overnight-t2-b4-lr2e6-8ep-seed1"),
    @("p3_stage1_overnight_t2_b4_lr3e6_8ep_seed1_b.yaml", "20260915-b-overnight-t2-b4-lr3e6-8ep-seed1"),
    @("p3_stage1_overnight_lora_last2_r16a32_lr1e4_8ep_seed1_b.yaml", "20260915-b-overnight-lora-last2-r16a32-lr1e4-8ep-seed1"),
    @("p3_stage1_overnight_lora_last2_r32a64_lr5e5_8ep_seed1_b.yaml", "20260915-b-overnight-lora-last2-r32a64-lr5e5-8ep-seed1"),
    @("p3_stage1_overnight_lora_last2_r32a64_lr1e4_8ep_seed1_b.yaml", "20260915-b-overnight-lora-last2-r32a64-lr1e4-8ep-seed1"),
    @("p3_stage1_overnight_lora_last2_r32a64_lr2e4_8ep_seed1_b.yaml", "20260915-b-overnight-lora-last2-r32a64-lr2e4-8ep-seed1"),
    @("p3_stage1_overnight_lora_last3_r32a64_lr1e4_8ep_seed1_b.yaml", "20260915-b-overnight-lora-last3-r32a64-lr1e4-8ep-seed1"),
    @("p3_stage1_overnight_lora_last4_r32a64_lr1e4_8ep_seed1_b.yaml", "20260915-b-overnight-lora-last4-r32a64-lr1e4-8ep-seed1")
)

$logRoot = Join-Path $root "artifacts\queues\20260915-b-overnight"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

foreach ($job in $jobs) {
    $config = Join-Path $root ("configs\experiment\" + $job[0])
    $output = Join-Path $root ("artifacts\experiments\" + $job[1])
    $metrics = Join-Path $output "metrics.json"
    $resume = Join-Path $output "last.pt"
    $log = Join-Path $logRoot ($job[1] + ".log")

    if (Test-Path -LiteralPath $metrics) {
        Write-Host "SKIP completed: $($job[1])"
        continue
    }

    $arguments = @("scripts\train_peft.py", "--config", $config)
    if (Test-Path -LiteralPath $resume) {
        $arguments += @("--resume", $resume)
        Write-Host "RESUME: $($job[1])"
    } elseif ((Test-Path -LiteralPath $output) -and (Get-ChildItem -LiteralPath $output -Force)) {
        throw "Non-empty, non-resumable output directory: $output"
    } else {
        Write-Host "START: $($job[1])"
    }

    & $Python @arguments 2>&1 | Tee-Object -FilePath $log -Append
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Training failed with exit code ${exitCode}: $($job[1])"
    }
    if (-not (Test-Path -LiteralPath $metrics)) {
        throw "Training returned success without metrics.json: $($job[1])"
    }
}

Write-Host "B overnight queue completed. Do not run test inference or additional experiments."
