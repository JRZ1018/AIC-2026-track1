# 任务单：20260915-C 通宵队列修复后恢复

## 覆盖声明

**本任务单覆盖** `docs/task_orders/20260915-B-overnight-8h-peft-search.md` 中的训练 commit 与队列启动层。旧训练 commit `42203b84a0bc00594113e75c2d4ff32ae1b0b850` 禁止原样重跑。

新的冻结训练 commit 为：

`c6c6cf9776a9012a814a2daa26b24422c3eb923f`

15 份训练配置、执行顺序、120 epochs、数据、H1 初始化器、seed、学习率、batch、输出目录、资源预算、停止线和回传要求全部不变。唯一修改是队列脚本不再用 PowerShell 管道合并 Python stderr，而改用独立隐藏进程、分离 stdout/stderr，并通过主进程对象取得退出码。故障分析见 `docs/47_B通宵队列启动故障分析与恢复.md`。

## 现场处置

- 保留旧 `queue.pid`、`queue.stdout.log`、`queue.stderr.log` 和空实验目录，不删除、不改名。
- 已确认无 Python 进程残留、GPU 已释放后方可继续。
- 首项空输出目录不构成冲突；新脚本允许在空目录中首次启动。
- 原运行没有 checkpoint，因此本次首项从 epoch 1 正常开始，不使用 `--resume`。
- 已建成并验收通过的 `.venv` 可以继续使用，无需重新安装依赖。

## 更新与门禁

先从最新 `main` 阅读本任务单，然后切换到冻结训练 commit：

```powershell
Set-Location D:\aicomp\AICtrack1
git fetch origin
git switch main
git pull --ff-only origin main
git switch --detach c6c6cf9776a9012a814a2daa26b24422c3eb923f
git rev-parse HEAD
git status --short
.\.venv\Scripts\python.exe -m pytest tests\test_b_overnight_queue_plan.py tests\test_partial_unfreeze_plan.py tests\test_peft.py tests\test_peft_checkpoint.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe scripts\check_governance.py
```

要求 HEAD 精确匹配、工作区为空、21 项目标测试和治理检查通过。配置及输入哈希继续使用被覆盖任务单中的原值；15 份配置未改变。

## 独立进程启动命令

```powershell
Set-Location D:\aicomp\AICtrack1

$logDir = "D:\aicomp\AICtrack1\artifacts\queues\20260915-b-overnight"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "D:\aicomp\AICtrack1\scripts\run_b_overnight_queue.ps1",
    "-Python", "D:\aicomp\AICtrack1\.venv\Scripts\python.exe"
)

$process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $arguments `
    -WorkingDirectory "D:\aicomp\AICtrack1" `
    -WindowStyle Hidden `
    -RedirectStandardOutput "$logDir\queue-retry.stdout.log" `
    -RedirectStandardError "$logDir\queue-retry.stderr.log" `
    -PassThru

$process.Id | Set-Content "$logDir\queue-retry.pid"
$process
```

启动后约 2–3 分钟只读确认：

```powershell
$pidValue = Get-Content "$logDir\queue-retry.pid"
Get-Process -Id $pidValue
Get-Content "$logDir\queue-retry.stdout.log" -Tail 30
Get-Content "$logDir\queue-retry.stderr.log" -Tail 30
Get-ChildItem "$logDir" -Filter "20260915-b-overnight-t2-b2-lr1e6-8ep-seed1.*.stderr.log"
```

PIL `UserWarning` 可以留在单项 stderr 日志中，不再视为训练失败。只有 Python 主进程非零退出、进程再次静默消失、指标/哈希异常、OOM、NaN/Inf 或 test 访问才触发停止线。

正常后仍按每 1–2 小时只读检查，只有启动成功、异常、资源接近停止线或全部完成时回报。不得修改配置、切换 commit、并发训练或运行 test。
