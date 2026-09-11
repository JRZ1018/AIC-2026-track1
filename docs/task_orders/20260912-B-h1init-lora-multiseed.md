# B H1→LoRA 多种子确认入口

下发时间：2026-09-11（Asia/Shanghai）

本轮只完成 H1 线性头初始化 + rank-4 LoRA 的 seed2/seed3 稳定性确认。A 已在修复后的 seed1 得到 68.9615%；B 不重复 seed1，不访问 test，不自行扩展实验。

## 固定版本与私有输入

- 训练 commit：`532d585efa6f4a18c70f7f93308af73d7c281687`
- 必需修复祖先：`a63d287617d1cdda71ce0a8dd3d5f10dbe3e2d20`
- H1 checkpoint 路径：`artifacts/experiments/20260911-p3-h1-linear-adam-e100-seed42/best.pt`
- H1 checkpoint SHA-256：`e85d53d446fef914c7cec74faa1ede792b7774bb71ec057dee2b551d6cf485af`
- 训练 manifest SHA-256：`212f645a0d027cc4f2aa1a6d2f81f0175e8a3441b6fa6d001947038e00ba75ef`
- 固定 split SHA-256：`9eb97e2b89da76f64323c2800d6c524a8455e31eaa9b0dab1917b7e4b877fb31`
- OpenAI CLIP 权重 SHA-256：`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`

旧任务 `20260911-B-H1init-LoRA-seed1` 和 commit `dd444453...` 已作废，禁止执行。

## 顺序

1. `docs/task_orders/20260912-B-h1init-lora-seed2.md`
2. `docs/task_orders/20260912-B-h1init-lora-seed3.md`

必须串行。seed2 失败、哈希不符或需要人工修改时，停止，不运行 seed3。

## B 开始前

```powershell
Set-Location D:\aicomp\AICtrack1
git switch main
git pull --ff-only origin main
git switch --detach 532d585efa6f4a18c70f7f93308af73d7c281687
git rev-parse HEAD
git status --short
.\.venv\Scripts\python.exe -m pytest tests\test_h1init_peft_plan.py tests\test_peft.py tests\test_peft_checkpoint.py -q
```

HEAD 必须完全一致、工作区必须干净、测试必须通过。若 Windows pytest 临时目录权限失败，只回传完整错误，不修改代码或系统权限。

## 总资源与回传

- 每项预计 25–32 分钟，总计约 50–64 分钟。
- 单卡串行，PyTorch 峰值约 533 MiB，单项预算上限 1 GiB。
- 两项合计磁盘增量预计小于 20 MiB。
- 每项回传完整输出目录、stdout/stderr 原始日志、起止时间、耗时、GPU、峰值显存、最佳 epoch、Top-1、macro accuracy、best/last SHA-256 和异常事实。
- checkpoint、日志和私有输入只通过私有通道回传，不进入 GitHub。

A 的最终门禁：三个 seed 必须全部高于 H1 seed42 的 68.6997%，三种子平均 Top-1 必须不低于 68.8997%，且平均 macro 不低于 H1 seed42 的 68.6825%。结果到达后不改变门槛；即使通过，也不自动授权 test 推理或全量重训。
