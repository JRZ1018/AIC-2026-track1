# 任务单：20260911-B-H1init-LoRA-seed1

- 负责人：A 设计 / B 执行
- 比赛阶段：preliminary
- 实验问题：只检验“以 H1 线性 Adam 最佳头初始化并冻结分类头”是否能让相同 rank-4 LoRA 配方超过 H1。
- Git commit：`dd4444537469f59664ebf45b134c871715db5536`
- 要求工作区：detached HEAD、clean
- 配置：`configs/experiment/p3_stage1_h1init_peft_lora_only_3ep_seed1.yaml`
- 配置 LF SHA-256：`e084d11d34d24d4bf87fe417b36d2b369eb05932043c49eacbef93062cac19eb`
- 配置 CRLF SHA-256：`8d8cf8e51274ebbcdd62683fbd95332c8c4fc194cb7357db4f158a41c7873bc1`
- 训练 manifest SHA-256：`212f645a0d027cc4f2aa1a6d2f81f0175e8a3441b6fa6d001947038e00ba75ef`
- 固定 split SHA-256：`9eb97e2b89da76f64323c2800d6c524a8455e31eaa9b0dab1917b7e4b877fb31`
- H1 初始化头 SHA-256：`e85d53d446fef914c7cec74faa1ede792b7774bb71ec057dee2b551d6cf485af`
- OpenAI CLIP 权重 SHA-256：`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`
- 输出目录：`artifacts/experiments/20260911-b-h1init-lora-only-3ep-seed1/`

## 开始前

1. 从 A 私下取得 H1 输入包，恢复到任务单规定的相对路径；checkpoint 不进入 GitHub。
2. 在 `main` 阅读本任务单，再锁定上述完整 commit。
3. 确认 HEAD、工作区、配置双哈希之一、H1 checkpoint 哈希和输出目录为空。
4. 运行以下轻量检查：

```powershell
git switch --detach dd4444537469f59664ebf45b134c871715db5536
git rev-parse HEAD
git status --short
.\.venv\Scripts\python.exe -m pytest tests\test_h1init_peft_plan.py tests\test_heads.py -q
```

## 唯一训练命令

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_peft_lora_only_3ep_seed1.yaml
```

- 预计 B 节点高占用时间：25–32 分钟；超过 40 分钟且无正常进展即停止回报。
- 预计峰值显存：约 533 MiB，预算上限 1 GiB。
- 预计磁盘增量：小于 10 MiB。
- 允许续跑：仅在 A 明确确认后使用训练器的 `--resume`；不得自行重跑或改参数。

## 验收和回传

训练成功要求：初始验证 Top-1 在 `0.686996996 ± 0.002` 内，完成 3 epochs，无 NaN/Inf/OOM/test 读取，保存完整 `best.pt`、`last.pt`、`metrics.json`、`config.yaml` 和 `command.txt`。

B 私下回传完整输出目录和原始日志，并报告 commit、起止时间、耗时、GPU、峰值显存、最佳 epoch、Top-1、macro accuracy、两个 checkpoint SHA-256 与异常事实。B 不执行测试推理、不生成提交包、不根据结果自行启动 seed2/seed3。

A 的预注册晋级门槛：Top-1 不低于 68.8997%，且 macro accuracy 相对 H1 seed42 下降不超过 0.10 点。未通过即停止该路线；通过也只获得设计 seed2/seed3 的资格。
