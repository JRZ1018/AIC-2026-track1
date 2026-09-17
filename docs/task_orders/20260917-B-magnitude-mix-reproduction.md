# 任务单：20260917 B 全局损失量级混合复现（item L，seed 20260910）

本单要求你在**你自己的机器**上跑一个与 A 正在跑的完全相同的实验。目的有两个，按重要性排序：

1. **首要目的是环境等价性校验**：这个配置在 A 的机器上已经跑过一次（item K，同种子、同配置、仅缺
   `magnitude_mix`），Top-1 为 `0.719480276`。你的结果必须落在它附近；**若明显不符，说明你的机器
   与 A 不等价，在查明原因前，任何跨机器实验结果都不可采信**。
2. 次要目的：给出 `magnitude_mix` 的机器间一致性证据。

## 冻结信息

- **训练 commit：`b07da0b2dea7125da584060c56a23e854d35351b`**（必须是这 40 位，不得使用 `main` 浮动头）
- 配置：`configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix50_12ep_seed1_l.yaml`
- 对照参考（A 已跑，你**不要**重跑）：item K `0.719480276`，配置
  `configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_12ep_seed1_k.yaml`
- 运行参数：单卡单进程、12 epochs、batch 64、seed 20260910、lr 1.75e-6、`last_n_blocks` 4、
  `min_weight` 0.1、新增 `magnitude_mix` 0.5
- 预计耗时 45–60 分钟（A 的 12-epoch 实测约 55 分钟），峰值显存约 1.2 GiB
- 输出目录：`artifacts\experiments\20260917-l-t2-last4-lr175e6-cwh-mw10-magmix50-12ep-seed1`
- **不访问 test、不推理、不生成提交包、不改任何参数**

## 开始前必须逐项确认并回报

1. `git rev-parse HEAD` 等于 `b07da0b2dea7125da584060c56a23e854d35351b`，且工作区干净；
2. 四个输入哈希与 A 一致（配置文件里写明的路径）：
   - `artifacts\audits\stage1\train\samples.csv`
   - `artifacts\audits\stage1\train\split.csv`
   - `artifacts\audits\stage1\train\label_conflicts.csv`
   - `artifacts\experiments\20260911-p3-h1-linear-adam-e100-seed42\best.pt`
     的 SHA-256 必须是 `e85d53d446fef914c7cec74faa1ede792b7774bb71ec057dee2b551d6cf485af`
3. 环境自检通过：

```powershell
$env:PYTEST_DEBUG_TEMPROOT="D:\aicomp\AICtrack1\tmp\pytest-temproot"
.\.venv\Scripts\python.exe -m pytest tests\test_confidence.py tests\test_t2_cwh_magnitude_mix_plan.py -q
```

任一项不符，立即停止并回报 A，不要自行修复。

## 唯一命令

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix50_12ep_seed1_l.yaml
```

## 完成后回传

1. 完整实验目录（`config.yaml` / `command.txt` / `metrics.json` / `best.pt` / `last.pt`）
2. 训练日志
3. 上述五个文件的 SHA-256

**特别回报这三个数字**（用于环境等价性判定）：

- `metrics.json` 的 `best_top1`
- `metrics.json` 的 `initial_validation.all.top1`（应为 `0.686996996` 附近，配置里设了 ±0.002 的初始守卫）
- 逐 epoch 的 `history[*].validation.all.top1` 全序列

## 判定规则（A 执行，你不需要判断）

| 你的 `best_top1` | 结论 |
|---|---|
| 与 `0.719480276` 之差 ≤ 0.0013 | 机器等价，你的后续结果可直接与 A 的结果合并比较 |
| 之差 > 0.0013 且工作区/哈希全部正确 | 机器不等价，暂停跨机器并行，A 先定位差异来源 |

不要根据本单结果自行安排下一项实验；下一单由 A 在验收后下发。
