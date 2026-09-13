# 任务单：20260913-B LoRA 容量四小时串行队列

## 目标与顺序

在 B 的单卡上执行两个预注册、互相独立的固定验证集容量消融。顺序不可交换：

1. 末三层注意力、rank 4、alpha 8；
2. 末两层注意力、rank 8、alpha 16（保持 alpha/rank=2）。

两组均先执行 seed1，严格按下述数值门禁决定是否追加 seed2/seed3。不得访问 test、不得生成提交、不得根据线上分数或预测分布修改参数。

## 冻结输入

- 训练 commit：`7a6ec202b23ed1c5ad0f13edd253f5591928b73c`
- train manifest SHA-256：`212f645a0d027cc4f2aa1a6d2f81f0175e8a3441b6fa6d001947038e00ba75ef`
- split SHA-256：`9eb97e2b89da76f64323c2800d6c524a8455e31eaa9b0dab1917b7e4b877fb31`
- conflicts SHA-256：`d06a03bf23b54e663bfc849236f855dfeb61251262329210a72274b2c4df70f1`
- H1 `best.pt` SHA-256：`e85d53d446fef914c7cec74faa1ede792b7774bb71ec057dee2b551d6cf485af`
- OpenAI CLIP ViT-B/32 SHA-256：`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`

配置及 SHA-256：

| 顺序 | 配置 | LF / CRLF SHA-256 |
|---:|---|---|
| 1 | `configs/experiment/p3_stage1_h1init_peft_lora_last3_r4_3ep_seed1_b_planned.yaml` | `07e868720bc8ee937de17f2b695d7848cd8b86423b732be83d6460f930f2c469` / `4ab2fa2c97517fb33bd430825d944a76fe57e2da2f3405f4d6f89ceb22d7e0d1` |
| 2 | `configs/experiment/p3_stage1_h1init_peft_lora_last3_r4_3ep_seed2_b_planned.yaml` | `00b80b1a5e1569b62092ed61873b5acc10ef41c56653399de9397253594d5ccb` / `14e9d489cf8437bbfa5d679a7dbfade8d0c064d7b0a47ca1591a3d9e5ded6543` |
| 3 | `configs/experiment/p3_stage1_h1init_peft_lora_last3_r4_3ep_seed3_b_planned.yaml` | `f07919f9b51080a058aeba389b999cb1aaf2332ce23307f72d44fc43913cac5c` / `55b47aebe72327b165da08a9f72a0d8a764abdcb55d6b2e3f41e971f8df65ae3` |
| 4 | `configs/experiment/p3_stage1_h1init_peft_lora_last2_r8a16_3ep_seed1_b_planned.yaml` | `cb8246996722cc246c17bf2dbfbf27680bf6513262addf5ffd406ec40c0cc638` / `492d8f0b4322a5bfc4fa82dda68a82b0d2bd5e907de43a59957b1bc76fff30d6` |
| 5 | `configs/experiment/p3_stage1_h1init_peft_lora_last2_r8a16_3ep_seed2_b_planned.yaml` | `654dabf3256e1d31893895150190cf4a698cdb58a8d9430fde46080b81cb3b4d` / `646a8397c2ea4eca4f35182522156ed7f221820ef4a4af398ca36eb28e81ed81` |
| 6 | `configs/experiment/p3_stage1_h1init_peft_lora_last2_r8a16_3ep_seed3_b_planned.yaml` | `d16a36ab0f65d3507a3dc7c96b9e4591e2e545e7b91a783791bcbf21472615d5` / `2e3d2150c88e756e2543192d7c3252c0f3e58eda223b507ce08000650e2d4cfe` |

## 开始前门禁

```powershell
Set-Location D:\aicomp\AICtrack1
git fetch origin
git switch --detach 7a6ec202b23ed1c5ad0f13edd253f5591928b73c
git rev-parse HEAD
git status --short
.\.venv\Scripts\python.exe -m pytest tests\test_b_next_queue_plan.py tests\test_peft.py tests\test_peft_checkpoint.py -q
.\.venv\Scripts\python.exe scripts\check_governance.py
```

HEAD 必须精确匹配、工作区必须为空、测试与治理必须通过、六个配置和四项私有输入哈希必须匹配。任一不符立即停止，不自行修复。

## 训练命令

每次只运行一条并等待完成：

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_peft_lora_last3_r4_3ep_seed1_b_planned.yaml
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_peft_lora_last3_r4_3ep_seed2_b_planned.yaml
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_peft_lora_last3_r4_3ep_seed3_b_planned.yaml
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_peft_lora_last2_r8a16_3ep_seed1_b_planned.yaml
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_peft_lora_last2_r8a16_3ep_seed2_b_planned.yaml
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_peft_lora_last2_r8a16_3ep_seed3_b_planned.yaml
```

不要无条件运行六条；每个家族按下一节门禁。输出目录一旦非空只允许训练器原生续跑，不删除、不覆盖、不改任务 ID。

## 无人值守固定门禁

匹配的末两层 rank-4 基线 Top-1 为：seed1 `0.691214979`、seed2 `0.690536201`、seed3 `0.690972567`。

对每个家族：

1. 运行 seed1。
2. 若 `best_top1 >= 0.692214979`，且 seed1 最佳 epoch 的 macro 不低于 `0.690583431`、冲突 Top-1 不低于 `0.305000329`，则运行 seed2、seed3。
3. 若 `0.691714979 <= best_top1 < 0.692214979` 且满足同样两个保护阈值，只运行 seed2。随后仅当 seed1/seed2 相对各自基线的平均增益至少为 `0.00075`，并且 seed2 最佳 epoch 的 macro 不低于 `0.689910733`、冲突 Top-1 不低于 `0.305663013`，才运行 seed3。
4. 其他情况停止该家族的后续 seed，继续下一个家族的 seed1。
5. 任意 NaN/Inf、OOM、读取 test、样本数不是 82,592/20,626、参数量不是末三层 55,296 或 rank-8 73,728，立即停止整个队列。

`best_top1` 在 `metrics.json` 顶层；macro 和冲突 Top-1 取 `history` 中 `validation.all.top1` 等于 `best_top1` 的 epoch，字段分别为 `validation.all.macro_accuracy` 和 `validation.cross_label_exact_conflicts.top1`。不得用其他 epoch 绕过门禁。

## 资源与四小时停止线

- 严格单进程串行，不并发。
- 每项预计 25–35 分钟；最多六项，训练约 2.5–3.5 小时，峰值显存预计低于 0.8 GiB，磁盘新增约 3–6 MiB/项。
- 从第一项开始计时，达到 3 小时 25 分时不再启动新任务，预留至少 35 分钟让当前任务完成与打包；总工作窗口不超过四小时。
- 每个 epoch 有 `last.pt` 可恢复。异常中断时保留现场，停止整个队列并报告，不猜测性重跑。

## 私下回传

将所有实际运行的完整实验目录、原始日志、起止时间、GPU/环境、每项配置/best/last 哈希和一份门禁判断清单打包私下回传 A。不要推送 checkpoint、日志或 ZIP 到 GitHub。明确列出哪些 seed 因门禁未运行。
