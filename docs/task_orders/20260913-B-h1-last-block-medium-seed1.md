# 任务单：20260913-B H1 末块中强度微调 seed1

## 目标

执行一次固定验证集筛选：从 H1 线性头初始化，冻结分类头，只训练 CLIP ViT-B/32 最后一个完整视觉 Transformer 块、`ln_post` 与 `proj`。本任务只允许 seed1，不追加 seed、不访问 test、不生成提交。

本任务是新的中强度路线首项，而不是继续微调 LoRA rank/alpha：7,482,624 个可训练视觉参数约为当前 rank-8 LoRA 的 101.5 倍。它仍以初赛 Top-1 提分为直接目标，同时验证受控部分解冻是否比轻量适配更有容量；配置不硬编码类别数，后续可在新阶段从官方权重重新运行。

## 冻结输入

- 训练 commit：`5668419b92eb39ea8d35acc723c982647937b5da`
- 配置：`configs/experiment/p3_stage1_h1init_last_block_frozen_head_lr3e6_3ep_seed1_planned.yaml`
- 配置 SHA-256（LF）：`79f06c723b6c303d58948908dedd0b925e508c9d0e945abdb670a289d7750e9f`
- 配置 SHA-256（CRLF）：`61fec185a500e9c1d4251b32c05022c16c62e1eba9ee6918e53508fefeca2a0f`
- train manifest SHA-256：`212f645a0d027cc4f2aa1a6d2f81f0175e8a3441b6fa6d001947038e00ba75ef`
- split SHA-256：`9eb97e2b89da76f64323c2800d6c524a8455e31eaa9b0dab1917b7e4b877fb31`
- conflicts SHA-256：`d06a03bf23b54e663bfc849236f855dfeb61251262329210a72274b2c4df70f1`
- H1 `best.pt` SHA-256：`e85d53d446fef914c7cec74faa1ede792b7774bb71ec057dee2b551d6cf485af`
- OpenAI CLIP ViT-B/32 SHA-256：`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`

配置是已批准的固定方法包：3 epochs、seed `20260910`、batch size 64、视觉学习率 `3e-6`、分类头学习率 0、weight decay 0.01、gradient clip 1.0。不得修改任何字段。

## 开始前门禁

```powershell
Set-Location D:\aicomp\AICtrack1
git fetch origin
git switch --detach 5668419b92eb39ea8d35acc723c982647937b5da
git rev-parse HEAD
git status --short
.\.venv\Scripts\python.exe -m pytest tests\test_h1init_peft_plan.py tests\test_peft.py tests\test_peft_checkpoint.py -q
.\.venv\Scripts\python.exe scripts\check_governance.py
```

HEAD 必须精确匹配、工作区必须为空、测试和治理必须通过；配置哈希允许匹配上方 LF 或 CRLF 之一，四项私有输入哈希必须匹配。任一不符立即停止并回报 A，不自行修复。

## 唯一训练命令

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_last_block_frozen_head_lr3e6_3ep_seed1_planned.yaml
```

输出目录一旦非空，只允许训练器原生续跑；不得删除、覆盖或另改 run ID。训练期间不得 `git pull` 或切换 commit。

## 成功条件与停止线

- 固定划分数量必须为训练 82,592、验证 20,626。
- 可训练视觉参数必须为 7,482,624，范围只能是 `proj`、完整 `resblocks.11` 与 `ln_post`；分类头保持冻结。
- 必须完成 3 epochs，保存 `best.pt`、`last.pt`、`metrics.json` 和完整日志。
- 任意 NaN/Inf、OOM、读取 test、参数范围或样本数不符，立即停止并保留现场。
- 不因中间 epoch 表现不佳提前停止，也不得自行增加 epoch、改学习率或追加 seed。

A 的晋级门槛为：最佳 Top-1 至少 `0.694184627`（比当前 rank-8 seed1 高 0.20 点），并且最佳 epoch 的 macro 不低于 `0.691577219`、冲突子集 Top-1 不低于 `0.305825727`。B 只原样回传结果，不负责依据门槛安排后续实验。

## 资源预算

- 单卡、单进程；预计 20–30 分钟。
- 峰值显存上限 1.5 GiB；磁盘新增预计 80–150 MiB。
- 每 epoch 的 `last.pt` 支持续跑。异常中断时保留完整输出目录并回报，不猜测性重跑。

## 私下回传

训练完成后，将完整实验目录、原始日志、起止时间、GPU/环境信息、配置/best/last SHA-256 和参数范围清单打包，通过私有渠道回传 A。checkpoint、日志和 ZIP 不得推送 GitHub。

如果 B 仍保留上一轮已验哈希的数据、H1 checkpoint 和 OpenAI CLIP 权重，本次无需 A 另发私有输入；若任一文件缺失或哈希不符，先停下向 A 索取，不重新下载或自行替代。
