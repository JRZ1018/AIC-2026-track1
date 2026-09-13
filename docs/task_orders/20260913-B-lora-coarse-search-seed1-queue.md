# 任务单：20260913-B LoRA 粗筛六项 seed1 串行队列

## 目标

在 H1 初始化、冻结分类头、3 epochs 的固定配方上，只沿三个轴做单 seed 粗筛：LoRA rank、`alpha/rank` 比例、末尾注意力块数。共六项，全部为 seed1，单卡单进程串行执行。

本任务只做固定验证集筛选，不追加 seed、不访问 test、不生成提交、不做全量复训。六项之间没有中途门禁：本队列一旦启动就把六项跑完，A 在队列结束后统一按 `docs/36_LoRA三四天粗筛队列预注册.md` 的选模门禁排序，最多挑两个候选进入 seed2/seed3。

上游触发项：`docs/04_上游方案调研与复用登记.md` 中 LoRA 相关条目（OpenAI CLIP 官方权重 + 自研最小 LoRA 实现，无第三方代码引入，无 `THIRD_PARTY_NOTICES.md` 变更）。

## 冻结输入

- 训练 commit：`452cf43aba094ea5bc77a7ba8aabe1437afb4b03`
- 要求工作区：clean（`git status --short` 必须为空）
- train manifest SHA-256：`212f645a0d027cc4f2aa1a6d2f81f0175e8a3441b6fa6d001947038e00ba75ef`
- split SHA-256：`9eb97e2b89da76f64323c2800d6c524a8455e31eaa9b0dab1917b7e4b877fb31`
- label_conflicts SHA-256：`d06a03bf23b54e663bfc849236f855dfeb61251262329210a72274b2c4df70f1`
- 数据配置 SHA-256：`a5919df165c557ecaf94e649a757e8081dfbf06ce84774104ff243e8a4c55251`
- H1 `best.pt` SHA-256：`e85d53d446fef914c7cec74faa1ede792b7774bb71ec057dee2b551d6cf485af`
- OpenAI CLIP ViT-B/32 SHA-256：`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`

队列内六项共享完全相同的 `training` 段：3 epochs、seed `20260910`、batch size 64、workers 4、视觉学习率 `1e-4`、分类头学习率 `0`、weight decay 0.01、gradient clip 1.0；并共享相同的 `head_checkpoint`、`manifest`、`split`、`label_conflicts`、`expected_initial_top1` 与 `device: cuda`。

## 六项顺序与各自哈希

严格按下列顺序串行执行。配置哈希允许匹配 LF 或 CRLF 之一（Git 在 Windows 上可能以 CRLF 检出）。

| 顺序 | 单一问题 | 配置文件 | run_id / 输出目录名 | LF SHA-256 | CRLF SHA-256 |
|---:|---|---|---|---|---|
| 1 | `alpha/rank` 由 2 降到 1 | `configs/experiment/p3_stage1_h1init_peft_lora_last2_r8a8_3ep_seed1_b_planned.yaml` | `20260913-b-h1init-lora-last2-r8a8-3ep-seed1` | `3087e754e800950bfa0f07fc12ab46fc9270b77d425ecd13faef661e1abfadbf` | `4bbdd841c89408e6257e34b0726f51fa4c32a2852a09e5f1ca81b453a9d8602a` |
| 2 | `alpha/rank` 由 2 升到 4 | `configs/experiment/p3_stage1_h1init_peft_lora_last2_r8a32_3ep_seed1_b_planned.yaml` | `20260913-b-h1init-lora-last2-r8a32-3ep-seed1` | `e7a40027dbfed1c40bddd1c5e1660efe6825d82fbe2e19a8611583f4099a9aae` | `f14786a8e6627a42dd5d6c21fa616cb95479b00ffda279cea4040792f61df149` |
| 3 | 固定比例 2、提高 rank 到 16 | `configs/experiment/p3_stage1_h1init_peft_lora_last2_r16a32_3ep_seed1_b_planned.yaml` | `20260913-b-h1init-lora-last2-r16a32-3ep-seed1` | `b0cede489a4d9754007117fe313525cce22781f6e884eb8f627f13e2cbdb1d3c` | `34b6bb35e8dd6516aacb51e3d6b46ed486f8e5e1a182d08df605ed6a80223341` |
| 4 | 固定比例 2、继续提高 rank 到 32 | `configs/experiment/p3_stage1_h1init_peft_lora_last2_r32a64_3ep_seed1_b_planned.yaml` | `20260913-b-h1init-lora-last2-r32a64-3ep-seed1` | `917d426ef8b23bed645c148176a6e148bedb0dd3694ccfe2461e9b48ba8b4f71` | `231fa5d12ef41bee2617db1bdec3c5b445d2ede85181f5522fe2c89bf707219c` |
| 5 | 固定 rank/比例、扩到末三层 | `configs/experiment/p3_stage1_h1init_peft_lora_last3_r8a16_3ep_seed1_b_planned.yaml` | `20260913-b-h1init-lora-last3-r8a16-3ep-seed1` | `cb29bf006e08539b1806156fc9b817d3920e1df3725f80a919cbf64793f08d79` | `d33143ef0fff06c0b297b8249a262af1c5ef470bb87b10a74e6c5a11ab9be2d9` |
| 6 | 固定 rank/比例、扩到末四层 | `configs/experiment/p3_stage1_h1init_peft_lora_last4_r8a16_3ep_seed1_b_planned.yaml` | `20260913-b-h1init-lora-last4-r8a16-3ep-seed1` | `fb4a0b7532ac1e2bfde1f04fef73fae0abb32107c76e9c7f7d10c6e9f52ee549` | `474179243f434e3bf073ab9c4354961ed79efb3f32d42f84fb8eb1b45c870ef8` |

六项均为 `execution_enabled: true`。文件名中的 `_planned` 只是预注册来源标记，不表示仍被禁用。

期望的可训练视觉参数（A 已独立核算，与预注册表一致）：

每个注意力块的可训练 LoRA 参数为 `4 × rank × 768`，共 8 个张量：`in_proj_weight` 是 qkv 融合权重（形状 2304×768），贡献 `rank × (2304 + 768)`；`out_proj.weight`（768×768）贡献 `rank × (768 + 768)`。

| 顺序 | 项 | 每块参数 | 可训练视觉参数 | 张量数 | 参数名块范围 |
|---:|---|---:|---:|---:|---|
| 1 | last2 r8 α8 | 36,864 | 73,728 | 8 | resblocks.10–11 attn |
| 2 | last2 r8 α32 | 36,864 | 73,728 | 8 | resblocks.10–11 attn |
| 3 | last2 r16 α32 | 73,728 | 147,456 | 8 | resblocks.10–11 attn |
| 4 | last2 r32 α64 | 147,456 | 294,912 | 8 | resblocks.10–11 attn |
| 5 | last3 r8 α16 | 36,864 | 110,592 | 12 | resblocks.9–11 attn |
| 6 | last4 r8 α16 | 36,864 | 147,456 | 16 | resblocks.8–11 attn |

这些数值与三处独立证据一致：A 端在提交 `452cf43` 上直接实例化 OpenAI CLIP 视觉塔并按配置逐项实测；`docs/36` 中容量上界两项的启用态 CUDA smoke；以及历史实验台账中同一实现路径的既有记录（last2 rank4 为 36,864、last2 rank8 alpha16 为 73,728、last3 rank4 为 55,296）。第 4 项同时是峰值显存上界，实测 505.24 MiB。

## 开始前门禁

```powershell
Set-Location D:\aicomp\AICtrack1
git fetch origin
git switch --detach 452cf43aba094ea5bc77a7ba8aabe1437afb4b03
git rev-parse HEAD
git status --short
.\.venv\Scripts\python.exe -m pytest tests\test_b_next_queue_plan.py tests\test_h1init_peft_plan.py tests\test_peft.py tests\test_peft_checkpoint.py -q
.\.venv\Scripts\python.exe scripts\check_governance.py
```

HEAD 必须精确等于 `452cf43aba094ea5bc77a7ba8aabe1437afb4b03`，工作区必须为空，测试与治理必须通过；六项配置哈希必须逐项匹配上方 LF 或 CRLF 之一；五项私有输入哈希必须匹配。任一不符立即停止并回报 A，不自行修复。

## 唯一训练命令

按顺序逐项执行，每项完成后才开始下一项：

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_peft_lora_last2_r8a8_3ep_seed1_b_planned.yaml
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_peft_lora_last2_r8a32_3ep_seed1_b_planned.yaml
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_peft_lora_last2_r16a32_3ep_seed1_b_planned.yaml
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_peft_lora_last2_r32a64_3ep_seed1_b_planned.yaml
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_peft_lora_last3_r8a16_3ep_seed1_b_planned.yaml
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_peft_lora_last4_r8a16_3ep_seed1_b_planned.yaml
```

不得并发。输出目录固定为 `artifacts/experiments/<run_id>/`，即上表第五列。目录一旦非空只允许训练器原生续跑；不得删除、覆盖或改 run ID。训练期间不得 `git pull` 或切换 commit。

## 成功条件

- 每项完成 3 epochs，生成 `config.yaml`、`command.txt`、`metrics.json`、`best.pt`、`last.pt` 与完整日志。
- 每项固定划分数量必须为训练 82,592、验证 20,626。
- 每项可训练视觉参数必须等于上方参数表对应值，范围只能是各配置声明的末 N 个注意力块的 LoRA；分类头保持冻结。
- 预训练权重、H1 初始化、manifest、split 与 label_conflicts 哈希全部匹配。

## 停止线

- 任意 NaN/Inf、CUDA OOM、读取 `DataSetStage1/test`、样本数不符、参数范围或数量不符、输出目录非空或属于另一任务、依赖缺失或需要安装新包，立即停止当前项并保留现场回报 A，不猜测性修复、不删除 checkpoint、不重置 Git。
- 不因中间 epoch 表现不佳提前停止当前项，也不得自行增加 epoch、改学习率、追加 seed 或叠加任务单外方法。
- 时间停止线：从第 1 项启动起算，若已运行满 3 小时 25 分而第 6 项尚未启动，不启动第 6 项，直接把已完成的项回传并回报 A（与上一轮队列同一停止规则）。
- 若某项异常中断，`last.pt` 支持用同一条命令原生续跑；续跑只在任务单给出的输出目录内进行，并如实报告实际续跑情况。

## 资源预算

- 单卡、单进程串行；六项合计预计 3–3.7 小时（单项约 24–32 分钟）。
- 峰值显存上限 0.9 GiB。A 端预检实测：rank 32/2 块为 505.24 MiB、rank 8/4 块为 619.04 MiB，均低于上限。
- 磁盘新增预计 30–90 MiB（含每项 best/last checkpoint）。
- 每 epoch 的 `last.pt` 支持续跑。

## 私下回传

训练完成后，将六个完整实验目录、原始日志、起止时间、GPU/环境信息、每项配置与 best/last 的 SHA-256、以及每项的实际可训练参数与参数名范围打包，通过私有渠道回传 A。checkpoint、日志和 ZIP 不得推送 GitHub。

回传时按 `docs/17_双人协作与训练执行交接手册.md` §4.4 的格式逐项列出事实（任务 ID、实际 commit、状态、起止与总耗时、峰值显存、最佳 epoch、Top-1、macro、冲突子集 Top-1、best/last SHA-256、输出目录、是否异常），不自行给出"应进入下一阶段"之类的选模结论。

若 B 仍保留上一轮已验哈希的数据、H1 checkpoint 和 OpenAI CLIP 权重，本次无需 A 另发私有输入；若任一文件缺失或哈希不符，先停下向 A 索取，不重新下载或自行替代。
