# 任务单：20260916 批 2（机制归因 + min_weight 搜索）

## 依据

- 预注册：`docs/57_批2预注册_机制归因与min_weight搜索.md`（门禁在其中写死）
- 上游：`docs/56_C候选cwh三种子通过全量复训与提交验收.md`（线上 59.6107）

## 冻结版本

- **训练 commit（完整 40 位）：`228c154bccdd9625fc9d8e819cfd9e65297a79f9`**
- 该 commit 包含两个配置、差异单测 `tests/test_t2_cwh_attribution_plan.py` 与预注册文档 `docs/57_…`。
- 本任务单位于更新的 commit（仅文档），训练必须 checkout 上述 commit。

## 两项定义

公共：`fixed_split_validation`、无泄漏 H1 `best.pt`、T2 `last_n_blocks`/末 4 块、
冻结分类头、batch 64、workers 4、seed `20260910`、wd 0.01、clip 1.0、6 epochs、不访问 test。

| 角色 | 配置文件 | 唯一变量 | 输出目录 |
| --- | --- | --- | --- |
| D | `p3_stage1_h1init_t2_last4_lr15e6_cwh_6ep_seed1_d.yaml` | lr1.5e-6 + classwise history | `artifacts/experiments/20260916-d-t2-last4-lr15e6-cwh-6ep-seed1` |
| E | `p3_stage1_h1init_t2_last4_lr175e6_cwh_mw35_6ep_seed1_e.yaml` | lr1.75e-6 + cwh，`min_weight=0.35` | `artifacts/experiments/20260916-e-t2-last4-lr175e6-cwh-mw35-6ep-seed1` |

软权重公共参数：`enabled: true`、`warmup_epochs: 1`、`momentum: 0.8`、`history_mix: 0.5`。

## 开始前门禁

```powershell
Set-Location D:\aicomp\AICtrack1
git fetch origin
git switch --detach 228c154bccdd9625fc9d8e819cfd9e65297a79f9
git rev-parse HEAD
git status --short
.\.venv\Scripts\python.exe -m pytest tests\test_t2_cwh_attribution_plan.py tests\test_t2_lr175e6_cwh_full_plan.py tests\test_peft.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe scripts\check_governance.py
```

要求：HEAD 精确匹配、工作区为空、测试与治理检查通过；
manifest `212f645a…75ef`、split `9eb97e2b…fb31`、H1 无泄漏初始化器 `e85d53d4…85af` 一致。

## 唯一命令（两项串行）

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr15e6_cwh_6ep_seed1_d.yaml
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_cwh_mw35_6ep_seed1_e.yaml
```

单卡必须串行；不得改变顺序、参数、seed、epoch、batch、输出目录或 commit。

## 资源预算（用户已预先批准至今晚产出）

- 2 项 × 约 30 分钟 ≈ **60 分钟**；峰值显存约 1.15 GiB；磁盘约 0.9 GiB。

## 判定

按 `docs/57` 第 3 节做归因解释，按第 4 节 seed1 门禁判定；
最多晋级两个候选进入 seed2/seed3（同一 commit，避免跨 commit 问题）。
`min_weight` 向上搜索在第 4 节附加停止线触发时终止，不再测试 0.5。

## 停止线

NaN/Inf、CUDA OOM、test 读取、样本数/参数范围不符、checkpoint 无法保存、
HEAD/配置/输入哈希不一致、同一 GPU 出现第二个训练进程 → 立即停止并保留现场。
