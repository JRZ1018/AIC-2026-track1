# 任务单：20260916 批 3（min_weight=0.5 与 8 epochs 延长）

## 依据

- 预注册：`docs/58_批3预注册_mw50与8epochs延长.md`（含基线切换声明与门禁）
- 上游：`docs/56_C候选cwh三种子通过全量复训与提交验收.md`（线上 59.6107）

## 冻结版本

- **训练 commit（完整 40 位）：`c7c3c620d3dafcbe07a50406becf0878c26428d7`**
- 该 commit 包含两个配置、差异单测 `tests/test_t2_cwh_mw_and_epochs_plan.py`
  与预注册文档 `docs/58_…`。
- 本任务单位于更新的 commit（仅文档），训练必须 checkout 上述 commit。

## 基线（本批起切换）

seed1 门禁基线改用已通过三种子门禁的 C 候选 seed1：

| 指标 | 基线值 |
| --- | ---: |
| Top-1 | `0.711286723613739` |
| macro | `0.7111637592315674` |
| 冲突子集 | `0.31013917922973633` |

**不得追溯适用于批 2**（批 2 已按旧基线冻结）。

## 两项定义

公共：`fixed_split_validation`、无泄漏 H1 `best.pt`、T2 `last_n_blocks`/末 4 块、
冻结分类头、batch 64、workers 4、seed `20260910`、wd 0.01、clip 1.0、
visual lr `1.75e-6`、软权重 `enabled/warmup 1/momentum 0.8/history_mix 0.5`、不访问 test。

| 角色 | 配置文件 | 唯一变量 | 输出目录 |
| --- | --- | --- | --- |
| F | `p3_stage1_h1init_t2_last4_lr175e6_cwh_mw50_6ep_seed1_f.yaml` | `min_weight=0.5`（6 epochs） | `artifacts/experiments/20260916-f-t2-last4-lr175e6-cwh-mw50-6ep-seed1` |
| G | `p3_stage1_h1init_t2_last4_lr175e6_cwh_8ep_seed1_g.yaml` | 8 epochs（`min_weight=0.2`） | `artifacts/experiments/20260916-g-t2-last4-lr175e6-cwh-8ep-seed1` |

## 开始前门禁

```powershell
Set-Location D:\aicomp\AICtrack1
git fetch origin
git switch --detach c7c3c620d3dafcbe07a50406becf0878c26428d7
git rev-parse HEAD
git status --short
.\.venv\Scripts\python.exe -m pytest tests\test_t2_cwh_mw_and_epochs_plan.py tests\test_t2_cwh_attribution_plan.py tests\test_peft.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe scripts\check_governance.py
```

要求：HEAD 精确匹配、工作区为空、测试与治理检查通过；
manifest `212f645a…75ef`、split `9eb97e2b…fb31`、H1 无泄漏初始化器 `e85d53d4…85af` 一致。

## 唯一命令（两项串行）

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_cwh_mw50_6ep_seed1_f.yaml
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_cwh_8ep_seed1_g.yaml
```

单卡必须串行；不得改变顺序、参数、seed、epoch、batch、输出目录或 commit。

## 资源预算（用户已批准至今晚）

- F：6 epochs 约 30 分钟；G：8 epochs 约 40 分钟；合计约 **70 分钟**
- 峰值显存约 1.15 GiB；磁盘约 0.9 GiB

## 判定

按 `docs/58` 第 2 节新基线门禁判定，最多晋级两个；
G 的定位是"受控延长对照"——若 8 epochs 不能同时改善 Top-1 与冲突指标，判定延长无效。

## 停止线

NaN/Inf、CUDA OOM、test 读取、样本数/参数范围不符、checkpoint 无法保存、
HEAD/配置/输入哈希不一致、同一 GPU 出现第二个训练进程 → 立即停止并保留现场。
