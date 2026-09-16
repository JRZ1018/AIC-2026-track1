# 任务单：20260916 多种子确认（C 候选 lr1.75e-6+cwh 与 B 候选 lr1.65e-6）

## 依据

- 第一批结果与晋级决定：`docs/55_第一批seed1结果与多种子确认预注册.md`
- 上游预注册门槛：`docs/54_第一批seed1短筛预注册.md` 第 4 节（未修改）
- 上游交接：`docs/52_新AI交接_当前最佳58.8577与后续追分方案.md` 第 9 节

## 冻结版本

- **训练 commit（完整 40 位）：`b408aa7639def5932c14f0f2435521360eaa4dde`**
- 该 commit 包含四个多种子配置、差异单测
  `tests/test_t2_lr_balance_multiseed_plan.py` 与预注册文档 `docs/55_…`。
- 本任务单本身位于更新的 commit（仅文档），训练必须 checkout 上述 commit。

## 四项定义

公共：`fixed_split_validation`、无泄漏 H1 `best.pt`、T2 `last_n_blocks`/末 4 块、
冻结分类头、batch 64、workers 4、wd 0.01、clip 1.0、6 epochs、不访问 test。

| 候选 | 配置文件 | seed | 输出目录 |
| --- | --- | ---: | --- |
| C | `p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_seed2_c.yaml` | 20260911 | `artifacts/experiments/20260916-c-t2-last4-lr175e6-cwh-6ep-seed2` |
| C | `p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_seed3_c.yaml` | 20260912 | `artifacts/experiments/20260916-c-t2-last4-lr175e6-cwh-6ep-seed3` |
| B | `p3_stage1_h1init_t2_last4_lr165e6_6ep_seed2_b.yaml` | 20260911 | `artifacts/experiments/20260916-b-t2-last4-lr165e6-6ep-seed2` |
| B | `p3_stage1_h1init_t2_last4_lr165e6_6ep_seed3_b.yaml` | 20260912 | `artifacts/experiments/20260916-b-t2-last4-lr165e6-6ep-seed3` |

C 候选额外携带：`confidence_weighting`（`enabled: true`、`warmup_epochs: 1`、
`momentum: 0.8`、`history_mix: 0.5`、`min_weight: 0.2`）。
B 候选为纯 CE，无 `confidence_weighting`。

## 第一批 seed1 对照（已验收，本批只做确认）

| 候选 | seed1 Top-1 | seed1 macro | seed1 冲突子集 |
| --- | ---: | ---: | ---: |
| C | 0.711286724 | 0.711163759 | 0.310139179 |
| B | 0.706244528 | 0.706124783 | 0.308813781 |

## 开始前门禁

```powershell
Set-Location D:\aicomp\AICtrack1
git fetch origin
git switch --detach b408aa7639def5932c14f0f2435521360eaa4dde
git rev-parse HEAD
git status --short
.\.venv\Scripts\python.exe -m pytest tests\test_t2_lr_balance_multiseed_plan.py tests\test_t2_lr_balance_plan.py tests\test_t2_lr175e6_plan.py tests\test_peft.py tests\test_peft_checkpoint.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe scripts\check_governance.py
```

要求：HEAD 精确匹配、工作区为空、全部测试与治理检查通过；
manifest `212f645a…75ef`、split `9eb97e2b…fb31`、
H1 无泄漏初始化器 `e85d53d4…85af` 哈希一致。

## 唯一命令（四项串行，不得并发）

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_seed2_c.yaml
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_seed3_c.yaml
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr165e6_6ep_seed2_b.yaml
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr165e6_6ep_seed3_b.yaml
```

本机仅 1 张 GPU，必须串行；一项结束后才开始下一项。
不得改变顺序、参数、seed、epoch、batch、输出目录或 commit。

## 资源预算（用户已预先批准，直至产出下一个 pred_results）

- 4 项 × 约 31 分钟 ≈ **2 小时 5 分钟**
- 单项峰值显存约 **1.15 GiB**
- 单项磁盘约 0.45 GiB，合计约 **1.8 GiB**

## 三种子门禁（每个候选独立判定）

对照 lr1.5e-6 三种子基线：Top-1 均值 `0.705468833`、macro 均值 `0.705364744`、
冲突子集均值 `0.308813790`。

1. 三个 seed 的 Top-1 都高于各自 lr1.5e-6 基线；
2. Top-1 均值 ≥ `0.705968833`；
3. macro 均值 ≥ `0.705364744`；
4. 冲突子集均值 ≥ `0.308801864`；
5. 三项训练完整、history epoch 连续；6. 无 NaN/Inf；7. checkpoint 可加载；
8. 配置/输入/commit 哈希一致；9. 未访问 test。

任何一项失败即停止该候选，不启用其 full config。
若两个候选都通过，按 Top-1 均值最高者进入全量。

## 停止线

NaN/Inf、CUDA OOM、test 读取、样本数/参数范围不符、checkpoint 无法保存、
HEAD/配置/输入哈希不一致、同一 GPU 出现第二个训练进程 → 立即停止并保留现场。

异常恢复：仅当同一 commit、同一配置、同一输出目录存在可加载 `last.pt` 时，
才允许用原配置加 `--resume` 恢复。
