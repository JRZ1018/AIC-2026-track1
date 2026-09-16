# 任务单：20260916 第一批 seed1 学习率平衡短筛（A/B/C 三项串行）

## 覆盖与依据

- 预注册：`docs/54_第一批seed1短筛预注册.md`（门禁在其中写死，结果出来不得修改）
- 上游交接：`docs/52_新AI交接_当前最佳58.8577与后续追分方案.md` 第 6 节
- 本任务单替代 `docs/task_orders/20260916-ABC-t2-last4-lr-short-screen.md` 作为**本批**执行依据

## 冻结版本

- **训练 commit（完整 40 位）：`4de83671f904b93a8a28c44a3a1d7d69089afb13`**
- 该 commit 包含三个配置文件、配置差异单测 `tests/test_t2_lr_balance_plan.py`
  与预注册文档 `docs/54_第一批seed1短筛预注册.md`。
- 本任务单本身位于更新的 commit（仅文档），训练必须 checkout 上述 commit。

## 三项定义（唯一变量原则）

公共：`fixed_split_validation`、无泄漏 H1 `best.pt`、T2 `last_n_blocks`/末 4 块、
冻结分类头、batch 64、workers 4、seed `20260910`、wd 0.01、clip 1.0、6 epochs、不访问 test。

| 角色 | 配置文件 | 唯一变量 | 输出目录 |
| --- | --- | --- | --- |
| A | `p3_stage1_h1init_t2_last4_lr16e6_6ep_seed1_a.yaml` | visual lr `1.60e-6` | `artifacts/experiments/20260916-a-t2-last4-lr16e6-6ep-seed1` |
| B | `p3_stage1_h1init_t2_last4_lr165e6_6ep_seed1_b.yaml` | visual lr `1.65e-6` | `artifacts/experiments/20260916-b-t2-last4-lr165e6-6ep-seed1` |
| C | `p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_seed1_c.yaml` | 增加 classwise history 软权重 | `artifacts/experiments/20260916-c-t2-last4-lr175e6-cwh-6ep-seed1` |

C 的软权重：`enabled: true`、`warmup_epochs: 1`、`momentum: 0.8`、`history_mix: 0.5`、`min_weight: 0.2`。

## 开始前门禁

```powershell
Set-Location D:\aicomp\AICtrack1
git fetch origin
git switch --detach 4de83671f904b93a8a28c44a3a1d7d69089afb13
git rev-parse HEAD
git status --short
.\.venv\Scripts\python.exe -m pytest tests\test_t2_lr_balance_plan.py tests\test_t2_lr175e6_plan.py tests\test_t2_lr15e6_multiseed_plan.py tests\test_peft.py tests\test_peft_checkpoint.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe scripts\check_governance.py
```

要求：HEAD 精确匹配、工作区为空、全部测试与治理检查通过；
四个输入哈希与 H1 初始化器哈希一致（见预注册文档第 2 节）。

## 唯一命令（三项串行，不得并发）

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr16e6_6ep_seed1_a.yaml
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr165e6_6ep_seed1_b.yaml
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_seed1_c.yaml
```

本机仅 1 张 RTX 4060 Laptop（8 GiB），**必须串行**：一项结束后才开始下一项。
不得改变顺序、参数、seed、epoch、batch、输出目录或 commit。

## 资源预算（用户已预先批准）

- 3 项 × 约 31 分钟 ≈ **95 分钟** GPU 墙钟
- 单项峰值显存约 **1.15 GiB**（安全上限 5 GiB）
- 单项磁盘约 0.45 GiB，合计约 **1.4 GiB**
- 每项支持 epoch 边界 `last.pt` 恢复

## 成功条件与停止线

成功：每项完成 6 epochs，生成 `config.yaml`、`command.txt`、`metrics.json`、`best.pt`、`last.pt`；
每项初始验证落在 `0.686996996 ± 0.002`；分类头冻结；参数范围仅限末 4 个完整视觉块。

立即停止并保留现场：NaN/Inf、CUDA OOM、test 读取、样本数/参数范围不符、
checkpoint 无法保存、HEAD/配置/输入哈希不一致、同一 GPU 出现第二个训练进程。

异常恢复：仅当同一 commit、同一配置、同一输出目录存在可加载 `last.pt` 时，
才允许用原配置加 `--resume` 恢复；不得删除有效 checkpoint、不得切换 commit、不得自行改代码。

## 完成后

按 `docs/54` 第 3 节 seed1 门禁判定，最多晋级两个候选进入 seed2/seed3；
未过门禁则按第 5 节条件实验或第 10 节分层学习率路线推进。
本阶段不访问 test、不生成提交包、不做全量复训。
