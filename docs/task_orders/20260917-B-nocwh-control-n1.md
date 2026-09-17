# 任务单：20260917 B 关闭类内历史软加权（item N1，seed 20260910）

## 本单目的

这条路线上的每个实验都带着类内历史软加权，**它值多少分从未被测过**。本单把它整块关掉，
其余一切不变，直接对照已知的 `0.719480276`（item K，同种子、同头、同数据、同预算）。

**这是归因对照，不是新方向。** 三种结果都有用：高于对照说明机制有害、接近说明机制中性、
低于说明机制有正贡献。

## 冻结信息

- **训练 commit：待下发（见任务单末尾的 SHA，必须逐字核对）**
- 配置：`configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_nocwh_12ep_seed1_n.yaml`
- 唯一变化：**整块删除 `confidence_weighting`**；epochs 12、lr 1.75e-6、
  `last_n_blocks` 4、batch 64、seed 20260910、wd 0.01、clip 1.0、冻结头、H1 seed42 `best.pt` 全部不变
- 输出目录：`artifacts\experiments\20260917-n-t2-last4-lr175e6-nocwh-12ep-seed1`
- **耗时预算：按你机器的实测吞吐估 ≈180–216 分钟（15–18 min/epoch × 12）。** 上次你实测
  721–840 s/epoch，明显慢于 A 机的 309 s。若你机器上有其他负载，请先说明，我会调整预算。

## 开始前必须逐项确认并回报

1. `git rev-parse HEAD` 等于任务单末尾给定的 40 位 SHA，且工作区干净；
2. 四个输入哈希与 A 一致（第四个必须精确等于给定值）：
   - `artifacts\audits\stage1\train\samples.csv`
   - `artifacts\audits\stage1\train\split.csv`
   - `artifacts\audits\stage1\train\label_conflicts.csv`
   - `artifacts\experiments\20260911-p3-h1-linear-adam-e100-seed42\best.pt`
     → SHA-256 = `e85d53d446fef914c7cec74faa1ede792b7774bb71ec057dee2b551d6cf485af`
3. 输出目录启动前不存在；
4. 环境自检通过：

```powershell
$env:PYTEST_DEBUG_TEMPROOT="D:\aicomp\AICtrack1\tmp\pytest-temproot"
.\.venv\Scripts\python.exe -m pytest tests\test_t2_nocwh_control_plan.py tests\test_confidence.py -q
```

任一项不符，立即停止并回报 A，不要自行修复。

## 唯一命令

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_nocwh_12ep_seed1_n.yaml
```

### 中断与续跑条款（本次已明确写入）

若运行被中断（休眠、断电、手动停止），**不要重新跑第一条命令**（会撞 `FileExistsError`），
改用续跑命令；续跑只从 epoch 边界恢复，被中断的那个 epoch 会整段重算，这是正确的：

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_nocwh_12ep_seed1_n.yaml --resume artifacts\experiments\20260917-n-t2-last4-lr175e6-nocwh-12ep-seed1\last.pt
```

**已知副作用（可接受）**：续跑会使 `metrics.json` 的 `initial_validation` 变为 `null`。
无需为此重跑。

## 预期数字（用于自查，不是判定）

- `initial_validation.all.top1` 应 ≈ **0.686996996**（配置里有 ±0.002 守卫，不符会立刻报错）
- epoch 1 的 Top-1 应 = **0.695432961**（与 A 的历史运行逐位相同；因为机制在 warmup 阶段本来就不生效）
- **从 epoch 2 起应与 A 的历史轨迹分叉**（因为加权已经关掉）

**若 epoch 1 不是 0.695432961，或 epoch 2 与 `0.703238606` 相同，请立刻停止回报 A**——
那说明配置没生效。

## 完成后回传

1. 完整实验目录：`config.yaml` / `command.txt` / `metrics.json` / `best.pt` / `last.pt`
2. 训练日志
3. 上述五个文件的 SHA-256
4. 三个数字：
   - `best_top1`
   - `history[*].validation.all.top1` 全 12 个值
   - **`history[*].validation.cross_label_exact_conflicts.top1` 全 12 个值**（冲突子集，必须报告）
5. 前置核对结果（HEAD、工作区、四个哈希、自检是否通过）

打包与回传方式同上一单（`SHA256SUMS.txt` 用 `<sha256>  <size>  <path>` 三段格式即可）。
**checkpoint 若超过 500 MB 可只传 `best.pt`，`last.pt` 可选**——A 侧已有同构副本可对照。

## 边界

- 不访问 test、不推理、不生成提交包、**任何参数都不许改**
- 中途不 `git pull`、不切 commit
- 本单只跑这一项，**不得自行追加 N2（全量那一项）**——N2 是否执行由 A 按预注册阈值决定
- 跑完即停，等 A 下发下一单
