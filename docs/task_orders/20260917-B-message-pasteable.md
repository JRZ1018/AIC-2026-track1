# 给 B 的消息（可直接复制粘贴）

B，你上线了，先跑一个**校准实验**，不是新方向。任务单在仓库里：
`docs\task_orders\20260917-B-magnitude-mix-reproduction.md`

## 你要做什么

**第一步，同步代码，检查是不是这个 commit：**

```powershell
git fetch origin
git checkout b07da0b2dea7125da584060c56a23e854d35351b
git rev-parse HEAD
```

`git rev-parse HEAD` 必须原样输出 `b07da0b2dea7125da584060c56a23e854d35351b`。
**不要用 `main`**，`main` 会往前动。工作区必须干净。

**第二步，确认这四个输入文件在，并且哈希和 A 一致**（前三个请回报 SHA-256，第四个必须等于给定值）：

- `artifacts\audits\stage1\train\samples.csv`
- `artifacts\audits\stage1\train\split.csv`
- `artifacts\audits\stage1\train\label_conflicts.csv`
- `artifacts\experiments\20260911-p3-h1-linear-adam-e100-seed42\best.pt`
  → SHA-256 必须是 `e85d53d446fef914c7cec74faa1ede792b7774bb71ec057dee2b551d6cf485af`

**第三步，环境自检：**

```powershell
$env:PYTEST_DEBUG_TEMPROOT="D:\aicomp\AICtrack1\tmp\pytest-temproot"
.\.venv\Scripts\python.exe -m pytest tests\test_confidence.py tests\test_t2_cwh_magnitude_mix_plan.py -q
```

**第四步，跑训练（唯一命令，别加任何参数）：**

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix50_12ep_seed1_l.yaml
```

预计 45–60 分钟，峰值显存约 1.2 GiB。

## 红线（不要做）

- 不访问 test 目录、不推理、不生成提交包
- **任何参数都不许改**（`magnitude_mix`、seed、epochs、batch、lr 全部按配置文件）
- 中途不 `git pull`、不切 commit
- 上面任何一步对不上（commit 不对、哈希不符、测试不过、目录不存在）→ **立即停下回报 A，不要自行修复**

## 跑完回传什么

**1）五个文件的 SHA-256：**
```
config.yaml  command.txt  metrics.json  best.pt  last.pt
```
加上训练日志。

**2）三个数字**（在 `metrics.json` 里，这是 A 真正要看的）：

| 数字 | 位置 | 期望 |
|---|---|---|
| `best_top1` | `metrics.json` 顶层 | 应接近 **0.719480276** |
| `initial_validation.all.top1` | `metrics.json` | 应 ≈ **0.686996996** |
| 逐 epoch Top-1 全序列 | `history[*].validation.all.top1` | 12 个数，全部列出 |

**3）前置核对结果**：HEAD、工作区状态、三个哈希、测试是否通过。

## 为什么是这个实验

A 在自己机器上跑的**就是这个配置**（15:21 启动，约 55 分钟出结果）。两边跑同一个东西，
是为了确认**你的机器和 A 的机器跑出来的数是否可比**。

A 手上已有一个同配置的已知参考值 `0.719480276`（只少 `magnitude_mix` 这一项）。

判定（A 来做，你不需要判断）：

| 你的 `best_top1` 与 0.719480276 之差 | 结论 |
|---|---|
| ≤ 0.0013 | 机器等价，后续可以两台并行、结果可合并 |
| > 0.0013 且哈希全对 | 机器不等价，先暂停跨机器并行，A 定位差异 |

**如果环境没齐（缺 `DataSetStage1\`、CLIP 权重、私有 checkpoint 或 `.venv`），先直接回报，
不要尝试自己搭**——首次同步流程见 `docs\17_双人协作与训练执行交接手册.md` §2。

跑完这一单先停，**不要自行安排下一项实验**，下一单由 A 验收后下发。
