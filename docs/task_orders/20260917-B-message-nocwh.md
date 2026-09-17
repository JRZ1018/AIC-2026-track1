# 给 B 的消息（可直接复制粘贴）—— item N1

B，上一单验收通过：**你的机器和我的等价**（12 个 epoch 的 Top-1、全部 48 个权重张量、
RNG 状态都与我的逐位相同），所以下面这一单的结果可以直接和我的合并比较。

这一单是**归因对照**，不是新方向。任务单全文在仓库里：
`docs\task_orders\20260917-B-nocwh-control-n1.md`

## 这一单要回答什么

这条路线上的每个实验都带着"类内历史软加权"，**它到底值多少分，我们从来没测过**。
本单把它**整块关掉**，其余一切不变。三种结果都有用。

## 你要做什么

**第一步，同步代码，核对 commit：**

```powershell
git fetch origin
git checkout 3f02eebfa99752020e7b2eeed742945e1648129c
git rev-parse HEAD
```

必须原样输出 `3f02eebfa99752020e7b2eeed742945e1648129c`。**不要用 `main`**，工作区要干净。

**第二步，核对四个输入：**

- `artifacts\audits\stage1\train\samples.csv`
- `artifacts\audits\stage1\train\split.csv`
- `artifacts\audits\stage1\train\label_conflicts.csv`
- `artifacts\experiments\20260911-p3-h1-linear-adam-e100-seed42\best.pt`
  → SHA-256 必须是 `e85d53d446fef914c7cec74faa1ede792b7774bb71ec057dee2b551d6cf485af`

并确认输出目录**启动前不存在**：
`artifacts\experiments\20260917-n-t2-last4-lr175e6-nocwh-12ep-seed1`

**第三步，环境自检：**

```powershell
$env:PYTEST_DEBUG_TEMPROOT="D:\aicomp\AICtrack1\tmp\pytest-temproot"
.\.venv\Scripts\python.exe -m pytest tests\test_t2_nocwh_control_plan.py tests\test_confidence.py -q
```

**第四步，跑训练（唯一命令）：**

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_nocwh_12ep_seed1_n.yaml
```

**耗时预算：按你机器的实测吞吐，预计 180–216 分钟**（15–18 min/epoch × 12）。
如果你机器上还有别的负载，开工前先告诉我。

**如果中途被中断**（休眠/断电/手动停止），**不要重跑上面的命令**（会撞 `FileExistsError`），
改用：

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_nocwh_12ep_seed1_n.yaml --resume artifacts\experiments\20260917-n-t2-last4-lr175e6-nocwh-12ep-seed1\last.pt
```

上次任务单漏写了这条，是我的疏漏，这次已写明。续跑会让 `metrics.json` 的
`initial_validation` 变成 `null`，这是正常副作用，不用为此重跑。

## 自查用的预期数字（不是判定）

- `initial_validation.all.top1` 应 ≈ **0.686996996**（配置里有 ±0.002 守卫，不符会立刻报错）
- **epoch 1 的 Top-1 应 = `0.695432961`**（机制在 warmup 阶段本来就不生效，所以与我的历史运行相同）
- **从 epoch 2 起应与 `0.703238606` 分叉**（因为加权已关闭）

**如果 epoch 1 不是 `0.695432961`，或 epoch 2 仍等于 `0.703238606`，立刻停下报我**——
那说明配置没生效。

## 红线

- 不访问 test、不推理、不生成提交包、**任何参数都不许改**
- 中途不 `git pull`、不切 commit
- **本单只跑这一项，不要自行追加全量那一项**（是否跑全量由我按预注册阈值决定）
- 跑完即停，等我的下一单

## 跑完回传什么

**1）五个文件的 SHA-256**：`config.yaml`、`command.txt`、`metrics.json`、`best.pt`、`last.pt`
（`SHA256SUMS.txt` 用 `<sha256>  <size>  <path>` 三段格式，上次的格式没问题）
**2）训练日志**
**3）三个数字**：
- `best_top1`
- `history[*].validation.all.top1` 全 12 个值
- **`history[*].validation.cross_label_exact_conflicts.top1` 全 12 个值** ← 冲突子集，**必须报**

**4）前置核对结果**：HEAD、工作区状态、四个哈希、自检是否通过

**打包体积**：`last.pt` 是 436 MB，如果上传费劲，**只传 `best.pt` 也可以**——我这边有同构副本可以对照。
