# 给 D 的消息（可直接复制粘贴）—— 环境关卡 + item M

D，欢迎加入。你的机器是全新的，所以**在跑正式任务前必须先过一道约 25 分钟的校验关卡**。
原因很简单：如果数据或环境和主节点有任何一处不一致，一道 4 小时以上的训练会白跑，
而且可能产出"看起来正常、实际不可比"的数字。任务单全文在仓库里：
`docs\task_orders\20260917-D-setup-parity-and-16ep.md`

**前提**：你的环境应已按 `docs\17_双人协作与训练执行交接手册.md` §2 备好
（Git clone 的仓库、`DataSetStage1\`、OpenAI CLIP 权重、私有 H1 初始化头、仓库内 `.venv`）。
**环境没备齐就不要开始**，直接回报，不要自己搭。

---

## 第 0 步：环境自检

```powershell
git fetch origin
git checkout <下方 SHA>
git rev-parse HEAD
```

必须原样输出该 SHA，工作区要干净（`git status --short` 为空）。**不要用 `main`。**

**SHA：`<见任务单末尾，A 会在下发给你的消息里替换>`**

核对这五个文件的 SHA-256：

| 文件 | 要求 |
|---|---|
| `artifacts\audits\stage1\train\samples.csv` | 回报，A 比对 |
| `artifacts\audits\stage1\train\split.csv` | 回报 |
| `artifacts\audits\stage1\train\label_conflicts.csv` | 回报 |
| `artifacts\experiments\20260911-p3-h1-linear-adam-e100-seed42\best.pt` | 必须 = `e85d53d446fef914c7cec74faa1ede792b7774bb71ec057dee2b551d6cf485af` |
| `artifacts\weights\ViT-B-32.pt` | 必须 = `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` |

确认测试清单存在即可（`artifacts\audits\stage1\test\samples.csv`，24,967 行）。
**不要读测试图片、不要推理。**

单元测试：

```powershell
$env:PYTEST_DEBUG_TEMPROOT="D:\aicomp\AICtrack1\tmp\pytest-temproot"
.\.venv\Scripts\python.exe -m pytest tests\test_d_setup_parity_plan.py tests\test_confidence.py -q
```

**任何一项不符：停下报我，不要自行修复。**

---

## 第 1 步（关卡）：3-epoch 一致性校验

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_d_setup_parity_3ep_seed1.yaml
```

预计 **20–35 分钟**。

**三个 epoch 的 Top-1 必须精确等于：**

| epoch | 必须 = |
|---:|---|
| 1 | **`0.695432961`** |
| 2 | **`0.703238606`** |
| 3 | **`0.709589839`** |

这三个值已在**两台互相独立的机器**（我的和 B 的，Python 3.12.10 / 3.12.3，torch 版本也不同）
上**逐位复现**。第一个 epoch 之所以能与我的完全相同，是因为软加权在 warmup 阶段本就不生效；
第 2 个 epoch 起机制生效，但只要配置和数据一致，仍然逐位相同。

**这三个数只要有一个不符，立刻停下报我，不要进第 2 步。** 那通常意味着数据、split、
初始化头或 CLIP 权重与主节点不一致——这正是这道关卡的意义。

**过关后**把该实验目录的 `metrics.json` 和 `command.txt` 发我。

---

## 第 2 步：正式任务 item M（16 epochs）

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix50_16ep_seed1_m.yaml
```

预计 **240–290 分钟**。**请在第 1、2 个 epoch 跑完后告诉我实际每 epoch 耗时**，我会校正预算。

**这一项在测什么**：把训练时长从 12 epochs 推到 16。时长是目前**唯一有重复线上证据**的轴
（6→8 epochs 线上 +0.1642；8→12 epochs 线上 +0.6969），但本地增量在衰减
（8→10 是 +0.002909，10→12 只剩 +0.001697），所以我们要看它到底还有没有油水。

**如果中途被中断**（休眠/断电/手动停止），**不要重跑上面的命令**（会撞 `FileExistsError`），改用：

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix50_16ep_seed1_m.yaml --resume artifacts\experiments\20260917-m-t2-last4-lr175e6-cwh-mw10-magmix50-16ep-seed1\last.pt
```

续跑会让 `metrics.json` 的 `initial_validation` 变成 `null`，这是正常副作用，不用为此重跑。

**自查**：epoch 1 应 = `0.695432961`，epoch 2 应 = `0.703238606`。
**epoch 3 起不需要和任何数字比对**——你的 16-epoch 轨迹会与 12-epoch 的历史轨迹分叉，
这是预期的（epoch 预算不同，余弦学习率调度随之改变）。但 epoch 1、2 必须仍然精确命中。

---

## 红线

- **不访问测试集**：不读测试图片、不推理、不生成提交包
- **任何参数都不许改**：epochs、seed、lr、batch、workers、`min_weight`、`magnitude_mix` 全部按配置
- 中途不 `git pull`、不切 commit、不装依赖
- **关卡没过不得进第 2 步**
- 跑完即停，**不要自行追加实验**（包括不要把 16-epoch 改成全量口径），下一单等我的

## 回传什么（第 1 步、第 2 步分别报）

1. 五个文件的 SHA-256：`config.yaml`、`command.txt`、`metrics.json`、`best.pt`、`last.pt`
   （`SHA256SUMS.txt` 用 `<sha256>  <size>  <path>` 三段格式）
2. 训练日志
3. 三个数字：
   - `best_top1`
   - `history[*].validation.all.top1` **全序列**（关卡 3 个，M 有 16 个）
   - `history[*].validation.cross_label_exact_conflicts.top1` **全序列** ← 冲突子集，**必须报**
4. 前置核对结果：HEAD、工作区、五个输入哈希、自检是否通过

**打包**：`last.pt` 约 436 MB、`best.pt` 约 108 MB，两个都要（我要按哈希验签）；
若传输困难，可先只发 `best.pt` + `metrics.json`，`last.pt` 后补。

## 一个已知的"假警报"

你的 `metrics.json` 里 `config_sha256` **可能**和我的不同。我和 B 之间已经出现过：
两边配置内容**逐字节相同**，但这个字段不同——因为它对**文件字节**取哈希，
而不同平台的 Git 检出会用 CRLF 或 LF、还可能带 BOM。

**判定等价性的依据是"配置内容比对 + 指标逐位比对"，不是 `config_sha256`。**
所以回报时**把 `config.yaml` 原样带上**就行。**不要为了让它一致去改文件或转换行尾。**
