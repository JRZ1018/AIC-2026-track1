# 任务单：20260917 D 节点 setup 校验 + item M（16 epochs）

> 前置说明：本单假设你机器的环境已按 `docs/17_双人协作与训练执行交接手册.md` §2 备好
> （Git clone 的仓库、`DataSetStage1/`、OpenAI CLIP 权重、私有 H1 初始化头、仓库内 `.venv`）。
> **环境未备齐就不要开始**，直接回报，由 A 判断如何补。

> **冻结训练 commit：`e48c1894c6612f57fabce589f1a0cf70aa39bf75`**
> （必须逐字核对，不得使用 `main` 浮动头）

**关键前提：你是一台全新的机器，所以在跑正式任务之前必须先过一道校验关卡。**
如果数据或环境与 A/B 有任何一处不一致，一道 2 小时以上的训练会白跑，而且可能产出
看起来正常、实际不可比的数字。校验关卡只花约 25 分钟，**必须先过**。

---

## 第 0 步：环境自检（必须全过）

```powershell
git fetch origin
git checkout <任务单末尾给定的 40 位 SHA>
git rev-parse HEAD
```

必须原样输出该 SHA，且工作区干净（`git status --short` 为空）。**不要用 `main`。**

然后确认四个输入文件存在，并核对其 SHA-256：

| 文件 | 必须等于 |
|---|---|
| `artifacts\audits\stage1\train\samples.csv` | 回报即可（A 会比对） |
| `artifacts\audits\stage1\train\split.csv` | 回报即可 |
| `artifacts\audits\stage1\train\label_conflicts.csv` | 回报即可 |
| `artifacts\experiments\20260911-p3-h1-linear-adam-e100-seed42\best.pt` | `e85d53d446fef914c7cec74faa1ede792b7774bb71ec057dee2b551d6cf485af` |
| `artifacts\weights\ViT-B-32.pt` | `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` |

再确认测试集**只做只读存在性确认**（`artifacts\audits\stage1\test\samples.csv` 存在即可，24,967 行）。
**不得读取测试图片本身、不得推理。**

最后跑单元测试：

```powershell
$env:PYTEST_DEBUG_TEMPROOT="D:\aicomp\AICtrack1\tmp\pytest-temproot"
.\.venv\Scripts\python.exe -m pytest tests\test_d_setup_parity_plan.py tests\test_confidence.py -q
```

**任一项不符：停下回报 A，不要自行修复、不要猜。**

---

## 第 1 步（关卡）：3-epoch 一致性校验

**唯一命令：**

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_d_setup_parity_3ep_seed1.yaml
```

预计 **20–35 分钟**（按你机器的实际吞吐；若你机器上仍有其他负载，先说明）。
输出目录：`artifacts\experiments\20260917-d-setup-parity-3ep-seed1`

**判据（必须三个全中，允许小数点后 9 位的显示精度）：**

| epoch | `history[].validation.all.top1` 必须等于 |
|---:|---|
| 1 | **`0.695432961`** |
| 2 | **`0.703238606`** |
| 3 | **`0.709589839`** |

这三个值已在**两台互相独立的机器**（A 与 B，Python 3.12.10/3.12.3、torch 版本也不同）上
**逐位复现**。第 1 个 epoch 之所以能与 A 完全相同，是因为软加权机制在 warmup 阶段本就不生效；
第 2 个 epoch 起机制生效，但只要配置与数据一致，仍然逐位相同。

另外 `metrics.json` 的 `initial_validation.all.top1` 应为 `0.687724…`
（配置里有 `expected_initial_top1: 0.686996996` ± 0.002 的守卫，不符会立刻 `RuntimeError`）。

**若三个数有任何不符，立刻停止，回报 A，不要继续第 2 步。** 这通常意味着数据、split、
初始化头或 CLIP 权重与主节点不一致——这正是这道关卡存在的意义。

**关卡通过后**，把该实验目录的 `metrics.json` 与 `command.txt` 一并回报。

---

## 第 2 步：item M（正式任务，16 epochs）

**唯一命令：**

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix50_16ep_seed1_m.yaml
```

输出目录：`artifacts\experiments\20260917-m-t2-last4-lr175e6-cwh-mw10-magmix50-16ep-seed1`
预计 **240–290 分钟**（16 min/epoch × 16，按你机器的实测吞吐估；请以第一、二个 epoch 的
实际耗时校正后告诉我，我会更新预算）。

### 中断与续跑条款

若运行被中断（休眠、断电、手动停止），**不要重跑上面的命令**（会撞 `FileExistsError`），改用：

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix50_16ep_seed1_m.yaml --resume artifacts\experiments\20260917-m-t2-last4-lr175e6-cwh-mw10-magmix50-16ep-seed1\last.pt
```

续跑只从 epoch 边界恢复，被中断的那个 epoch 会整段重算，这是正确的。
**已知副作用（可接受）**：续跑会使 `metrics.json` 的 `initial_validation` 变成 `null`，无需为此重跑。

### 自查用的预期数字

- epoch 1 应为 **`0.695432961`**，epoch 2 应为 **`0.703238606`**（与关卡一致）
- epoch 3 起 **不再需要**与任何数字比对——你的 16-epoch 轨迹会与 12-epoch 的历史轨迹分叉，
  这是**预期**的（epoch 预算不同，余弦学习率调度随之改变）
- 但 epoch 1、2 必须仍然精确命中；**若不符，停下回报**

---

## 边界（不许做）

- **不访问测试集**：不读测试图片、不推理、不生成提交包、不做任何预测
- **任何参数都不许改**：epochs、seed、lr、`min_weight`、`magnitude_mix`、batch、workers 全部按配置
- 中途不 `git pull`、不切 commit、不装依赖
- 第 1 步关卡未通过**不得**开始第 2 步
- 跑完即停，**不得自行追加实验**（包括不要让 16-epoch 跑成全量口径），下一单由 A 下发

---

## 完成后回传什么

**第 1 步（关卡）与第 2 步（M）分别回报：**

1. 五个文件的 SHA-256：`config.yaml`、`command.txt`、`metrics.json`、`best.pt`、`last.pt`
   （`SHA256SUMS.txt` 用 `<sha256>  <size>  <path>` 三段格式）
2. 训练日志
3. 三个数字：
   - `best_top1`
   - `history[*].validation.all.top1` **全序列**（关卡 3 个，M 有 16 个）
   - `history[*].validation.cross_label_exact_conflicts.top1` **全序列**（冲突子集，**必须报**）
4. 前置核对结果：HEAD、工作区状态、五个输入哈希、自检是否通过

**打包体积**：`last.pt` 约 436 MB，`best.pt` 约 108 MB。**两个都要**（A 需要按哈希验签），
但若传输困难，可先只传 `best.pt` 与 `metrics.json`，`last.pt` 后补。

---

## 关于 `config_sha256` 的已知差异（不要慌张）

你的 `metrics.json` 里 `config_sha256` **可能**与 A 的不同。A 与 B 之间已经出现过这种情况：
两者配置内容**逐字节相同**，但该字段不同——因为它是对**文件字节**取哈希，
而不同平台的 Git 检出会用 CRLF 或 LF、并可能带 BOM。

**判定等价性的正确依据是"配置文件内容比较 + 指标逐位比较"，不是 `config_sha256`。**
所以：回报时**请把 `config.yaml` 原样包含在内**，A 会做内容比对。
若你的 `config_sha256` 与 A 不同，**不要试图修文件、不要转换行尾**，照原样回报即可。
