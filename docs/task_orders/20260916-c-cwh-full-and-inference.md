# 任务单：20260916 C 候选三种子门禁通过后的全量复训与推理

## 依据

- 三种子预注册门禁：`docs/55_第一批seed1结果与多种子确认预注册.md` 第 5 节
- 上游交接 SOP：`docs/52_新AI交接_当前最佳58.8577与后续追分方案.md` 第 9、11 节
- 第一批结果：`docs/55` 第 1、3 节

## 一、三种子门禁判定（C 候选）

候选定义：T2 解冻末 4 个完整视觉块、visual lr `1.75e-6`、classwise history 软权重
（`warmup_epochs: 1`、`momentum: 0.8`、`history_mix: 0.5`、`min_weight: 0.2`）、
冻结分类头、batch 64、seed `20260910`、6 epochs。

| seed | Top-1 | macro | 冲突子集 |
| ---: | ---: | ---: | ---: |
| 20260910 | 0.711286723613739 | 0.711163759231567 | 0.310139179229736 |
| 20260911 | 0.710559487342834 | 0.710445106029510 | 0.313452631235123 |
| 20260912 | 0.711238265037537 | 0.711119711399078 | 0.314778000116348 |
| **均值** | **0.711028158664703** | **0.710909525553385** | **0.312789936860402** |

对照 lr1.5e-6 三种子基线：Top-1 均值 `0.705468833`、macro 均值 `0.705364744`、
冲突子集均值 `0.308813790`。

| # | 门禁 | 阈值 | 实际 | 判定 |
| ---: | --- | ---: | ---: | --- |
| 1 | 三 seed Top-1 均高于各自基线 | 逐 seed | +0.005769 / +0.005236 / +0.005672 | 通过 |
| 2 | Top-1 均值 | ≥ 0.705968833 | 0.711028159 | 通过（+0.0051） |
| 3 | macro 均值 | ≥ 0.705364744 | 0.710909526 | 通过（+0.0055） |
| 4 | 冲突子集均值 | ≥ 0.308801864 | 0.312789937 | 通过（+0.0040） |
| 5 | 三项训练完整、history 连续 | 6/6 | 6/6/6 | 通过 |
| 6 | 无 NaN/Inf | — | 指标均有限 | 通过 |
| 7 | checkpoint 可加载 | — | 三项均可加载 | 通过 |
| 8 | 配置/输入/commit 哈希一致 | — | seed1 见下方说明 | 见说明 |
| 9 | 未访问 test | — | `scope` 声明且 command 无 test | 通过 |

### 第 8 条的处置说明

seed2/seed3 的训练 commit 为 `b408aa7639def5932c14f0f2435521360eaa4dde`；
seed1 的原始运行在第一批 commit `4de83671f904b93a8a28c44a3a1d7d69089afb13`。
两份配置除 `run_id`、`output_dir`、`training.seed` 外完全一致，
由 `tests/test_t2_lr_balance_multiseed_plan.py` 锁定。

为**严格满足**门禁字面要求，追加一次同 commit 重跑：

- 配置：`p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_seed1_v2_c.yaml`
- 输出：`artifacts/experiments/20260916-c-t2-last4-lr175e6-cwh-seed1-v2`
- 该配置与 seed1 仅 `run_id`、`output_dir` 不同，seed 同为 `20260910`，
  由 `tests/test_t2_lr175e6_cwh_full_plan.py::test_seed1_v2_reruns_the_same_candidate_config_under_a_new_identity` 锁定。
- 原 seed1 目录**不删除、不覆盖**，保留作审计与交叉核验。

**v2 结果只用于补齐 commit 一致性，不用于挑选更有利的数字。**
若 v2 与 seed1 出现实质性差异（Top-1 差超过 0.002），停止全量并回报复核。

## 二、B 候选的处理

B 候选（lr1.65e-6 纯 CE）亦在完成三种子，但按预注册规则
"若两个候选都通过，按 Top-1 均值最高者进入全量"，
C 候选均值 `0.711028159` 显著高于 B 候选（seed1 `0.706244528`、seed2 `0.705614269`），
因此**全量对象唯一确定为 C 候选**，B 候选只做台账记录，不进入全量、不推理、不提交。

## 三、唯一全量配置

```text
configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_full_seed1_c.yaml
```

- `training_scope: full_train_fixed_epochs`、`expected_train_count: 103218`
- 初始化器：`artifacts/experiments/20260912-b-final-h1-linear-full-e100-seed1/final.pt`
  （见过全部 103,218 张 train 的 H1 全量头）
- 保持候选结构：T2 last4、visual lr `1.75e-6`、classwise history 同参数、冻结头、
  6 epochs、batch 64、seed `20260910`
- 输出：`artifacts/experiments/20260916-c-final-h1init-t2-last4-lr175e6-cwh-full-6ep-seed1`
- 全量训练**不使用验证集选模**，推理取最后 epoch 的 **`final.pt`**，不是 `best.pt`

## 四、唯一命令

```powershell
# 1) 补齐 seed1 的 commit 一致性（约 30 分钟）
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_seed1_v2_c.yaml

# 2) 全量复训（约 34 分钟）
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_full_seed1_c.yaml
```

两项**串行**，不得并发。完成后由 A 执行唯一一次 test 推理。

## 五、最终推理与打包

```powershell
.\.venv\Scripts\python.exe scripts\infer_and_build_submission.py `
  --experiment-config configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_full_seed1_c.yaml `
  --checkpoint artifacts\experiments\20260916-c-final-h1init-t2-last4-lr175e6-cwh-full-6ep-seed1\final.pt `
  --test-manifest artifacts\audits\stage1\test\samples.csv `
  --output-dir submissions\20260916_h1init_t2_last4_lr175e6_cwh_full_final `
  --batch-size 128 --workers 4
```

提交校验必须包括：CSV 恰好 24,967 行、无表头、文件名顺序与 test manifest 的
`relative_path` 完全一致、标签均为四位数字、无重复/缺失/额外文件、
ZIP 内只有 `pred_results.csv`；保存 CSV/ZIP/checkpoint/配置/commit 的 SHA-256 与生成时间。

## 六、线上分数口径

只有严格高于 `58.8577` 才能替换当前最佳。若未超过，保留 58.8577 为唯一最佳，
**不得降低门禁或事后修改门槛强行提交**。

## 七、停止线

NaN/Inf、CUDA OOM、test 被训练流程访问、样本数/参数范围不符、checkpoint 无法保存、
HEAD/配置/输入哈希不一致、同一 GPU 出现第二个训练进程、v2 与 seed1 差异超过 0.002
→ 立即停止并保留现场。
