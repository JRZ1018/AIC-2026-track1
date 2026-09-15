# 任务单：20260917 B/C T2 last4 lr1.5e-6 三种子确认

## 目标

对短筛选中最优的 T2 last4 / visual lr `1.5e-6` 做 seed2、seed3 固定验证确认。只使用初赛 train 的固定训练/验证划分，不访问 test、不生成提交、不做全量训练。

| 执行者 | 配置 | seed | 输出目录 |
| --- | --- | ---: | --- |
| B | `p3_stage1_h1init_t2_last4_lr15e6_6ep_seed2_b.yaml` | 20260911 | `artifacts/experiments/20260917-b-t2-last4-lr15e6-6ep-seed2` |
| C | `p3_stage1_h1init_t2_last4_lr15e6_6ep_seed3_c.yaml` | 20260912 | `artifacts/experiments/20260917-c-t2-last4-lr15e6-6ep-seed3` |

## 冻结版本与资源

- 训练 commit：`56437f44d556976d904baf701eeab00038bfe532`
- 单卡单进程，6 epochs，batch 64；每项预计 25–35 分钟，峰值显存约 1.2 GiB，新增磁盘约 0.4 GiB。
- B/C 可在不同机器并行；同一机器不得并发 GPU 训练。
- 训练中不得切换 commit、修改配置、改变 seed/lr/epoch/batch 或删除现场。

## 开始前门禁

```powershell
Set-Location D:\aicomp\AICtrack1
git fetch origin
git switch --detach 56437f44d556976d904baf701eeab00038bfe532
git rev-parse HEAD
git status --short
.\.venv\Scripts\python.exe -m pytest tests\test_t2_lr15e6_multiseed_plan.py tests\test_partial_unfreeze_plan.py tests\test_peft.py tests\test_peft_checkpoint.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe scripts\check_governance.py
```

## 唯一命令

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\<对应配置文件>
```

异常时保留现场；只有同一输出目录存在可加载 `last.pt` 时，才允许用原配置加 `--resume` 恢复。完成后回传整个实验目录、日志及 `config.yaml`、`command.txt`、`metrics.json`、`best.pt`、`last.pt` 的 SHA-256 清单。A 将统一验签并按预注册门禁决定是否启用全量配置。

## 预注册确认门禁

当前三种子基线是已验收的 T2 last4 / lr1e-6 / 8 epochs，均值 Top-1 `0.704450687`、macro `0.704344849`、冲突子集 `0.310801864`。候选通过条件：

1. 三种子平均 Top-1 高于 `0.704950687`（至少提升 0.0005）；
2. 三种子平均 macro 不低于 `0.704344849`；
3. 三种子平均冲突子集 Top-1 不低于 `0.308801864`（基线均值减 0.002）；
4. 每项训练完整、指标有限、配置/输入/checkpoint 哈希一致且声明未访问 test。

只有四项全部通过，A 才能启用 `p3_stage1_h1init_t2_last4_lr15e6_6ep_full_seed1_a_planned.yaml` 的等价全量配置，完成 103,218 张 6 epochs 复训并最终推理。
