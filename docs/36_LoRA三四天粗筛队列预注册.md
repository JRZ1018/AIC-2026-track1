# LoRA 三四天粗筛队列预注册

日期：2026-09-13

## 目标与边界

用户明确要求不停止 LoRA rank/alpha 搜索，并允许在未来 3–4 天继续寻找增益。本轮先预注册 6 个 seed1 粗筛点，只研究三个轴：rank、`alpha/rank` 比例、末层块数。所有配置复用 H1、固定 manifest/split、seed、3 epochs、学习率、batch size、冻结分类头和 OpenAI CLIP ViT-B/32；不访问 test、不生成提交。

用户已明确批准本轮资源预算，六个配置统一切换为 `execution_enabled: true`；启用后重新执行差异测试、最大组合 CUDA smoke 和治理检查，再冻结 commit 与 B 任务单。

## 六个候选与顺序

统一参考为末两层 rank-8/alpha-16 seed1：Top-1 `0.692184627`、macro `0.692077219`、冲突子集 Top-1 `0.306825727`。

| 顺序 | 候选 | 单一问题 | 可训练视觉参数 |
|---:|---|---|---:|
| 1 | last2 rank8 alpha8 | 把 `alpha/rank` 从 2 降到 1 是否更稳 | 73,728 |
| 2 | last2 rank8 alpha32 | 把 `alpha/rank` 从 2 升到 4 是否增强适配 | 73,728 |
| 3 | last2 rank16 alpha32 | 固定比例 2，提高 rank | 147,456 |
| 4 | last2 rank32 alpha64 | 固定比例 2，继续提高 rank | 294,912 |
| 5 | last3 rank8 alpha16 | 固定 rank/比例，扩大到末三层 | 110,592 |
| 6 | last4 rank8 alpha16 | 固定 rank/比例，扩大到末四层 | 147,456 |

这不是完整笛卡尔积。六点用于分别观察尺度、强度和深度趋势；在看到结果前不交叉组合最优值。

## 统一选模门禁

- 六项均完成 seed1 后按固定验证 Top-1 排序；macro 与冲突子集作为保护和解释指标。
- 只有 Top-1 至少达到 `0.692684627`（比 rank-8 参考高 0.05 点），且 macro 不低于 `0.691577219`、冲突子集不低于 `0.305825727` 的候选才有资格进入复验。
- 最多两个候选进入 seed2/seed3；若无候选达标，保留现有 rank-8，不用线上分数反向扩张本轮网格。
- 本轮不直接生成全量 checkpoint 或 pred_results；多种子结果确认后最多选择一个候选进入全量复训。

## 回传结果与晋级决定（2026-09-14）

B 的六项回传包 SHA-256 为 `a92b14f0b63829cfbf005bdab68268250568e0e4425c86a252dec0b745adbf59`。六项 commit、配置、输入、样本数、参数范围、metrics 与 best/last 哈希均通过验签；两项用户主动暂停后均从 epoch 边界的 `last.pt` 原生续跑，最终 history 完整，无 test 读取、NaN、Inf 或 OOM。

相对 last2 rank8/alpha16 seed1，rank32/alpha64 的 Top-1 提高 0.3830 点、macro 提高 0.3816 点、冲突子集提高 0.2651 点；last4 rank8/alpha16 的整体 Top-1 最高，提高 0.3976 点，但冲突子集下降 0.4639 点。按预注册三项门槛，rank16/alpha32 与 rank32/alpha64 合格，last3/last4 因冲突保护线失败不晋级。

rank32/alpha64 在整体与冲突指标上均支配 rank16/alpha32，因此只晋级 rank32/alpha64 到 seed2/seed3。多种子未完成前，不执行全量复训或 test 推理。

## 资源预估

- 六项单卡单进程串行，每项预计 24–32 分钟；训练总计约 2.5–3.2 小时，连同检查与打包按 3–3.7 小时申报。
- 峰值显存预计低于 0.9 GiB；六项磁盘新增预计 30–90 MiB。
- 每 epoch 保存 `last.pt`，支持训练器原生续跑；不得并发，不得因结果自行追加 seed 或配置。
- 任何 NaN/Inf、OOM、test 读取、82,592/20,626 样本数不符、参数量不符或资源明显越界，停止队列并保留现场。

## 启用前预检结果

六份配置启用后已重新执行差异测试、完整测试套件与治理检查：

- `tests/test_b_next_queue_plan.py`、`tests/test_h1init_peft_plan.py` 共 26 项通过；完整测试套件 164 项通过；`scripts/check_governance.py` 为 `PASSED`。
- 参考配置 `last2 rank8 alpha16 seed1` 保持 `execution_enabled: true` 不变（它就是 B 已执行并已验收的晋升运行），因此不再构成字段级差异；差异集断言相应改为 `{run_id, output_dir, peft}`，并显式断言参考与候选均为启用态。

按上轮惯例对容量上界两个点做受限 CUDA smoke（128 样本 / 4 steps / batch 32，仅训练集与验证集，未访问 test）：

| smoke | 可训练视觉参数 | 峰值显存 | 吞吐 | trainable_state.pt SHA-256 |
|---|---:|---:|---:|---|
| last2 r32 α64 | 294,912 | 505.24 MiB | 83.67 samples/s | `69d3569cecdb1ad9d3fa8658fcd65254db50ad3f346f201d7ac1fbd1a07d4140` |
| last4 r8 α16 | 147,456 | 619.04 MiB | 93.02 samples/s | `d709a08d4e22de374c7d8964ed9f6e1c24ea6586aabe3ed5ab8ec64e52bbbf91` |

两项分别在启用态下完成，可从各自 `result.json` 的 `command` 字段原样重跑；OpenAI CLIP 权重、train manifest、split 与数据配置哈希均与任务单要求一致。实测峰值显存 619.04 MiB 低于申报上限 0.9 GiB。`trainable_visual_parameters` 与预注册表逐项相符，参数名范围分别为 resblocks.10–11 与 resblocks.8–11 的注意力 LoRA。
