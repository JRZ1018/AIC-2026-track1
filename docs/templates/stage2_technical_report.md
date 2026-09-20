# 赛题 4 复赛技术报告

> 状态：模板。方括号内容必须由可追溯产物替换；不得预填未发生的结果。

## 1. 摘要

- 最终方案：[唯一配置路径]
- 最终 checkpoint：[路径与 SHA-256]
- 本地固定验证 Top-1：[值；来源 metrics 字段]
- 线上 Top-1：[值；来源平台记录]
- 单模型/单次推理声明：[复核人]

## 2. 任务、规则与合规边界

- 数据阶段与官方来源：[公告/下载记录]
- 骨干与官方权重：[OpenAI CLIP ViT-B/32 SHA-256]
- 数据使用、test 隔离、禁止融合：[证据路径]
- 未决规则及保守处置：[问题编号/答复]

## 3. 数据审计与切分

- 训练/测试/类别数量：[audit JSON 字段]
- 类别映射来源与 SHA-256：[mapping manifest]
- 坏图、重复簇、跨标签冲突：[audit report]
- split 算法、seed、重复簇零跨侧证明：[split summary]

## 4. 长尾与噪声诊断

- B09 L0/L1/L2 结论与触发项：[long-tail readiness JSON]
- min/p1/p10/median/p90/max、Gini、effective number：[字段]
- 冲突/非冲突支持和指标：[字段]
- 每类 loss、平均权重、sum(weight)、降权率、采样次数、recall：[字段]

## 5. 方法

- H1 冻结特征基线：[唯一配置]
- 部分解冻基线：[唯一配置]
- 噪声鲁棒机制：[唯一变量与公式]
- 条件式长尾机制：[若未触发，明确写“未启用”]
- checkpoint 选择规则：[固定 epoch 或固定验证规则]

## 6. 实验矩阵与消融

| run_id | 唯一变化 | seed | Top-1 | macro | head | mid | tail | conflict | duration | peak VRAM | decision | 来源 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| [填] | [填] | [填] | [填] | [填] | [填] | [填] | [填] | [填] | [填] | [填] | [填] | `metrics.json` |

## 7. 最终训练与推理

- 锁模时间、commit 与工作区状态：[台账]
- 环境哈希与依赖锁：[environment identity]
- 精确训练命令：[command.txt]
- 精确推理命令：[submission ledger]
- CSV/ZIP/manifest 哈希：[submission manifest]

## 8. 资源与复现

- 硬件、系统、Python/PyTorch/CUDA/cuDNN：[environment JSON]
- 数据、审计、特征、训练、推理耗时：[结构化字段]
- 显存、磁盘峰值与恢复点：[metrics/capacity report]
- 从官方原始权重复现步骤：[命令清单]

## 9. 失败实验、限制与风险

- 未晋级实验及停止理由：[index/任务单]
- 固定验证分辨率、种子方差与线上限制：[事实]
- 长尾/噪声方法适用边界：[事实]
- 未解决但采取保守策略的规则：[问题清单]

## 10. 结论

[只总结有证据支持的结论，不把压力夹具或初赛结果当作复赛事实。]

## 附录 A：复现身份表

| 项 | 值 | 来源 |
|---|---|---|
| data SHA-256 | [填] | [填] |
| split SHA-256 | [填] | [填] |
| mapping SHA-256 | [填] | [填] |
| config SHA-256 | [填] | [填] |
| environment SHA-256 | [填] | [填] |
| CLIP weight SHA-256 | [填] | [填] |
| checkpoint SHA-256 | [填] | [填] |
| Git commit | [填] | [填] |
| CSV/ZIP SHA-256 | [填] | [填] |
