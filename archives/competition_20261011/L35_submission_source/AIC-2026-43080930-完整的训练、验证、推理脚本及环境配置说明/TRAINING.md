# AIC-2026-43080930 完整训练与验证复现

本文件说明方案的两阶段训练流程与训练派生验证的复现方法。命令均在作品根 `AIC-2026-43080930-两股王霸之气-复赛-JRZ1018/` 执行，路径均从作品根起算。

训练只使用赛事官方 `DataSetStage2/train/四位类别ID/*.jpg`，共 148,695 张、750 类。训练配置把 test 指向不可访问目录，不提供测试集训练入口。官方图片需从赛事授权入口取得后放入作品根 `DataSetStage2/`。

## 一、方案结构

1. **分类头初始化。** 用官方 CLIP 对全部官方训练图提取固定 336 特征，训练 512→750 线性分类头 100 轮，取固定第 100 轮状态。产物为 `AIC-2026-43080930-可执行模型文件或容器化复现环境/h1_full100.pt`。
2. **联合鲁棒微调。** 视觉部分从官方 CLIP 权重初始化，分类头从上述初始化头出发，联合更新完整视觉编码器与分类头 20 轮，取固定末轮导出。产物为 `final.pt` 与完整恢复状态 `last.pt`。

视觉输入为 336×336，位置编码按双三次插值由 7×7 网格改为 10×10。输入增强为随机裁剪（最短边比例下限 0.8）与 0.5 概率水平翻转。损失为对称交叉熵 SCE，并叠加仅由训练样本历史预测生成的置信度软权重。

## 二、快速重训最终模型

```bash
python 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/train.py' full --data-root 'DataSetStage2' --work-dir './runtime_outputs/full' --device cuda --workers 8
```

输出为 `runtime_outputs/full/checkpoints/last.pt`、`final.pt` 和 `metrics.json`。固定第 20 轮导出单个学生模型；完整状态还含优化器、余弦日程、AMP scaler、逐样本置信度权重与随机状态。

固定超参数：

| 项目 | 值 |
|---|---|
| 损失 | SCE `α=0.1`、`β=1.0`、`log_zero=−4.0`，启用历史置信度加权 |
| 视觉 / 分类头学习率 | `3.5e-6` / `5e-6` |
| 优化器与日程 | AdamW（weight decay 0.01）、余弦退火 `T_max=20`、梯度裁剪 1.0 |
| batch / 轮数 | 64 / 20，固定末轮导出 |
| 图像 | 336×336 |
| seed | `20261010` |
| 数据范围 | 全部 148,695 张官方训练图，验证 0 张 |

`data/metadata/full_split.csv` 保留历史上的 118,958/29,737 划分用于来源识别；全量训练由 `full_train_fixed_epochs` 显式合并两分区，实际优化全部 148,695 张。

## 三、从固定特征重建分类头

由官方权重提取全部官方训练图的固定 336 特征，再训练 100 轮分类头。特征提取包含位置编码 7×7 到 10×10 双三次插值、resize336/中心裁剪336、L2 归一化；缓存为 FP16，训练时读取为 FP32。

```bash
python 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/train.py' features --data-root 'DataSetStage2' --work-dir './runtime_outputs/features_full' --split full --device cuda --workers 8
python 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/train.py' h1 --cache './runtime_outputs/features_full/cache' --work-dir './runtime_outputs/h1_full' --scope full --device cuda
python 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/train.py' full --data-root 'DataSetStage2' --work-dir './runtime_outputs/retrained' --head-checkpoint './runtime_outputs/h1_full/last.pt' --device cuda --workers 8
```

分类头训练只优化标准 512→750 线性层：Adam、学习率 0.01、batch 4096、weight decay 0、常数日程、seed `20261008`，不启用 AMP，取固定第 100 轮状态。`--scope full` 合并历史 train/validation 分区作为全部官方训练数据。

## 四、训练派生持出验证

最终全量模型使用了全部官方训练图，`scripts/validate.py` 会拒绝用它在训练派生持出集上报告结果。验证需重新训练对应划分的分类头与候选模型。`first` 与 `second` 是两份独立固定划分，各含 118,958 张训练样本与 29,737 张持出样本。

第二划分示例：

```bash
python 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/train.py' features --data-root 'DataSetStage2' --work-dir './runtime_outputs/features_second' --split second --device cuda --workers 8
python 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/train.py' h1 --cache './runtime_outputs/features_second/cache' --work-dir './runtime_outputs/h1_second' --scope heldout --device cuda
python 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/train.py' heldout --data-root 'DataSetStage2' --work-dir './runtime_outputs/second' --split second --head-checkpoint './runtime_outputs/h1_second/last.pt' --device cuda --workers 8
python 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/validate.py' --checkpoint './runtime_outputs/second/checkpoints/final.pt' --data-root 'DataSetStage2' --split second --output './runtime_outputs/second_validation' --device cuda --workers 4
```

第一划分使用 `--split first` 与独立新目录。分类头只由该划分的 118,958 个训练 ID 拟合；持出 ID 不参与优化，仅在固定 100 轮后计算初始化一致性。视觉联合训练 first seed 为 `20261008`，second seed 为 `20261010`，均固定 20 轮，不选择 best。

验证按冻结自然查询 CSV 的 sample_id 原顺序重排持出预测，再使用固定 query_indices：natural 为 29,737×750；balanced 为 738 类各 20 张，共 14,760 个不重复查询。12 个支持不足的类别保持披露，不做复制回填。每个查询单独建固定图，先选择对应列，再映回全部 750 类。查询元数据与基线对照预测在 `data/validation/first/` 与 `data/validation/second/`。

固定四视图只做单 checkpoint 的四次 forward，同次 hook 提取特征，不产生额外 forward；FP32 平均 logits 后按原诊断存为 FP16，再转 FP32 解码。后处理固定 `k=10`、`λ=0.3`、3 轮、温度 1、`ln(4)` 原始 margin 保护与类别均衡（0.5/0.75/60），不扫描参数。报告 raw argmax、raw 类别均衡和完整组合三种结果，并与同查询顺序的基线对照预测配对比较；缺少对照缓存或身份绑定时入口退出。统计包含 Top-1、逐类、head/mid/tail、新增零召回、重复组及 500 次分组 bootstrap 置信区间。

训练派生持出验证相对基线对照方案的净新增正确数：探索划分（`first`）均衡 +98、自然 +238；独立确认划分（`second`）均衡 +152、自然 +292。

## 五、中断恢复

保留原工作目录、数据根、设备及 workers，指定该次运行自己的完整状态文件。配置与来源绑定变化会被拒绝续训。最终参数导出只用于推理，优化器恢复须使用该次训练的完整状态。

```bash
python 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/train.py' full --data-root 'DataSetStage2' --work-dir './runtime_outputs/full' --device cuda --workers 8 --resume './runtime_outputs/full/checkpoints/last.pt'
python 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/train.py' h1 --cache './runtime_outputs/features_full/cache' --work-dir './runtime_outputs/h1_full' --scope full --device cuda --resume './runtime_outputs/h1_full/last.pt'
python 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/train.py' features --data-root 'DataSetStage2' --work-dir './runtime_outputs/features_full' --split full --device cuda --workers 8 --resume
```

恢复沿原日程到固定终点，不额外增加轮数。完整状态包含模型、AdamW、余弦日程、AMP scaler、逐样本置信度权重、epoch 与随机状态。

保存前脚本检查至少 3 GiB 剩余空间并预留额外 10 GiB 工作空间。完整特征缓存约 150 MB，训练完整状态约 1.07 GB，最终导出约 353 MB。CPU 可做依赖与合成检查，全量训练使用 CUDA 环境。

## 六、运行自检

```bash
python 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/train.py' check
python 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/train.py' full --data-root 'DataSetStage2' --work-dir './runtime_outputs/preflight' --device cpu --workers 0 --prepare-only
```

`check` 校验训练实现的关键科学函数与依赖（SCE、置信度加权损失、合成梯度、优化器与调度器状态）；`--prepare-only` 核对路径、配置、来源与分类头元数据，不打开比赛图片、不执行训练。
