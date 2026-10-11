# AIC-2026-43080930 复赛作品

- 赛题：面向噪声标签数据的细粒度图像识别鲁棒微调（第八届 AIC 算法挑战赛道）
- 团队：两股王霸之气
- 参赛团队编号：`AIC-2026-43080930`
- 提交人：JRZ1018
- 复赛线上最佳 Top-1：**69.6773848947762**

骨干网络为 OpenAI 官方 CLIP ViT-B/32，方案为单一模型、单一推理流程。本方案在本地实验记录中的代号是 `L35_GRAPH`，命令与配置里出现的 `L35` 均指该方案。

## 目录结构

作品根为 `AIC-2026-43080930-两股王霸之气-复赛-JRZ1018/`。下文路径除注明外均从作品根起算，所有命令也在作品根执行。

```text
AIC-2026-43080930-两股王霸之气-复赛-JRZ1018/
├─ AIC-2026-43080930-可复现的代码与运行说明文档/
│  ├─ src/、vendor/、archived/、licenses/
│  ├─ README.md、运行说明.md、运行说明.pdf
│  ├─ THIRD_PARTY_NOTICES.md
│  └─ 提交结果/
│     ├─ pred_results.csv、pred_results.zip
│     ├─ submission_manifest.json
│     └─ README.md、AIC-2026-43080930-结果说明.pdf
├─ AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/
│  ├─ scripts/、portable_training/、configs/
│  ├─ data/、environment/、evidence/、reproduction_cache/
│  ├─ requirements-cpu.txt、requirements-cuda.txt
│  └─ README.md、TRAINING.md、环境与来源说明.md/.pdf、RELEASE_SHA256.json
└─ AIC-2026-43080930-可执行模型文件或容器化复现环境/
   ├─ final.pt、last.pt、openai_clip_ViT-B-32.pt、h1_full100.pt
   ├─ container-image.tar、container-image.tar.sha256
   └─ README.md、CONTAINER.md、Dockerfile、Dockerfile.cuda
```

## 阅读与执行顺序

1. 本文件说明作品构成与结果；`AIC-2026-43080930-可复现的代码与运行说明文档/提交结果/README.md` 说明赛题结果文档本身。
2. 按 `AIC-2026-43080930-可执行模型文件或容器化复现环境/CONTAINER.md` 加载容器镜像，或按 `AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/环境与来源说明.md` 安装 Python 环境。
3. 按 `AIC-2026-43080930-可复现的代码与运行说明文档/运行说明.md` 执行文件验签、预测复现与完整图片推理。
4. 按 `AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/TRAINING.md` 执行特征提取、分类头初始化、完整训练与持出验证。

无需官方图片即可运行的四项检查：

```bash
python 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/verify_release.py'
python 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/infer.py' --help
python 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/train.py' check
python 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/train.py' full --data-root 'DataSetStage2' --work-dir './runtime_outputs/preflight' --device cpu --workers 0 --prepare-only
```

## 方案概要

**分类头初始化。** 用官方 CLIP 对全部 148,695 张官方训练图提取固定 336 特征，训练 512→750 线性分类头 100 轮（Adam，学习率 0.01，batch 4096，常数日程），取固定第 100 轮状态。

**联合鲁棒微调。** 视觉部分从官方 CLIP 权重初始化，分类头从上述初始化头出发，联合更新完整视觉编码器与分类头 20 轮，共 154 个张量、88,233,966 个参数。损失为对称交叉熵 SCE（α=0.1），叠加由训练样本历史预测生成的置信度软权重；AdamW，余弦日程，batch 64，视觉学习率 3.5e-6、分类头 5e-6。使用全部 148,695 张官方训练图，不设验证集，固定第 20 轮导出。

**推理与后处理。** 对冻结的单个 `final.pt` 做固定四视图（原生 336、原生 336 水平翻转、短边缩放到 384 后中心裁剪 336、以及其水平翻转），四路 logits 等权平均；再做固定 k 近邻图平滑（k=10，λ=0.3，3 轮，`ln 4` 原始 margin 保护）与类别均衡调整（温度系数 0.5，偏置强度 0.75，60 轮）。后处理只作用于冻结的预测矩阵。

**训练派生验证。** 在官方训练集上构造固定划分（118,958 训练 / 29,737 持出）与均衡查询（738 类 × 20 张，共 14,760 张），与基线对照方案（代号 `J5`）做同硬件、同环境的配对比较：探索划分净新增正确数均衡 +98、自然 +238，独立确认划分均衡 +152、自然 +292。

## 数据与许可

训练只使用赛事官方训练集（148,695 张、750 类），官方测试集（37,444 张）只用于冻结模型推理。官方原始图片未随作品分发，复现方需从赛事授权入口取得后放入作品根 `DataSetStage2/`。第三方源码的版本、上游、许可证与逐文件 SHA 见 `AIC-2026-43080930-可复现的代码与运行说明文档/THIRD_PARTY_NOTICES.md`。运行输出统一写入作品根 `runtime_outputs/`。
