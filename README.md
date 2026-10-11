# AIC 算法挑战赛：噪声标签细粒度图像识别鲁棒微调

本目录为本道赛题的主工作文件夹。

## 2026-10-11 最终作品与恢复

最终作品采用 L35_GRAPH，用户报告复赛最高成绩为 **69.6773848947762**。项目源码、研究配置、台账、技术报告与冻结科学代码归档已整理。

- [从 GitHub 与 F2 恢复项目](docs/恢复与继续工作_20261011.md)
- [最终 L35 技术报告](reports/20261011_L35_final_technical_report.md)
- [复赛源码归档](archives/competition_20261011/README.md)
- [历史技术报告 Word](docs/技术报告.docx)

完整模型、容器镜像、提交 CSV/ZIP 及复现资产保存在独立的最终作品文件夹，按恢复说明与 GitHub 源码配合使用。下文各阶段记录保留其原始日期与实验语境。

- 官方赛道通知：[3629](https://www.aicomp.cn/notice/notice-1/3629.html)
- 赛题规则：[3714](https://www.aicomp.cn/tracks/tracks-1/3714.html)
- 资料提取日期：2026-09-09

## 开始工作前

1. 阅读根目录 [`AGENTS.md`](AGENTS.md)。
2. 阅读 [`docs/HANDOFF.md`](docs/HANDOFF.md)。
3. 阅读两份规则摘要与 [`docs/赛题规则.pdf`](docs/赛题规则.pdf)。
4. 按 [`docs/00_获奖导向总体工作计划.md`](docs/00_获奖导向总体工作计划.md) 的门禁顺序执行。
5. 对规则有疑问时，先登记到 [`docs/compliance_questions.md`](docs/compliance_questions.md)，未经确认采用保守解释。
6. 开始 P3 或 M1-M5 前，检查 [`docs/04_上游方案调研与复用登记.md`](docs/04_上游方案调研与复用登记.md) 和 [`docs/upstream_registry.yaml`](docs/upstream_registry.yaml)。
7. 换机开发按 [`docs/16_换机迁移与恢复指南.md`](docs/16_换机迁移与恢复指南.md) 分离 GitHub 源码和 U 盘私有状态，并逐文件校验 SHA-256。
8. 两台电脑并行开发按 [`docs/17_双人协作与训练执行交接手册.md`](docs/17_双人协作与训练执行交接手册.md) 执行：A 负责核心研发与提交，B 只执行冻结训练任务单。

## 当前状态

- 初赛官方数据已取得，保存在本地 `DataSetStage1/`；该目录整体由 Git 忽略，训练集和测试集必须严格隔离。
- 已验证 OpenAI CLIP ViT-B/32 权重及 RTX 4060 Laptop GPU CUDA 环境。
- P0/P0.5 治理以及 P1/P2 合成工程验收已完成。
- 初赛数据只读审计、固定划分、冻结特征基线、模拟提交闭环和 A2/A3 鲁棒损失对照均已完成。
- P3 全量训练特征与 A0/A1 已完成；A1 余弦头以 67.909% noisy-validation Top-1 成为后续统一基线，详见 [`docs/12_P3_A0_A1_正式基线报告.md`](docs/12_P3_A0_A1_正式基线报告.md)。
- PEFT/M0 已完成：冻结 A1 分类头、只训练最后视觉注意力 rank-4 LoRA 三轮，固定验证集 Top-1 为 68.4282%、macro 为 68.4163%；下一工作包为 M1 类内可信度，详见 [`docs/14_P3_PEFT资源与短程筛选报告.md`](docs/14_P3_PEFT资源与短程筛选报告.md)。
- 严格冻结头的 M0 v2 为 68.3894%；M1 类内历史软权重单种子最佳为 68.4427%，暂选为首次真实提交模型，详见 [`docs/15_M1首轮筛选与首次提交选模报告.md`](docs/15_M1首轮筛选与首次提交选模报告.md)。该微小增益仍需多种子确认。
- 首份真实提交包已生成在 `submissions/20260911_m1_first/`，24,967 行并通过严格本地校验；尚未上传，记录见 `submissions/ledger.csv`。
- 正式训练、全量特征提取或其他大量占用系统资源的过程，必须先报告预计高占用时间和资源，再等待用户批准。

## 治理检查

使用项目 Python 运行：

```powershell
D:\aicomp\python312\python.exe scripts\check_governance.py
```

检查器会验证上游候选的固定 commit、许可证边界、触发阶段和交付物字段，并阻止数据、权重、密钥或大文件进入 Git 候选集。

只读同步上游参考源码：

```powershell
D:\aicomp\python312\python.exe scripts\manage_upstream_references.py sync
D:\aicomp\python312\python.exe scripts\manage_upstream_references.py check
```

源码保存在 `references/upstream/` 并被主项目 Git 忽略。同步只检出注册表锁定的 commit，不初始化子模块、不下载 Git LFS 对象，也不安装或运行上游代码。

## 新电脑恢复

公开源码从 GitHub 克隆，官方数据从比赛官方渠道重新获取；权重、checkpoint、审计划分、真实提交包和可选特征缓存通过 U 盘迁移。不要复制旧 Python 环境。完整命令见 [`docs/16_换机迁移与恢复指南.md`](docs/16_换机迁移与恢复指南.md)，依赖版本锁定在 [`requirements-lock.txt`](requirements-lock.txt)。

新电脑作为第二训练节点长期协作时，不按两名自由开发者处理。A 统一负责代码、配置、实验决策和测试提交；B 使用任务单指定的 commit 和命令执行长训练、多 seed 与固定消融。权限、Git 流程、产物回传和近期排期见 [`docs/17_双人协作与训练执行交接手册.md`](docs/17_双人协作与训练执行交接手册.md)。

## P1/P2 数据与噪声夹具

官方配置模板是 `configs/data/template.yaml`。它只定义显式契约，不代表官方压缩包的真实结构。

```powershell
# 生成完全程序化的细粒度、长尾、噪声标签夹具
D:\aicomp\python312\python.exe scripts\generate_synthetic_fixture.py --config configs\data\synthetic_smoke.yaml

# 只读审计并检测坏图、精确重复和近重复
D:\aicomp\python312\python.exe scripts\audit_dataset.py --config configs\data\synthetic_smoke.yaml --source train

# 以重复簇为不可拆分单元创建验证划分
D:\aicomp\python312\python.exe scripts\build_split.py --config configs\data\synthetic_smoke.yaml --manifest artifacts\audits\p1p2_smoke\train\samples.csv --output artifacts\audits\p1p2_smoke\train\split.csv

# 运行 CE、LogitClip、ELR 和类内可信度的已知噪声行为验收
D:\aicomp\python312\python.exe scripts\run_p2_benchmark.py --config configs\experiment\p2_closeout.yaml --output artifacts\audits\p2_closeout\results.json

# 运行全部 CPU/CUDA 测试
D:\aicomp\python312\python.exe -m pytest
```

生成器拒绝覆盖非空目录。正式阶段默认对坏图和同内容异标签严格失败；合成配置显式允许记录并排除注入的坏图。合成图片、审计结果和 pytest 临时文件位于 `artifacts/`，不会进入 Git。

## P3 冻结 CLIP 基线

正式命令、续跑方式和执行前预算维护在 [`docs/11_P3_冻结特征基线实施与资源门禁.md`](docs/11_P3_冻结特征基线实施与资源门禁.md)，结果和诊断维护在 [`docs/12_P3_A0_A1_正式基线报告.md`](docs/12_P3_A0_A1_正式基线报告.md)。模型入口固定为 `configs/model/openai_clip_vit_b32.yaml`；脚本拒绝隐式下载权重，训练脚本也拒绝加载测试特征缓存。

## P3 PEFT / M0

当前 M0 复现命令：

```powershell
D:\aicomp\python312\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_peft_b3_lora_only_3ep.yaml
```

该配置只训练最后视觉注意力的 LoRA，分类头冻结；完整消融、资源与 checkpoint 哈希见 [`docs/14_P3_PEFT资源与短程筛选报告.md`](docs/14_P3_PEFT资源与短程筛选报告.md)。

## 复赛迁移：长尾压力与分层诊断

协议与门禁见 [`docs/41_复赛迁移长尾压力与分层诊断预注册.md`](docs/41_复赛迁移长尾压力与分层诊断预注册.md)，实测证据见 [`docs/42_复赛迁移长尾压力与分层诊断验收.md`](docs/42_复赛迁移长尾压力与分层诊断验收.md)。全程只使用当前阶段训练部分，不访问 `DataSetStage1/test`。

```powershell
# 1) 由固定 train 划分生成确定性长尾压力划分（只读官方数据，写 artifacts/）
D:\aicomp\python312\python.exe scripts\build_long_tail_split.py --config configs\data\long_tail_stress.yaml

# 2) 对已锁定 checkpoint 做一次只读分层诊断（head/mid/tail、最差类召回、每类保留率）
D:\aicomp\python312\python.exe scripts\diagnose_long_tail.py --experiment-config configs\experiment\p3_stage1_h1init_peft_lora_last2_r32a64_3ep_full_seed1_a_planned.yaml --checkpoint artifacts\experiments\20260914-a-final-h1init-lora-last2-r32a64-full-3ep-seed1\final.pt --long-tail-config configs\data\long_tail_stress.yaml --output-dir artifacts\reports\20260915-long-tail-diagnostics --batch-size 256 --workers 0
```

诊断结果只作工程口径与失效模式证据，不替代固定 noisy-validation 选模，也不得据其改动已锁定的训练样本、阈值或提交后处理。
