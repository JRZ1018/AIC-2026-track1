# AICtrack1 交接文档

更新时间：2026-09-14（Asia/Shanghai）

## 1. 给新对话的首要指令

工作目录固定为 `D:\aicomp\AICtrack1`。开始任何实现前，必须依次阅读：

1. `docs/01_算法挑战赛道通知.md`
2. `docs/02_噪声标签细粒度图像识别鲁棒微调_赛题规则.md`
3. `docs/赛题规则.pdf`
4. `AGENTS.md`
5. `docs/00_获奖导向总体工作计划.md`
6. 换机时阅读 `docs/16_换机迁移与恢复指南.md`
7. 双人并行时阅读 `docs/17_双人协作与训练执行交接手册.md`

项目目标：参加第八届全球校园人工智能算法精英大赛算法挑战赛道的“面向噪声标签数据的细粒度图像识别鲁棒微调”，做出合规、单模型、可复现、可提交的方案。

2026-09-14 组委会《关于举办算法挑战赛道复赛的通知》及日程附件已经确认：初赛报名于 2026-09-20 18:00 截止，初赛结果文档于当日 20:00 截止；复赛数据于 2026-09-21 11:00 公布。赛题 4 的复赛结果文档提交窗口为 2026-09-22 09:00 至 2026-10-11 20:00，复赛作品文档于 2026-10-11 23:59 截止。当前仍处于初赛官方数据阶段。

## 2. 绝对合规边界

- 核心骨干只能是 **OpenAI 官方 CLIP ViT-B/32**，不得换成更大模型、OpenCLIP 或 LAION 权重。
- 最终只能是单模型、单一推理流程；不得集成、投票或融合。
- 只能使用当前比赛阶段由官方提供的数据；不得使用额外公开/私有数据。
- 测试数据只能推理，不能用于训练、自监督、伪标签、调参、BatchNorm 更新或后处理。
- 不得寻找、下载、使用或传播他人分享的比赛数据。数据需在报名缴费后由队长在“我的参赛信息 → 详情”领取。
- 任何自动数据筛样都必须写入最终代码并可复现；人工清洗不能成为最终训练前置步骤。

规则不明确时，在 `docs/compliance_questions.md` 记录问题并建议用户向组委会确认；不擅自采用宽松解释。

## 3. 当前状态

### 已完成

- 已选定赛题并整理官方通知、赛题规则 Markdown。
- 已将项目级工程与合规规范迁移至根目录：`AGENTS.md`。
- 已初始化 Git 主分支并建立防泄漏 `.gitignore`；初赛审计报告提交为 `f02fd87`，旧 `data/` 占位移除提交为 `aea9535`，P3 冻结 CLIP 基线工程提交为 `d34c650`，float32 分类诊断提交为 `cab1fc0`，A0/A1 正式结果文档提交为 `29117d9`。
- 已建立最小目录骨架、实验索引和提交台账。
- 已将旧无数据阶段计划重构并改名为 `docs/00_获奖导向总体工作计划.md`，按初赛真实数据和 A0/A1 结果重排当前关键路径。
- 已完成 P0/P0.5 治理落地：上游方案固定版本登记、许可证边界、强制触发阶段、最小复用规格和 Git 防泄漏检查器均已建立。
- 已将 11 个上游参考仓库按锁定 SHA 同步到 `references/upstream/`；源码由主项目 Git 忽略，使用 `scripts/manage_upstream_references.py check` 可离线核验。
- 已完成 P1 数据契约、变换隔离、严格只读审计、重复簇隔离划分，以及 P2 程序化噪声、类内可信度和 CE/ELR/LogitClip 行为基准；其工程验收已完成，正式 CLIP A0/A1 结果另见 P3 报告。
- 初赛官方数据已经取得并完成全量只读审计：训练 103,218 张/500 类、测试 24,967 张，全部可解码；固定划分为 82,592/20,626，报告见 `docs/10_初赛数据审计报告.md`。
- 已安装 Python 训练环境，且真实 CUDA 运算验证通过。
- 已安装本地提供的 OpenAI CLIP 源码。
- 已下载 OpenAI 官方 ViT-B/32 权重：`artifacts/weights/ViT-B-32.pt`（约 337.6 MiB）。
- 已修复环境检查脚本中的 PowerShell/Python 引号问题：`scripts/check_environment.ps1`。
- P3 全量训练特征与 A0/A1 正式实验已完成；A1 余弦头的 float32 noisy-validation Top-1 为 67.909%、macro accuracy 为 67.894%，选择为后续统一基线。正式报告见 `docs/12_P3_A0_A1_正式基线报告.md`。
- S1 模拟提交生成、严格校验、确定性 ZIP 和 manifest 已完成；A2 GCE、A3 LogitClip 均未超过 A1，已经停止，见 `docs/13_S1与A2_A3验收报告.md`。
- PEFT 参数白名单、LoRA 折叠、可恢复 checkpoint、确定性训练增强、资源冒烟和正式筛选已完成。当前唯一 M0 为“冻结 A1 分类头，只训练最后视觉注意力 rank-4 LoRA 三轮”，noisy-validation Top-1 68.4282%、macro 68.4163%，见 `docs/14_P3_PEFT资源与短程筛选报告.md`。
- 已修正零学习率分类头仍参与梯度裁剪的问题并补跑严格 M0 v2（68.3894%）；M1 类内历史软权重单种子最佳为 68.4427%，暂用于首次真实提交，见 `docs/15_M1首轮筛选与首次提交选模报告.md`。该微小增益仍需多种子确认。
- 已在锁模后执行一次真实测试推理，生成 `submissions/20260911_m1_first/pred_results.csv` 与 ZIP；共 24,967 行，CSV SHA-256 为 `bcd3db88d044583b90628e8dfd0a7ff4ab1ec43ad00a0306e18bbf30a389c8ea`。文件已通过严格本地校验但尚未上传。
- 已补齐换机恢复链路：`requirements-lock.txt` 锁定当前环境，`scripts/setup_environment.ps1` 创建仓库内虚拟环境，`scripts/transfer_workspace.py` 逐文件校验并迁移 Git 忽略的私有状态。完整流程见 `docs/16_换机迁移与恢复指南.md`。
- 已确定双人协作模式：A 负责核心研发、Git 集成、实验决策和唯一测试提交；B 作为第二算力节点，只执行 A 冻结的训练任务单。首次同步、权限边界、Git 和产物规范、故障门禁及近期并行节奏见 `docs/17_双人协作与训练执行交接手册.md`。
- 已冻结 B 首轮四项配对复验代码 commit `6dc7ddf74df5198fe7c3d4bcb3bbc7850b8d2758`：按 M0 v2 seed2、M1 seed2、M0 v2 seed3、M1 seed3 顺序串行执行。任务总入口为 `docs/task_orders/20260911-B-first-round.md`；四项配置差异单测、治理检查、两步 CUDA smoke 和完整 105 项测试均已通过。
- 已完成队友线性头优化预算的固定划分公平复验：H1 线性 Adam 三种子 Top-1 均值 68.6916%、总体标准差 0.0114 点，显著推翻旧 A0 作为线性能力代表；同预算 H2 余弦 Adam 均值 67.6412%，说明该配方只适合线性头。证据与榜分差异边界见 `docs/18_公平冻结头复验与榜分差异分析.md`。
- 已准备 B 产物严格验签与三种子配对汇总工具；验收计划为 `configs/evaluation/b_first_round.yaml`。H1 线性头初始化 LoRA 的单变量配置已预注册但以 `execution_enabled: false` 锁住，未运行。门禁、命令和资源边界见 `docs/19_B结果验收与H1_LoRA预注册.md`。
- B 首轮四项结果已正式验签：M1 三组配对增益全部为正，但平均仅 +0.0695 点，低于 +0.10 点门槛，故停止 M1 扩张；H1 仍是更强参考。完整结论见 `docs/20_B首轮验收与H1冻结头提交.md`。
- H1 冻结线性头第二份提交线上分数为 56.8751，相对首次 M1 的 55.0006 提高 1.8745 点。B 的 H1→LoRA seed2/seed3 已验签；与 A seed1 汇总后 Top-1 均值 68.9761%、标准差 0.0241 点、macro 均值 68.9634%，完整通过预注册三种子门禁。详见 `docs/21_H1_LoRA多种子验收.md`。
- B 的最终全量 H1→LoRA attempt2 已通过 A 端验收，并在用户批准后完成一次最终测试推理。有效提交线上分数 **57.8924**，比 H1 高 1.0173 点、比队友 baseline 高 0.8451 点，晋级为当前最佳。详见 `docs/23_H1_LoRA最终全量复训验收.md`。
- rank-8 H1→LoRA 全量复训线上分数为 **58.1247**，是当前最佳；比 rank-4 的 57.9926 高 0.1321 点。
- Attention+MLP LoRA seed1 固定验证仅比匹配 rank-4 高 0.0097 点，已经停止，不运行多种子或提交。
- 已批准并冻结首个中强度候选：H1 初始化、冻结头、解冻最后一个完整视觉块、视觉学习率 `3e-6`、3 epochs。B 任务单为 `docs/task_orders/20260913-B-h1-last-block-medium-seed1.md`，训练 commit 为 `d473e858afe6a00e9590171dfa65677e801d11b3`。
- B 的末块 seed1 已验签：Top-1 69.2815%，比 rank-8 LoRA 同 seed 高 0.0630 点，但冲突子集低 0.4639 点，未过预注册门槛；不直接晋级多种子或提交。因三个 epoch 连续上升，可另行预注册更长的探索性对照。
- B 的 LoRA 六点粗筛队列已全部验签并归档。last2 rank32/alpha64 seed1 的 Top-1 为 69.6015%、macro 为 69.5893%、冲突子集 Top-1 为 30.9476%，相对匹配 rank8 seed1 分别提高 0.3830、0.3816、0.2651 点；它在合格候选中支配 rank16，因此是唯一晋级多种子确认的候选。last3/last4 虽有较高整体 Top-1，但未过预注册冲突保护线。
- rank32 seed2/seed3 已完成并严格验收：相对匹配 rank8 的三组 Top-1 增益均为正，均值 +0.4493 点，macro 均值 69.6534%、冲突子集均值 30.7930%，通过预注册门禁。全量 rank32/alpha64 配置已获本次对话预算批准并启用；全量训练、验签与唯一一次 A 端 test 推理仍须按 `docs/38_rank32多种子门禁验收与全量复训冻结.md` 顺序完成。
- rank32/alpha64 全量三轮复训已严格验签，随后由 A 执行唯一一次 test 推理。`submissions/20260914_h1init_lora_last2_r32a64_full_final/pred_results.zip` 已生成并通过本地严格格式校验，尚未上传；详见 `docs/39_rank32全量复训与提交包验收.md`。
- rank32/alpha64 提交已获线上 **58.4011**，较 rank8/alpha16 的 58.1247 提高 **0.2764**，成为当前线上最佳。后续交接、禁止项和下一轮实验决策见 `docs/40_rank32线上结果与下班交接计划.md`。
- 2026-09-15 已完成复赛迁移验收第一批：只读核验 `main`、rank32 全量 checkpoint、提交 CSV/ZIP、H1 头、OpenAI CLIP 权重与全部输入哈希一致；rank32/alpha64 正式冻结为当前最佳与唯一回退，本轮未重训三种子、未重新推理、未重复提交。
- 已按预注册协议生成确定性长尾压力划分（`configs/data/long_tail_stress.yaml`，seed `20260915`，仅用固定 train 划分）：长尾训练侧 23,880 张/500 类、长尾评估侧 9,823 张；head 100 类 / mid 150 类 / tail 250 类；饿死类 0、最小每类保留率 0.0482、tail/head 保留率比 0.0492，跨侧重复簇泄漏 0，保留率门禁通过。
- 已实现 head/mid/tail、类均准确率、最差类召回、召回分位、零召回类数与每类保留率诊断（`src/aictrack1/evaluation/long_tail.py`、`scripts/diagnose_long_tail.py`）；rank32 只读诊断结果：固定验证 Top-1 84.2335%（head 87.11 / mid 82.26 / tail 84.26，mid 为最弱分层），长尾评估 Top-1 82.7039%（mid 82.05 / tail 83.12，head 因支持不足被标记为不可用）。这些数字是乐观上界，只用于工程口径与失效模式，不用于选模。
- 已预注册但保持锁定两个 T2 候选（末 1–2 个完整视觉块解冻，`last_n_blocks`，H1 全量初始化、同 split、同 seed、3 epochs、lr `2e-6`、冻结头）：`configs/experiment/p3_stage1_h1init_last_block_count1_lr2e6_3ep_seed1_planned.yaml` 与 `..._count2_...yaml`，两者 `execution_enabled: false`。完整协议与门禁见 `docs/41_复赛迁移长尾压力与分层诊断预注册.md`，实测证据见 `docs/42_复赛迁移长尾压力与分层诊断验收.md`。
- 已按 `docs/41` 第 6 节完成真实长尾压力测试（4 个线性头，同配方、只换训练子集，共 32.3 秒 GPU，未访问 test）：
  - L1 长尾压力头（23,880 张，249 类仅 8 张）：固定验证 Top-1 43.6246%，tail 21.35%，**31 个零召回类**，tail 最差类召回 0；
  - L2 支持数匹配对照（同池每类恰好 8 张）：Top-1 45.3214%，tail 45.11%，零召回类 4；
  - L3 预算对照（每类 20 张）：Top-1 55.7306%，tail 55.31%；
  - H1 参照（每类约 166 张）：Top-1 68.6997%，tail 68.51%。
  - **结论**：长尾确实饿死尾类（触发 `docs/41` 停止线：8 张桶类均召回 21.29% < 0.30、零召回类 31 > 25）；在同等 8 张/类稀缺度下，类频偏斜还额外使 tail 降低约 23.8 点。复赛机制必须先补“每类最低保留 + 类感知阈值”，不得原样送入复赛。
  - 支撑工程：`scripts/train_frozen_head.py` 新增 `long_tail_stress` / `balanced_size_matched_control` / `balanced_full_train_control` 三个训练范围并把实际每类训练支持写入 checkpoint 元数据；`scripts/evaluate_long_tail_heads.py` 支持一次评估多个头并输出按训练支持数分桶的饥饿探针；指标见 `artifacts/reports/20260915-long-tail-heads/long_tail_heads.json`。
- 已按 `docs/41` 第 5 节完成 T2 视觉容量线的严格配对验收，结论是**停止**：在 H1 全量初始化、冻结头、同固定划分、同 seed `20260910`、同 3 epochs / batch 64 条件下，完全冻结的初始化头为 **82.8857%**，而解冻末 1 块（7,087,872 参数）为 82.6675%、解冻末 2 块（14,175,744 参数）为 82.6384%、LoRA rank32 对照（294,912 参数）为 82.3815%，即**容量越大反而越低**，三者均未通过相对 rank32 全量头（84.2335%）的预注册门禁。因此不冻结 seed2/seed3、不申请全量复训、不申请 test 推理。详见 `docs/43_T2视觉容量线与H1头初始化验收.md`。
- 同时确认：**头质量取决于训练数据规模而不是训练轮数**。同一冻结特征缓存上把平衡头从 `100 epochs / Adam lr 0.01 / constant` 换成 `300 epochs / Adam lr 0.001 / cosine`，固定验证从 68.6997% 掉到 63.4054%。H1 全量头（在 103,218 张上训练）的 82.9240%（raw-image 同口径）才是当前最强冻结参照。
- 口径提醒：本项目 P3.5 之前的固定验证数字（H1 seed42 68.70%、rank32 LoRA 69.60% 等）基于较弱的 `20260911-p3-h1-linear-adam-e100-seed42/best.pt` 初始化；用于 T2 与提交的强初始化是 `20260912-b-final-h1-linear-full-e100-seed1/final.pt`（同一次前向 raw-image 82.9240%）。**两代数字不可跨代直接比较**，新实验请统一使用强初始化与固定划分。
- 当前仍处于**初赛官方数据阶段**；继续以初赛固定验证和线上 Top-1 追分，同时所有新方法必须通过复赛可迁移门禁。不得利用测试预测统计继续调参。

### 当前不可做

- 不用任何外部真实图像、公开数据集或他人分享数据来训练/调参。
- 不把跨标签重复比例直接解释为完整噪声率，不在训练前宣称真实性能或得分。
- 不预设官方数据目录结构、真实类别编号映射、提交是否有表头等细节。

## 4. 已验证环境

| 项目 | 已验证值 |
|---|---|
| Python | `D:\aicomp\python312\python.exe`，Python 3.12.10 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8 GiB，compute capability 8.9 |
| NVIDIA 驱动 | NVIDIA-SMI 610.88；CUDA UMD 13.3 |
| PyTorch | `2.11.0+cu128` |
| torchvision | `0.26.0+cu128` |
| PyTorch CUDA | 12.8；`torch.cuda.is_available() == True` |
| OpenAI CLIP 包 | `clip 1.0`，来自用户本地 `CLIP-main.zip` |
| 其他主要依赖 | numpy 2.5.2、Pillow 12.3.0、pandas 3.0.5、scikit-learn 1.9.0、scipy 1.18.1、PyYAML 6.0.3、tqdm 4.70.0、TensorBoard 2.21.0、matplotlib 3.11.1、pytest 9.1.1、ftfy 6.3.1、regex 2026.9.3 |

验证命令：

```powershell
powershell -ExecutionPolicy Bypass -File "D:\aicomp\AICtrack1\scripts\check_environment.ps1"
```

旧电脑继续显式使用 `D:\aicomp\python312\python.exe`。新电脑应运行 `scripts/setup_environment.ps1` 创建 `.venv`，随后显式使用 `.\.venv\Scripts\python.exe`。不需要单独安装 CUDA Toolkit；预编译 PyTorch CUDA 轮子已具备训练所需运行时，只有编译自定义 CUDA 扩展时才需要 Toolkit。

## 5. 数据获取状态

官方参赛流程说明：成功报名、缴费后，队长登录报名系统，在“我的参赛信息 → 详情”获取数据下载链接与提取码。

初赛数据已由用户取得并存放在 `DataSetStage1/`，该目录整体被 Git 忽略。只读审计和固定划分已完成，派生产物位于 `artifacts/audits/stage1/`。训练集存在 1,032 个跨标签精确重复哈希，划分时全部按重复簇隔离；不得人工删改。原始 ZIP 当前不在工作区，尚无压缩包 SHA-256。

## 6. 新对话的建议第一件事

先做只读核验：`git status -sb`、`git rev-parse HEAD`，确认 `main` 干净并已推送，再核对 `docs/42`、`docs/43` 中列出的 rank32 checkpoint 与提交哈希。rank32/alpha64 已冻结为当前最佳与唯一回退，**不得重训其 seed1–seed3、不得重复 test 推理、不得重复提交同一包**。

随后按以下优先级推进，全部先预注册再做 seed1 配对固定验证：

1. **分类头初始化质量**（当前最强杠杆）：H1 全量头直接推理 82.8857%，任何视觉微调都没有超过它；而头质量随训练数据规模变化。任何新的头方案必须先与 `20260912-b-final-h1-linear-full-e100-seed1/final.pt` 在同划分上比较，且注意 P3.5 之前的旧数字不可跨代比较。
2. **长尾机制**：`docs/42` 已证明当前流程会在长尾下饿死尾类（8 张/类桶类均召回 21.29%、31 个零召回类），复赛前必须补“每类最低保留 + 类感知阈值”并重跑同一压力测试。
3. 新一轮稳健性机制候选（噪声抑制/表征锚定）只有在上述两项有结论后才进入。

不得在全量复训验签前触碰 test；B 目前没有待执行任务单。本批已发布停训通知与下一批前置条件：`docs/task_orders/20260915-B-hold-and-next-batch-preconditions.md`。重新下发训练前必须先冻结新 commit、配置、任务单与资源预算。

## 7. 上游成熟方案入口

- 人类可读审计与实验回填：`docs/04_上游方案调研与复用登记.md`
- 最小模块接口与退出条件：`docs/05_上游最小复用规格.md`
- 机器可检查的版本和触发点：`docs/upstream_registry.yaml`
- P0/P0.5 完成证据：`docs/07_P0_P0.5_验收报告.md`
- 上游本地快照验收：`docs/08_上游本地快照验收.md`
- P1/P2 实施验收：`docs/09_P1_P2_实施验收报告.md`
- 初赛真实数据审计：`docs/10_初赛数据审计报告.md`
- P3 实施状态与资源门禁：`docs/11_P3_冻结特征基线实施与资源门禁.md`
- P3 A0/A1 正式结果：`docs/12_P3_A0_A1_正式基线报告.md`
- S1 与 A2/A3 验收：`docs/13_S1与A2_A3验收报告.md`
- P3 PEFT/M0 筛选：`docs/14_P3_PEFT资源与短程筛选报告.md`
- M1 首筛与首次提交选模：`docs/15_M1首轮筛选与首次提交选模报告.md`
- 实际复制或改写第三方代码后才创建：`THIRD_PARTY_NOTICES.md`

上游候选不能只列为参考：到达强制触发阶段后必须形成最小实现/正式对照实验或有证据的排除结论。

## 8. 工作日志与文件说明

- `pytorch-install.stdout.log` / `pytorch-install.stderr.log`：本次 PyTorch 安装过程日志，可保留用于排错；不是训练实验记录。
- `docs/技术报告.docx`：项目现有文档，尚未进行内容审查或修改；若后续需要编辑 `.docx`，应遵守文档渲染与校验流程。
- 已创建并正式运行冻结 CLIP、端到端 PEFT 和模拟提交生成链路；真实测试推理与线上提交仍未运行，不得把 noisy-validation 指标误报为干净测试或排行榜成绩。
- P0/P0.5 已完成治理建设；报名、数据权限和组委会答复仍是外部待办，不应被误报为已完成。
