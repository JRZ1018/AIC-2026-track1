# AICtrack1 交接文档

更新时间：2026-09-10（Asia/Shanghai）

## 1. 给新对话的首要指令

工作目录固定为 `D:\aicomp\AICtrack1`。开始任何实现前，必须依次阅读：

1. `docs/01_算法挑战赛道通知.md`
2. `docs/02_噪声标签细粒度图像识别鲁棒微调_赛题规则.md`
3. `docs/赛题规则.pdf`
4. `AGENTS.md`
5. `docs/00_获奖导向总体工作计划.md`

项目目标：参加第八届全球校园人工智能算法精英大赛算法挑战赛道的“面向噪声标签数据的细粒度图像识别鲁棒微调”，做出合规、单模型、可复现、可提交的方案。

用户当前设定的初赛冲刺目标是 2026-09-20；该日期应在报名系统或最新官方通知中再次核验，不要把它当作已由本地文档确认的正式截止时间。

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
- 当前已经进入**初赛官方数据阶段**并完成 M0，但尚无测试推理或排行榜提交；下一技术工作包是 M1 类内损失与历史稳定性软权重。

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

注意：不要直接使用系统默认 `python`，它仍指向 Python 3.13。项目命令应显式使用 `D:\aicomp\python312\python.exe`。不需要单独安装 CUDA Toolkit；预编译 PyTorch CUDA 轮子已具备训练所需运行时，只有编译自定义 CUDA 扩展时才需要 Toolkit。

## 5. 数据获取状态

官方参赛流程说明：成功报名、缴费后，队长登录报名系统，在“我的参赛信息 → 详情”获取数据下载链接与提取码。

初赛数据已由用户取得并存放在 `DataSetStage1/`，该目录整体被 Git 忽略。只读审计和固定划分已完成，派生产物位于 `artifacts/audits/stage1/`。训练集存在 1,032 个跨标签精确重复哈希，划分时全部按重复簇隔离；不得人工删改。原始 ZIP 当前不在工作区，尚无压缩包 SHA-256。

## 6. 新对话的建议第一件事

下一步在不读取测试图片的前提下进入 M1：先实现类内归一化损失、历史预测稳定性和软权重状态的最小模块与单测，再从 M0 最佳 checkpoint 做单种子短程正式筛选。M1 只允许相对 M0 增加可信度加权，不同时叠加视觉原型或标签修正；有清晰增益才进入多种子/M2，无增益则回退 M0。

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
- 实际复制或改写第三方代码后才创建：`THIRD_PARTY_NOTICES.md`

上游候选不能只列为参考：到达强制触发阶段后必须形成最小实现/正式对照实验或有证据的排除结论。

## 8. 工作日志与文件说明

- `pytorch-install.stdout.log` / `pytorch-install.stderr.log`：本次 PyTorch 安装过程日志，可保留用于排错；不是训练实验记录。
- `docs/技术报告.docx`：项目现有文档，尚未进行内容审查或修改；若后续需要编辑 `.docx`，应遵守文档渲染与校验流程。
- 已创建并正式运行冻结 CLIP、端到端 PEFT 和模拟提交生成链路；真实测试推理与线上提交仍未运行，不得把 noisy-validation 指标误报为干净测试或排行榜成绩。
- P0/P0.5 已完成治理建设；报名、数据权限和组委会答复仍是外部待办，不应被误报为已完成。
