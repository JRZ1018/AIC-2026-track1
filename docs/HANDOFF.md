# AICtrack1 交接文档

更新时间：2026-09-09（Asia/Shanghai）

## 1. 给新对话的首要指令

工作目录固定为 `D:\aicomp\AICtrack1`。开始任何实现前，必须依次阅读：

1. `docs/01_算法挑战赛道通知.md`
2. `docs/02_噪声标签细粒度图像识别鲁棒微调_赛题规则.md`
3. `docs/赛题规则.pdf`
4. `AGENTS.md`
5. `docs/00_无官方数据阶段工作计划.md`

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
- 已初始化 Git 主分支并建立防泄漏 `.gitignore`；治理、上游快照、P1/P2 初版及验收补齐提交依次为 `e5fe5bc`、`fd45b8a`、`b1d5e6d`、`1a3a752`、`99b1b1f`。
- 已建立最小目录骨架、实验索引和提交台账。
- 已将无数据阶段计划升级为获奖导向路线，并建立合规问题与规则变更日志。
- 已完成 P0/P0.5 治理落地：上游方案固定版本登记、许可证边界、强制触发阶段、最小复用规格和 Git 防泄漏检查器均已建立。
- 已将 11 个上游参考仓库按锁定 SHA 同步到 `references/upstream/`；源码由主项目 Git 忽略，使用 `scripts/manage_upstream_references.py check` 可离线核验。
- 已完成无数据阶段 P1 数据契约、变换隔离、严格只读审计、重复簇隔离划分，以及 P2 程序化噪声、类内可信度和 CE/ELR/LogitClip 行为基准；44 项 CPU/CUDA 测试通过，正式 CLIP 效果实验尚未开始。
- 已安装 Python 训练环境，且真实 CUDA 运算验证通过。
- 已安装本地提供的 OpenAI CLIP 源码。
- 已下载 OpenAI 官方 ViT-B/32 权重：`artifacts/weights/ViT-B-32.pt`（约 337.6 MiB）。
- 已修复环境检查脚本中的 PowerShell/Python 引号问题：`scripts/check_environment.ps1`。
- 当前**没有任何官方比赛数据**，尚未开始真实训练、验证或排行榜提交。

### 当前不可做

- 不用任何外部真实图像、公开数据集或他人分享数据来训练/调参。
- 不对真实效果、得分或噪声比例作出结论。
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

当前用户尚未缴费，因此没有合规的数据集访问权限。拿到官方数据后，先执行 `docs/00_无官方数据阶段工作计划.md` 的“官方数据到位后的 24 小时清单”。数据应原样放在 `data/<stage>/raw/`；不提交到 Git，不传播链接或文件。

## 6. 新对话的建议第一件事

在没有官方数据的情况下：队长继续处理报名缴费、数据权限和组委会规则确认；工程实现下一步进入 P3 计算经济型 CLIP 基线，只能做加载、吞吐、恢复和合成夹具验证，不能宣称真实比赛效果。开始 P3 或 M1-M5 前必须运行 `scripts/check_governance.py` 并回查上游登记。

在用户提供官方数据后：先停止开发新算法，执行只读数据审计并确认阶段、类别映射、文件格式和提交规则；之后跑线性分类头 CE 基线。基线可复现前不实施复杂噪声学习。

## 7. 上游成熟方案入口

- 人类可读审计与实验回填：`docs/04_上游方案调研与复用登记.md`
- 最小模块接口与退出条件：`docs/05_上游最小复用规格.md`
- 机器可检查的版本和触发点：`docs/upstream_registry.yaml`
- P0/P0.5 完成证据：`docs/07_P0_P0.5_验收报告.md`
- 上游本地快照验收：`docs/08_上游本地快照验收.md`
- P1/P2 实施验收：`docs/09_P1_P2_实施验收报告.md`
- 实际复制或改写第三方代码后才创建：`THIRD_PARTY_NOTICES.md`

上游候选不能只列为参考：到达强制触发阶段后必须形成最小实现/正式对照实验或有证据的排除结论。

## 8. 工作日志与文件说明

- `pytorch-install.stdout.log` / `pytorch-install.stderr.log`：本次 PyTorch 安装过程日志，可保留用于排错；不是训练实验记录。
- `docs/技术报告.docx`：项目现有文档，尚未进行内容审查或修改；若后续需要编辑 `.docx`，应遵守文档渲染与校验流程。
- 尚未创建正式 CLIP 训练引擎、模型封装或提交生成器；新对话不得假设它们存在。P1/P2 数据和损失基础设施已经存在并有测试。
- P0/P0.5 已完成治理建设；报名、数据权限和组委会答复仍是外部待办，不应被误报为已完成。
