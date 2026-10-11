# 文献与冠军方案复核（2026-10-08 发起，2026-10-09 完成）

本文件记录实际读取的原文、作者代码与迁移判断。没有训练、推理、调用识别 API 或下载作者模型权重。所有群聊收益仅是线索，不能转成对本项目的收益保证。训练集和验证集中的噪声、长尾、750 类匿名映射与资源条件必须另行核对。

## 一、指定知乎原文：已完整读取，10 张正文图逐张查看

- 原文：[大比分领先！ACCV 2022 大规模细粒度图像分类冠军方案](https://zhuanlan.zhihu.com/p/591298885)。普通网页检索抓取失败，Codex 浏览器正常读取成功；未登录、未绕验证码或付费限制。
- 离线副本：`C:\Users\JRZha\Downloads\ACCV 2022大规模细粒度图像分类冠军方案 - 知乎.html`，417566 字节；全文与浏览器正文一致。
- 本文作者为 OpenMMLab 社区投稿者 ABCDEFG，描述 ACCV 2022 FGIA 冠军；“往年国一”不是本文自身提供的赛事身份。
- 10 张正文图片全部存在于同名 `_files` 目录；原尺寸查看：榜单、长尾分布、噪声图、低分辨率例、多标签图、方法流程、训练技巧表、预测分布变化、累计消融图、实验日志。广告头像不属算法正文。
- 图片中没有新增 GitHub URL。正文去重 GitHub URL 只有以下两条（均出现两次）；后续第三条来自第二个项目 README 的明确链接。

文章的可迁移摘要：成功去掉不可读和重复多标签图片；均衡 sampler、聚类剔除离群没有明显收益。技巧图包含 CE logit adjustment、动态 ArcFace、SoftEQL、蒸馏、权重平均、RRC/Flip/RandAug/RandomErase/Mixup/CutMix、FlipTTA。模型是 ViT-L/Swin-v2，最终 17 模型加权；测试类别均匀先验用于调整预测分布。其消融是累计方案，并非所有单项均有独立匹配对照。[原文](https://zhuanlan.zhihu.com/p/591298885)

### GitHub 全清单和实际读取边界

| 源 URL | 实际访问、读取 | 本赛可参考 | 不能直接采用/缺口 |
|---|---|---|---|
| [MMSelfSup / dev-1.x / fgia_accv2022_1st](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/projects/fgia_accv2022_1st) | 项目 README 全文；项目 MAE 配置 120 行全文；主仓 README 与 Apache-2.0 LICENSE 全文 | 实验配置、断点/日志管理；只在允许训练集上自己训练的自监督思想 | 该配置用 MAE ViT-L、另一个预训练权重、1600 epochs。作者用了 16×80G A100。不能替换规定 CLIP，也不能取其 checkpoint；24 小时不复现此预训练。历史分支 LICENSE 直链抓取失败，但主仓许可已核对 |
| [MMClassification / dev-1.x / fgia_accv2022_1st](https://github.com/open-mmlab/mmclassification/tree/dev-1.x/projects/fgia_accv2022_1st) | 重定向 [MMPreTrain 同路径](https://github.com/open-mmlab/mmpretrain/tree/dev-1.x/projects/fgia_accv2022_1st)；项目 README、主仓 README、Apache-2.0 LICENSE 全文 | 单模型 TTA、合法训练集损失、单权重平均、推理后处理思想 | 项目入口链接第三个作者仓库，其预训练/测试训练/ensemble 不可照搬。未读取整个 MMPreTrain 每个文件；没有声称全仓逐文件审查 |
| [Ezra-Yu / ACCV2022_FGIA_1st](https://github.com/Ezra-Yu/ACCV2022_FGIA_1st) | 旧名 ACCV_workshop 链接重定向；README、Reproduce.md 全文；下列 8 个相关代码文件全文 | 训练集筛样、损失和后处理的算法思路 | 没有看到根目录 LICENSE；尝试 LICENSE 直链失败，不能声称整个作者仓是 Apache 授权。含 OpenMMLab 版权头的文件与工具代码仍应分别保留来源；本轮只研究，后续优先自己实现公式 |

作者仓实际代码读取清单：

1. [arcface_head.py](https://github.com/Ezra-Yu/ACCV2022_FGIA_1st/blob/master/src/models/arcface_head.py)：每类多中心取最大 cosine；训练加入角度 margin；自适应 margin 由训练类频率的逆四次方根构造。可以在 CLIP 后加随机初始化分类头，但噪声标签可能放大 margin 的负作用，不能直接带上所有技巧。
2. [linear_head_lt.py](https://github.com/Ezra-Yu/ACCV2022_FGIA_1st/blob/master/src/models/linear_head_lt.py)：由训练标注计算类频率，训练 logits 加 log prior，推理按模式加/减 prior。先核对本项目已存在 balanced_prior，不能把相同机制当新方向反复实现。
3. [eql.py](https://github.com/Ezra-Yu/ACCV2022_FGIA_1st/blob/master/src/models/eql.py)：记录各类正/负梯度量，调节非目标类分数；`collect_grad()` 在该文件 forward 中没有被调用。需要审查外部 hook，不能把这个文件照抄后声称实现完整 SoftEQL；本轮未核实全仓是否有外部调用。
4. [tta_classifier.py](https://github.com/Ezra-Yu/ACCV2022_FGIA_1st/blob/master/src/models/tta_classifier.py)：原图+水平翻转的 logits 平均，参数不更新。
5. [model_soup.py](https://github.com/Ezra-Yu/ACCV2022_FGIA_1st/blob/master/tools/model_soup.py)：多个同结构权重平均，产物是一套参数。自己合法训练的兼容 checkpoint 可以探讨；LoRA A/B 因子的直接平均不等于平均完整增量，必须先合并到有效权重后核对；不能把多个网络的预测集成冒充权重平均。
6. [creat_pseudo.py](https://github.com/Ezra-Yu/ACCV2022_FGIA_1st/blob/master/tools/creat_pseudo.py)：作者将高置信 testa/testb 伪标签回写训练标注。本赛明确测试不得训练时，不可采用该路径。
7. [emsemble.py](https://github.com/Ezra-Yu/ACCV2022_FGIA_1st/blob/master/tools/emsemble.py) 与 [re-distribute-label.py](https://github.com/Ezra-Yu/ACCV2022_FGIA_1st/blob/master/tools/re-distribute-label.py)：前者多模型预测平均和按分数加权；后者按低于 K 的类依次补高分样本。后者是贪心下限补样，不是 Sinkhorn，也不是精确硬配额；竞争类可能争同一图片，原始计数未动态重算，顺序覆盖可能出现。只参考均衡思想，不能原样移植。此项合计 2 文件，作者仓共 8 个算法相关文件读完。

[Reproduce.md](https://github.com/Ezra-Yu/ACCV2022_FGIA_1st/blob/master/Reproduce.md) 明写 MAE 使用 train+testa，Swin 使用 ImageNet21k 权重，并将 testa/testb 伪标签迭代加入训练。作者复现中预测分布调整使 78.15→78.94，即 +0.79 个百分点；这不是本赛的收益估计。其 ViT-L 预训练约一周、16×80G A100，微调约 20 小时；最终体系不符合本项目的权重/测试训练/单模型限制。

## 二、Jo-SRC exact v1：全文及官方训练/推理主实现已读

- [arXiv:2103.13029v1，24 Mar 2021](https://arxiv.org/abs/2103.13029v1)、[全文 HTML](https://arxiv.org/html/2103.13029v1)、[exact v1 PDF](https://arxiv.org/pdf/2103.13029v1)。全文包括方法、实验、消融、结论和参考文献。
- [CVPR 官方论文页](https://openaccess.thecvf.com/content/CVPR2021/html/Yao_Jo-SRC_A_Contrastive_Approach_for_Combating_Noisy_Labels_CVPR_2021_paper.html)。
- [作者官方 GitHub](https://github.com/NUST-Machine-Intelligence-Laboratory/Jo-SRC)：README、main.py、demo.py 逐行读取；配置目录页抓取曾失败。没有看到明确 LICENSE，不能默认许可；本赛不能用作者 Clothing1M/Food101N checkpoint，也无需移植老 Python3.6/Apex 训练栈。
- exact v1 PDF 已另行渲染并原图核对 Fig.1–5、Table 1–6：共享 student 与 EMA teacher 的关系、双视图筛选、ID/OOD 消融和高噪声下 OOD 选择精度均与正文交叉确认；不是只读摘要或 HTML 提取文字。

原文方法摘要：同一 student 对一个样本做两次随机增强；预测与平滑给定标签的 JS 距离用于全局 clean 判断。非 clean 样本中双视图 argmax 一致判 ID、不一致判 OOD。clean 用平滑标签，ID 用 EMA teacher 的软标签，OOD 目标趋于均匀。分类项之外使用对称 KL，clean/ID 奖励一致，OOD 鼓励不一致；EMA 衰减 0.99。算法输出一个 student，demo 也只加载一个网络推理。消融说明 clean 筛样与 ID/OOD 利用共同贡献，但 OOD 选择精度在高噪声条件下可能很低。[v1 原文](https://arxiv.org/html/2103.13029v1)

本项目迁移提案（自主设计，尚未实测）：

- 不能照搬 epsilon=0.6、tau_max=0.95 或“视图不一致即 OOD”；750 细粒度类、尾类困难样本、裁剪丢主体都会造成干净样本不一致。
- 先用训练分区的 JS/EMA 可靠度作软降权，给每类保留最低权重或最低样本量，禁止把少数类全部清空；筛选对照应与现有 confidence weighting 对照。
- 第一轮只 clean+ID / 正一致性；完整 OOD 负一致性是独立变量，确认 ID 流程收益后再开启。不得把无收益的完整方法拆装后称为“原版 Jo-SRC 已通过”。
- 双 student 视图反传+两次 teacher 前向，成本与显存均高于普通单视图；先测真实吞吐再承诺 4090/4060 时长。它不是两个独立学生的 Co-teaching，也不要求推理 ensemble。
- 24 小时顺序建议仍是推理后处理、筛样保护下的短全视觉+头微调、然后 Jo-SRC 模块或完整小规模对照。没有证据把完整 Jo-SRC 排在最快两条之前。

## 三、OTMatch 与语义标签

查证的正确论文是 [OTMatch: Improving Semi-Supervised Learning with Optimal Transport，ICML 2024](https://proceedings.mlr.press/v235/tan24f.html)，Zhiquan Tan、Kaipeng Zheng、Weiran Huang；[论文 PDF](https://arxiv.org/pdf/2310.17455) 已读方法、算法、实验和附录。没有从论文页面/正文确认官方 GitHub，不随意标注非作者实现为官方。

14 页 PDF 的全文、附录证明已读；Fig.1–2 和 Algorithm 1 已渲染原图核对。图中类间相似性来自分类头，不需要外部语义类名。具体 class-cost 是归一化分类头向量的余弦距离再作 EMA，不能与使用测试真标签、或者推理时的 Sinkhorn 均衡混为一谈。

论文方法摘要：在 FreeMatch 上加入类间代价；代价来自归一化分类头向量间的 cosine 距离并做 EMA 更新。对高置信无标签训练样本，加入 weak-view 伪类别到 strong-view 分布的代价加权和。它处理训练时相近类别与遥远类别的代价差异，不等同推理后 Sinkhorn 类别均衡；论文也不要求先拿到测试真标签或类名。运行成本比对照略增，但其 CIFAR 时间不能直接移算本项目。[原论文](https://arxiv.org/pdf/2310.17455)

本项目提案：没有合法语义类名时也可由分类头向量构造 class-cost；若取得的是官方公开训练类别名称及可靠映射，可以用规定 CLIP text tower 构造语义代价/文本原型。图像“被预测成鸟而不是树”只是类间语义代价直觉，不能据此认为取得测试答案被允许。语义 CE/代价惩罚属于另一个训练变量，先在训练/验证分区比较；模型 text tower 没有监督信号时，盲目解冻它不产生有效更新。

## 四、全参微调、损失阶段与筛样

[TURN: Fine-tuning Pre-trained Models for Robustness under Noisy Labels，IJCAI 2024](https://www.ijcai.org/proceedings/2024/0403.pdf) 的方法/算法/实验已读：先冻结提取器做鲁棒线性探测，再按类对样本损失做 GMM 清洗，随后短全参数微调；包含 CLIP 预训练提取器实验。论文不是不加区别地对所有噪声样本长期 CE 解冻。其均衡子集策略在 750 类真实长尾上不能直接照搬“每类最小干净数”，否则一个低样本类可让总训练量坍缩。

文献支持的本项目最高优先训练方向：保留现最佳作为回退；从规定官方权重/合法本阶段初始化分支重走“保护预训练→筛样或软权重→所有视觉层和分类头联合短微调”，与同数据量、同增强、同预算的现流程比较。尝试分层学习率、短 warmup、训练 EMA 或末段权重平均；不能用测试损失/测试负载均衡更新参数。先纯 CE 对照，再加入一个鲁棒损失/筛样项，避免一次混合多个变量导致无法归因。

[LIFT: Long-Tailed Recognition with Foundation Model Fine-Tuning](https://arxiv.org/pdf/2309.10019) 的摘要/作者仓 README 表明全参微调也可能损害尾类；本轮没有宣称读完其 26 页全文。群聊“全参拟合好”和该结果不矛盾：训练拟合不能替代固定验证和尾类指标。

## 五、注意力与 backbone 改造

已找到最贴近“长尾、多类不均衡注意力”的候选 [Attentive Feature Augmentation for Long-Tailed Visual Recognition](https://bura.brunel.ac.uk/handle/2438/24436)（Weiqiu Wang、Zhicheng Zhao、Pingyu Wang、Fei Su、Hongying Meng，TCSVT 2022，DOI 10.1109/TCSVT.2022.3161427）。作者 PDF 已下载到本审查目录 `afa_author.pdf`；主 agent 已全文读完 12 页，并逐图核对 Fig.1–6、Table I–VIII（第 3/4/7/8/9/10/11 页）。这是团队完整原文审查，不声称找到群聊唯一方法。没有确认作者官方 GitHub。[作者 PDF](https://bura.brunel.ac.uk/bitstream/2438/24436/3/FullText.pdf)

完整 AFA 包含 CAFD 特征分解、SFG 特征生成与 LBLS 分类策略。Table IV 的 ImageNet-LT CE+cRT 49.6→完整 AFA 52.6 是 +3.0pp，不能转成本赛承诺；Table VI 中去掉特征生成的 44.8 对比基线 44.7 仅 +0.1pp，完整方法为 46.8。因此“单 SE/注意力能 +3”不由本文成立。[作者 PDF](https://bura.brunel.ac.uk/bitstream/2438/24436/3/FullText.pdf)

迁移草案需标成改造实验：保留 CLIP ViT-B/32，尝试在最后层 7×7 patch tokens/输出 embedding 上实现随机初始化的特征分解与受控残差；先单独比较分解模块，再研究生成和 LBLS。750 类、batch 64 时多数类单批频次为 1，原来的批内频次驱动生成可能失效；用训练总体频次替代、限制合成量和保护噪声尾类是本项目额外设计，不能称原版复现。完整生成体系在 24 小时内优先级低于后处理和筛样保护的短全参微调。

已访问原论文/作者仓：

- [ECA-Net 原论文](https://arxiv.org/abs/1910.03151)、[作者实现](https://github.com/BangguWu/ECANet)：通道池化后局部 Conv1D，不做 SE 的降维；作者仓 MIT。论文 ImageNet 收益不是本赛收益。
- [SE 原论文](https://arxiv.org/abs/1709.01507)、[作者实现](https://github.com/hujie-frank/SENet)：通道 squeeze/excitation；作者仓 Apache-2.0。此层可从随机初始化加入规定 CLIP，不需要作者预训练权重。
- [EMA 注意力论文](https://arxiv.org/abs/2305.13563)：Efficient Multi-Scale Attention with Cross-Spatial Learning，与 EMA 参数滑动平均是两件事，不混用简称。
- [EfficientNet 原论文](https://arxiv.org/abs/1905.11946)、[ConvNeXt 作者仓](https://github.com/facebookresearch/ConvNeXt)。不能把外部 EfficientNet/ConvNeXt 预训练替换规定 CLIP；作者冠军仓中 EfficientNetv2 没有收益，不能据此保证“efficient+SE”有效。组委会允许一定改造，不等于 blanket 禁止任何附加层。

可实现的轻量草案（自主设计）：保留整个官方 CLIP visual，在最后层 patch tokens 或输出 embedding 后附加随机初始化的 residual SE/ECA gate，使用零初始化 residual scale 保持初始函数接近原模型；只用赛内训练数据。先测试一个 gate，和无 gate 完全匹配。AFA 更复杂，需要先完成分解/合成特征及尾类噪声防护的完整设计，不能把一个 SE gate 改名为 AFA。整体替换 backbone 成 ConvNeXt 的合规性没有由“局部魔改”一句确认，不列为当前默认计划。

## 六、合规分界与 24 小时优先顺序

以下来自用户转述的组委会澄清，与文献支持/实测收益分开：TTA、训练 EMA、两个规定初始化网络的 Co-teaching 训练且最终一个模型推理、测试均衡先验下的参数冻结 Sinkhorn/硬配额是可探讨范围。测试加入模型训练的 MoE 负载损失或伪标签更新模型，与“只正则化、不引入数据集”的澄清不是同一行为；该例不能被 blanket 认定合法。训练集自动过滤和追加随机层不用外部训练均不应被旧规则错误排除。第 13 条 BDE 已由用户明确忽略，不再猜缩写或要求补充。

研究优先顺序：

1. **推理分布校正**：复用现有验证 logits，全域评估温度/软均衡→更强均衡→硬配额；增加类均召回、差类别变化、翻转样本的置信 margin。只处理固定分数，不更新模型。验证集若本身不均匀，需先构造均匀评估条件，不强行用测试先验评估原长尾验证。硬配额 N/C 必须核对真实样本/类数量与官方先验强度，不机械取 50。
2. **保护噪声与尾类后的全视觉+头联合短微调**：短筛约 70% 训练分区可做，但现基线也要用同一个 70% 对照；不能把更多数据、更多 epoch、换初始化和新损失一次混合。先真实吞吐估计，筛后可给两轮扩展空间，不能因一个 seed 单点失败关闭整个容量方向。
3. Jo-SRC clean/ID/EMA 或完整小规模独立对照；若完整 OOD 较昂贵/误判高则保留模块证据，不虚称原版有效。
4. 合法语义代价/OTMatch 迁移、训练集图像增强设计。
5. ECA/SE/AFA 轻量注意力有余量才推进；不在第一天重建 ConvNeXt 预训练体系。

参数搜索应预留局部扩展规则，不以本赛规则名义禁止广搜；但也不能用线上分数无限挑超参。使用固定且无初始化泄漏的验证、同硬件或分层比较、完整 final/last 推理协议后选下一份提交。目标 >71 是目标，不是文献或群聊能保证的结果。

## 七、额外文档全文覆盖：task_orders 38/38

`literature_extra_coverage.json` 记录每个文件 SHA-256、字节、行数和完整读取范围。全部已逐行读取；初始组合输出裁掉第 0/1 文件，两者已单独完整重读，没有把目录扫描当内容阅读。

历史任务单显示：

- [Attention+MLP 单 seed 单](D:/aicomp/AICtrack1/docs/task_orders/20260913-A-attention-mlp-lora-seed1.md) 第 9 行规定未达 +0.20pp 不搜索 MLP 超参；[视觉原型单](D:/aicomp/AICtrack1/docs/task_orders/20260912-B-m2-visual-prototype-seed1.md) 第 75 行规定未达 +0.10pp 立即停止路线。这能支持“早停过严、一个配方代表整方向”的质疑。
- [通宵队列](D:/aicomp/AICtrack1/docs/task_orders/20260915-B-overnight-8h-peft-search.md) 曾做 15×8=120 epochs 串行粗筛；不能说从来不扩展搜索。搜索域长期围绕末 N 块/LoRA 且冻结分类头是更具体的问题。
- [旧容量停训单](D:/aicomp/AICtrack1/docs/task_orders/20260915-B-hold-and-next-batch-preconditions.md) 第 3 行已作废全量 H1 泄漏固定验证下的停止结论；不能拿原表淘汰全参数容量。第 5–6 行澄清人工 8 样本/类压力测试不能当复赛真实分布。
- [20260916 C 全量单](D:/aicomp/AICtrack1/docs/task_orders/20260916-c-cwh-full-and-inference.md) 明确 full 取 final，best 仅固定验证选模；不应因回传只传 best 的旧建议而做 fixed-final 诊断替代。
- 这些是初赛历史协议，不是复赛当前最佳/分数的直接证据；现况由主 agent 的 docs 全量审查确定。

## 实际读取状态/缺口

完整：指定知乎正文+10 图、两个关联项目 README、作者仓 README/Reproduce/8 算法文件、OpenMMLab 两个主仓 LICENSE；Jo-SRC v1 全文、全部方法/消融图和官方 main/demo；OTMatch 全文/附录证明与两张方法图；AFA 全文与 Fig.1–6/Table I–VIII（主 agent 并行完成）；task_orders 全部 38。

部分：TURN 主方法/算法/实验；SE/ECA/EfficientNet/EMA 注意力/ConvNeXt 原论文摘要或作者 README/相关机制，未宣称每篇每行全读；LIFT 仅摘要/README。

具体缺口：Ezra-Yu 作者仓和 Jo-SRC 仓未发现明确许可证；SoftEQL 外部梯度 hook 尚未验证；OTMatch/AFA 官方作者 GitHub 未确认。以上不是文章正文漏链；文章正文两个真实 GitHub 链接均已成功读取，不需要用户再补该文章文件。
