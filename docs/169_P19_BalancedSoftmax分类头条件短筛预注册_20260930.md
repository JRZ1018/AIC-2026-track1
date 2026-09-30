# P19：Balanced Softmax 分类头条件短筛（2026-09-30，待资源审批）

> **2026-10-01 结果：**已获批并完成两折 H1 训练及 train 派生验证，主指标与近重复组 CI 未过原门槛，P19 止损。下文“待审批/未启动”为预注册历史状态；实际证据与 SHA 见文末执行结果。

## 当前状态和不能混淆的结论

本方案按用户要求**假定 P18 未过门禁来规划后继**；截至本文写作，`docs/168` 中的 P18 实际状态仍为队友 RTX4090 上运行中，不能把假设写成失败结果，也不能停、覆盖或并行占用该队列。P19 当前仅完成本机源码/配置与 CPU 单元测试，**没有启动训练、验证或 test 推理，没有新提交包**。两份实验配置均为 `execution_enabled: false`。

P7 单 checkpoint 的用户回报线上分为 68.19784211088559；这是历史成绩，不用来调本项门槛/系数。P13 两折 14 轮优于 8 轮，但 P7 本身已是 14 轮。R4 `tau=.5` 训练先验校正未晋级、D2 `count^-0.5` 后训练头总体无增益、Day4 effective-number H1 接 PEFT 8 轮失败；因此本项**成功先验偏低**。Balanced Softmax 原式的 `log(n_c)` 加在训练 CE 内，既不同于 R4 的半强度校正，也不同于样本 CE 加权；不把上述负结果称为原式已证伪，也不因“不同”而假定它会提高分数。P7 已用冻结的 `uniform_soft_t0.5_s0.75` 推理校正，新头可能被其重复校正抵消。

## 唯一变量与数据边界

- 两折分别使用现成且已完整生成的 Stage2 train-only 官方 OpenAI CLIP ViT-B/32 336 冻结特征缓存；折2/3 对应 `train_openai_vit_b32_fold{2,3}_336_20260925`。每折仅其 train 分区统计 750 类 `n_c`，任何缺类、非正、非有限计数硬失败。验证不参与 `n_c`。
- 与同折已有线性 H1 CE 完全相同的模型、初始化 seed20260921、100 epoch、batch4096、Adam 0.01、constant scheduler、零 weight decay 和数据划分。唯一算法变量是训练损失从 `CE(z,y)` 变为 `CE(z + log(n_c),y)`；验证及推理始终使用**原始** `z`，不再手工减去 `log(n_c)`。新方法的训练来源计数写入 checkpoint/metrics，续训要求逐元素相同。
- 不使用 Stage1、额外数据、非 OpenAI 权重、多 checkpoint 平均/融合，不碰 test、线上榜分或 P18 活跃源码/锁。P19 配置：`configs/experiment/stage2_p19_fold2_h1_balanced_softmax_336_seed20260921.yaml` 与折3同名配置。原始 CE 对照为 `stage2_fold{2,3}_c5_h1_336_seed20260921.yaml` 的已验 best；如来源 SHA 或缓存身份不能核对，必须同配置重跑匹配 CE，不拼凑跨折结果。
- 本仓库是旧工作区，`docs/168` 所述队友最近 P13–P18 的主工作副本在 `E:\AICtrack1_p13_20260929`，本机未核验该目录可用；不得将本地代码静默当成队友服务器当前代码。任何迁移需逐文件 SHA 与训练锁核对。

## 冻结的筛查与门槛

先运行本机两折 H1，逐项验配置/缓存/官方权重/manifest/split/映射 SHA、100/100 轮、有限 loss、完整 best/last、可安全读取的模型权重及原子锁释放。对各折同一 29,737 张 train 派生固定验证样本作配对评估；记录原始中心视图 Top1、macro、tail 类均、ordinary 及零召回，并用样本近重复组而非单图作为 bootstrap 单位（500 次，seed20260925）。按 P7 冻结的 `uniform_soft_t0.5_s0.75` 在同折无标签 logits 上另计算**描述性**校正视图，绝不扫描强度；长尾原始验证集直接拟合的校正视图不是 P7 的类别平衡代理，不能充当本阶段晋级门槛。此分析只读 train-derived 验证，不借 test。

H1 只是廉价排除筛查：**两折原始中心视图 Top1 均不得低于匹配 CE，折均须至少 +0.20pp，折均 macro>0，折均 tail 类均不低于 −0.25pp，折均 ordinary 不低于 −0.25pp，合并近重复组配对 bootstrap 的 95% 下界须>0。** 任一未过则 P19 止损，不改阈值、不扫 class-prior 温度或 alpha。中心视图与四视图不可混称；通过 H1 也只是允许申请下一阶段，不证明最终模型或线上收益。校正视图如与原始收益方向冲突，记录风险并在后继 P7 四视图验证前不推进全量。

若 H1 全过，下一阶段另做资源预算：同一折内从官方 OpenAI CLIP 与各自折内 H1 开始，比较 P13/P7 匹配的 14 轮 SCE+AC PEFT，仅 H1 初始化不同；保留每轮 AC、完整 last/best、源 SHA。对两折真实 best 做固定384四视图和 P7 冻结校正，在全 29,737 行 train-derived 验证上配对并给出普通/尾类与 bootstrap CI。PEFT 两折的最低晋级线预先定为：折均 Top1 ≥+0.15pp、两折各自差值≥0、合并95%下界>0、折均 macro>0、每折 tail 类均≥−0.25pp、每折 ordinary≥−0.25pp、AC每轮非L2。旧实验使用过的折不算独立盲复现；需增加不同种子或明确方差局限。若过门，再单独冻结全量头/PEFT和唯一单 checkpoint 推理；全量训练不可直接复用折内权重，test 只在最终流程冻结后读一次，不自主线上提交。不以 68.1978 线上分数决定模型或回调门槛。

## 当前可审批的小批资源预算与停止条件

本机 RTX4060 Laptop 只做上述两折 frozen H1 短筛，使用现成缓存，不重提特征；预计 GPU 高占用约 5–12 分钟，**硬截止 20 分钟**，CPU/内存/磁盘读缓存，新增产物预期 <100 MiB、**硬上限 1 GiB**，D: 起始空闲需≥5 GiB。只在全系统无其他训练进程、仓库训练锁空、P18 不与本机共享同一 GPU 时，通过现有 `run_serial_queue` 原子锁串行启动。每折有完整原子 last，可在来源/配置完全相同且另行确认后从轮界继续；异常不自动清锁/续训/扩大预算。训练结束立刻逐文件 SHA 和 CPU `weights_only=True` 安全加载验 best/last。新 frozen-head last 已将 NumPy RNG 编码为安全原语与张量；旧 pickle 格式 last 不自动加载，旧实验的续训需要单独审计迁移，不能用 `weights_only=False` 悄悄回退。

上述 GPU 训练须按 `AGENTS.md` §7.1 在本次对话披露预算并等用户明确批准；本文和 `execution_enabled: false` 不构成批准。P18 若真实完成且通过，应先比较其预注册结论，再决定 P19 是否仍值得消耗预算。

## 本地准备验收

新增纯函数 `src/aictrack1/methods/balanced_softmax.py`；`src/aictrack1/engine/frozen_head.py` 和 `scripts/train_frozen_head.py` 新增一种可显式选用的损失、train 分区计数绑定，并将新 last 的 NumPy RNG 存成 `weights_only=True` 可读格式。`scripts/run_stage2_p19_balanced_h1_pair.py` 使用仓库原子训练锁，`scripts/audit_stage2_p19_head_pair.py` 仅支持 train-derived fold2/3、验来源 SHA 后报告原始中心视图与描述性校正视图，`scripts/aggregate_stage2_p19_head_pair.py` 以跨折共用的近重复组 ID 合并充分统计后作固定500次 bootstrap；这些脚本均未运行正式实验。`tests/test_balanced_softmax.py` 的 9 项 CPU 测试通过：等类计数回归 CE、精确 `log(n_c)`、非法支持拒绝、续训来源计数匹配、安全 last 内存往返、跨折共用组 bootstrap、两份配置只改变预注册字段。连同旧 `tests/test_frozen_head.py` 在可写临时目录下完整回归，**24 项 CPU 测试全部通过**；默认 sandbox 的 pytest 临时目录 ACL 曾拒绝访问，随后对同一小型 CPU 测试使用获准的正常 Windows 临时目录权限完成复验，无 GPU 训练或竞赛数据读取。

预注册实现文件 SHA-256（正式运行前必须逐项复核；不匹配就停止并重新审阅，不事后静默改代码）：

下列两份配置当前禁用；若获批，唯一允许的配置变更是将 `execution_enabled: false` 改为 `true`，并在启动单中另记执行态 SHA。源码与队列 SHA 不应改变。

| 文件 | SHA-256 |
|---|---|
| `src/aictrack1/methods/balanced_softmax.py` | `d552dc9ba307729e9c6fa3669f8555f645ec0ffe5b20c40c19cd6443fe76f5fb` |
| `src/aictrack1/engine/frozen_head.py` | `81ae970dab76aa0ead2ed46fe7d6bca1db238e5956a0bc5b4be0bcef7abc964c` |
| `scripts/train_frozen_head.py` | `f9a8bd37075ba2fa28b2dcc14566894ffb65a45739aa946bba5a3e8912ac6de4` |
| `scripts/run_stage2_p19_balanced_h1_pair.py` | `441b4a20146a2607dd4a8721c594165f73d5da5b39a557ea119092ad5d605e9d` |
| `scripts/audit_stage2_p19_head_pair.py` | `32901796faa704e39ac2769078400578d41ad47fc78c3b7ee42beb5ff8670126` |
| `scripts/aggregate_stage2_p19_head_pair.py` | `7e830f446141c11b11217a64cfc14da30c5dc31ac47a4e45d77abb561c370e40` |
| 折2配置 | `5c04d90c4424191706d2eb750d6ea5b331879d99f28c420c5723f74929cce57e` |
| 折3配置 | `426fa058175ea297b96e4aa312cc7e43392c3399229d7ac5c4cf0e1352622990` |

## 2026-10-01 获批后的执行前变更记录

用户已明确批准上文限定的本机两折 H1 短筛及 train 派生验证。两份配置只将 `execution_enabled` 改为 `true`。运行前为落实原定 20 分钟硬截止，队列入口增加单调时钟截止守卫；超时仅终止本队列亲自启动的 trainer，保留已原子保存的轮次 checkpoint，不改训练算法、损失、数据、种子或门槛。故上表队列 SHA 为准备态，不再是执行态；执行态 SHA 和配置 SHA 在启动记录中另记。若超时或任一来源校验失败，停止本批而非扩大预算。

执行态预启 SHA-256：队列 `b3b646946cabc72c8c2a9b5e02a9062c040fc4e8720866b0764a10e8d5afbc15`；折2配置 `a13ef44d5b7d5d1fd57f1915c497cdab757a04ef8cd4618397375fec4e5c5aff`；折3配置 `b378c95e1031ff0ea635889d3453e386bb7e1a339029a275ad752fc3b5b79f02`。9项 P19 CPU 测试在审批启用后的配置上通过；原 CE best.pt 与已有 metrics 内 SHA 相符。此后本批运行中不得再改这些源码或配置。

## 2026-10-01 执行结果与止损

本机原子串行队列于00:09:50启动、00:14:14正常完成，锁已释放；折2和折3各100/100轮，best/last 均经逐文件 SHA 与 CPU `torch.load(weights_only=True)` 验收，last 含 optimizer/scheduler/scaler 与 Python/NumPy/Torch/CUDA RNG。真实 best SHA：折2 `7f5798639713fa7d899ff41b2ef9a9ecd3433016629832175272a986d0109239`，折3 `8f348ee63a960730a88e65db1bd23bc92746debff95bdd8e0b96c449ee257df6`；last SHA：折2 `6e02370baabce2a24ec5d2f24365e55ebdb220adcdc47e08b3280a0b1d3796fb`，折3 `4538b0f43233515b398b043bf09ee0dd72872fb367ed70e2dcb16be1248b758e`。

两个固定29,737行 Stage2 train派生验证报告为 `artifacts/reports/20261001-p19-fold{2,3}-balanced-h1-screen.json`，合并门禁报告 `artifacts/reports/20261001-p19-balanced-h1-aggregate.json` SHA `7e009f07c92d2efcd0b65281ea64934d2ce84034a8dcf88739cc2078820bebcd`。原始中心视图相对匹配 CE：折2 Top1 **−0.2690pp**，折3 **−0.1782pp**，折均 **−0.2236pp**；合并近重复组 bootstrap500 的95% CI为 **[−0.3488,−0.1069]pp**。macro折均+0.1987pp、tail类均+1.8986pp，不能抵消官方主指标和置信区间的失败。预注册六项中 Top1 两折非负、折均至少+0.20pp、CI下界>0 三项失败；**P19止损**，不调整门槛或先验系数，不申请本方向 PEFT/全量/test。固定P7校正视图只是描述性诊断，不作为救回门槛。
