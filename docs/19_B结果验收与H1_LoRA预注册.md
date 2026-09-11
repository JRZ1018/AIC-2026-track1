# B 结果验收与 H1→LoRA 预注册

更新时间：2026-09-11（Asia/Shanghai）

实现 commit：`a87527c0b4414a7c29c554a9f0b0cb94214b6746`

## 当前结论

A 端等待 B 首轮 M0 v2/M1 配对复验期间，只完成结果验收工具和下一轮单变量实验设计。当前没有启动 H1→LoRA 正式训练、全量重训、测试推理或提交。

后续状态：B 首轮已验签，M1 平均配对增益 +0.0695 点，未达 +0.10 点门槛；H1→LoRA seed1 已获用户批准并另建启用配置，详见 `docs/20_B首轮验收与H1冻结头提交.md`。

## B 返回产物的自动验收

验收计划：`configs/evaluation/b_first_round.yaml`。

只检查当前到件状态：

```powershell
D:\aicomp\python312\python.exe scripts\aggregate_b_results.py --plan configs\evaluation\b_first_round.yaml --check-readiness
```

四项 B 结果完整复制到各自唯一输出目录后，生成一次不可覆盖的验收报告：

```powershell
D:\aicomp\python312\python.exe scripts\aggregate_b_results.py --plan configs\evaluation\b_first_round.yaml --output artifacts\reports\20260911-b-first-round-accepted.json
```

工具逐项拒绝以下问题：

- 缺少 `config.yaml`、`command.txt`、`metrics.json`、`best.pt` 或 `last.pt`；
- 返回配置与 Git 中的锁定配置在规范化 LF 后不同；原始 LF/CRLF 哈希仍必须与训练记录一致；
- run ID、seed、commit、训练/验证数量不符；
- manifest、split、配置或 checkpoint SHA-256 不符；
- 命令不是锁定的 `train_peft.py --config ...`，或出现测试/提交入口；
- scope 没有声明 test 未访问；
- 指标存在 NaN/Inf，或 `best_top1` 与逐轮历史不一致；
- M0/M1 未按同 seed 成对。

当前 readiness 的预期状态是：历史 seed1 的 M0 v2/M1 已就绪，B 的 seed2/seed3 四项待回传。

## M1 预注册门禁

三组配对统一计算 `M1 Top-1 - M0 v2 Top-1`，不在结果到达后改变标准：

1. 三个 seed 的配对增益必须全部大于 0；
2. 平均配对增益必须至少为 `0.001`，即 **0.10 个百分点**；
3. 不满足任一条件，输出 `stop_m1_unstable_or_too_small`，停止 M1；
4. 若 M1 稳定超过 M0 v2、但三种子均值仍不超过 H1 的 68.6916%，只保留机制价值，输出 `retain_m1_mechanism_only_h1_remains_stronger`；
5. 只有 M1 同时稳定超过 M0 v2 和 H1，才输出 `m1_is_stable_and_exceeds_h1_reference`。

自动结论只是预注册门禁结果，不自动启动下一项训练，也不自动修改实验台账；最终验收仍由 A 完成。

## H1→LoRA 单变量设计

预备配置：`configs/experiment/p3_stage1_h1init_peft_lora_only_3ep_seed1_planned.yaml`。

相对严格 M0 v2，它保持相同数据、split、增强、LoRA rank/alpha、3 epochs、训练 seed、学习率、batch size、裁剪和冻结头策略；唯一方法变化是把初始化分类头从 A1 余弦头换成 H1 线性 Adam seed42 最佳头。

初始化头固定证据：

- H1 checkpoint：`artifacts/experiments/20260911-p3-h1-linear-adam-e100-seed42/best.pt`
- SHA-256：`e85d53d446fef914c7cec74faa1ede792b7774bb71ec057dee2b551d6cf485af`
- 固定验证 Top-1：68.6997%
- OpenAI CLIP 权重、类别映射和 split 元数据均已通过真实 checkpoint 只读加载 smoke。

训练器与最终推理器现在共用同一个分类头 checkpoint 加载器，支持线性/余弦两种头，并严格核对 OpenAI CLIP 权重、类别映射、split 和模型状态。旧 `a1_head_checkpoint` 配置继续兼容；新配置使用通用 `head_checkpoint`。

## 防误启动和后续门禁

预备配置含 `execution_enabled: false`。当前直接调用训练命令会在读取数据、创建输出目录或占用 GPU 前明确失败。正式运行必须在收到 B 结果并由用户再次批准后，复制成新的启用配置与新 run ID，不修改本预注册快照。

若未来批准单种子初筛，预注册晋级条件为：

- 相对配对 H1 seed42 的 68.6997%，至少提高 **0.20 个百分点**，即 Top-1 不低于 68.8997%；
- macro accuracy 不得降低超过 0.10 个百分点；
- 无 NaN/OOM、测试读取、哈希漂移或资源超限；
- 单种子通过后仍只获得“允许设计 seed2/seed3”的资格，不自动进入全量训练或测试推理。

预估单次正式运行约 15–17 分钟，PyTorch 峰值显存约 0.53 GiB，产物小于 10 MiB；这些资源尚未在本轮使用。
