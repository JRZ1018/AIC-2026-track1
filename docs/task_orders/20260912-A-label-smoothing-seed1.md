# A 本机代执行任务单：H1→LoRA 轻度标签平滑 seed1

下发日期：2026-09-12

## 1. 任务边界

B 已下班，本轮由 A 在本机代执行一个固定验证集单种子筛选。以 H1 线性头初始化最后视觉注意力 rank-4 LoRA，仅将交叉熵 `label_smoothing` 从 0 改为 0.05。不得修改 seed、epoch、学习率、增强或输出目录；不得读取测试集、生成提交文件或根据本轮结果自动继续多种子实验。

- 锁定训练 commit：`5a70bfa0221dd70c2799d379d539da70c69e34cd`
- 配置：`configs/experiment/p3_stage1_h1init_peft_lora_label_smoothing_3ep_seed1_a.yaml`
- 输出：`artifacts/experiments/20260912-a-h1init-lora-label-smoothing-005-3ep-seed1/`
- 资源预算：GPU 训练约 15–18 分钟；峰值显存约 0.55 GiB；新增磁盘约 2–5 MiB。
- 恢复：每个 epoch 写入 `last.pt`；异常退出时保留现场，不删除输出目录。

## 2. 身份与输入预检

训练必须在上述锁定 commit 的干净工作树执行。输入哈希如下：

| 文件 | SHA-256 |
|---|---|
| 配置 LF | `b890835be17e6c089781088f24d11ce32c9ddb0744e32e7dfc04586e581c6e54` |
| 配置 CRLF | `d73d5e32c1da50dccb3f3309eeb708bd0ce28bf37e3212297a1e082a07b260dc` |
| H1 `best.pt` | `e85d53d446fef914c7cec74faa1ede792b7774bb71ec057dee2b551d6cf485af` |
| 训练 manifest | `212f645a0d027cc4f2aa1a6d2f81f0175e8a3441b6fa6d001947038e00ba75ef` |
| 固定 split | `9eb97e2b89da76f64323c2800d6c524a8455e31eaa9b0dab1917b7e4b877fb31` |
| label conflicts | `d06a03bf23b54e663bfc849236f855dfeb61251262329210a72274b2c4df70f1` |
| OpenAI CLIP 权重 | `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` |

输出目录在启动前必须不存在；若已存在则停止，不覆盖或删除。

## 3. 训练前测试

```powershell
D:\aicomp\python312\python.exe -m pytest tests\test_h1init_peft_plan.py tests\test_peft.py tests\test_confidence.py -q -p no:cacheprovider
```

必须全部通过。

## 4. 唯一训练命令

```powershell
D:\aicomp\python312\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_peft_lora_label_smoothing_3ep_seed1_a.yaml
```

标准输出与错误同时保存到 `a-label-smoothing-seed1.stdout.log`。训练期间不启动其他 GPU 任务。

## 5. A 端晋级门禁

参考基线为同 seed H1→LoRA：Top-1 68.9615%、macro 68.9492%、最佳 epoch ECE 2.2925%。候选最佳 Top-1 必须至少达到 69.0615%，macro 不低于 68.9492%，ECE 不高于 2.2925%，且跨标签冲突子集不得异常退化。Top-1 未过门槛时，即使校准改善也停止。

通过本门禁后才申请 seed2/seed3；未经用户再次批准，不运行多种子、全量复训或测试推理。

## 6. A 本机首次启动失败与等价重试

首次启动在初始验证 DataLoader 创建 Windows 多进程管道时因沙箱 ACL 返回 `WinError 5`，尚未开始训练、未生成 `last.pt`。保留原空输出目录和日志，不覆盖现场。等价重试只改变运行标识与输出目录；代码、数据、seed 和全部训练超参数不变。重试需在沙箱外运行，以允许 PyTorch DataLoader 创建工作进程。

- 重试锁定训练 commit：`8ce862c7e2dccc98b104d50416d805030e9b2fef`
- 重试配置：`configs/experiment/p3_stage1_h1init_peft_lora_label_smoothing_3ep_seed1_a_v2.yaml`
- 重试输出：`artifacts/experiments/20260912-a-h1init-lora-label-smoothing-005-3ep-seed1-v2/`
- 配置 LF / CRLF SHA-256：`fd2e516739cb15271d4605ccf767aac8642898dd3a4165f586c3a614f4cc32a8` / `75e7fb921967865448cbcca0ac7b9659135e1099a5b773649fe847c7b0a940b7`

重试唯一训练命令：

```powershell
D:\aicomp\python312\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_peft_lora_label_smoothing_3ep_seed1_a_v2.yaml
```
