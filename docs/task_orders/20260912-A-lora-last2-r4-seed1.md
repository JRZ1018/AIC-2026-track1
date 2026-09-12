# A 本机任务单：H1→LoRA 末两层注意力 rank-4 seed1

下发日期：2026-09-12

## 任务边界

只执行一个固定验证集单种子实验。相对当前 H1→LoRA 基线，唯一变化是把 rank-4 LoRA 从最后 1 个视觉注意力块扩展到最后 2 个。不得修改配置、seed、epoch、学习率或输出目录；不得读取测试集或生成提交。

- 锁定训练 commit：`127a922ffb519a98a7300fc035ba406e45f1df76`
- 配置：`configs/experiment/p3_stage1_h1init_peft_lora_last2_r4_3ep_seed1_a.yaml`
- 配置 LF / CRLF SHA-256：`560bfda5eb531d419d608c0d788f3eb6ce36afdabc5cb44b3a2ce58c19c42f3e` / `ef5191bd8750435bb536ab83c5bc0d0881497ff46e5852c26d32ff3718ac602d`
- 输出：`artifacts/experiments/20260912-a-h1init-lora-last2-r4-3ep-seed1/`
- 输入哈希沿用 `docs/task_orders/20260912-A-label-smoothing-seed1.md` 中的 H1、manifest、split、conflicts 与 OpenAI CLIP 锁定值。
- 资源：预计 16–20 分钟，峰值显存低于 0.8 GiB，磁盘新增 3–6 MiB；每 epoch 可恢复。

## 训练前门禁

完整 144 项测试、治理检查和 256+256 张 CUDA 冒烟均已通过。冒烟确认视觉块 10/11 共 36,864 个 LoRA 参数，峰值显存 415 MiB，测试数据未访问。正式输出目录启动前必须不存在。

## 唯一命令

```powershell
D:\aicomp\python312\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_peft_lora_last2_r4_3ep_seed1_a.yaml
```

需在沙箱外运行以允许 Windows DataLoader 创建多进程管道，并保存合并日志 `a-lora-last2-r4-seed1.stdout.log`。

## 晋级门禁

同 seed 基线 Top-1 68.9615%、macro 68.9492%。候选最佳 Top-1 至少达到 69.0615%，macro 不低于 68.9492%，且冲突子集不异常退化。未过即停止；通过后才运行 seed2/seed3。多种子、全量复训和测试推理仍需分别记录决策。
