# 任务单：20260913-A Attention+MLP LoRA seed1

- 训练 commit：`d6d9b89373fd54aa853741aa671b20ae7ef11de7`
- 配置：`configs/experiment/p3_stage1_h1init_peft_lora_last2_attn_mlp_r4_3ep_seed1_a.yaml`
- 配置 SHA-256：`3ea801c8ba49578f46df449bd723285837b307929759b6e503428529a300f432`
- 单变量：末两层 rank-4 LoRA 从 Attention 扩展到 Attention+MLP；其余训练设置与匹配基线相同。
- 资源：预计 18–25 分钟，峰值显存低于 0.8 GiB，新增磁盘低于 5 MiB，每 epoch 可恢复。
- 成功条件：82,592/20,626 固定划分、98,304 个可训练视觉参数、3 epochs、checkpoint 与哈希完整、无 NaN/Inf/OOM、test 未访问。
- 晋级门槛：相对 69.1215% 提升至少 0.20 点；macro 退化不超过 0.05 点，冲突子集退化不超过 0.10 点。未通过则停止，不搜索 MLP 超参数。
