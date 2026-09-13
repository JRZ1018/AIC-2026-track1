# 任务单：20260913-A 代执行末两层 LoRA 全量复训

## 唯一目标

B 未上班期间，由 A 在本机按冻结方案完成全量 H1 初始化的末两层注意力 rank-4 LoRA 固定 3 epochs 复训；验签通过后由 A 执行唯一一次测试推理并生成 `pred_results.zip`。

## 冻结输入

- 训练 commit：`b61f00287d9daa6f5628ba5b2a1b083136666765`
- 配置：`configs/experiment/p3_stage1_h1init_peft_lora_last2_r4_3ep_full_seed1_a.yaml`
- 配置 SHA-256：`6416537cf5c6b3a134aade275ce085cd956257256ba3abccd5495e717c91485d`
- 全量 H1：`artifacts/experiments/20260912-b-final-h1-linear-full-e100-seed1/final.pt`
- H1 SHA-256：`f436e5d2a33f0ff3135a77a0acb6669d309576f4f8b334b8444319fafdff16c9`
- 固定算法：最后两个视觉注意力块、rank 4、alpha 8、3 epochs、seed `20260910`，分类头冻结。
- 固定数据：全部 103,218 张训练图片，不验证选模，不根据测试预测调整参数。

## 执行与门禁

1. 在冻结 commit 上运行完整回归测试；已通过 147 项。
2. 运行 `scripts/train_peft.py --config configs/experiment/p3_stage1_h1init_peft_lora_last2_r4_3ep_full_seed1_a.yaml`。
3. 必须得到 `training_scope=full_train_fixed_epochs`、`train_count=103218`、`validation_count=0`、36,864 个可训练视觉参数以及有限数值的 epoch3 checkpoint。
4. 验签训练 commit、配置、H1、OpenAI 权重及 checkpoint 哈希后，才允许一次测试推理。
5. 提交包必须为 24,967 行、无表头、四位类别编码，通过独立解压复核；不得基于输出分布继续调参。

用户已授权本机持续运行至产出下一份 `pred_results`，并允许调度系统性能。
