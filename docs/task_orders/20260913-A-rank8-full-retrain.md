# 任务单：20260913-A rank-8 LoRA 全量复训与提交

- 训练 commit：`53e12deae4c435f4abc4658429369cf043fc44b7`
- 配置：`configs/experiment/p3_stage1_h1init_peft_lora_last2_r8a16_3ep_full_seed1_a.yaml`
- 配置 SHA-256：`8c7b2d3f61e70a930fa85e1a6913126dee361c55772ef7ce3f23430bc44d383b`
- 全量 H1 SHA-256：`f436e5d2a33f0ff3135a77a0acb6669d309576f4f8b334b8444319fafdff16c9`
- 数据 manifest SHA-256：`212f645a0d027cc4f2aa1a6d2f81f0175e8a3441b6fa6d001947038e00ba75ef`
- OpenAI CLIP SHA-256：`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`

固定使用全部 103,218 张训练图、末两层注意力 rank 8、alpha 16、seed 20260910、3 epochs，冻结分类头，无验证选择。预计训练 15–25 分钟、峰值显存低于 0.9 GiB、磁盘新增低于 8 MiB，每 epoch 可恢复。

训练结束后必须验签 commit、配置、H1、权重、73,728 个可训练视觉参数、有限数值、固定 epoch3 checkpoint。全部通过后只执行一次测试推理，生成 24,967 行无表头四位标签的 `pred_results.zip` 并独立解压校验。不得根据预测分布修改任何参数。
