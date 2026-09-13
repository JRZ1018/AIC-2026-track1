# Attention+MLP LoRA 预注册

日期：2026-09-13

当前线上最佳为末两层注意力 rank-8 的 58.1247。继续增加注意力 LoRA 容量已进入约 0.1 分的边际收益区，本轮不再改变层数或 rank，而是在相同末两层、rank-4、alpha-8 下，将 LoRA 从注意力 `in_proj/out_proj` 扩展到 MLP `c_fc/c_proj`，检验适配位置是否比单纯扩容更有效。

候选严格复用末两层 rank-4 的 H1、数据、split、seed、3 epochs、学习率、batch size、增强和冻结分类头设置；唯一主要变量为 LoRA 适配位置。CUDA smoke 实测只训练 98,304 个视觉参数，仍为单模型、单次推理，可在提交前折叠全部 LoRA；不使用额外数据、文本语义或测试统计。

Seed1 相对匹配的末两层 rank-4 基线 69.1215% 必须提升至少 0.20 点，macro 不得退化超过 0.05 点，冲突子集不得退化超过 0.10 点，才进入 seed2/seed3。若主指标增益不足，停止该路线并转入单模型 MixUp/强增强预注册，不搜索 MLP rank、alpha、层数或学习率。
