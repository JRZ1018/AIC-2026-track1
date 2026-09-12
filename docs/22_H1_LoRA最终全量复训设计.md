# H1→LoRA 最终全量复训设计

更新时间：2026-09-12（Asia/Shanghai）

## 冻结决策

H1→LoRA 已通过三种子门禁。为避免按单个 best checkpoint 乐观选择，最终 LoRA epoch 使用每个 seed 在同一固定 epoch 的均值预注册：epoch 1/2/3 分别为 68.9324% / **68.9421%** / 68.9179%，因此固定为 2 epochs。随机种子固定为首个预注册 seed `20260910`，不选择单次表现最高的 seed3。

最终训练链为单模型、单一推理流程：

1. 从已审计的 OpenAI 官方 CLIP ViT-B/32 冻结特征缓存开始，使用全部 103,218 张初赛训练图片训练线性 H1 头 100 epochs。
2. 从同一 OpenAI 官方权重重新加载视觉骨干，以全量 H1 头初始化分类器并冻结分类头。
3. 只训练最后视觉注意力的 rank-4 LoRA，共 2 epochs，最终选择规则是固定第 2 epoch，不运行验证选择。
4. 训练过程继续保留固定 split 文件作为完整样本清单和来源哈希，但 train 与 validation 两部分均合并进入全量训练；测试 manifest 不进入训练配置。

## 工程与合规边界

- 新模式名为 `full_train_fixed_epochs`；它与历史 `fixed_split_validation` 路径显式分离，历史实验默认行为不变。
- 全量模式要求 `expected_train_count: 103218`，并拒绝缺失样本、未知 partition、测试特征缓存和非固定 epoch 配置。
- 全量 H1 输出 `final.pt`，全量 LoRA 输出 `final.pt`；不生成会被误解为验证集选出的 `best.pt`。
- 两阶段由 `scripts/train_final_h1_lora.py` 单命令串行运行；中断后重复同一命令会从各阶段 `last.pt` 恢复，并校验已完成阶段的配置哈希。
- 未增加第三方依赖或复制第三方代码。本轮上游回查只采用 OpenAI CLIP 官方底座，以及此前已采用的 OpenCLIP 工程检查项（AMP、断点和哈希日志）；不采用 OpenCLIP 权重或训练代码。
- 本轮不授权测试推理，不根据 55.0006、56.8751 或 57.0473 调整任何训练参数。

## 冻结代码与验证

- 当前训练 commit：`6800050b9549993babccaf3e8b88c0c591989a2a`（替代未逐 epoch 保存 H1 状态的旧 `c3d54d2...`）
- H1 全量配置 LF / CRLF SHA-256：`b5fa9cb08ee31d292a0c2ac52c779551f115fd3f643cafda1301f11d47444d8f` / `fda56b2f4873c49603ecaf8a644c64abd4d917333a2a3a1696bbe77bd95bd8fc`
- LoRA 全量配置 LF / CRLF SHA-256：`98370139679806954a9926cb6688e540be99b2559ca33a4238cd9b9e6ee413b8` / `29be5fd7331c88680c1fae9a5ea5376f8af513e2853ad236012ef38b4d976276`
- 单命令入口 LF / CRLF SHA-256：`210cd1bf8c1eebf3bfd012a91ee8efcbc37147be0885d3b9389e31b141099483` / `ddfe7caf3700b1867dce5e4005a93c33fef9a03afe90444cfdbb4cb6a961e660`
- A 端定向测试：38 项通过。
- A 端完整测试：134 项通过。
- 治理检查：通过；没有数据、权重、checkpoint 或大文件进入 Git。

B 的唯一执行入口为 `docs/task_orders/20260912-B-final-h1-lora-full-retrain.md`。结果回传后由 A 验签；通过前不生成测试预测。

首次 B 端启动在约 60 秒遭遇 Intel Fortran runtime `window-CLOSE event`，且 H1 输出目录为空。A 判定为控制台/父进程被外部关闭，同时发现旧入口没有兑现 H1 每 epoch 恢复承诺。新 commit 已增加缓存加载提示、每 epoch 进度和每 epoch 原子 `last.pt`；B 获准保留失败日志、删除经验证为空的精确目录，并在持久终端用既往成功的 `cmd /c` 重定向方式进行一次重试。
