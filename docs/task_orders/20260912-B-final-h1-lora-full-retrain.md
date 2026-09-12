# 任务单：20260912-B-H1→LoRA 最终全量复训

## 唯一目标

从冻结代码和 OpenAI 官方 CLIP ViT-B/32 权重出发，串行完成全量 H1 线性头和全量 rank-4 LoRA 固定 epoch 复训。不得访问 test，不得改参数，不得增加实验。

## 冻结输入

- 训练 commit：`c3d54d2ba19889509f8299ada0954dbcf72500ec`
- H1 配置：`configs/experiment/p3_stage1_h1_linear_adam_e100_full_seed1.yaml`
  - SHA-256：`b5fa9cb08ee31d292a0c2ac52c779551f115fd3f643cafda1301f11d47444d8f`
- LoRA 配置：`configs/experiment/p3_stage1_h1init_peft_lora_only_2ep_full_seed1_b.yaml`
  - SHA-256：`98370139679806954a9926cb6688e540be99b2559ca33a4238cd9b9e6ee413b8`
- 唯一入口：`scripts/train_final_h1_lora.py`
  - SHA-256：`210cd1bf8c1eebf3bfd012a91ee8efcbc37147be0885d3b9389e31b141099483`
- train manifest SHA-256：`212f645a0d027cc4f2aa1a6d2f81f0175e8a3441b6fa6d001947038e00ba75ef`
- split SHA-256：`9eb97e2b89da76f64323c2800d6c524a8455e31eaa9b0dab1917b7e4b877fb31`
- label conflicts SHA-256：`d06a03bf23b54e663bfc849236f855dfeb61251262329210a72274b2c4df70f1`
- class mapping SHA-256：`ff8754d203c97e59adbc5103df83e28ac954ccda7482e31fba696db507fe1ee6`
- feature-cache completion manifest SHA-256：`26b96ff14b66468919fc7318e113dd150f0a79b254227489692fed9403757fe1`
- OpenAI CLIP 权重 SHA-256：`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`

固定算法：H1 linear Adam、100 epochs、seed `20260910`；随后冻结分类头，只训练最后视觉注意力 rank-4 LoRA、2 epochs、seed `20260910`。两阶段均使用全部 103,218 张初赛训练图片，不进行验证选模。

## 运行前门禁

1. `git checkout --detach c3d54d2ba19889509f8299ada0954dbcf72500ec`，确认 `git status --short` 为空；训练中途禁止 pull 或切换 commit。
2. 上述文件哈希必须全部一致，feature cache 必须完整可读。
3. 以下两个输出目录必须都不存在：
   - `artifacts/experiments/20260912-b-final-h1-linear-full-e100-seed1/`
   - `artifacts/experiments/20260912-b-final-h1init-lora-full-2ep-seed1/`
4. 运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_final_retrain_plan.py tests\test_h1init_peft_plan.py tests\test_frozen_head.py tests\test_peft.py tests\test_peft_checkpoint.py tests\test_heads.py -q
```

定向测试不通过即停止，不得修改代码或配置。

## 唯一训练命令

```powershell
.\.venv\Scripts\python.exe scripts\train_final_h1_lora.py --head-config configs\experiment\p3_stage1_h1_linear_adam_e100_full_seed1.yaml --peft-config configs\experiment\p3_stage1_h1init_peft_lora_only_2ep_full_seed1_b.yaml 2>&1 | Tee-Object b-final-h1-lora-full.stdout.log
```

预计 GPU 高占用约 25–40 分钟，峰值显存约 0.6 GiB，新增磁盘通常低于 10 MiB。任务支持 epoch 边界恢复；若意外中断且两个输出目录内状态完整，只重复同一命令，入口会自动使用 `last.pt`。不得自行删除或改写现场。

## 成功与停止条件

成功必须同时满足：

- H1 `metrics.json`：`training_scope=full_train_fixed_epochs`、`train_count=103218`、`validation_count=0`、完成 100 epochs、生成 `final.pt` 和 `last.pt`。
- LoRA `metrics.json`：同一 training scope、`train_count=103218`、`validation_count=0`、完成 2 epochs、只训练 18,432 个视觉参数、生成 `final.pt` 和 `last.pt`。
- 两项均记录 test 未访问，配置、数据、权重、H1 初始化和 checkpoint 哈希完整，无 NaN/Inf/OOM/traceback。

任一哈希不符、样本数不符、出现 test 读取或需要改参数时立即停止并原样回报 A。

## 私下回传

将以下内容原样打包私下回传，不推送 GitHub：

- 两个完整实验输出目录；
- `b-final-h1-lora-full.stdout.log`；
- 定向 pytest 原始输出；
- 起止时间、总耗时、Python/PyTorch/CUDA/GPU；
- `final.pt`、`last.pt`、配置和回传 ZIP 的 SHA-256；
- 所有 warning、异常、中断或资源抢占事实。

B 不运行测试推理、不生成 `pred_results`，完成后等待 A 验收。
