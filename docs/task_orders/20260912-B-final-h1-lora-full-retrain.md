# 任务单：20260912-B-H1→LoRA 最终全量复训

## 唯一目标

从冻结代码和 OpenAI 官方 CLIP ViT-B/32 权重出发，串行完成全量 H1 线性头和全量 rank-4 LoRA 固定 epoch 复训。不得访问 test，不得改参数，不得增加实验。

## 冻结输入

- 训练 commit：`6800050b9549993babccaf3e8b88c0c591989a2a`
- H1 配置：`configs/experiment/p3_stage1_h1_linear_adam_e100_full_seed1.yaml`
  - LF SHA-256：`b5fa9cb08ee31d292a0c2ac52c779551f115fd3f643cafda1301f11d47444d8f`
  - CRLF SHA-256：`fda56b2f4873c49603ecaf8a644c64abd4d917333a2a3a1696bbe77bd95bd8fc`
- LoRA 配置：`configs/experiment/p3_stage1_h1init_peft_lora_only_2ep_full_seed1_b.yaml`
  - LF SHA-256：`98370139679806954a9926cb6688e540be99b2559ca33a4238cd9b9e6ee413b8`
  - CRLF SHA-256：`29be5fd7331c88680c1fae9a5ea5376f8af513e2853ad236012ef38b4d976276`
- 唯一入口：`scripts/train_final_h1_lora.py`
  - LF SHA-256：`210cd1bf8c1eebf3bfd012a91ee8efcbc37147be0885d3b9389e31b141099483`
  - CRLF SHA-256：`ddfe7caf3700b1867dce5e4005a93c33fef9a03afe90444cfdbb4cb6a961e660`
- H1 训练入口：`scripts/train_frozen_head.py`
  - LF SHA-256：`374f820c5da49609f5c9198a44d9048b4c25380017bf80c3e1a0ea0e0775b13c`
  - CRLF SHA-256：`d385a11e60380e4b14e32b20de75e13b269e6e1c53b011e3c56a892e4b17e7a1`
- LoRA 训练入口：`scripts/train_peft.py`
  - LF SHA-256：`02bd25e41e51ed7101e776397da8826f29bbd089468303cf927877f3a9b313ab`
  - CRLF SHA-256：`54d2b796cd740cfd57c9e7d09fdb4bdbac8a1066f0bc6fd42ffbc2a42f893d81`
- train manifest SHA-256：`212f645a0d027cc4f2aa1a6d2f81f0175e8a3441b6fa6d001947038e00ba75ef`
- split SHA-256：`9eb97e2b89da76f64323c2800d6c524a8455e31eaa9b0dab1917b7e4b877fb31`
- label conflicts SHA-256：`d06a03bf23b54e663bfc849236f855dfeb61251262329210a72274b2c4df70f1`
- class mapping SHA-256：`ff8754d203c97e59adbc5103df83e28ac954ccda7482e31fba696db507fe1ee6`
- feature-cache completion manifest SHA-256：`26b96ff14b66468919fc7318e113dd150f0a79b254227489692fed9403757fe1`
- OpenAI CLIP 权重 SHA-256：`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`

固定算法：H1 linear Adam、100 epochs、seed `20260910`；随后冻结分类头，只训练最后视觉注意力 rank-4 LoRA、2 epochs、seed `20260910`。两阶段均使用全部 103,218 张初赛训练图片，不进行验证选模。

## 运行前门禁

1. `git checkout --detach 6800050b9549993babccaf3e8b88c0c591989a2a`，确认 `git status --short` 为空；训练中途禁止 pull 或切换 commit。旧 commit `c3d54d2...` 已因 H1 未逐 epoch 落盘而作废。
2. 上述五个 Git 文本文件的原始哈希允许对应的 LF 或 CRLF 值；若为 CRLF，规范化为 LF 后必须精确匹配 LF 值。该差异仅允许由 Git 换行转换产生，B 不修改或重新签出文件。其余二进制/数据文件哈希必须逐字节精确一致，feature cache 必须完整可读。
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
cmd /d /c ".venv\Scripts\python.exe scripts\train_final_h1_lora.py --head-config configs\experiment\p3_stage1_h1_linear_adam_e100_full_seed1.yaml --peft-config configs\experiment\p3_stage1_h1init_peft_lora_only_2ep_full_seed1_b.yaml > b-final-h1-lora-full-attempt2.stdout.log 2>&1"
```

必须在不会被 60 秒超时或自动关闭的持久终端中运行并等待命令完成。预计 GPU 高占用约 25–40 分钟，峰值显存约 0.6 GiB，新增磁盘通常低于 10 MiB。新 commit 在每个 H1/LoRA epoch 后原子保存 `last.pt` 并输出进度；若意外中断且输出目录内存在完整 `last.pt`，只重复同一命令，入口会自动恢复。不得自行删除或改写非空现场。

## 首次 forrtl 中断的批准恢复

首次尝试在约 60 秒收到 `forrtl: error (200): window-CLOSE event`，没有 Python traceback、训练输出或 checkpoint；这表明控制台/父进程被外部关闭，而不是数值训练报错。B 必须先将原始日志改名保留为 `b-final-h1-lora-full-attempt1.forrtl.log`。随后只允许删除已经确认包含 0 个文件的精确目录：

```powershell
$failed = (Resolve-Path -LiteralPath "artifacts\experiments\20260912-b-final-h1-linear-full-e100-seed1").Path
$expected = (Join-Path (Resolve-Path ".").Path "artifacts\experiments\20260912-b-final-h1-linear-full-e100-seed1")
if ($failed -ne $expected) { throw "unexpected recovery target: $failed" }
if ((Get-ChildItem -LiteralPath $failed -Force | Measure-Object).Count -ne 0) { throw "recovery target is not empty" }
Remove-Item -LiteralPath $failed
```

若目录不存在则无需删除；若非空则停止并回报 A。删除后按新 commit 重新运行定向测试和上述 attempt2 唯一训练命令。若同一 `window-CLOSE` 再次发生，不进行第三次尝试，保留现场回报 A。

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
