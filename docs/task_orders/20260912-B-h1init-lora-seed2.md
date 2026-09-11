# 任务单：20260912-B-H1init-LoRA-seed2

- Git commit：`532d585efa6f4a18c70f7f93308af73d7c281687`
- 配置：`configs/experiment/p3_stage1_h1init_peft_lora_only_3ep_seed2_b.yaml`
- 配置 LF SHA-256：`ee3fdadb84dd38ab411e1e61edd8e703cf694bc5978555caa096a2022e010448`
- 配置 CRLF SHA-256：`05fb2c9476ed604ec1cd23da179a12aa1e7d1f3c4f5430c48f3f1e8ff67f5994`
- seed：`20260911`
- 输出目录：`artifacts/experiments/20260912-b-h1init-lora-only-3ep-seed2/`
- 预计耗时：25–32 分钟；超过 40 分钟且无合理进展即停止回报。

运行前确认配置原始哈希属于上述 LF/CRLF 之一，H1 checkpoint/manifest/split/CLIP 哈希与总入口一致，并且输出目录不存在。

唯一命令：

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_peft_lora_only_3ep_seed2_b.yaml
```

成功条件：初始 Top-1 通过 `0.686996996 ± 0.002` 门禁，完成 3 epochs，生成五项必需产物，无 NaN/Inf/OOM/test 读取。B 只报告事实，不依据结果调整参数或启动任务单外实验。
