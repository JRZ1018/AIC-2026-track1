# 任务单：20260912-B-H1init-LoRA-seed3

- Git commit：`532d585efa6f4a18c70f7f93308af73d7c281687`
- 配置：`configs/experiment/p3_stage1_h1init_peft_lora_only_3ep_seed3_b.yaml`
- 配置 LF SHA-256：`b01cf8b7f97116fc2dc52d8ecf106b1e2ba4e3ce2f6bda779e4e2cf2a8e80690`
- 配置 CRLF SHA-256：`600b5fdd27362d9bb3f7cd87d15b198edf2c8bbabf49e75efbdd007dc2a19f49`
- seed：`20260912`
- 输出目录：`artifacts/experiments/20260912-b-h1init-lora-only-3ep-seed3/`
- 预计耗时：25–32 分钟；超过 40 分钟且无合理进展即停止回报。

只有 seed2 成功完成且没有哈希、数据、环境或运行异常时才能开始本项。运行前再次确认配置原始哈希属于上述 LF/CRLF 之一，H1 checkpoint/manifest/split/CLIP 哈希与总入口一致，并且输出目录不存在。

唯一命令：

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_peft_lora_only_3ep_seed3_b.yaml
```

成功条件：初始 Top-1 通过 `0.686996996 ± 0.002` 门禁，完成 3 epochs，生成五项必需产物，无 NaN/Inf/OOM/test 读取。B 只报告事实，不依据结果调整参数或启动任务单外实验。
