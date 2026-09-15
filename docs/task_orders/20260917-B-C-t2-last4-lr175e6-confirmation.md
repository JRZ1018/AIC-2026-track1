# 任务单：20260917 B/C T2 last4 lr1.75e-6 确认

这是今晚第二个独立候选。它与 lr1.5e-6 候选分开验收，不能根据任何线上分数临时改参。

- 训练 commit：`fae5ebaa54d48998f906e8428d66f48906b44ac6`
- B 配置：`configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_6ep_seed2_b.yaml`
- C 配置：`configs\experiment\p3_stage1_h1init_t2_last4_lr175e6_6ep_seed3_c.yaml`
- 单卡单进程、6 epochs、batch 64、预计每项 25–35 分钟；不访问 test、不生成提交。

开始前必须确认 HEAD、工作区、四个输入哈希、治理检查和 `tests\test_t2_lr175e6_plan.py` 全部通过。唯一命令为：

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\<对应配置文件>
```

完成后回传完整实验目录、日志，以及 config/command/metrics/best/last 五个文件的 SHA-256。A 统一验签后才决定是否启用对应 full config。
