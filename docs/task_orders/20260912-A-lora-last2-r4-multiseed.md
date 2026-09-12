# A 本机任务单：H1→LoRA 末两层注意力 seed2/seed3

下发日期：2026-09-12

## 状态与边界

任务已冻结，等待用户对扩大资源范围再次批准。批准后严格串行执行 seed2、seed3；不得修改代码、配置、学习率、epoch 或输出目录，不得读取测试集或生成提交。

- 锁定训练 commit：`8b42fd833ab1f46e59d829c5ba80c274507777f4`
- Seed2 配置：`configs/experiment/p3_stage1_h1init_peft_lora_last2_r4_3ep_seed2_a.yaml`
- Seed2 LF / CRLF SHA-256：`1d8c7876cad9476200e98607602a27c4a0e81c6ee6b601e454d6f899221300f5` / `b4a2f73741cd36638ad7cb9169f1f41ea9d0856bd1065114d52a64e53f63d831`
- Seed3 配置：`configs/experiment/p3_stage1_h1init_peft_lora_last2_r4_3ep_seed3_a.yaml`
- Seed3 LF / CRLF SHA-256：`b23288c533cb8b71b7845bd75dcff37bb6c72ffb90bc52faeeecb83985c7387c` / `4e3a23a1031947f0a83c37fdf6c92eed02bef98539fbe5a5ac4b537c64d92278`
- 两项输出目录启动前均已确认不存在。
- 资源：两次合计预计 35–45 分钟，峰值显存约 0.65 GiB，磁盘新增 6–12 MiB；每 epoch 可恢复。

## 唯一命令与顺序

```powershell
D:\aicomp\python312\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_peft_lora_last2_r4_3ep_seed2_a.yaml
D:\aicomp\python312\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_peft_lora_last2_r4_3ep_seed3_a.yaml
```

两项均在沙箱外运行并分别保存合并日志；seed2 失败时停止，不自动启动 seed3。

## 三种子门禁

与一层 LoRA 的同 seed 结果配对。要求三种子平均 Top-1 增益至少 +0.10 点，任一 seed 不得落后其配对基线超过 0.05 点，平均 macro 不退化，冲突子集不出现系统性退化。只有通过后才设计全量复训；测试推理仍禁止。
