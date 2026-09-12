# B 任务单：M2 视觉原型软降权 seed1

下发日期：2026-09-12

## 1. 任务边界

只执行一个固定验证集单种子实验：以已验证的 H1 线性头初始化最后视觉注意力 rank-4 LoRA，并仅增加训练集视觉原型静态软权重。不得修改代码、配置、seed、epoch、阈值或输出目录；不得读取测试集、生成提交文件或根据结果继续安排实验。

- 训练 commit：`95a138bdcef9d193d60ccc28e3d5af291b4ff3e6`
- 配置：`configs/experiment/p3_stage1_h1init_peft_lora_visual_prototype_3ep_seed1.yaml`
- 输出：`artifacts/experiments/20260912-a-h1init-lora-visual-prototype-3ep-seed1/`
- 资源预算：可信度预计算 10–20 秒；GPU 训练约 15–18 分钟；峰值显存约 0.55 GiB；新增磁盘约 2–5 MiB。
- 恢复：每个 epoch 写入 `last.pt`；异常退出后先保留现场并联系 A，不自行删除或猜测性修改。

## 2. Git 与输入预检

先在 `main` 阅读本任务单，然后执行：

```powershell
git fetch origin
git checkout --detach 95a138bdcef9d193d60ccc28e3d5af291b4ff3e6
git status --porcelain
git rev-parse HEAD
```

要求 `git status --porcelain` 无输出，HEAD 与上述 40 位 commit 完全一致。若不一致立即停止。

核对以下 SHA-256；配置因 Git 换行策略允许 LF/CRLF 两个已登记值之一，其余必须精确一致：

| 文件 | SHA-256 |
|---|---|
| 配置 LF | `9b4a10c28ed508133f9526529db0eabfc5a2db0072dfa86fbf668a180281d68c` |
| 配置 CRLF | `5b9e54b77c03c76fdb0723310485afa644f9488922b4d0864fce281328b1861f` |
| H1 `best.pt` | `e85d53d446fef914c7cec74faa1ede792b7774bb71ec057dee2b551d6cf485af` |
| 训练 manifest | `212f645a0d027cc4f2aa1a6d2f81f0175e8a3441b6fa6d001947038e00ba75ef` |
| 固定 split | `9eb97e2b89da76f64323c2800d6c524a8455e31eaa9b0dab1917b7e4b877fb31` |
| label conflicts | `d06a03bf23b54e663bfc849236f855dfeb61251262329210a72274b2c4df70f1` |
| 特征缓存 metadata | `8b6919ee0e80250cf6ca18f32020f2f95075f662e65bc9ef29f46fe3eae698a1` |
| 特征缓存 complete | `26b96ff14b66468919fc7318e113dd150f0a79b254227489692fed9403757fe1` |
| OpenAI CLIP 权重 | `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` |

若输出目录已经存在，不得覆盖或删除，停止并回报 A。

## 3. 训练前测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_visual_prototype_weighting.py tests\test_confidence.py tests\test_peft.py -q -p no:cacheprovider
```

必须全部通过。测试失败时停止，不得训练。

## 4. 唯一训练命令

在独立持久 `cmd` 终端运行，并保存合并日志：

```powershell
cmd /d /c ".venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\p3_stage1_h1init_peft_lora_visual_prototype_3ep_seed1.yaml > b-m2-visual-prototype-seed1.stdout.log 2>&1"
```

禁止同时运行其他 GPU 任务。若发生 OOM、NaN/Inf、traceback、窗口关闭或输出停滞，保留日志与目录并停止，联系 A。

## 5. 回传要求

通过私有通道回传一个 ZIP，至少包含：

- 完整实验目录中的 `config.yaml`、`command.txt`、`metrics.json`、`best.pt`、`last.pt`；
- `b-m2-visual-prototype-seed1.stdout.log`；
- 训练前 pytest 完整输出；
- 一份结果说明，列出 HEAD、起止时间、GPU、峰值显存、最佳 epoch、Top-1、macro accuracy、冲突子集指标，以及所有 checkpoint SHA-256。

不得将 checkpoint、训练日志、官方数据或特征缓存推送到 GitHub。A 验签前不得启动 seed2/seed3、全量复训或测试推理。

## 6. A 端晋级门禁

参考基线为同 seed 的 H1→LoRA：Top-1 68.9615%、macro 68.9492%。候选最佳 Top-1 必须至少达到 69.0615%，macro 不低于 68.9492%，且冲突子集不得异常退化；否则立即停止该路线。门禁只由 A 在收到完整结果后判定。
