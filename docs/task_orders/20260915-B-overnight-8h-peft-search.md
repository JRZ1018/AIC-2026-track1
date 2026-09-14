# 任务单：20260915-B 通宵 8 小时以上 PEFT 固定验证搜索

## 目标

B 在单卡单进程上按固定顺序连续执行 15 项 seed1 固定验证训练，共 120 个完整训练 epoch。预计总墙钟 10–14 小时，训练工作量不得擅自删减至 8 小时以下。本任务只做训练侧/固定验证侧粗筛，不读取 test、不生成提交、不做全量复训，也不根据夜间中间结果自行改参数或追加配置。

## 冻结版本与资源

- **训练 commit：`42203b84a0bc00594113e75c2d4ff32ae1b0b850`**
- 分支来源：`origin/main`
- 要求工作区：clean（`git status --short` 必须为空）
- Python：B 机器项目 `.venv\Scripts\python.exe`
- 规模：15 项 × 8 epochs = 120 epochs；每项训练 82,592、验证 20,626，seed 均为 `20260910`
- 资源：单卡单进程、batch 64；预计峰值显存不高于约 1.8 GiB；预计新增磁盘 3–5 GiB
- 恢复：每项每 epoch 原子保存 `last.pt`；队列脚本重启时跳过已有 `metrics.json` 的完成项，并从当前项 `last.pt` 恢复

请在开始前确保剩余磁盘至少 8 GiB、电源接通、系统不会自动睡眠。不要在同一 GPU 上并发其他训练。

本任务单所在提交晚于训练 commit。B 从最新 `main` 阅读任务单后，训练必须 checkout 上述完整 40 位 commit；该 commit 已包含全部启用配置、测试和队列脚本。

## 固定输入哈希

- 数据配置：`a5919df165c557ecaf94e649a757e8081dfbf06ce84774104ff243e8a4c55251`
- train manifest：`212f645a0d027cc4f2aa1a6d2f81f0175e8a3441b6fa6d001947038e00ba75ef`
- split：`9eb97e2b89da76f64323c2800d6c524a8455e31eaa9b0dab1917b7e4b877fb31`
- label conflicts：`d06a03bf23b54e663bfc849236f855dfeb61251262329210a72274b2c4df70f1`
- class mapping 语义哈希：`e69951b88640b31298ef7786fb9d3654e18114010bfe02769f9481103a550fcb`
- 固定 split H1 `best.pt`：`e85d53d446fef914c7cec74faa1ede792b7774bb71ec057dee2b551d6cf485af`
- OpenAI CLIP ViT-B/32：`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`

## 配置规范化 SHA-256

以下哈希按 UTF-8 文本先把 CRLF 规范化为 LF 再计算，避免 Windows checkout 换行差异：

| 配置文件 | SHA-256 |
| --- | --- |
| `p3_stage1_overnight_t2_b2_lr1e6_8ep_seed1_b.yaml` | `36f9853b71aaa4346fd57bbeedbcbd4ccf07810e62dc6170d5dd3f40d32cbfb6` |
| `p3_stage1_overnight_t2_b2_lr2e6_8ep_seed1_b.yaml` | `1d5e24c2be60bc0ef6237b6f45cdd1a7e4156052d66c44388f5a8eca6e06d076` |
| `p3_stage1_overnight_t2_b2_lr3e6_8ep_seed1_b.yaml` | `fe2b356a6eab5ccc5e05c7c6ad4d07c0c9d87c25cd0926fc7f6fe7f1cc715e96` |
| `p3_stage1_overnight_t2_b3_lr1e6_8ep_seed1_b.yaml` | `0e900897f4365a4448ebe9922ce7fd795d75b67c69b3f799f9df7d5e611d89ae` |
| `p3_stage1_overnight_t2_b3_lr2e6_8ep_seed1_b.yaml` | `44f81080a4bc569d417efa3a1d1dba754538dd18d0660039d6d91b01fee472a1` |
| `p3_stage1_overnight_t2_b3_lr3e6_8ep_seed1_b.yaml` | `f3de8acd922b48681d72b256a1bd8b15c677fce29772702a31e6253d726b893e` |
| `p3_stage1_overnight_t2_b4_lr1e6_8ep_seed1_b.yaml` | `6642275771d926acef3f87165f75dcb5bb324c14c1d63b0ea2a04074d42615e4` |
| `p3_stage1_overnight_t2_b4_lr2e6_8ep_seed1_b.yaml` | `fcb7051aefe918405c9964651a1aaf06bd26be521bdc800cf5f0401ed2e2e4d6` |
| `p3_stage1_overnight_t2_b4_lr3e6_8ep_seed1_b.yaml` | `115afb0914c78e7953783e9aef79a4fdcc4c39914504b7031ccb0d457cb0336b` |
| `p3_stage1_overnight_lora_last2_r16a32_lr1e4_8ep_seed1_b.yaml` | `e2a603a83a9ac974b7a77bddcf4e3e11d232f781155b828ead2004e29cb67062` |
| `p3_stage1_overnight_lora_last2_r32a64_lr5e5_8ep_seed1_b.yaml` | `ccd7fa5dbb6d0c75623b1533f30ae154bdaba26c33f919f0dce0146c76b76fc0` |
| `p3_stage1_overnight_lora_last2_r32a64_lr1e4_8ep_seed1_b.yaml` | `0374a8589add8e54bf335395580cfd850623e9961e5ce641067f129e0454242b` |
| `p3_stage1_overnight_lora_last2_r32a64_lr2e4_8ep_seed1_b.yaml` | `d70d8b9d2753049aa5f98e3dddf354b0ed1f310c4a369df9dd660c1a4a929deb` |
| `p3_stage1_overnight_lora_last3_r32a64_lr1e4_8ep_seed1_b.yaml` | `6b904bcb64faaa3446cd5d6a695a019fdb72b4b453923b1d20b61bab01ef32a2` |
| `p3_stage1_overnight_lora_last4_r32a64_lr1e4_8ep_seed1_b.yaml` | `1ff3cbb0b71c87bd50f953dab378f3f48080cdeb9f7a204d5f281f46357ab058` |

## 开始前门禁

```powershell
Set-Location D:\aicomp\AICtrack1
git fetch origin
git switch --detach 42203b84a0bc00594113e75c2d4ff32ae1b0b850
git rev-parse HEAD
git status --short
.\.venv\Scripts\python.exe -m pytest tests\test_b_overnight_queue_plan.py tests\test_partial_unfreeze_plan.py tests\test_peft.py tests\test_peft_checkpoint.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe scripts\check_governance.py
```

只有 HEAD 精确匹配、工作区为空、测试/治理通过、输入哈希和 15 个规范化配置哈希全部匹配时才可开始。任一不符即停止并回报 A，不自行修改、安装替代依赖、重建输入或更换 checkpoint。

## 唯一队列命令

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_b_overnight_queue.ps1 -Python .\.venv\Scripts\python.exe
```

只运行这一个队列命令。顺序已经冻结：先 9 个 T2，再 6 个 LoRA。首次启动时 15 个输出目录必须均不存在或为空；若夜间异常退出，保留现场，排除机器故障后原样重跑同一队列命令即可从当前 `last.pt` 恢复。禁止删除部分输出后重来，禁止跳项、换序、并发、改 batch/epoch/seed/lr。

## 成功条件与停止线

- 每项完成 8 epochs，生成 `config.yaml`、`command.txt`、`metrics.json`、`best.pt`、`last.pt` 和队列日志。
- 每项初始验证通过 `0.686996996 ± 0.002`，训练/验证数为 82,592/20,626，分类头冻结。
- T2 参数范围只允许相应末 2/3/4 个完整视觉块；LoRA 参数范围只允许相应末 2/3/4 个视觉 attention。
- 任意 NaN/Inf、CUDA OOM、test 读取、样本数/参数范围不符或 checkpoint 无法保存，立即停止整个队列并保留现场。
- 不因某项中间指标较高或较低提前改变后续队列；B 不判断晋级，不运行 seed2/seed3。

## 私下回传

队列完成后回传：

1. 15 个完整实验目录；
2. `artifacts/queues/20260915-b-overnight/` 下全部日志；
3. 起止时间、实际总墙钟、GPU/环境信息；
4. 每项配置、`best.pt`、`last.pt` 的 SHA-256，以及最佳 epoch、Top-1、macro、冲突子集 Top-1、峰值显存、耗时；
5. 是否发生恢复、异常或手工操作的事实记录。

产物经 U 盘或团队私有存储回传，不得把 checkpoint、日志或压缩包推到 GitHub。B 不运行 test、不生成 `pred_results.zip`，也不依据结果自行追加实验。
