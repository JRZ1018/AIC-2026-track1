# B 首轮训练入口

下发时间：2026-09-11（Asia/Shanghai）

本轮目标是用两个新增随机种子，成对复验严格冻结头 M0 v2 与 M1 历史置信度加权。B 只负责原样执行和回传事实；A 负责验收、汇总和路线判断。

## 固定版本

- Git commit：`6dc7ddf74df5198fe7c3d4bcb3bbc7850b8d2758`
- 要求：detached HEAD、工作区干净，不修改任何 `.py`、`.yaml`、依赖或固定划分。
- 训练只可读取 `DataSetStage1/train` 及任务单列出的训练/验证派生产物；不得运行测试推理或打开测试预测。

## 执行顺序

| 顺序 | 任务 | 任务单 |
|---:|---|---|
| 1 | M0 v2 / seed2 | `docs/task_orders/20260911-b-m0v2-seed2.md` |
| 2 | M1 / seed2 | `docs/task_orders/20260911-b-m1-seed2.md` |
| 3 | M0 v2 / seed3 | `docs/task_orders/20260911-b-m0v2-seed3.md` |
| 4 | M1 / seed3 | `docs/task_orders/20260911-b-m1-seed3.md` |

必须串行执行。每个 seed 先跑 M0 v2、再跑 M1，以保证同 seed 对照；一项失败或哈希不符时，停止后续任务并把现场交给 A。

## 首次开始前

先在 `main` 阅读并保存本页和四张任务单，然后锁定代码：

```powershell
Set-Location D:\aicomp\AICtrack1
git fetch origin
git switch --detach 6dc7ddf74df5198fe7c3d4bcb3bbc7850b8d2758
git rev-parse HEAD
git status --short
.\.venv\Scripts\python.exe -m pytest tests\test_b_task_configs.py -q
.\.venv\Scripts\python.exe scripts\check_governance.py
```

若 pytest 因 Windows 默认临时目录权限失败，保留完整错误并联系 A，不自行更改系统权限或命令。

## 共用输入哈希

- 训练 manifest：`212f645a0d027cc4f2aa1a6d2f81f0175e8a3441b6fa6d001947038e00ba75ef`
- 固定 split：`9eb97e2b89da76f64323c2800d6c524a8455e31eaa9b0dab1917b7e4b877fb31`
- A1 初始化 checkpoint：`6dbdc41c37b3422800aa5f11c30a7b70422849cd6aa62ce86803928c06dbec55`
- OpenAI CLIP ViT-B/32 权重：`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`

## 回传

每项完成后，按单项任务单记录 commit、起止时间、耗时、GPU、峰值显存、最佳 epoch、Top-1、macro accuracy、异常情况以及 `best.pt`/`last.pt` SHA-256。完整输出目录通过 U 盘或团队私有存储回传，不进入 GitHub。
