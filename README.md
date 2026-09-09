# AIC 算法挑战赛：噪声标签细粒度图像识别鲁棒微调

本目录为本道赛题的主工作文件夹。

- 官方赛道通知：[3629](https://www.aicomp.cn/notice/notice-1/3629.html)
- 赛题规则：[3714](https://www.aicomp.cn/tracks/tracks-1/3714.html)
- 资料提取日期：2026-09-09

## 开始工作前

1. 阅读根目录 [`AGENTS.md`](AGENTS.md)。
2. 阅读 [`docs/HANDOFF.md`](docs/HANDOFF.md)。
3. 阅读两份规则摘要与 [`docs/赛题规则.pdf`](docs/赛题规则.pdf)。
4. 按 [`docs/00_无官方数据阶段工作计划.md`](docs/00_无官方数据阶段工作计划.md) 的门禁顺序执行。
5. 对规则有疑问时，先登记到 [`docs/compliance_questions.md`](docs/compliance_questions.md)，未经确认采用保守解释。
6. 开始 P3 或 M1-M5 前，检查 [`docs/04_上游方案调研与复用登记.md`](docs/04_上游方案调研与复用登记.md) 和 [`docs/upstream_registry.yaml`](docs/upstream_registry.yaml)。

## 当前状态

- 尚无官方比赛数据，不得用任何外部真实图像替代训练或调参。
- 已验证 OpenAI CLIP ViT-B/32 权重及 RTX 4060 Laptop GPU CUDA 环境。
- 当前优先级是完成报名缴费与数据领取，同时建设合成数据下的可复现训练、评估和提交闭环。
- P0/P0.5 治理与上游方案审计已落地；下一工程阶段是 P1 数据契约与 P2 可控噪声夹具。

## 治理检查

使用项目 Python 运行：

```powershell
D:\aicomp\python312\python.exe scripts\check_governance.py
```

检查器会验证上游候选的固定 commit、许可证边界、触发阶段和交付物字段，并阻止数据、权重、密钥或大文件进入 Git 候选集。

只读同步上游参考源码：

```powershell
D:\aicomp\python312\python.exe scripts\manage_upstream_references.py sync
D:\aicomp\python312\python.exe scripts\manage_upstream_references.py check
```

源码保存在 `references/upstream/` 并被主项目 Git 忽略。同步只检出注册表锁定的 commit，不初始化子模块、不下载 Git LFS 对象，也不安装或运行上游代码。
