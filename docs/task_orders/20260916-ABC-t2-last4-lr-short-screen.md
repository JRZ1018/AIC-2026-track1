# 任务单：20260916 T2 last4 学习率短筛选

## 目标与冻结边界

本轮只做固定验证短筛选，不访问 test、不生成提交、不做全量训练。三个配置均从仅见固定 train 的 H1 `best.pt` 初始化，只改变 visual learning rate，统一解冻末 4 个完整视觉块、冻结分类头、batch 64、6 epochs、seed `20260910`。6 epochs 是筛选预算，不是最终训练预算。

| 执行者 | 配置 | 学习率 | 输出目录 |
| --- | --- | ---: | --- |
| A | `p3_stage1_h1init_t2_last4_lr075e6_6ep_seed1_a.yaml` | `7.5e-7` | `artifacts/experiments/20260916-a-t2-last4-lr075e6-6ep-seed1` |
| B | `p3_stage1_h1init_t2_last4_lr125e6_6ep_seed1_b.yaml` | `1.25e-6` | `artifacts/experiments/20260916-b-t2-last4-lr125e6-6ep-seed1` |
| C | `p3_stage1_h1init_t2_last4_lr15e6_6ep_seed1_c.yaml` | `1.5e-6` | `artifacts/experiments/20260916-c-t2-last4-lr15e6-6ep-seed1` |

## 冻结版本与资源

- 训练 commit：`d27e91cf710f9fc04c26587bbba46c2187932df9`
- Python：B/C 使用 `D:\aicomp\AICtrack1\.venv\Scripts\python.exe`；A 使用 `D:\aicomp\python312\python.exe`
- 单卡单进程；每项预计 25–35 分钟，峰值显存约 1.2 GiB，新增磁盘约 0.4 GiB；支持 epoch 边界断点恢复。
- 用户已批准本轮资源。不得并发第二个训练进程，不得改变参数或配置。

## 唯一命令

```powershell
.\.venv\Scripts\python.exe scripts\train_peft.py --config configs\experiment\<对应配置文件>
```

若异常中断，仅在同一输出目录有可加载 `last.pt` 时，用原配置加 `--resume <输出目录>\last.pt` 恢复；不得删除有效 checkpoint 或切换 commit。

## 回传与筛选

B/C 完成后回传完整实验目录、日志和五个核心文件 SHA-256。A 统一验签，按固定验证 Top-1、macro、冲突子集 Top-1 和有限性排序；只有进入前三且保护指标合格的候选才会进入三种子确认。短筛选阶段不根据 test 或线上分数改参数。
