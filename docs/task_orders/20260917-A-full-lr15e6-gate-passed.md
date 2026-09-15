# 任务单：20260917 A lr1.5e-6 全量复训与推理

三种子预注册门禁已通过，允许启用等价 full 配置。不得改变 commit、配置、seed、epoch、batch 或输出路径。

## 门禁证据

- seed1：`20260916-c-t2-last4-lr15e6-6ep-seed1`，Top-1 0.7055172920，macro 0.7054080963，冲突 0.3088137805。
- seed2：`20260917-b-t2-last4-lr15e6-6ep-seed2`，Top-1 0.7053233981，macro 0.7052209377，冲突 0.3068257272。
- seed3：`20260917-c-t2-last4-lr15e6-6ep-seed3`，Top-1 0.7055658102，macro 0.7054651976，冲突 0.3108018637。
- 均值：Top-1 0.7054688334、macro 0.7053647439、冲突 0.3088137905；阈值分别为 >0.704950687、≥0.704344849、≥0.308801864。

## 唯一执行配置

`configs\experiment\p3_stage1_h1init_t2_last4_lr15e6_6ep_full_seed1_a.yaml`

该配置使用 103,218 张训练样本、6 epochs、T2 last4、visual lr 1.5e-6，输出目录为：

`artifacts\experiments\20260917-a-final-h1init-t2-last4-lr15e6-full-6ep-seed1`

训练完成后仅使用该目录的 `best.pt` 做最终 test 推理，生成新的 `pred_results.zip`；按既有提交校验流程检查 24,967 行、文件名顺序、四位标签和 ZIP 内容。
