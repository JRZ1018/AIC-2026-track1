# H1→LoRA 多种子验收

更新时间：2026-09-12（Asia/Shanghai）

## 线上锚点

H1 冻结线性头提交 `submissions/20260911_h1_linear_seed42_v2/pred_results.zip` 的网页分数为 **56.8751**。相对首次 M1 提交的 55.0006 提高 1.8745 点，距离队友 baseline 的 57.0473 低 0.1722 点。该结果只作为稀疏外部验证登记，不用于改变已经预注册的 H1→LoRA 门槛或继续搜索参数。

## B 回传验签

B 回传包 `B-h1init-multiseed-20260912.zip` 的 SHA-256 为 `80882838479169aa4a52d290129128edddf094a11cd40594fb7de20f7d2bf92a`。原包、汇总和合并控制台日志归档在 `artifacts/returns/20260912-b-h1init-multiseed/`，两项实验目录原样归档到 `artifacts/experiments/`。

验签结果：

- 两项均锁定训练 commit `532d585efa6f4a18c70f7f93308af73d7c281687`，run ID、seed、配置哈希、H1 初始化 checkpoint、manifest、split、样本数和 test 隔离声明均符合任务单。
- seed2/seed3 的 `best.pt`、`last.pt` 实算 SHA-256 均与 `metrics.json` 一致；初始 Top-1 均为 68.7724%，处于 `68.6997% ± 0.2` 点门禁内。
- 两项均完成 3 epochs，无 NaN、Inf、OOM、traceback 或 test 读取。日志里的 `error` 命中仅来自 `expected_calibration_error` 字段，`test` 命中仅来自范围声明。
- B 未单独回传 pytest transcript；A 使用已验证本机环境补跑任务单的 14 项定向测试，全部通过。控制台日志包含 PIL 的 stderr warning，说明回传的是合并控制台流，因此缺少独立空 stderr 文件不构成结果拒收。
- seed2 第 2 epoch 因外部 CPU/磁盘 I/O 抢占而异常减速，但训练轨迹完整、配置未变且 checkpoint 可验；seed3 耗时正常。该性能异常不改变指标有效性。

## 三种子结果与预注册门禁

| 运行 | seed | 最佳 epoch | Top-1 | macro accuracy | 相对 H1 seed42 |
|---|---:|---:|---:|---:|---:|
| A seed1 | 20260910 | 2 | 68.9615% | 68.9492% | +0.2618 点 |
| B seed2 | 20260911 | 3 | 68.9567% | 68.9448% | +0.2570 点 |
| B seed3 | 20260912 | 1 | 69.0100% | 68.9961% | +0.3103 点 |
| 三种子汇总 | — | — | **68.9761%** | **68.9634%** | **+0.2764 点** |

总体 Top-1 标准差为 0.0241 点。预注册门禁逐项判定：

1. 三个 seed 全部高于 H1 seed42 的 68.6997%：通过，最差 seed 仍高 0.2570 点。
2. 三种子平均 Top-1 不低于 68.8997%：通过，高 0.0764 点。
3. 三种子平均 macro 不低于 68.6825%：通过，高 0.2809 点。

结论：**H1→LoRA 短程候选通过三种子稳定性门禁，可以进入锁模后的下一阶段。** 这不自动授权全量重训或测试推理；二者都必须由用户另行明确批准。下一阶段不得依据 56.8751 的线上结果搜索参数。
