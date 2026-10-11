# P1 固定终轮统一诊断：部署与续接契约

本目录是独立工程，不属于活跃 `training_source`。本地只完成源码准备、CPU 单元测试和已验收 C 缓存自配对，没有访问服务器、执行 GPU、打开图片、更新模型或访问 test。真正的 P1 结果必须等 J5/J20 训练完成后，由 root 或既有心跳控制器执行下述阶段。

## 冻结身份与代码来源

- P1 活跃训练 `source_manifest.json` SHA256：`2fee29fc5e5c282a331883de2b3cd4d5865c28e38a8cead89be247ec95a6c9fe`。运行前校验全部 83 个源文件和注册资产，诊断不修改它们。
- `reference/fv2_single_model.py` 原文 SHA：`4315a435dd7bdf3b9c3856d1b68120fb3994e0dfec32e33f9168543fb26e3439`，从验收过的 `fullvisual_confirm_diagnostic_v6_20261008` 逐字复制。其先调用 `configure_classifier_trainability_from_state`，再严格加载 state，已有联合头恢复能力。
- `reference/diagnose_stage2_tta.py` 原文 SHA：`7fd977bb31678bdf1d9e1ef302f232fa99803deb0f4fcb90409e1494da68ad96`，从回传的实际 final-role engine 逐字复制。保持相同验证 transforms、精确水平 flip、336/384 两尺度四视图，各 float logits 等权平均。新增 wrapper 用 `final` 角色指向真正的终轮 `last.pt`。
- `reference/fv2_v6_wrapper.py` 是历史审查参考，**不要执行它**：旧 wrapper 的 fullvisual 审计硬编码 152 张量和冻结头，P1 新 wrapper 改为 154 张量 / 88,233,966 参数，其中视觉 152 / 87,849,216，完整头 weight/bias 共 384,750。所有恢复后的 named parameters 必须逐张量等于新 last state，头不能退回 H1。
- 合法 fold H1 100 轮 SHA `45266a98329a59f708b12f394775640e032e50be8c44077a1037048f28c88753`；split SHA `0c902ea38c04581eb33ef4b26fac2d37a09abb8a4fad4ae0893b89e989ac584c`。验证标签不参与 H1 或 P1 优化。
- C 只复用 **FV2 fullvisual20 固定 last**：checkpoint SHA `078bdae1654cecc076d19e8e67b2401c6a47b985d90c91f7b6ec25293002b6c9`，4V cache SHA `4813ad6727624f3072ff6e5d28d8ecaa9b4a9e919a5e0b489946dfeea08180b9`。拒绝 best、control20(last8)、其他 seed/split、全量模型及全量 H1。

## 自动阶段推进顺序

1. 等既有 J5→J20 串行训练链完全结束。某个 arm 仅到第 8 轮或 partial，不能进入此诊断，也不能冒称 20 轮失败。确认训练 metrics.history 和 last.history 一致、epoch=20、early_stop=null、last SHA 与 metrics.last_checkpoint_sha256 一致。
2. GPU 诊断前检查 canonical lock 与运行中的 trainer。wrapper 使用同一个 `/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json`，并复用 engine 的 `TrainingLease` 获取/释放。忙则退出并让控制器下次再检查，不抢训练链 GPU、不删锁。模型构造与恢复审计也在这个租约内。
3. 部署本目录到 `/root/AICtrack1_next_cycle_diagnostic_20261009`。只上传 `source_manifest.json` 的 files 列表及 manifest 本身，逐文件 SHA 校验；不要上传 `__pycache__` 或 tests/fixtures。所有相对路径必须保留，文件不得重写换行。目录可在正式训练进行期间上传，运行必须等步骤 1–2。
4. J5、J20 各一次固定终轮四视图，batch64、workers4（可选0只影响加载）。phase deadline 由 root 的已授权预算给出；默认 1800 秒、最多 3600 秒，另预留 120 秒。至少 2GiB 空闲和活跃训练源 host memory guard 通过。代码无 runtime downloads、无 test loader、无 optimizer step。
5. 完整结果成功才下载/进入 CPU 配对。GPU 报告成功但缺新 identity 或 result_manifest 不算验收成功。保留任何失败输出，不覆盖；如明确是基础设施故障，同一 checkpoint 可用 `--attempt-name retry1` 新路径作一次有限重试，不能借此改变 checkpoint/算法。
6. 配对前核验 source receipt 实际文件 SHA、新 head 恢复审计、cache SHA、两边 sample ID/targets/group/训练 counts/映射/split/4V 逐项一致。既有 C cache 直接复用，不重新跑 best 或 last8。
7. 阅读 `selection_protocol.json` 的事前规则和 paired report 的 eligibility。此 fold 是历史反复使用过的探索条件；bootstrap 只描述固定模型观测标签下的条件误差，不能声称 leaderboard 稳定提高。新 grouped split 若要确认，须新的 train-only H1 和同 seed 完整 20 轮，原查询文件不能直接复用到新 split。

## GPU 命令（由 root 在远端执行）

`LAST_SHA` 填该 arm `metrics.last_checkpoint_sha256`，并先与实际 last 文件 SHA 比对。`PHASE_DEADLINE` 是 root 控制器的 Unix 秒截止时间，不延长已有资源授权。

```bash
python /root/AICtrack1_next_cycle_diagnostic_20261009/run_p1_diagnostic.py --training-root /root/AICtrack1_next_cycle_20261009 --arm J5 --expected-last-sha256 LAST_SHA --deadline-unix PHASE_DEADLINE --hard-seconds 1800 --run
python /root/AICtrack1_next_cycle_diagnostic_20261009/run_p1_diagnostic.py --training-root /root/AICtrack1_next_cycle_20261009 --arm J20 --expected-last-sha256 LAST_SHA --deadline-unix PHASE_DEADLINE --hard-seconds 1800 --run
```

默认产物 `artifacts/diagnostics/P1_J5/` 或 `P1_J20/`：

- `fused_logits.pt`：同 C schema `aictrack1.stage2_train_derived_fused_logits.v1`；FP16 `[29737,750]` logits、int64 targets、sample_ids、train_class_counts、epoch20、checkpoint/config/split/manifest/mapping SHA、4V 字符串。
- `paired_predictions.csv`：原 forward FP32 四视图 argmax，与 cache FP16 argmax的差异单列。C 原 FP32逐行预测未回传，差异记未知，不能写零。
- `p1_checkpoint_audit.json`：154完整 state、头覆盖H1、全部参数相等、frozen eval、152+2 optimizer groups/有限矩、20轮Cosine与AC、训练 ID/标签绑定、源manifest、last SHA。
- `rows.csv`：sample_id,target,group,relative_path,near_duplicate_group,duplicate_group,four_view_prediction。group与 C 相同构造：near 优先，再 exact，再真实 singleton sample ID。禁止重采样重复图冒充独立支持。
- `input_identity.json`：继承已验收 P0 identity schema，新增 p1_arm、joint154/numel、`classifier_head_source=joint_fixed_last_state`、训练与诊断源manifest，所有产物用 SHA 正向绑定。
- `report.json`、`result_manifest.json`：仅完整四视图可产生身份与完成清单。retry 输出带 `_retry1`，模型身份不变。

CPU 配对至少下载以上 7 件到同一目录，保留相对 `rows.csv`/audit 路径，勿编辑 JSON 后假装原 receipt。C 使用本机现有 47MB cache 与已验 sidecar。CPU entry 不读取图片、不构造模型、不查询 CUDA。

## 本机 CPU 配对命令

以下 `P1_RESULT` 是回传后的 J5 或 J20 诊断目录；`NEW_OUTPUT` 必须不存在。可以在训练链运行期间对已回传完整结果做 CPU 配对。

```powershell
& 'D:\aicomp\AICtrack1\.venv\Scripts\python.exe' 'D:\aicomp\AICtrack1\artifacts\next_cycle_20261009\p1_diagnostic_source\compare_p1.py' --control-bundle 'D:\aicomp\stage2-server-transfer\fv2_results_20261008\diagnostic\artifacts\diagnostics\fullvisual\fused_logits.pt' --control-identity 'D:\aicomp\AICtrack1\artifacts\next_cycle_20261009\acceptance_source\fv2_input_identity.json' --candidate-bundle 'P1_RESULT\fused_logits.pt' --candidate-identity 'P1_RESULT\input_identity.json' --output 'NEW_OUTPUT' --threads 2
```

查询已经在 P1 完整结果出现之前用 C 的标签/重复组支持独立冻结，位于 `inputs/frozen_queries/`，禁止后续按预测替换：自然唯一ID 29737/750 类、29632 组；均衡支持子任务 737×20=14740 图且 14740 组，不放回、不共享组，13 排除类见 query_identity。均衡是这 737 类的闭集压力任务，不等于全 750 类准确率，也不是独立新 fold。每个模型分别在相同 query 上拟合无标签 P7 bias，P7 固定 temperature=.5/strength=.75/iter60。

输出每协议 C/J P7 与 raw、paired wins/losses/changes、group bootstrap 500次、95%CI、margin保护翻转、预测 counts/逐类召回/支持、精度差异及 query 身份。无网格、容量新求解、模型融合、TTA改动或 hidden labels。`.npz` 保存同 ID 顺序的类别输出以供审查。

依据 docs240，当前均衡 P7 ≥+.20pp 为探索继续信号；≥+.10pp 需 fresh 同向确认；并列取 J5。自然750 P7和不平衡先验失配另报，不能隐藏负向。原始argmax不用于选模；1–2图类的零召回变化不会独自硬否决。尚需 root 评估尾类广泛坍塌、来源、完整训练与最终出包，不自动部署或保证分数。

## CPU 验证与依赖

本机 runtime `D:\aicomp\AICtrack1\.venv\Scripts\python.exe`，Python3.12.10 / NumPy2.5.2 / SciPy1.18.1 / Torch2.11.0+cu128。`reference/p0_core.py` 原文复用已验 P0 的公式/采样/paired bootstrap，导入要求 SciPy，但本入口不调用容量 solver。GPU engine 额外需要训练源既有 YAML/Pillow/torchvision/CLIP 依赖，不安装新包、不下载模型。

8项 CPU 测试包含直接调用原 FV2 loader 的合成联合头恢复、partial/额外head拒绝、非有限/shape拒绝、epoch8/wrongsplit/best/last8拒绝、双optimizer参数组与momentshape、预测无关groupunique查询、sameC零差值与确定性、实际共享four_view的FP32等权与精确flip。`CPU_TEST_RECEIPT.json` 保存结果。`CPU_FULL_CACHE_SELFCHECK.json` 记录真实已验 C 47MB cache 在冻结的两个查询协议下自配对：零变化、零差值、CI=[0,0]；它是工程验证，不是新的 P1 结果。

```powershell
& 'D:\aicomp\AICtrack1\.venv\Scripts\python.exe' 'D:\aicomp\AICtrack1\artifacts\next_cycle_20261009\p1_diagnostic_source\tests\test_diagnostic.py'
```

源码清单冻结后不得编辑；若发现实际运行故障，另建明确版本并让 root 审核/绑定新manifest，不能静默替换活跃文件。
