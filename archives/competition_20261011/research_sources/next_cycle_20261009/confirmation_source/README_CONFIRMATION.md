# Fresh C/J 确认训练资产与执行接口

本包状态：**本地 split/features/源码工程准备通过，真实新 H1/C/J 尚未训练**。本子任务没有访问服务器、GPU、测试图片，也没有修改正在运行的 `training_source`。部署仅由 root 执行。

## 已核准的资产

- Stage2 官方 train manifest SHA：`1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d`，148,695 行、750 类。
- 新 `inputs/confirmation_split.csv`：seed **20261009**，seeded_random；SHA `57189c0bd5bf195fc9d582663eb1b31282448e82ab99ca068c76c53097a658b7`，train **118,958** / query **29,737**。
- 旧选参 split 为 seed20261003/SHA `0c902ea38c04581eb33ef4b26fac2d37a09abb8a4fad4ae0893b89e989ac584c`。新的查询排除旧 validation 所属的全部连通组；旧 validation 可以进入新 train。
- exact 图片 SHA、exact duplicate group、near duplicate group 先取连通分量。共有 139,996 个分量，7,565 个非单例分量；官方 priority grouping 与连通分量一致，合并额外 priority 组数为0。新 train/query 组交叉0，旧query/新query组及样本交叉0；750 类均达到每类20%查询目标，缺口0。
- `inputs/confirmation_rows.csv` 正向记录原 manifest 行序的 sample_id、真实 target、新 split、连通组和 relative_path。H1 binding：all `37b23205175174328fafbcfe18e9893df7eacc36a0580d56064281ce3eda7065`；train `1196a5653fcacd1f1387f0be84b540dedc2ae4f85b585d118655b73cb206f524`；query `42c7e7fe4931152790e726d866eb758542f21b61c8115769f8f7f78daa57d767`。

唯一核准复用的本地 feature cache 是 `D:\aicomp\AICtrack1\artifacts\features\stage2\train_official336_p5_20260928`。官方 OpenAI CLIP SHA `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`；native336、7x7→10x10 bicubic positional interpolation、normalized FP16 `[148695,512]`。没有 PEFT checkpoint/config。37 shards 共 **175,669,216 bytes**，每 shard SHA/大小以及全部真实 ID、标签、sample_index、relative_path 已逐行验证，feature 有限且归一。

原 metadata SHA `cea6c9ab89cbadec3999cbae49531d3d1a58d9add05927b90edb87a3dd71b885`，原 complete SHA `d2b0bd3be0011139ecf3e5b2cd8c0e512c1d8e8b3c8decc17d1c5c8628c51c73`，原 split `c8cd...`。**没有修改原 cache**；新 metadata 与 rows 独立写在本包 inputs 下，H1 loader 在正向绑定全部行之后才重设内存中的 partition/new split metadata，并保留原 feature identity。不能把 PEFT 特征按这个流程跨 split 复用。远端是否存在同哈希 cache 尚由 root 独立核准；不把本地可用等同远端可用。

## 部署与固定训练协议

Bootstrap ROOT 必须是 `/root/autodl-tmp/AICtrack1_next_cycle_confirm_20261009`，禁止输出跨盘 symlink。H1 输出 `ROOT/artifacts/experiments/confirmation_H1/{last.pt,h1_receipt.json,metrics.json}`。实际 H1 epoch100、Adam lr .01/batch4096、seed20261008、线性头、CE、只训练新train；整个拟合期间不计算query指标、不选best。终点固定后仅计算一次 query cache parity，作为后续 raw-image loader 的数值校验，不能回调训练配方。

所有GPU阶段通过原封不动的 `aictrack1.engine.training_queue.run_serial_queue` 与唯一锁 `/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json`；同机已有训练或锁不明即停止，不清锁、不绕守卫。H1 trainer 用原队列已有的 `scripts/train_frozen_head.py` 名称，无需新增队列别名。控制器使用进程组超时退出和原锁的失败处理。

实际当前 fold 独立选模结束后，仅当 J5 或 J20 入围才准备 C/J 确认。`prepare_confirmation_visual.py` 要求 recent 的 `aictrack1.p1.full_selection.v1` SHA绑定回执：真实old-fold fixedlast20、selected_arm、验证依据、无test/榜分使用，并重核实际 checkpoint/config/identity/paired report/training_source 的哈希。它不自行挑winner。

脚本会在 `ROOT/prepared_visual` 生成另一份独立冻结的视觉源码，H1 bootstrap 不被覆盖。两个正式训练输出都在数据盘的 `ROOT/prepared_visual/artifacts/experiments/P1_C|P1_<winner>`。C+入围J从**官方 CLIP + 同一个新train H1** fresh初始化，seed20261008、20轮完整日程、native336、batch64、同增强/损失/AC/批序，visual LR1.75e−6。唯一算法差异是 head LR：C=0，J5=5e−6或J20=2e−5。不载入旧视觉last/final，不缩短日程，不以best替代last20。C152张量，J154张量/88,233,966参数，末端新head来自joint trainable_state。

父进程在 torch import 之前设置 `OMP_NUM_THREADS=4`、`MKL_NUM_THREADS=4`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`；guard设置并实测 `torch.get_num_threads()==4`。原 train_n1 主体与 canonical queue 逐字保留。actual runtime 回执写 `prepared_visual/artifacts/runtime_receipts/C.json|<winner>.json`，绑定当次 source/config/split/H1 SHA，记录 interop 实测值；C/J runtime要一致。

## Root 的复制与命令

只按 `CONFIRMATION_DELIVERY.json.files` 上传（不要上传 `test_fixtures`，不要 rglob 共享 artifacts）。`external_feature_assets` 是私有 cache 的另行复用/传输清单，不在源码包中。已有远端源 metadata/complete/全部37 shard 哈希相同，可保留原路径；若无同哈希源，可将本地175.7MB原cache私有传输，仍不重新提取、重写原分片。

以下命令为 Linux/root 操作模板。`PYTHON` 必须使用当前已经验收的训练环境；`OFFICIAL336_CACHE` 必须先定位并逐SHA核准。deadline 和 initial-free 是 root 当前预算回执的实际值，不能重置截止时间或伪造当前剩余空间。

```bash
cd /root/autodl-tmp/AICtrack1_next_cycle_confirm_20261009
"$PYTHON" scripts/bind_confirmation_deployment.py \
  --manifest /root/autodl-tmp/AICtrack1_q0_fold2_inputs_20261002/artifacts/audits/stage2/train/samples.csv \
  --class-mapping /root/autodl-tmp/AICtrack1_q0_fold2_inputs_20261002/artifacts/audits/stage2/metadata/classes.csv \
  --label-conflicts /root/autodl-tmp/AICtrack1_q0_fold2_inputs_20261002/artifacts/audits/stage2/train/label_conflicts.csv \
  --clip /root/autodl-tmp/AICtrack1_q0_fold2_inputs_20261002/artifacts/weights/ViT-B-32.pt \
  --dataset-root /root/autodl-tmp/AICtrack1_q0_fold2_inputs_20261002/DataSetStage2 \
  --feature-cache "$OFFICIAL336_CACHE"
"$PYTHON" scripts/run_confirmation_h1.py --deadline-unix "$CONFIRM_DEADLINE" --initial-free-bytes "$CONFIRM_INITIAL_FREE"
"$PYTHON" scripts/prepare_confirmation_visual.py --selection-receipt "$OLD_FOLD_SELECTION" --selection-receipt-sha256 "$OLD_FOLD_SELECTION_SHA"
cd prepared_visual
"$PYTHON" scripts/run_confirmation_pair.py --deadline-unix "$CONFIRM_DEADLINE" --initial-free-bytes "$CONFIRM_INITIAL_FREE"
```

初次运行不带 `--resume`。中断恢复只允许同一arm的确切last/config/source；原子锁/退出确认不明时仍先人工核验。H1/视觉各自完整source冻结会在绑定真实路径/实际H1之后产生；当前文件里的“not yet trained”状态不能伪装真实H1 provenance。

磁盘预算沿 root 最新分盘决定：确认data盘至少预留3GiB剩余和完整atomic-save峰值，fresh C/J预计至少4.2GiB checkpoint+atomic峰值；每次保存用实际tensor存储量重新验预算。最终full另在其独立source/root，不使用跨盘 artifacts 软链接。

## 检查和剩余接口

本机CPU tests覆盖连接组桥接、旧query全组排除、支持不足不偷偷回填旧query、错误split泄漏、行标签/顺序绑定、H1 fixed100与full/best拒绝、oldfold last20 winner、原queue SHA、**实际调用源生成器**保证仅两匹配配置及runtime guard。源解析与四个真实CLI `--help` 通过。曾出现仅toy测试的Windows默认GBK读UTF8错误，已显式UTF8修复并通过，不等同训练算法失败。

真实H1/C/J GPU训练与其正式CPU恢复尚未执行。history 正在另制 `confirmation_diagnostic_source`，其必须使用新split、新H1、fresh C/J fixedlast20与新source identity；现有 old0c902/078控制cache不能冒用。确认只评估已经冻结的P7/P0算法，不在新query重扫超参。最终full必须等待 fresh paired 独立正向确认回执，统一记录观察标注一致率、head/mid/tail风险和实际matched runtime；不能把本包工程PASS当成涨分PASS。
