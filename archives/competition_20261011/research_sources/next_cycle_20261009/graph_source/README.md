# P3条件图平滑工程包

仅工程与小型CPU验收。没有运行真实29,737节点图，没有服务器/GPU/test访问，也没有修改训练、全量训练、7bb或fresh诊断冻结包。整个模块自行实现，LaplacianShot仅为docs240列出的冻结特征推理背景，不复制上游代码或引入额外骨干。

## 已冻结的算法

事前协议是 `graph_protocol.json`。候选严格为k{10,20}×lambda{.1,.3}四组；三轮同步残差 `Qnext=(1-lambda)P+lambda WQ`，每轮锚定原P。输入P来自原四视图分数的native温度1 softmax。输出为log概率回到原score单位，P7只使用既有.5/.75/60一次，不额外扫描温度。

每个固定查询协议在自身唯一行及闭集class columns上单独建图，不能将一张图复制50次，也不能把全量模型反评train派生持出。原始embedding为同一个frozen fixed-last20模型已有四次forward的FP32归一化均值。模型、头、source/config/checkpoint/split/manifest/mapping、文件SHA和sample ID顺序全绑定。不加载第二模型、不更新参数、不读标签构图；缓存targets只用于行身份核验和预测完成后的observed-label evaluation。

分块cosine top-k排除self与zero embedding；同分按原query row较小者优先，不加epsilon改相似度。只保留严格正的互惠边，将两方向权重平均后得到对称邻接，再row normalize。row-stochastic W一般不对称；对称性指未归一化邻接。k20仅计算一次，k10取其前10再重检互惠，不能截取k20的互惠图冒充k10。

raw原分数top1−top2>=ln4（native odds>=4:1）和zero-degree行每轮恢复原P；输出这些行恢复**逐位原logits**。报告zero feature/degree、degree histogram、正向边/互惠边、实际similarity block峰值、protected行数、原/末零概率、float32_tiny floor条目、raw保护翻转0与margin差0。graph+P7的全局bias仍可能改变protected行相对于base+P7的预测，单独统计protected正确→错误/错误→正确，供root审核，不能声称其也被raw保护完全禁止。

四候选按既定balanced_supported_unique的**固定P7** observed Top1排序，同分先lambda低、再k低。raw与natural均完整诊断，不加natural>=0或CI下界>0硬门。选一组后只比较base/raw、base/P7、chosen_graph/raw、chosen_graph/P7四组合；复用已算预测，不把单项收益相加。负向候选不生成可确认的正向status。独立确认只执行原选择的一点，不重新扫四组。

## 资产绑定与运行

必须先有真实的 `fused_features.pt`、`fused_logits.pt`、`input_identity.json`、原rows/FP32预测回执，以及实际checkpoint/config/training与diagnostic source。特征schema与7bb/fresh诊断一致：`aictrack1.stage2_train_derived_4v_features.v1`。旧7bb support-only unique queries可复用；fresh确认需其新split frozen queries。资产缺失不会生成可启动占位输入。

```sh
python bind_graph_assets.py \
  --features /actual/diagnostic/fused_features.pt \
  --logits /actual/diagnostic/fused_logits.pt \
  --identity /actual/diagnostic/input_identity.json \
  --checkpoint /actual/training/artifacts/experiments/P1_J5/last.pt \
  --config /actual/training/configs/experiment/p1_J5.yaml \
  --training-source /actual/training/source_manifest.json \
  --diagnostic-source /actual/diagnostic_source/source_manifest.json \
  --query-directory /actual/diagnostic_source/inputs/frozen_queries \
  --output /actual/p3/graph_assets.json
python run_graph.py --stage explore --assets /actual/p3/graph_assets.json \
  --assets-sha256 ASSET_SHA --output /actual/p3/exploration \
  --budget-ledger /actual/p3/aggregate_budget.json
python run_graph.py --stage confirm --assets /actual/p3/fresh_assets.json \
  --assets-sha256 FRESH_ASSET_SHA --output /actual/p3/confirmation \
  --choice /actual/p3/exploration/graph_choice.json --choice-sha256 CHOICE_SHA \
  --budget-ledger /actual/p3/aggregate_budget.json
```

above `/actual`为真实已完成资产位置的命令模板，不是已存在文件或可直接launch的占位receipt。CPU默认并固定2线程（OMP/MKL/OpenBLAS与Torch），similarity block上限64MiB/默认256行，永不构造N×N稠密相似度/邻接。29,737行时一块约29MiB；CSR图是O(Nk)，概率矩阵是O(NC)，不能把blocked graph宣称整个算法无O(NC)内存。原cache若为FP16 logits，仅转FP32算，报告与原forward FP32 argmax不一致数，不能声称恢复丢失精度。

explore+confirm共用一份两小时wall ledger，最多7200秒累计，不能每候选/协议重置。每块/迭代检查剩余时间；中断会计入已耗wall，异常保留partial report，不生成正式complete choice。预算lease用独立原子CPU锁，不触碰GPU训练canonical lock。旧CPU进程退出不明/lock不明时不自动清理。不要为恢复重设spent_wall_seconds；真实图无中间block恢复，失败重跑须扣剩余原预算。

输出：候选/基线raw+P7 paired报告、每类独立支持/召回、train-count strata、重复组bootstrap 500draws、margin/保护风险、一个chosen score cache与predictions、完整result_manifest。choice必须绑定complete report+manifest；confirmation限定seed20261009新split SHA57189c...658b7。确认报告只声明primary observed gain，automatic_deployment_authorized=false，必须由root审阅高margin错翻和尾类风险；不能把主指标正值自动当最终准入PASS。只允许heldout研究，本包没有正式test入口；最终test流程须另由root根据已冻结选择集成。

## 本机验证

```powershell
& D:\aicomp\python312\python.exe .\tests\test_graph.py
```

科学fixture覆盖directed非互惠边剔除、对称原邻接与row normalize、零向量/零度保留、原行tie与不同block/标量参考一致、同步三轮原P残差、native T1与T.5的差异、log概率回score/P7仅一次、ln4逐位保护、概率零/floor统计、无label anchor/参数更新、同checkpoint/ID order/额外forward/test拒绝、累计预算与过期、真实CLI和小型注入cache运行器四候选。注入cache最大80行，仅验工程，不能作为赛题涨分证据。

部署按 `source_manifest.json.files` 严格复制文本和source_manifest，可附CPU_TEST_RECEIPT；排除artifacts/pycache、fixture输出、checkpoint/数据。真实大图吞吐、内存峰值、候选收益和独立确认尚未验证。原7bb、2fee、677 manifest保持原样。
