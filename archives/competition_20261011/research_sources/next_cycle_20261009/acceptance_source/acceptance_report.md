# FV2 本机恢复、提交契约及运行守护验收（2026-10-09）

本任务仅写本目录。没有服务器访问、真实训练、GPU计算或test图像前向，没有改历史资产/receipt、其他代理源码或原始数据。

## 已实际完成

- [CPU恢复凭据](D:/aicomp/AICtrack1/artifacts/next_cycle_20261009/acceptance_source/cpu_restore_549bd5ce336bba4c.json)：完整重建官方CLIP视觉+全量H1+FV2最后轮状态；152个视觉张量、87,849,216参数均与last/final逐项一致且有限。AdamW152及H1 Adam2每个moment、scheduler/scaler、CPU RNG、AC20及148,695原始ID/标签集合检查通过。两张原train的SHA验签、336预处理和CPU前向[1,750]通过。
- 实际Python3.12.10、torch2.11.0+cu128，2 CPU线程；恢复主体6.42秒，进程峰值2,981,023,744B≈2.78GiB。CUDA API禁用，CUDA RNG只验证序列化，不声称GPU位级续训。重复命令返回`reused_complete=true`，没有再次恢复模型或前向。
- 238个回传文件重新逐SHA；full103和delivery88源码按各自原始字节SHA全部通过。[源码版本清单](D:/aicomp/AICtrack1/artifacts/next_cycle_20261009/acceptance_source/fv2_source_inventory.json)包含完整源文件、主仓对照和真实metrics SHA。full source manifest=`bc706203a179b6e9f420af3088e92faf54dc40a66751bdddfc92fed44700819c`；delivery=`ff9dc078be4e297fb6459e27d191c33bffa830c2788cce595370ad3ff179556b`。
- [包验收](D:/aicomp/AICtrack1/artifacts/next_cycle_20261009/acceptance_source/package_acceptance_a5594f1be75006fd.json)：37,444行/750类、完整文件名与顺序、无表头/BOM、UTF-8 LF、官方四位ID、ZIP仅根CSV且字节一致、CSV/ZIP/manifest/模型身份通过。它不代表线上提交或新增分数。
- `test_acceptance.py`的9项合成破坏测试通过：错行序、非法类、缺行、BOM、嵌套ZIP、错误checkpoint、缺来源及两种NumPy RNG安全格式。通过结果为`acceptance_contract_tests_<test SHA前16>.json`；旧失败结果保留，不能把它当当前PASS。

模型身份：final=`85f4a7c50da95e78397ed0a1067632fe0fbfc410b46a4cd35bcf61df621cb8b0`；last=`226557f79475b8e2f02c69738972aad2d75c87b82bb14150ce1f7076daaaa450`；全量H1=`fa4623670a6f48731cf54e62dfb5ce5df44b232f4530f6c13b886efe0579e605`。这些不是后续P1候选的身份。

## 复用资产与缺口

[FV2 fold input identity](D:/aicomp/AICtrack1/artifacts/next_cycle_20261009/acceptance_source/fv2_input_identity.json)和[逐行查询清单](D:/aicomp/AICtrack1/artifacts/next_cycle_20261009/acceptance_source/fv2_rows.csv)已交P0/P1代理复用。

- 缓存：`D:\aicomp\stage2-server-transfer\fv2_results_20261008\diagnostic\artifacts\diagnostics\fullvisual\fused_logits.pt`，FP16[29737,750]，SHA=`4813ad6727624f3072ff6e5d28d8ecaa9b4a9e919a5e0b489946dfeea08180b9`；有targets、sample_ids、train_class_counts以及四视图来源。顶层report的`view_protocol`描述2V，正确4V字段是`four_view_protocol`，bundle自身view_protocol也是4V。
- 实际绑定固定epoch20的`last.pt`，SHA=`078bdae1654cecc076d19e8e67b2401c6a47b985d90c91f7b6ec25293002b6c9`。上游把角色写`final`，但actual last文件/metrics.last_checkpoint_sha/CPU receipt一致。trainer metrics.selected_checkpoint另为best，不能拿其SHA绑定此缓存。
- 官方manifest/split查询顺序、labels、训练类计数、query ID唯一、train/query精确与近重复组零交叉均实际检查。fold H1=`45266a98329a59f708b12f394775640e032e50be8c44077a1037048f28c88753`，100轮，仅118,958 fold train，正向row binding核准，没有query优化/选模。
- [特征清单](D:/aicomp/AICtrack1/artifacts/next_cycle_20261009/acceptance_source/feature_cache_inventory.json)列11个Stage2本地缓存。分片格式为features/labels/sample_indices/sample_ids/relative_paths/partitions；一个实际shard为FP16[4096,512]。多数为官方冻结CLIP224/336，另有旧C2微调特征。它们不是FV2微调特征；其旧split metadata必须正向重绑定，不能直接用于FV2图平滑或新fold H1来源。
- FV2本机diagnostic未找到微调视觉embedding或原FP32逐行4V argmax。只有FP16 logits时不能伪造“原FP32预测”；P3需要同一合规fold checkpoint新增提取或可验签回传。只回传metadata/complete并不意味着37个shard已在本机。

## 运行守护及重入

服务器训练应继续使用冻结P1源码的`run_serial_queue`及统一`/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json`。该冻结queue识别`train_n1_peft.py`；主仓旧queue仅识别三个旧入口，不能把它替换进P1调度。P1实际入口为n1，复用已验fold H1，不另起H1入口；这一点已与P1负责代理交叉确认。后续换入口名必须同步process-probe识别范围。

原子O_EXCL获取锁后再全系统训练进程探测；探测失败或未知锁均拒绝启动，lease只释放匹配token。不得自动删陈旧锁、终止未知PID或把所有Python当训练器。监测身份应绑定PID/父PID/创建时间或Linux startticks/完整argv/锁token，并维持原截止与累计预算。

本机恢复不跑旧delivery.main：其中有硬编码远端锁和test一次性入口。`accept_fv2.py`只在本目录留独立input+script SHA journal；完成凭据可复用；未完成stage须先看失败和进程身份，不能盲重启。工程修复不重置首次CPU stage的600秒截止。

历史`LOCAL_CPU_STAGE.json`保留原事实；本次新增CPU恢复凭据在本目录，不追改过期旧receipt。首次预检发现venv无psutil，改用Windows标准API获取内存/RSS/自身创建时间。实际H1 NumPy RNG是安全dict、visual是安全tuple；已分别适配，空AMP scaler按disabled恢复。一次fixture失败来自Python tempfile的700模式在Windows沙箱ACL下不可访问，改用继承ACL的普通mkdir和精确非递归清理，旧空失败目录保留。

## 可复用命令

```powershell
& D:\aicomp\AICtrack1\.venv\Scripts\python.exe -B D:\aicomp\AICtrack1\artifacts\next_cycle_20261009\acceptance_source\accept_fv2.py package
& D:\aicomp\AICtrack1\.venv\Scripts\python.exe -B D:\aicomp\AICtrack1\artifacts\next_cycle_20261009\acceptance_source\accept_fv2.py cpu
```

新的提交包用`package --package <新目录> --expected-checkpoint-sha256 <事前冻结的新final SHA>`。输出凭据按manifest SHA分名，不能覆盖旧包结果。该命令校验包契约；新的P1模型恢复按旁边P1契约验收，不能沿用FV2固定152断言。
