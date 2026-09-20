# Stage 2 数据到手头 4 小时任务单

状态：`execution_enabled: false`（只允许复制后填写，不可直接执行）

## 0:00–0:30 来源与原包

- 官方页面/通知时间：`<value>`
- 下载责任人/复核人：`<A>/<B>`
- 原包文件名、字节数、SHA-256：`<values>`
- 下载/解压唯一目录：`<DataSetStage2 paths>`
- D: 实测余量与 B14 决策：`<GO/NO-GO>`
- 平台资格、下载入口、提交入口：`<confirmed/stop>`

停止：来源不明、哈希/大小异常、容量门禁未过、账号资格或入口不明。

## 0:30–1:30 只读审计

```powershell
<python> scripts\audit_dataset.py --config <stage2-data-config> --source train
<python> scripts\audit_dataset.py --config <stage2-data-config> --source test
```

- train/test/class 实测数量：`<values>`
- 坏图、重复、冲突、层级与 basename 冲突：`<values>`
- train/test manifest SHA-256：`<values>`

停止：数量或目录异常无法由官方材料解释；坏图策略未冻结。

## 1:30–2:00 映射与提交契约

- 官方映射来源/哈希：`<value>`
- ID 正则：`\d{4}`（仅在样例再次确认后保留）
- 首列：`<basename/relative_path>`；表头/空格/顺序：`<values>`
- 合成契约负向测试：`<passed>`

停止：需要按目录排序猜 ID、映射重复/缺失、路径模式碰撞。

## 2:00–3:00 split 与长尾级别

```powershell
<python> scripts\build_split.py --config <stage2-data-config> --manifest <train-manifest> --output <split>
<python> scripts\report_long_tail_readiness.py --manifest <train-manifest> `
  --class-mapping <mapping> --output-json <report.json> --output-md <report.md>
```

- split SHA-256：`<value>`
- B09 判定与触发指标：`<L0/L1/L2 + evidence>`
- 首轮允许矩阵：`<H1/CE/AC/conditional LT>`

## 3:00–4:00 冻结首个 H1 任务

- 从 `docs/templates/stage2_task_order.md` 复制完整任务单。
- 把三个 Stage 2 模板的数量、manifest/split/mapping/权重哈希写实后另存正式配置。
- 运行配置、checkpoint provenance、训练锁、磁盘、电源和 CUDA 门禁。
- A 启动唯一训练；B 只复核。任何占位符或 `execution_enabled: false` 未处理即停止。
