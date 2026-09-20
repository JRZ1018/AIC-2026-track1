# Stage 2 冻结任务单

> 每次正式训练、恢复、推理或出包各用一张任务单。所有占位符填写完成并由执行人与复核人
> 共同核对后才可执行；不得用“最新代码”“默认配置”等可变描述。

## 身份与目标

- 任务 ID：`<YYYYMMDD-owner-stage2-method-seed>`
- 任务类型：`audit / train / resume / evaluate / infer / package`
- 本次唯一问题：`<one-question-only>`
- 主负责人：`<name>`
- 执行人：`<name>`
- 独立复核人：`<name>`
- 替补：`<name>`

## 冻结输入

- Git commit（40 位）：`<sha>`
- 工作区要求：`clean`
- 唯一配置：`<path>`
- 配置 SHA-256：`<sha256>`
- competition stage：`stage2`
- train manifest SHA-256：`<sha256-or-N/A>`
- test manifest SHA-256：`<sha256-or-N/A>`
- split SHA-256：`<sha256-or-N/A>`
- class mapping SHA-256：`<sha256>`
- OpenAI CLIP 权重 SHA-256：`<sha256>`
- 输入 checkpoint 路径/哈希：`<path-and-sha256-or-none>`

## 执行边界

- 唯一命令：`<copy-paste-command>`
- 唯一输出目录：`<path>`
- 预计开始/结束：`<time-range>`
- 预计 GPU/峰值显存：`<device-and-MiB>`
- 预计磁盘增量：`<GiB>`
- 允许续跑：`yes/no`
- 唯一续跑命令：`<command-or-none>`
- 成功条件：`<measurable-condition>`
- 停止条件：`<measurable-condition>`
- 禁止事项：改参数、读 test 做选择、覆盖目录、并发训练、任务外续跑。

## 回传与验收

- 必须回传：command、config、environment、git state、metrics、日志、checkpoint 及 SHA-256。
- 实际状态：`success / resumable / failed / stopped`
- 实际起止时间：`<time-range>`
- 峰值显存/磁盘：`<values>`
- 关键指标：`<values-or-N/A>`
- 产物清单与哈希：`<manifest-path>`
- 异常与日志：`<path-or-none>`
- 复核结论：`accepted / rejected / needs-investigation`
- 台账行：`<experiments/index.csv-or-submissions/ledger.csv-row>`
