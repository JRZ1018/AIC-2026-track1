# 独立 fresh confirmation 回执生成器

这是 bootstrap `DEPLOYMENT_FILES.json` 之外的追加源包。不修改已部署H1/visual训练源码，不重置截止时间，不启动训练或推理。

真实 fresh C/J 训练、4V/P7配对诊断和root独立风险审查全部完成后，由root在已验收Python环境运行：

```bash
python write_confirmation_receipt.py --request ACTUAL_CONFIRMATION_REQUEST.json --output FRESH_CONFIRMATION_RECEIPT.json
```

Request schema：`aictrack1.p1.fresh_confirmation.request.v1`，`selected_arm`是先前冻结的J5/J20。`evidence`需19个真实绝对路径，可以是字符串或 `{path,sha256}`；writer总是自行重hash真实文件，有给定sha则必须相符。

证据键：`checkpoint_C/config_C/identity_C/checkpoint_audit_C/checkpoint_J/config_J/identity_J/checkpoint_audit_J/paired_report/training_source/diagnostic_source/selection_protocol/frozen_postprocess/split/h1/h1_receipt/runtime_C/runtime_J/root_review`。

writer核对全部源文件SHA、fresh split57189与train/query/旧选参query隔离、真实train-only H1100与源链、仅头LR不同的C/J配置、runtime receipts绑定及4线程相等、实际CPU safe/mmap fixedlast20张量/优化器组/20轮scheduler/全global AC ID标签，以及history新audit/identity/report来源。它不构建CLIP、不前向、不使用GPU或test。完整模型恢复的正向证明来自另行真实diagnostic audit，writer也复核checkpoint实际状态。

输出schema `aictrack1.p1.fresh_confirmation.v1`。硬门为既定均衡unique/P7主指标 `primary_delta_pp>0` 与完整来源/恢复通过。**不增加 natural>=0 或CI下界>0门禁**；natural指标须有限并与原report一致。source保留自然750、分层、尾类、margin与CI供root审查。`root_review`必须真实文件 `{status:"pass",reviewer:"root",reason:"非空的独立审查理由"}`，并以path/SHA进evidence；CI跨零、单seed等不确定性不得被回执包装成稳定收益，writer固定 `stable_gain_claim=false`。

实际H1/C/J/paired结果未产生时，不生成fake PASS。CPU测试只测试合成gate/文件，不代表新模型已通过。

Bootstrap 的实际传输白名单是已冻结的 `confirmation_source/DEPLOYMENT_FILES.json`（b41a...，99文件）；bootstrap README 中一处旧称 `CONFIRMATION_DELIVERY.json` 应按这个实际文件名理解。此追加包另有自己的 `receipt_source/source_manifest.json`，不能覆盖已绑定的H1 canonical source manifest。
