# P1 可训练分类头：固定last诊断及最终包最小修订

已与history_audit、recent_audit对齐。8项CPU toy破坏测试通过，详`p1_restore_toy_tests_<test SHA前16>.json`；测试只用小合成模型，不能代替未来真实P1 checkpoint验收。本任务不访问服务器，也未改P1冻结源码。

## 核心契约

J5/J20由最后轮`last.pt`诊断，epoch20、mode=full_visual、fixed-split train118958/query29737；不能替换best或第8轮部分结果。head initializer SHA为fold H1 `45266a...88753`，split=`0c902e...584c`，config各自严格SHA。152视觉+完整`classifier.linear.weight`和`classifier.linear.bias`=154可学习张量、88,233,966参数。C的152张量和既有缓存保持其原身份。

“head_initializer_sha256”只是初始头的溯源，不代表最终分类器参数；最终两项必须来自新last/final的trainable_state。

## 最小实现变化

1. **现有v6 loader无需重写**：先安全读取H1用于结构/归一化/来源，再调用`configure_classifier_trainability_from_state(head,state)`，构造PEFT模型，`load_trainable_state_dict`严格恢复全部154。先逐项确认classifier两张量与checkpoint相等，再统一冻结requires_grad并eval。不要在完整恢复前永久冻结头，不要把H1权重重新覆盖到已恢复模型。
2. **外层audit改152常量**：按配方明确C152、J154，不接受任意张量数；名字、shape、dtype、有限性和参数总数都校验。新head相对H1的变化量记录为诊断，精确head一致性以新checkpoint为准。
3. **优化器恢复保留两个group**：原152 visual group→2 classifier group，按训练时模型参数顺序恢复，逐moment核准；scheduler20的base_lrs应对应visual与head LR。不能用FV2的一组AdamW断言验收J。
4. **final导出保留154完整状态**：从已验20轮last直接保存新final及来源元数据，核准final/last所有学习张量（尤其头）相等、配置/CLIP/映射/数据来源相同。不得过滤成`visual.*`的152，也不得在推理全部freeze后重新调用trainable_state_dict得到空状态。
5. **正式全量阶段与fold阶段分开**：最终全148695的H1和split/manifest来源按该全量链核验，不能硬套fold H1 `45266...`到完整全量模型。如果复用当前合法全量H1，应记录实际`fa462...`及全量row binding，不冒称重新训练。
6. **新delivery绑定新freeze**：expected final SHA、head initializer SHA、config SHA及selection报告均更新为实际新链；无新参数的P0也需新后处理来源/算法JSON。旧P7选择报告及硬编码152 audit不能简单复用。CSV仍37444/750/官方ID，包校验脚本可直接接受新expected checkpoint SHA。

## 接受端工具及测试范围

[accept_p1_state.py](D:/aicomp/AICtrack1/artifacts/next_cycle_20261009/acceptance_source/accept_p1_state.py)提供新head严格恢复与两组optimizer逐项恢复检查。它使用冻结P1的实际engine.peft/models.peft API，不改该源码。真实模型还须独立核准官方CLIP重建、154名字/shape、输入SHA、scheduler/scaler/RNG/AC和两张train CPU前向；本工具的轻量export函数不是单独的完整架构/来源证明，应在严格模型恢复之后使用。

8测试覆盖：安全序列化后新head覆盖H1、只有visual的错误导出、缺head bias、错head shape、非有限head、epoch8、两组optimizer及坏group、完整154导出保留head。首轮仅有一条测试预期错误消息的正则不匹配，修复测试文本后8/8通过，旧失败记录保留；没有将真实模型失败改成PASS。
