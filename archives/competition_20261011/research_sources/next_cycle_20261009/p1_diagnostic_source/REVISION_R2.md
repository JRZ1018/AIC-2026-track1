# R2：真实入口依赖修复，保持算法与查询不变

R1 manifest SHA `f11dc78480c74e6aba8c6fc15b426591df4bac44c554ee40d965e4c2623f20c0` 已封存于 `revision_history/r1_f11dc784/source_manifest.json`，字节及SHA相同，相关旧入口/README/测试/manifest builder/CPU receipt也封存。R1未部署运行。

主代理独立审查发现 R1 `main()` 从 `training_source/scripts/mainline_guard` 导入旧版 `memory_snapshot` / `memory_reason`，实际冻结 2fee 训练包只提供训练入口/保存资源守卫，GPU推理前就会 ImportError。8项函数/toy测试没有检查实际入口导入，R1不能视作可运行验收。

R2改为独立标准库 `diagnostic_resources.py`：Windows GlobalMemoryStatusEx；Linux `/proc/meminfo` 加 cgroup v2/v1 限额，取host与cgroup可用量最小值，至少4GiB。无psutil或旧guard依赖，无GPU枚举。活跃训练包字节完全不变。

新增真正的 CPU 子模式 `run_p1_diagnostic.py --training-root <冻结2fee目录> --dependency-check`。该模式校验实际83个训练源文件、配置、wrapper全部静态 imports 的真实对象和路径、所有跨包调用形参；导入原FV2 loader和原final四视图engine；用原engine main里实际parser构造节点解析生产argv，并执行原engine `--help` 子进程。没有mock公共API，没有调用GPU/图片/checkpoint/远端资产/锁/训练主入口。生产argv抽成同一个函数，测试和真正运行不会两套分叉。

原来8项测试继续；新增真实生产入口子进程依赖检查、实际host读取和合成cgroup资源边界，以及下述同次forward feature hook，合计11项。新的源码清单重新冻结并带revision=2/previous_manifest_sha256；不静默覆盖旧f11证据。

主代理在R2未部署时明确请求可选 `--save-features`，默认关闭。只附着原模型visual forward hook，不改原4V engine、不调用model。生产观察4次B（该实际源码的布局），另验证两次2B原图/flip切分fixture。每视图raw embedding的CPU FP32 L2归一化均值再归一化，FP32保存同ID/同checkpoint的 `[29737,512]`。hook返回None、原logits严格相等、feature_file/SHA进入identity/result_manifest。供后续冻结单模型kNN，不是本阶段已实施P3。仍需root独立复核再部署。

4V、P7(.5,.75,60)、原C cache、154 joint head恢复契约、固定last20、自然/均衡支持query以及 selection_protocol SHA `b60a39fc5e966b45f97a04bae30e5d2b67fe70b83f96c01c1d97799622cd2a02` 均未改变。先由 root 独立复核新的 manifest 和dependency-check，再部署。
