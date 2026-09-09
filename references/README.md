# 上游参考源码

`references/upstream/` 保存论文作者或官方仓库的本地只读代码快照，便于后续核对实现。该目录整体被主项目 `.gitignore` 排除，不属于比赛代码，也不能成为正式运行时依赖。

## 使用规则

- 只通过 `scripts/manage_upstream_references.py` 同步。
- 每个仓库检出 `docs/upstream_registry.yaml` 中锁定的完整 commit SHA。
- 不直接 `git pull`，更新版本前先重新审计并修改注册表。
- 不初始化 submodule，不下载 Git LFS 对象、数据集或预训练权重。
- 不在上游目录运行训练/安装脚本，不把它加入 `PYTHONPATH`。
- 需要复用时，在 `src/aictrack1/` 中做最小、可测试的实现，并按许可证记录来源。
- 无许可证的 LogitClip、Co-teaching 只允许阅读；不得复制、修改或分发其源码。

## 命令

```powershell
D:\aicomp\python312\python.exe scripts\manage_upstream_references.py sync
D:\aicomp\python312\python.exe scripts\manage_upstream_references.py check
```

`sync` 只获取注册表指定的单个 commit；若本地参考仓库存在改动，会拒绝覆盖。`check` 不联网，只检查目录、远程地址、HEAD 和工作区状态。
