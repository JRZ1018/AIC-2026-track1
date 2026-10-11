# AIC-2026-43080930 完整训练、验证、推理脚本及环境配置说明

本目录提供训练、验证、推理与后处理的全部脚本，以及环境配置与数据身份元数据。命令均在作品根 `AIC-2026-43080930-两股王霸之气-复赛-JRZ1018/` 执行，路径均从作品根起算。

| 入口或配置 | 用途 |
|---|---|
| `scripts/train.py` | 官方特征提取、分类头初始化训练、全量或持出联合训练、断点恢复 |
| `scripts/validate.py` | 固定训练派生持出集的四视图、图平滑、类别均衡与配对统计 |
| `scripts/infer.py` | 加载最终模型，从测试图片生成 CSV/ZIP |
| `scripts/postprocess.py` | 从冻结 logits 与特征精确复现后处理 |
| `scripts/verify_release.py` | 文件 SHA、模型身份与提交结果格式校验 |
| `portable_training/` | 训练实现的便携运行入口 |
| `configs/`、`data/` | 固定配置、类别映射、训练/测试清单与验证划分 |
| `environment/`、`requirements-cuda.txt`、`requirements-cpu.txt` | 环境身份、原依赖锁与安装版本 |
| `evidence/`、`reproduction_cache/` | 训练与推理执行记录、冻结推理缓存 |

完整训练和验证步骤见本目录 `TRAINING.md`；环境安装与逐文件来源见本目录 `环境与来源说明.md`；完整推理步骤见 `AIC-2026-43080930-可复现的代码与运行说明文档/运行说明.md`。

脚本使用 `AIC-2026-43080930-可复现的代码与运行说明文档/src/`、`vendor/` 与 `archived/` 中的实现，并加载 `AIC-2026-43080930-可执行模型文件或容器化复现环境/` 中的模型。

```bash
python 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/train.py' --help
python 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/validate.py' --help
python 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/infer.py' --help
```
