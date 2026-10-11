# 环境与第三方来源

本作品由源码与运行说明、完整脚本与环境配置、模型与容器三个目录构成，共享同一作品根；所有命令在作品根 `AIC-2026-43080930-两股王霸之气-复赛-张延瑞/` 执行，路径从作品根起算。完整说明见 `AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/环境与来源说明.md`，运行步骤见 `AIC-2026-43080930-可复现的代码与运行说明文档/运行说明.md`。

## 环境

正式训练与正式推理为 Linux x86_64、Python 3.12.3、Torch 2.11.0+cu128、CUDA 12.8、cuDNN 91900、RTX 4080 SUPER；线程为 OMP_NUM_THREADS=4、MKL_NUM_THREADS=4、CUBLAS_WORKSPACE_CONFIG=:4096:8，Torch 4 threads。身份见本目录 `original_environment.json` 及训练与推理回执。

`requirements-original-lock.txt` 保留原环境锁字节；`../requirements-cuda.txt` 保存直接依赖的固定版本，其中 Torch/CUDA 由运行回执确认。

CPU 侧可执行模型加载、合成输入前向与冻结后处理，身份见本目录 `portable_CPU_environment.json`。

在作品根建立独立虚拟环境并选择对应依赖。Linux CUDA：

```bash
python3.12 -m venv './runtime_outputs/venv'
source './runtime_outputs/venv/bin/activate'
python -m pip install -r 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/requirements-cuda.txt'
```

PowerShell CPU：

```powershell
py -3.13 -m venv './runtime_outputs/venv_cpu'
& './runtime_outputs/venv_cpu/Scripts/Activate.ps1'
python -m pip install -r 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/requirements-cpu.txt'
```

## 第三方来源

OpenAI CLIP、ftfy、wcwidth 的源码在 `AIC-2026-43080930-可复现的代码与运行说明文档/vendor/`，许可在同级 `licenses/`；入口自动设置导入路径，无需从网络下载 CLIP。固定版本、commit 与逐文件 SHA 见本目录 `vendor_sources.json`。其他依赖通过 pip requirements 安装。第三方清单汇总见 `AIC-2026-43080930-可复现的代码与运行说明文档/THIRD_PARTY_NOTICES.md`。

## 容器

预构建容器归档位于 `AIC-2026-43080930-可执行模型文件或容器化复现环境/container-image.tar`，构建规格位于同目录 `Dockerfile` 与 `Dockerfile.cuda`；镜像工作根为 `/app`，自带代码和模型资源。完整命令与两种布局的入口差异见同目录 `CONTAINER.md`。

官方原始图片未随作品分发，须从赛事授权入口取得后放入作品根 `DataSetStage2/`；`scripts/verify_release.py` 与 `scripts/postprocess.py` 不需要图片即可执行。
