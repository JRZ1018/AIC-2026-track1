# AIC-2026-43080930 容器加载与复现

全部命令在作品根 `AIC-2026-43080930-两股王霸之气-复赛-JRZ1018/` 执行，涉及的宿主机路径均从该根起算。

## 一、加载预构建镜像（推荐）

`AIC-2026-43080930-可执行模型文件或容器化复现环境/container-image.tar` 是已导出的 Linux amd64 CPU 镜像，自带完整代码、四个模型、元数据与冻结缓存，可直接加载运行，不要求安装作者的 Python 环境。归档校验值在 `container-image.tar.sha256`。

```bash
docker load --input './AIC-2026-43080930-可执行模型文件或容器化复现环境/container-image.tar'
docker run --rm aic-l35-cpu python -B scripts/verify_release.py
docker volume create aic-l35-output
docker run --rm --mount type=volume,source=aic-l35-output,target=/outputs aic-l35-cpu python -B scripts/infer.py --synthetic-smoke --device cpu --workers 0 --output /outputs/cpu_smoke
docker run --rm --mount type=volume,source=aic-l35-output,target=/outputs aic-l35-cpu python -B scripts/postprocess.py --output /outputs/cache_replay
```

镜像工作目录为 `/app`，容器内入口为 `scripts/infer.py`、`scripts/train.py`、`scripts/validate.py`、`scripts/verify_release.py`、`scripts/postprocess.py`。`/app`、`/outputs` 与容器命令中的 `scripts/` 是容器内部路径，宿主机文件路径仍以作品根为起点。

取得赛事授权的官方测试图片后，可在作品根执行完整图片推理：

```bash
docker run --rm --mount "type=bind,source=$PWD/DataSetStage2,target=/app/DataSetStage2,readonly" --mount type=volume,source=aic-l35-output,target=/outputs aic-l35-cpu python -B scripts/infer.py --test-root DataSetStage2/test --device cpu --batch-size 64 --workers 0 --output /outputs/predictions_cpu
```

`$PWD` 在 PowerShell 和 bash 中均表示当前作品根。输出使用命名卷保存，复跑请换新的输出子目录。

## 二、从作品目录重新构建镜像

构建上下文为作品根，`AIC-2026-43080930-可执行模型文件或容器化复现环境/Dockerfile` 与 `Dockerfile.cuda` 为构建规格，排除规则位于与各自 Dockerfile 相邻的 `Dockerfile.dockerignore` 与 `Dockerfile.cuda.dockerignore`。

```bash
docker build -f './AIC-2026-43080930-可执行模型文件或容器化复现环境/Dockerfile' -t aic-l35-cpu .
docker run --rm aic-l35-cpu python -B 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/infer.py' --help
docker run --rm --mount type=volume,source=aic-l35-output,target=/outputs aic-l35-cpu python -B 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/infer.py' --synthetic-smoke --device cpu --workers 0 --output /outputs/cpu_smoke
```

新镜像内部保留作品根的三目录结构，入口路径因此与预构建镜像不同。`Dockerfile.cuda` 提供 CUDA 版本的重建规格；CUDA 训练与推理环境见 `AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/环境与来源说明.md`。

## 三、不使用容器

直接按 `AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/环境与来源说明.md` 安装对应 requirements，然后从作品根执行 `AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/` 下的入口，功能与容器一致。
