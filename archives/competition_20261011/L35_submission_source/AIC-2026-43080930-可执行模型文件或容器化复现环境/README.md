# AIC-2026-43080930 模型文件

本目录属于作品根 `AIC-2026-43080930-两股王霸之气-复赛-JRZ1018/`；以下宿主机路径均从该根起算。

## 一、模型文件

| 模型路径 | 角色 | SHA-256 |
|---|---|---|
| `AIC-2026-43080930-可执行模型文件或容器化复现环境/final.pt` | 最终视觉参数与 750 类分类头，正式推理加载的 checkpoint | `ebc130838f898e3f5bc5ad50c9b4d05966a7ce0afbe9bc4f08511b4c2f21d9f6` |
| `AIC-2026-43080930-可执行模型文件或容器化复现环境/last.pt` | 同一次训练第 20 轮的完整恢复状态：AdamW、余弦日程、AMP scaler、逐样本置信度权重与随机状态 | `5325982fad83513b11e6f2c9e94537272ae5f7ded295e46b80948bf9fea3532d` |
| `AIC-2026-43080930-可执行模型文件或容器化复现环境/openai_clip_ViT-B-32.pt` | 规定的官方 OpenAI CLIP ViT-B/32 结构与预训练权重 | `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` |
| `AIC-2026-43080930-可执行模型文件或容器化复现环境/h1_full100.pt` | 全部官方训练样本上固定 100 轮训练得到的分类头初始化 | `fa4623670a6f48731cf54e62dfb5ce5df44b232f4530f6c13b886efe0579e605` |

正式推理先构造官方 CLIP 图像编码器，再严格加载 `final.pt`。模型包含 154 个训练张量（视觉 152 + 分类头 2）、88,233,966 个参数，可用 Python/PyTorch 直接加载。加载合约、来源与逐字节校验由 `AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/verify_release.py` 完成。

## 二、预构建容器

`AIC-2026-43080930-可执行模型文件或容器化复现环境/container-image.tar` 是已导出的 Linux amd64 CPU 镜像，自带完整执行链与全部模型资源，可直接加载，不要求安装作者的 Python 环境。校验值在 `container-image.tar.sha256`。

```bash
docker load --input './AIC-2026-43080930-可执行模型文件或容器化复现环境/container-image.tar'
docker run --rm aic-l35-cpu python -B scripts/infer.py --help
```

镜像工作目录为 `/app`，内部入口为 `scripts/infer.py`、`scripts/train.py`、`scripts/validate.py`、`scripts/verify_release.py`、`scripts/postprocess.py`；这些是容器内部路径，宿主机路径仍以作品根为起点。完整的验签、缓存重放、图片推理与重新构建命令见本目录 `CONTAINER.md`。

不使用容器时，直接按 `AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/环境与来源说明.md` 安装依赖后执行同一批脚本即可。

```bash
python 'AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/infer.py' --synthetic-smoke --device cpu --workers 0 --output './runtime_outputs/cpu_smoke'
```
