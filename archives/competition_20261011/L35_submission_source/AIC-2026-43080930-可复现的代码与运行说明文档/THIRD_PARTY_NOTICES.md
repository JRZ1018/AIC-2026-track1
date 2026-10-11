# 第三方软件、模型与数据说明

以下文件路径均从作品根 `AIC-2026-43080930-两股王霸之气-复赛-张延瑞/` 出发，源码、脚本与模型按三个材料目录组织。

## 一、分发的第三方源码

| 软件 | 固定版本或提交 | 上游 | 许可证与本作品副本 |
|---|---|---|---|
| OpenAI CLIP | `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` / clip 1.0 | [openai/CLIP](https://github.com/openai/CLIP) | MIT；`AIC-2026-43080930-可复现的代码与运行说明文档/licenses/OpenAI-CLIP-MIT.txt` |
| ftfy | 6.3.1 | [python-ftfy](https://github.com/rspeer/python-ftfy) | Apache-2.0；`licenses/ftfy-NOTICE.txt` 与 `licenses/Apache-2.0.txt` |
| wcwidth | 0.8.3 | [wcwidth](https://github.com/jquast/wcwidth) | MIT 及内含 Unicode 表授权；`licenses/wcwidth-MIT.txt` |

`AIC-2026-43080930-可复现的代码与运行说明文档/vendor/` 仅含上述纯 Python 源码及 CLIP 官方 BPE 词表，保留原字节、版权与授权声明；逐文件 SHA 与来源核验在 `AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/environment/vendor_sources.json`。上游 CLIP 原文件与本作品副本只存在 Git CRLF 换行差别，两方字节 SHA 均保留；模型实现未作修改。

## 二、官方初始权重与本队训练权重

`AIC-2026-43080930-可执行模型文件或容器化复现环境/openai_clip_ViT-B-32.pt` 为规定的官方 OpenAI CLIP ViT-B/32 权重，SHA-256 为 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`，用于构造模型结构与初始化。本队训练权重 `AIC-2026-43080930-可执行模型文件或容器化复现环境/final.pt` 与 `h1_full100.pt` 由本作品代码在上述官方权重与官方训练数据上训练得到。未使用 OpenCLIP、LAION 或其他来源的预训练权重。

## 三、安装型依赖

PyTorch、Torchvision、NumPy、SciPy、Pillow、PyYAML、regex、tqdm、packaging 通过 `AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/requirements-cuda.txt` 或 `AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/requirements-cpu.txt` 安装，依赖通过对应发行包安装。原完整环境锁在 `AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/environment/requirements-original-lock.txt`。

## 四、容器镜像

`AIC-2026-43080930-可执行模型文件或容器化复现环境/container-image.tar` 为已导出的 CPU 运行镜像，内含本作品的代码与模型资源，其中的 Python 基础镜像与上述安装型依赖遵循各自发行方的许可。构建规格见 `AIC-2026-43080930-可执行模型文件或容器化复现环境/Dockerfile` 与 `AIC-2026-43080930-可执行模型文件或容器化复现环境/Dockerfile.cuda`。

## 五、数据资产

训练仅使用赛事官方 Stage 2 train（148,695 图、750 类），官方 test（37,444 图）仅用于冻结模型推理。作品中的数据资产为官方文件身份、类别映射、固定划分及训练派生查询元数据；官方原始图片未随作品分发，须从赛事授权入口获取。作品不含测试标签，验证标签均来自官方训练集。取得数据后的目录组织与运行命令见 `AIC-2026-43080930-可复现的代码与运行说明文档/运行说明.md`。
