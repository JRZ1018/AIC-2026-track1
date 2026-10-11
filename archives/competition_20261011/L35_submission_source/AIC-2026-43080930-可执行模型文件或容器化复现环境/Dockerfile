# Build context is the complete submitted work root, with all three categories.
ARG PYTHON_BASE=python:3.13-slim@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f
FROM ${PYTHON_BASE}
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4
WORKDIR /app
COPY ["AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/requirements-cpu.txt", "/tmp/requirements-cpu.txt"]
RUN python -m pip install --no-cache-dir --only-binary=:all: -r /tmp/requirements-cpu.txt
COPY . /app
CMD ["python", "-B", "AIC-2026-43080930-完整的训练、验证、推理脚本及环境配置说明/scripts/infer.py", "--help"]
