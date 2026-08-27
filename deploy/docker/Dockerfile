# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm

ARG TORCH_VERSION=2.10.0
ARG TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    MOBO_PROJECT_DIR=/app \
    MOBO_DATA_DIR=/app/data \
    MOBO_API_HOST=0.0.0.0 \
    MOBO_API_PORT=5000

WORKDIR /app

# 先复制依赖描述，充分利用 Docker 构建缓存。torch 使用 CPU wheel，避免引入 CUDA 运行时。
COPY pyproject.toml README.md ./
COPY requirements/runtime.txt requirements/server.txt ./requirements/
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_CPU_INDEX}" \
    && python -m pip install -r requirements/runtime.txt -r requirements/server.txt

# 包含 API、代理模型、优化、提取和 DEFORM 批处理源代码；DEFORM 本体不在 Linux 镜像中。
COPY . .
RUN python -m pip install . --no-deps \
    && mkdir -p /app/data /app/logs /tmp/matplotlib \
    && python -m pip check \
    && groupadd --system mobo \
    && useradd --system --gid mobo --home-dir /app --no-create-home mobo \
    && chown -R mobo:mobo /app/data /app/logs /tmp/matplotlib

USER mobo

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import json, os, urllib.request; port=os.environ.get('MOBO_API_PORT', '5000'); data=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3)); assert data['code'] == 0" || exit 1

CMD ["gunicorn", "--config", "deploy/docker/gunicorn.conf.py", "mobo.api.app:create_app()"]
