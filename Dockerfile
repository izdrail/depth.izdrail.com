# syntax=docker/dockerfile:1.7
FROM node:20.18.3-alpine AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS python-builder
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3.10-venv python3-pip git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /srv/marigold
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY marigoldv2 ./marigoldv2
COPY evaluation ./evaluation
COPY requirements-service.txt ./requirements-service.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128 && pip install -r requirements-service.txt

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive PATH=/opt/venv/bin:$PATH DEPTH_ASSETS_DIR=/opt/marigold-assets PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 curl libgomp1 libgl1 && rm -rf /var/lib/apt/lists/*
COPY --from=python-builder /opt/venv /opt/venv
WORKDIR /srv/marigold
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY marigoldv2 ./marigoldv2
COPY evaluation ./evaluation
COPY app ./app
COPY scripts/download_assets.py scripts/validate_inference_assets.py scripts/container-entrypoint.sh ./scripts/
COPY --from=frontend /build/frontend/dist ./app/static
RUN mkdir -p /opt/marigold-assets
VOLUME ["/opt/marigold-assets"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=45m --retries=3 CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1
ENTRYPOINT ["./scripts/container-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
