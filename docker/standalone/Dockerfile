FROM node:22-bookworm-slim AS node
FROM ghcr.io/astral-sh/uv:0.6.14 AS uv
FROM python:3.12-slim-bookworm AS source
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
ARG AKASHIC_REVISION
WORKDIR /source
RUN git init && git remote add origin https://github.com/kachofugetsu09/akashic-agent.git \
    && git fetch --depth=1 origin "${AKASHIC_REVISION}" && git checkout --detach FETCH_HEAD

FROM python:3.12-slim-bookworm AS build
COPY --from=node /usr/local /usr/local
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
# 只按锁文件和 SDK 内容建立依赖层；改 README 或业务源码时无需重装依赖。
COPY --from=source /source/docker/host-runtime/requirements.lock /tmp/requirements.lock
COPY --from=source /source/sdk/python /tmp/sdk
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock \
    && /opt/venv/bin/pip install --no-cache-dir --no-deps /tmp/sdk
COPY --from=source /source /source
WORKDIR /source
RUN /opt/venv/bin/python scripts/build_plugin_distribution.py --revision HEAD --output /opt/distribution \
    && /opt/venv/bin/python scripts/distribution_runtime.py prepare-wheels --distribution /opt/distribution \
    && /opt/venv/bin/python -c "from pathlib import Path; from scripts.install_plugin_distribution import extract_core; extract_core(Path('/opt/distribution'), Path('/opt/core'))"

FROM python:3.12-slim-bookworm
ARG AKASHIC_REVISION
LABEL org.opencontainers.image.source="https://github.com/kachofugetsu09/akashic-agent" \
      org.opencontainers.image.revision=$AKASHIC_REVISION
# 后半行是插件按需下载的 Chromium（visuals、design、screenshot）运行所需的系统库与中文字体，与生产镜像同一组。
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates zsh curl libstdc++6 libatomic1 ripgrep \
        libasound2 libatk-bridge2.0-0 libatk1.0-0 libatspi2.0-0 libcairo2 libcups2 libdbus-1-3 libdrm2 libgbm1 \
        libglib2.0-0 libnspr4 libnss3 libpango-1.0-0 libx11-6 libxcb1 libxcomposite1 libxdamage1 libxext6 \
        libxfixes3 libxkbcommon0 libxrandr2 fonts-liberation fonts-wqy-zenhei \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 akashic && useradd --uid 1000 --gid 1000 --create-home akashic \
    && mkdir /data && chown akashic:akashic /data
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/distribution /opt/distribution
COPY --from=build /opt/core /opt/core
# 插件源码保留临时目录的 700 权限；运行用户需要读取，但不应改写镜像制品。
RUN chmod -R a=rX /opt/distribution && /usr/local/bin/node --version && npm --version
# 限制 glibc arena 数量，避免后台线程的大块临时分配把 RSS 撑到 线程数 × 峰值（#1455）。
ENV MALLOC_ARENA_MAX=2
# 用户或 Agent 自己装的命令行工具（npm -g、uv tool install）都落在数据卷里，重建容器后还在。
ENV NPM_CONFIG_PREFIX=/data/tools UV_TOOL_DIR=/data/tools/uv UV_TOOL_BIN_DIR=/data/tools/bin
ENV PATH=/data/tools/bin:/opt/venv/bin:$PATH AKASHIC_EXECUTION_MODE=local PYTHONUNBUFFERED=1 AKASHIC_PLUGIN_DISTRIBUTION=/opt/distribution \
    AKASHIC_WORKSPACE=/data/workspace AKASHIC_PLUGIN_HOME=/data/plugin-home
USER akashic
WORKDIR /opt/core
EXPOSE 2236
ENTRYPOINT ["python", "scripts/start.py", "--distribution", "/opt/distribution", "--state", "/data", "--no-browser"]
