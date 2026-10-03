FROM node:22-bookworm-slim AS node
FROM python:3.12-slim-bookworm AS build
COPY --from=node /usr/local /usr/local
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
ARG AKASHIC_REVISION
WORKDIR /source
RUN git init && git remote add origin https://github.com/kachofugetsu09/akashic-agent.git \
    && git fetch --depth=1 origin "${AKASHIC_REVISION}" && git checkout --detach FETCH_HEAD
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --require-hashes -r docker/host-runtime/requirements.lock \
    && /opt/venv/bin/pip install --no-cache-dir --no-deps ./sdk/python
RUN /opt/venv/bin/python scripts/build_plugin_distribution.py --revision HEAD --output /opt/distribution \
    && /opt/venv/bin/python scripts/distribution_runtime.py prepare-wheels --distribution /opt/distribution \
    && /opt/venv/bin/python -c "from pathlib import Path; from scripts.install_plugin_distribution import extract_core; extract_core(Path('/opt/distribution'), Path('/opt/core'))"

FROM python:3.12-slim-bookworm
ARG AKASHIC_REVISION
LABEL org.opencontainers.image.source="https://github.com/kachofugetsu09/akashic-agent" \
      org.opencontainers.image.revision=$AKASHIC_REVISION
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates zsh curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 akashic && useradd --uid 1000 --gid 1000 --create-home akashic \
    && mkdir /data && chown akashic:akashic /data
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/distribution /opt/distribution
COPY --from=build /opt/core /opt/core
# 插件源码保留临时目录的 700 权限；运行用户需要读取，但不应改写镜像制品。
RUN chmod -R a=rX /opt/distribution
ENV PATH=/opt/venv/bin:$PATH AKASHIC_EXECUTION_MODE=local PYTHONUNBUFFERED=1 AKASHIC_PLUGIN_DISTRIBUTION=/opt/distribution
USER akashic
WORKDIR /opt/core
EXPOSE 2236
ENTRYPOINT ["python", "scripts/start.py", "--distribution", "/opt/distribution", "--state", "/data", "--no-browser"]
