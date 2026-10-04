# syntax=docker/dockerfile:1

FROM rust:1.88-bookworm@sha256:af306cfa71d987911a781c37b59d7d67d934f49684058f96cf72079c3626bfe0 AS desktop-build
WORKDIR /build
COPY docker/computer/native/Cargo.toml docker/computer/native/Cargo.lock ./
COPY docker/computer/native/src ./src
RUN cargo build --locked --release

# Selkies 客户端保持 MPL-2.0；固定上游源码，并保留补丁后的源码供审阅。
FROM ghcr.io/kachofugetsu09/akashic-computer@sha256:6a2cb0b57e912d48dae55b1b4c1d62c43d52108692bb330b038c1ba00bb6a365 AS stream-build
USER root
WORKDIR /build
ADD --checksum=sha256:cf91cb00117bc492bc8af55aa1f2157cecdc76c5c0f81a7bba18cee99dc467bd https://codeload.github.com/selkies-project/selkies/tar.gz/3ec56fb1538cf077c27156f5ab75b6595a83c461 /build/selkies.tar.gz
RUN tar -xzf selkies.tar.gz --strip-components=1
COPY docker/computer/build-stream-client.mjs /build/
WORKDIR /build/addons/selkies-web-core
RUN node /build/build-stream-client.mjs selkies-ws-core.js \
    && npm ci --no-audit --no-fund && npm run build

# 固定现有桌面环境；不重新安装 Chromium、XFCE 或 OpenCLI。
FROM ghcr.io/kachofugetsu09/akashic-computer@sha256:6fd3c605380a3daef5ddebb34f2905ee992d2b4e1490fbfb78dcce9f06a3dadb
USER root
COPY docker/computer/stream-requirements.txt /opt/computer/
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-venv libva2 libva-drm2 libva-x11-2 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/computer/stream-venv \
    && /opt/computer/stream-venv/bin/pip install --no-cache-dir -r /opt/computer/stream-requirements.txt
COPY --from=stream-build /build/addons/selkies-web-core/dist /opt/computer/stream-web
COPY --from=stream-build /build/addons/selkies-web-core/selkies-ws-core.js /opt/computer/stream-web/source/selkies-ws-core.js
COPY --from=stream-build /build/addons/selkies-web-core/lib/util.js /opt/computer/stream-web/source/util.js
COPY --from=stream-build /build/LICENSE /opt/computer/stream-web/source/LICENSE
COPY --from=desktop-build /build/target/release/akashic-desktop /opt/computer/bin/akashic-desktop
COPY --chown=1000:1000 docker/computer/package.json docker/computer/package-lock.json /opt/computer/
RUN cd /opt/computer && npm ci --omit=dev --no-audit --no-fund
COPY --chown=1000:1000 docker/computer/driver /opt/computer/driver
COPY --chown=1000:1000 docker/computer/gateway.mjs docker/computer/lifecycle.mjs /opt/computer/
COPY --chown=1000:1000 docker/computer/start.sh /opt/computer/start.sh
RUN chmod 0755 /opt/computer/start.sh
USER 1000:1000
WORKDIR /data
