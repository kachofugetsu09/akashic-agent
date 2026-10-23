# syntax=docker/dockerfile:1

# Selkies 客户端保持 MPL-2.0；固定上游源码，并保留补丁后的源码供审阅。
FROM node:22-bookworm-slim@sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5 AS stream-build
WORKDIR /build
ADD --checksum=sha256:cf91cb00117bc492bc8af55aa1f2157cecdc76c5c0f81a7bba18cee99dc467bd https://codeload.github.com/selkies-project/selkies/tar.gz/3ec56fb1538cf077c27156f5ab75b6595a83c461 /build/selkies.tar.gz
RUN tar -xzf selkies.tar.gz --strip-components=1
COPY docker/computer/build-stream-client.mjs /build/
WORKDIR /build/addons/selkies-web-core
RUN node /build/build-stream-client.mjs selkies-ws-core.js \
    && npm ci --no-audit --no-fund && npm run build

# 只有浏览器：一个 Chromium 带个人 profile；不装桌面环境，matchbox 只负责让每个窗口铺满屏幕。
FROM node:22-bookworm-slim@sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5

LABEL org.opencontainers.image.source="https://github.com/kachofugetsu09/akashic-agent"

ARG OPENCLI_VERSION=1.8.7
ARG OPENCLI_EXTENSION_VERSION=1.0.23
ARG OPENCLI_EXTENSION_SHA256=e3399db1e9dd626519a8719d638d3c3813494d1f030c1e96799e05ebe7ba5340

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:99 \
    HOME=/data/home \
    OPENCLI_AUTH_REFRESH_SITES=bilibili,gemini,github,jd,notebooklm,taobao,xianyu,youtube \
    OPENCLI_DAEMON_PORT=19825 \
    XDG_CACHE_HOME=/data/cache \
    XDG_CONFIG_HOME=/data/config

COPY docker/computer/stream-requirements.txt /opt/computer/
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates chromium dbus fonts-noto-cjk fonts-noto-color-emoji \
        libva2 libva-drm2 libva-x11-2 matchbox-window-manager procps python3-venv \
        tigervnc-standalone-server tini unzip wget x11-xserver-utils xclip xdotool \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/computer/stream-venv \
    && /opt/computer/stream-venv/bin/pip install --no-cache-dir -r /opt/computer/stream-requirements.txt \
    && npm install --global "@jackwener/opencli@${OPENCLI_VERSION}" \
    && wget -q -O /tmp/opencli-extension.zip \
        "https://github.com/jackwener/OpenCLI/releases/download/v${OPENCLI_VERSION}/opencli-extension-v${OPENCLI_EXTENSION_VERSION}.zip" \
    && echo "${OPENCLI_EXTENSION_SHA256}  /tmp/opencli-extension.zip" | sha256sum -c - \
    && mkdir -p /opt/opencli-extension \
    && unzip -q /tmp/opencli-extension.zip -d /opt/opencli-extension \
    && rm /tmp/opencli-extension.zip \
    && mkdir -p /data /tmp/.X11-unix \
    && chmod 1777 /tmp/.X11-unix \
    && chown -R 1000:1000 /data /opt/computer

COPY --from=stream-build /build/addons/selkies-web-core/dist /opt/computer/stream-web
COPY --from=stream-build /build/addons/selkies-web-core/selkies-ws-core.js /opt/computer/stream-web/source/selkies-ws-core.js
COPY --from=stream-build /build/addons/selkies-web-core/selkies-wr-core.js /opt/computer/stream-web/source/selkies-wr-core.js
COPY --from=stream-build /build/addons/selkies-web-core/lib/util.js /opt/computer/stream-web/source/util.js
COPY --from=stream-build /build/LICENSE /opt/computer/stream-web/source/LICENSE
COPY --chown=1000:1000 docker/computer/package.json docker/computer/package-lock.json /opt/computer/
RUN cd /opt/computer && npm ci --omit=dev --no-audit --no-fund
COPY --chown=1000:1000 docker/computer/driver /opt/computer/driver
COPY --chown=1000:1000 docker/computer/gateway.mjs docker/computer/lifecycle.mjs /opt/computer/
COPY --chown=1000:1000 docker/computer/start.sh /opt/computer/start.sh
RUN chmod 0755 /opt/computer/start.sh

USER 1000:1000
WORKDIR /data
EXPOSE 6081 8080 19826
# Only the supervisor receives stop; it closes Chromium before D-Bus and X11.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/opt/computer/start.sh"]
