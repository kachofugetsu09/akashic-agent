# syntax=docker/dockerfile:1

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

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates chromium dbus-x11 fonts-noto-cjk imagemagick procps \
        thunar tigervnc-standalone-server tini unzip websockify wget \
        x11-xserver-utils xclip xdotool xfce4-panel xfce4-session \
        xfce4-settings xfce4-terminal xfdesktop4 xfwm4 \
    && rm -rf /var/lib/apt/lists/* \
    && npm install --global "@jackwener/opencli@${OPENCLI_VERSION}" \
    && wget -q -O /tmp/opencli-extension.zip \
        "https://github.com/jackwener/OpenCLI/releases/download/v${OPENCLI_VERSION}/opencli-extension-v${OPENCLI_EXTENSION_VERSION}.zip" \
    && echo "${OPENCLI_EXTENSION_SHA256}  /tmp/opencli-extension.zip" | sha256sum -c - \
    && mkdir -p /opt/opencli-extension \
    && unzip -q /tmp/opencli-extension.zip -d /opt/opencli-extension \
    && rm /tmp/opencli-extension.zip \
    && mkdir -p /data /opt/computer /tmp/.ICE-unix /tmp/.X11-unix \
    && chmod 1777 /tmp/.ICE-unix /tmp/.X11-unix \
    && chown -R 1000:1000 /data /opt/computer

COPY --chown=1000:1000 docker/computer/gateway.mjs /opt/computer/gateway.mjs
COPY --chown=1000:1000 docker/computer/start.sh /opt/computer/start.sh
RUN chmod 0755 /opt/computer/start.sh

USER 1000:1000
WORKDIR /data
EXPOSE 6080 8080 19826
ENTRYPOINT ["/usr/bin/tini", "-g", "--"]
CMD ["/opt/computer/start.sh"]
