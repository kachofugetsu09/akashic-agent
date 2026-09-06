# syntax=docker/dockerfile:1

FROM rust:1.88-bookworm@sha256:af306cfa71d987911a781c37b59d7d67d934f49684058f96cf72079c3626bfe0 AS desktop-build
WORKDIR /build
COPY docker/computer/native/Cargo.toml docker/computer/native/Cargo.lock ./
COPY docker/computer/native/src ./src
RUN cargo build --locked --release

# 固定现有桌面环境；本次只升级 driver，不重新安装 Chromium、XFCE 或 OpenCLI。
FROM ghcr.io/kachofugetsu09/akashic-computer@sha256:6fd3c605380a3daef5ddebb34f2905ee992d2b4e1490fbfb78dcce9f06a3dadb
USER root
COPY --from=desktop-build /build/target/release/akashic-desktop /opt/computer/bin/akashic-desktop
COPY --chown=1000:1000 docker/computer/package.json docker/computer/package-lock.json /opt/computer/
RUN cd /opt/computer && npm ci --omit=dev --no-audit --no-fund
COPY --chown=1000:1000 docker/computer/driver /opt/computer/driver
COPY --chown=1000:1000 docker/computer/gateway.mjs /opt/computer/gateway.mjs
COPY --chown=1000:1000 docker/computer/start.sh /opt/computer/start.sh
RUN chmod 0755 /opt/computer/start.sh
USER 1000:1000
WORKDIR /data
