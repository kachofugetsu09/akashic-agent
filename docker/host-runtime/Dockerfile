ARG AKASHIC_BASE_IMAGE=archlinux@sha256:345a872f6c95e082d4b8c050af637eebb57402c6e2177b411c3acf7df84eb33b
FROM ${AKASHIC_BASE_IMAGE}

ARG AKASHIC_BASE_IMAGE
ARG AKASHIC_RUNTIME_UID=1000
ARG AKASHIC_RUNTIME_GID=1000
ARG AKASHIC_ARCH_SNAPSHOT=2026/08/09
ARG AKASHIC_PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG AKASHIC_NPM_REGISTRY=https://registry.npmmirror.com

ENV HOME=/home/huashen \
    PATH=/opt/venv/bin:/home/huashen/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN printf '%s\n' \
        'CacheServer = https://mirrors.tuna.tsinghua.edu.cn/archlinux/$repo/os/$arch' \
        'CacheServer = https://mirrors.ustc.edu.cn/archlinux/$repo/os/$arch' \
        "Server = https://archive.archlinux.org/repos/${AKASHIC_ARCH_SNAPSHOT}/\$repo/os/\$arch" \
        > /etc/pacman.d/mirrorlist \
    && pacman --disable-download-timeout -Syyu --noconfirm \
    && pacman --disable-download-timeout -S --noconfirm --needed \
        ca-certificates \
        curl \
        git \
        nodejs \
        npm \
        python \
        python-pip \
        util-linux \
        zsh \
    && pacman -Scc --noconfirm \
    && groupadd --gid "${AKASHIC_RUNTIME_GID}" huashen \
    && useradd \
        --uid "${AKASHIC_RUNTIME_UID}" \
        --gid "${AKASHIC_RUNTIME_GID}" \
        --create-home \
        --shell /usr/bin/zsh \
        huashen

WORKDIR /opt/akashic/source

COPY docker/host-runtime/requirements.lock /tmp/requirements.lock
RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install \
        --index-url "${AKASHIC_PYPI_INDEX_URL}" \
        --no-cache-dir --require-hashes -r /tmp/requirements.lock

ARG AKASHIC_SOURCE_COMMIT
ARG AKASHIC_SOURCE_TREE
ARG AKASHIC_SOURCE_ARCHIVE_SHA256
ARG AKASHIC_SOURCE_MANIFEST_SHA256
ARG AKASHIC_REQUIREMENTS_LOCK_SHA256
ARG AKASHIC_PACKAGE_LOCK_SHA256
COPY . /opt/akashic/source
RUN test -n "${AKASHIC_SOURCE_COMMIT}" \
    && test -n "${AKASHIC_SOURCE_TREE}" \
    && test -n "${AKASHIC_SOURCE_ARCHIVE_SHA256}" \
    && test -n "${AKASHIC_SOURCE_MANIFEST_SHA256}" \
    && test "$(sha256sum .akashic-source-manifest.json | cut -d' ' -f1)" = "${AKASHIC_SOURCE_MANIFEST_SHA256}" \
    && test "$(sha256sum docker/host-runtime/requirements.lock | cut -d' ' -f1)" = "${AKASHIC_REQUIREMENTS_LOCK_SHA256}" \
    && test "$(sha256sum package-lock.json | cut -d' ' -f1)" = "${AKASHIC_PACKAGE_LOCK_SHA256}" \
    && /opt/venv/bin/python docker/host-runtime/verify_release_source.py \
        --root /opt/akashic/source \
        --manifest /opt/akashic/source/.akashic-source-manifest.json \
        --expected-commit "${AKASHIC_SOURCE_COMMIT}" \
        --expected-tree "${AKASHIC_SOURCE_TREE}" \
        --expected-archive-sha256 "${AKASHIC_SOURCE_ARCHIVE_SHA256}" \
    && npm ci --ignore-scripts --registry "${AKASHIC_NPM_REGISTRY}" \
    && npm run build \
    && rm -rf node_modules /root/.npm \
    && PACMAN_DIGEST="$(pacman -Q | LC_ALL=C sort | sha256sum | cut -d' ' -f1)" \
    && PYTHON_VERSION="$(python --version | cut -d' ' -f2)" \
    && NODE_VERSION="$(node --version)" \
    && NPM_VERSION="$(npm --version)" \
    && printf '%s\n' \
        '{' \
        '  "schemaVersion": 2,' \
        "  \"sourceCommit\": \"${AKASHIC_SOURCE_COMMIT}\"," \
        "  \"sourceTree\": \"${AKASHIC_SOURCE_TREE}\"," \
        "  \"sourceArchiveSha256\": \"${AKASHIC_SOURCE_ARCHIVE_SHA256}\"," \
        "  \"sourceManifestSha256\": \"${AKASHIC_SOURCE_MANIFEST_SHA256}\"," \
        "  \"baseImage\": \"${AKASHIC_BASE_IMAGE}\"," \
        "  \"archSnapshot\": \"${AKASHIC_ARCH_SNAPSHOT}\"," \
        "  \"pypiIndexUrl\": \"${AKASHIC_PYPI_INDEX_URL}\"," \
        "  \"npmRegistry\": \"${AKASHIC_NPM_REGISTRY}\"," \
        "  \"pacmanDigest\": \"${PACMAN_DIGEST}\"," \
        "  \"requirementsLockSha256\": \"${AKASHIC_REQUIREMENTS_LOCK_SHA256}\"," \
        "  \"packageLockSha256\": \"${AKASHIC_PACKAGE_LOCK_SHA256}\"," \
        "  \"pythonVersion\": \"${PYTHON_VERSION}\"," \
        "  \"nodeVersion\": \"${NODE_VERSION}\"," \
        "  \"npmVersion\": \"${NPM_VERSION}\"" \
        '}' \
        > /opt/akashic/runtime-info.json \
    && chown -R root:root /opt/akashic \
    && chmod -R a-w /opt/akashic

LABEL org.opencontainers.image.revision="${AKASHIC_SOURCE_COMMIT}"

USER huashen:huashen
ENTRYPOINT ["/opt/akashic/source/docker/host-runtime/entrypoint.sh"]
CMD ["supervise"]
