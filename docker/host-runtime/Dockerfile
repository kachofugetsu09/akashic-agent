FROM archlinux:latest

ARG AKASHIC_SOURCE_COMMIT
ARG AKASHIC_SOURCE_TREE
ARG AKASHIC_RUNTIME_UID=1000
ARG AKASHIC_RUNTIME_GID=1000

ENV HOME=/home/huashen \
    PATH=/opt/venv/bin:/home/huashen/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pacman -Syu --noconfirm \
    && pacman -S --noconfirm --needed \
        ca-certificates \
        curl \
        git \
        nodejs \
        npm \
        python \
        python-pip \
        tini \
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

COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && /opt/venv/bin/python -m pip install --no-cache-dir mcp==1.28.1

COPY . /opt/akashic/source
RUN test -n "${AKASHIC_SOURCE_COMMIT}" \
    && test -n "${AKASHIC_SOURCE_TREE}" \
    && printf '%s\n' \
        '{' \
        '  "schemaVersion": 1,' \
        "  \"sourceCommit\": \"${AKASHIC_SOURCE_COMMIT}\"," \
        "  \"sourceTree\": \"${AKASHIC_SOURCE_TREE}\"" \
        '}' \
        > /opt/akashic/runtime-info.json \
    && chown -R root:root /opt/akashic \
    && chmod -R a-w /opt/akashic

USER huashen:huashen
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/akashic/source/docker/host-runtime/entrypoint.sh"]
CMD ["supervise"]
