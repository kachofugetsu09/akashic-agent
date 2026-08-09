FROM archlinux:latest

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

ARG AKASHIC_SOURCE_COMMIT
ARG AKASHIC_SOURCE_TREE
COPY . /opt/akashic/source
RUN test -n "${AKASHIC_SOURCE_COMMIT}" \
    && test -n "${AKASHIC_SOURCE_TREE}" \
    && mkdir -p /opt/akashic/source/static/dashboard /opt/akashic/source/static/chat \
    && find /opt/akashic/source/plugins -maxdepth 2 \
        \( -name 'dashboard_panel.ts' -o -name 'dashboard_panel.tsx' \
        -o -name 'dashboard_panel_inspector.ts' -o -name 'dashboard_panel_inspector.tsx' \) \
        -print0 \
        | while IFS= read -r -d '' source; do \
            target="${source%.*}.js"; \
            npx --yes esbuild "$source" \
                --outfile="$target" --bundle --platform=browser --target=es2020 \
                --format=esm --jsx=automatic \
                --external:react --external:react-dom --external:react-dom/client \
                --external:react/jsx-runtime --external:@akashic/dashboard-ui; \
        done \
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
ENTRYPOINT ["/opt/akashic/source/docker/host-runtime/entrypoint.sh"]
CMD ["supervise"]
