FROM archlinux:latest

ENV HOME=/sandbox/home \
    PATH=/opt/venv/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN printf '%s\n' \
        'Server = https://geo.mirror.pkgbuild.com/$repo/os/$arch' \
        'Server = https://mirrors.ustc.edu.cn/archlinux/$repo/os/$arch' \
        'Server = https://mirrors.tuna.tsinghua.edu.cn/archlinux/$repo/os/$arch' \
        > /etc/pacman.d/mirrorlist \
    && pacman-key --init \
    && pacman-key --populate archlinux \
    && pacman -Syu --noconfirm \
    && pacman -S --noconfirm --needed \
        base-devel \
        ca-certificates \
        curl \
        git \
        nodejs \
        npm \
        python \
        python-pip \
        util-linux \
    && pacman -Scc --noconfirm

COPY requirements.txt requirements-dev.txt /tmp/
RUN python -m venv /opt/venv \
    && python -m pip install pip==26.1.2 \
    && python -m pip install -r /tmp/requirements-dev.txt \
    && python -m pip install mcp==1.28.1

COPY docker/debug/entrypoint.sh /usr/local/bin/akashic-debug
RUN chmod +x /usr/local/bin/akashic-debug

ENTRYPOINT ["akashic-debug"]
CMD ["run"]
