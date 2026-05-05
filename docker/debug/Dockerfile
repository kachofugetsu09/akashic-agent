FROM archlinux:latest

ENV HOME=/sandbox/home \
    PATH=/opt/venv/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN printf '%s\n' \
        'Server = https://mirrors.ustc.edu.cn/archlinux/$repo/os/$arch' \
        'Server = https://mirrors.tuna.tsinghua.edu.cn/archlinux/$repo/os/$arch' \
        > /etc/pacman.d/mirrorlist \
    && pacman -Sy --noconfirm archlinux-keyring \
    && pacman -Syu --noconfirm \
    && pacman -S --noconfirm --needed \
        base-devel \
        ca-certificates \
        curl \
        git \
        python \
        python-pip \
        util-linux \
    && pacman -Scc --noconfirm

COPY requirements.txt requirements-dev.txt /tmp/
RUN python -m venv /opt/venv \
    && python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements-dev.txt

COPY docker/debug/entrypoint.sh /usr/local/bin/akashic-debug
RUN chmod +x /usr/local/bin/akashic-debug

ENTRYPOINT ["akashic-debug"]
CMD ["run"]
