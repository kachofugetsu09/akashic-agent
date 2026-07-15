FROM akashic-agent-debug:latest

ARG HOST_UID=1000
ARG HOST_GID=1000

RUN pacman -Sy --noconfirm \
    && pacman -S --noconfirm --needed \
        dbus \
        gnome-keyring \
        libsecret \
    && pacman -Scc --noconfirm \
    && groupadd --gid "$HOST_GID" akashic-lab \
    && useradd --uid "$HOST_UID" --gid "$HOST_GID" --home-dir /sandbox/home --no-create-home akashic-lab

COPY docker/mobile-lab/entrypoint.sh /usr/local/bin/akashic-mobile-lab
RUN chmod +x /usr/local/bin/akashic-mobile-lab

ENTRYPOINT ["akashic-mobile-lab"]
