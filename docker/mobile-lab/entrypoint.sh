#!/usr/bin/env bash
set -euo pipefail

SOURCE_CONFIG="${AKASHIC_MOBILE_LAB_SOURCE_CONFIG:-/source/config.toml}"
CONFIG="${AKASHIC_MOBILE_LAB_CONFIG:-/sandbox/config.toml}"
WORKSPACE="${AKASHIC_MOBILE_LAB_WORKSPACE:-/sandbox/workspace}"
HOST_UID="${AKASHIC_HOST_UID:-1000}"
HOST_GID="${AKASHIC_HOST_GID:-1000}"

test -f "$SOURCE_CONFIG"
test -n "${AKASHIC_MOBILE_LAB_PUBLIC_URL:-}"
test -n "${AKASHIC_MOBILE_LAB_LAN_HOSTNAME:-}"

mkdir -p /sandbox/home/.akashic-plugin "$WORKSPACE"
chown -R "$HOST_UID:$HOST_GID" /sandbox

setpriv --reuid "$HOST_UID" --regid "$HOST_GID" --clear-groups \
    python /app/docker/mobile-lab/prepare_config.py \
        --source "$SOURCE_CONFIG" \
        --target "$CONFIG" \
        --public-url "$AKASHIC_MOBILE_LAB_PUBLIC_URL" \
        --lan-hostname "$AKASHIC_MOBILE_LAB_LAN_HOSTNAME"

exec setpriv --reuid "$HOST_UID" --regid "$HOST_GID" --clear-groups \
    dbus-run-session -- bash -ceu '
        export XDG_RUNTIME_DIR=/tmp/akashic-mobile-lab-runtime
        mkdir -p "$XDG_RUNTIME_DIR"
        chmod 700 "$XDG_RUNTIME_DIR"
        eval "$(printf "\n" | gnome-keyring-daemon --unlock --components=secrets)"
        printf ready | secret-tool store \
            --label="Akashic Mobile Lab startup probe" \
            application akashic-mobile-lab secret startup-probe
        test "$(secret-tool lookup application akashic-mobile-lab secret startup-probe)" = ready
        secret-tool clear application akashic-mobile-lab secret startup-probe
        exec python main.py supervise \
            --config "$AKASHIC_MOBILE_LAB_CONFIG" \
            --workspace "$AKASHIC_MOBILE_LAB_WORKSPACE"
    '
