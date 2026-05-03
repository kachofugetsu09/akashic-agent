#!/usr/bin/env bash
set -euo pipefail

CONFIG="${AKASHIC_DEBUG_CONFIG:-/sandbox/config.toml}"
WORKSPACE="${AKASHIC_DEBUG_WORKSPACE:-/sandbox/workspace}"
SOCKET="/sandbox/akashic.sock"
DASHBOARD_HOST="${AKASHIC_DASHBOARD_HOST:-0.0.0.0}"
DASHBOARD_PORT="${AKASHIC_DASHBOARD_PORT:-2236}"
HOST_UID="${AKASHIC_HOST_UID:-1000}"
HOST_GID="${AKASHIC_HOST_GID:-1000}"

ensure_sandbox_path() {
    local path="$1"
    case "$path" in
        /sandbox/*) ;;
        *)
            echo "拒绝启动：调试路径必须位于 /sandbox 内：$path" >&2
            exit 2
            ;;
    esac
}

ensure_socket_config() {
    if [ ! -f "$CONFIG" ]; then
        return
    fi
    gosu "$HOST_UID:$HOST_GID" python - "$CONFIG" "$SOCKET" <<'PY'
from pathlib import Path
import sys
import toml
import tomllib

path = Path(sys.argv[1])
socket = sys.argv[2]
data = tomllib.loads(path.read_text(encoding="utf-8"))
channels = data.setdefault("channels", {})
channels["socket"] = socket
cli = channels.setdefault("cli", {})
cli["socket"] = socket
path.write_text(toml.dumps(data), encoding="utf-8")
PY
}

ensure_sandbox_path "$CONFIG"
ensure_sandbox_path "$WORKSPACE"
ensure_sandbox_path "$SOCKET"
mkdir -p /sandbox "$WORKSPACE" /sandbox/home
chown -R "$HOST_UID:$HOST_GID" /sandbox
cd /app

cmd="${1:-run}"
shift || true

case "$cmd" in
    setup)
        gosu "$HOST_UID:$HOST_GID" python main.py setup --config "$CONFIG" --workspace "$WORKSPACE" "$@"
        ensure_socket_config
        ;;
    init)
        gosu "$HOST_UID:$HOST_GID" python main.py init --config "$CONFIG" --workspace "$WORKSPACE" "$@"
        ensure_socket_config
        ;;
    reset-workspace)
        gosu "$HOST_UID:$HOST_GID" rm -rf "$WORKSPACE"
        gosu "$HOST_UID:$HOST_GID" python main.py init --config "$CONFIG" --workspace "$WORKSPACE" "$@"
        ensure_socket_config
        ;;
    run|gateway|serve)
        if [ ! -f "$CONFIG" ]; then
            echo "缺少 $CONFIG，请先运行：docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug setup" >&2
            exit 2
        fi
        ensure_socket_config
        exec gosu "$HOST_UID:$HOST_GID" python main.py --config "$CONFIG" --workspace "$WORKSPACE" "$@"
        ;;
    cli)
        if [ ! -f "$CONFIG" ]; then
            echo "缺少 $CONFIG，请先运行 setup。" >&2
            exit 2
        fi
        ensure_socket_config
        exec gosu "$HOST_UID:$HOST_GID" python main.py cli --config "$CONFIG" "$@"
        ;;
    dashboard)
        exec gosu "$HOST_UID:$HOST_GID" python main.py dashboard \
            --workspace "$WORKSPACE" \
            --host "$DASHBOARD_HOST" \
            --port "$DASHBOARD_PORT" \
            "$@"
        ;;
    *)
        exec gosu "$HOST_UID:$HOST_GID" "$cmd" "$@"
        ;;
esac
