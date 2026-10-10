#!/usr/bin/env bash
set -euo pipefail

CONFIG="${AKASHIC_DEBUG_CONFIG:-/sandbox/config.toml}"
WORKSPACE="${AKASHIC_DEBUG_WORKSPACE:-/sandbox/workspace}"
SOCKET="/sandbox/akashic.sock"
WEB_HOST="${AKASHIC_WEB_HOST:-0.0.0.0}"
WEB_PORT="${AKASHIC_WEB_PORT:-2236}"
HOST_UID="${AKASHIC_HOST_UID:-1000}"
HOST_GID="${AKASHIC_HOST_GID:-1000}"

as_host() {
    setpriv --reuid "$HOST_UID" --regid "$HOST_GID" --clear-groups "$@"
}

exec_as_host() {
    exec setpriv --reuid "$HOST_UID" --regid "$HOST_GID" --clear-groups "$@"
}

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

init_gateway_config() {
    if [ ! -f "$CONFIG" ]; then
        return
    fi
    as_host python - "$CONFIG" "$WORKSPACE" "$SOCKET" <<'PY_CONFIG'
from pathlib import Path
import os
import sys
import tomllib
from agent.plugin_composition.config_input import CONFIG_INPUT, save_config
from agent.plugins.distribution_sources import distribution_sources
from agent.plugins.manifest import plugins_root, workspace_plugin_data_dir
from agent.plugins.selection import PluginSelection
from agent.plugins.source_resolver import scan_plugin_sources

config, workspace, socket = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
# 旧 app_server 由 Gateway 迁移完整复制；调试入口不抢先建立另一份值。
if "app_server" in tomllib.loads(config.read_text(encoding="utf-8")):
    print("旧控制配置等待 Gateway 迁移；调试入口不改写 Core 配置。")
    raise SystemExit(0)
selection = PluginSelection(workspace)
reference = selection.read() if selection.path.exists() else None
if reference is not None:
    records = [selection.read_input(ref) for ref in selection.components(reference)]
    targets = [(str(record["plugin_id"]), workspace / str(record["data_dir"])) for record in records
               if str(record["plugin_id"]).partition("@")[0] == "gateway"]
else:
    home = plugins_root()
    distribution = distribution_sources(workspace, home)
    roots = [Path(value) for value in os.environ.get("AKASHIC_EXTRA_PLUGIN_DIRS", "").split(os.pathsep) if value]
    scan = scan_plugin_sources(roots, installed_cache_root=home / "cache",
                               fixed_sources=distribution.sources,
                               ignored_installed_roots=distribution.ignored_installed_roots)
    matches = [source for source in scan.sources if source.plugin_name == "gateway"]
    installed = [source for source in matches if source.source_type == "installed"]
    targets = [("gateway" + (f"@{source.marketplace}" if source.marketplace else ""),
                workspace_plugin_data_dir(workspace, "gateway", source.marketplace or "builtin"))
               for source in installed or matches[:1]]
if not targets:
    print("未选择 Gateway；调试入口不创建或启用业务配置。")
    raise SystemExit(0)
if len(targets) != 1:
    raise ValueError("调试 Gateway 配置需要唯一 provider: " + ", ".join(identity for identity, _ in targets))
_, data = targets[0]
# 只初始化缺席输入；保留现有 enabled、监听选择和其他插件设置。
if not (data / CONFIG_INPUT).exists():
    save_config(data, {"listen": socket})
PY_CONFIG
}

ensure_sandbox_path "$CONFIG"
ensure_sandbox_path "$WORKSPACE"
ensure_sandbox_path "$SOCKET"
mkdir -p /sandbox /sandbox/home/.akashic-plugin
chown "$HOST_UID:$HOST_GID" \
    /sandbox \
    /sandbox/home \
    /sandbox/home/.akashic-plugin
if [ -d "$WORKSPACE" ]; then
    chown -R "$HOST_UID:$HOST_GID" "$WORKSPACE"
fi
if [ -f "$WORKSPACE/replay/clock.json" ]; then
    export AKASHIC_REPLAY_CLOCK_FILE="$WORKSPACE/replay/clock.json"
    export AKASHIC_REPLAY_EVENTS_FILE="$WORKSPACE/replay/events.jsonl"
    export AKASHIC_REPLAY_OUTBOX_FILE="$WORKSPACE/replay/outbox.jsonl"
fi
cd /app

cmd="${1:-run}"
shift || true

case "$cmd" in
    setup)
        as_host python main.py setup --config "$CONFIG" --workspace "$WORKSPACE" "$@"
        init_gateway_config
        ;;
    init)
        as_host python main.py init --config "$CONFIG" --workspace "$WORKSPACE" "$@"
        init_gateway_config
        ;;
    reset-workspace)
        as_host rm -rf "$WORKSPACE"
        as_host python main.py init --config "$CONFIG" --workspace "$WORKSPACE" "$@"
        init_gateway_config
        ;;
    run|serve)
        if [ ! -f "$CONFIG" ]; then
            echo "未找到调试配置，正在初始化空白本地实例：$CONFIG"
            as_host python main.py init --config "$CONFIG" --workspace "$WORKSPACE"
        fi
        init_gateway_config
        exec_as_host python main.py --config "$CONFIG" --workspace "$WORKSPACE" "$@"
        ;;
    gateway)
        if [ ! -f "$CONFIG" ]; then
            echo "缺少 $CONFIG，请先运行 setup。" >&2
            exit 2
        fi
        exec_as_host python main.py supervise \
            --config "$CONFIG" \
            --workspace "$WORKSPACE" \
            "$@"
        ;;
    app-server)
        if [ ! -f "$CONFIG" ]; then
            echo "缺少 $CONFIG，请先运行 setup。" >&2
            exit 2
        fi
        exec_as_host python main.py app-server \
            --config "$CONFIG" \
            --workspace "$WORKSPACE" \
            "$@"
        ;;
    exec)
        init_gateway_config
        exec_as_host python main.py exec --config "$CONFIG" --workspace "$WORKSPACE" "$@"
        ;;
    dashboard)
        exec_as_host python main.py dashboard \
            --workspace "$WORKSPACE" \
            --host "$WEB_HOST" \
            --port "$WEB_PORT" \
            "$@"
        ;;
    gate-root-shell-cleanup)
        exec python -m pytest -q \
            tests/test_unified_exec.py::test_real_cross_uid_live_process_group_returns_eperm
        ;;
    *)
        exec_as_host "$cmd" "$@"
        ;;
esac
