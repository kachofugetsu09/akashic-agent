#!/bin/bash
set -euo pipefail

# 1. 解析运行版本和显式 workspace
SCRIPT_ROOT=$(cd "$(dirname "$0")" && pwd)
RUNTIME_ROOT=${AKASHIC_RUNTIME_ROOT:-$SCRIPT_ROOT}
CONFIG_PATH=${AKASHIC_CONFIG:-$SCRIPT_ROOT/config.toml}
WORKSPACE=${AKASHIC_WORKSPACE:-}
PYTHON=${AKASHIC_PYTHON:-$SCRIPT_ROOT/.venv/bin/python}

usage() {
    echo "用法: ./start.sh [--workspace PATH] [--config PATH] [--runtime-root PATH] [--python PATH]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --workspace|--config|--runtime-root|--python)
            [ "$#" -ge 2 ] || { echo "$1 缺少值" >&2; exit 2; }
            case "$1" in
                --workspace) WORKSPACE=$2 ;;
                --config) CONFIG_PATH=$2 ;;
                --runtime-root) RUNTIME_ROOT=$2 ;;
                --python) PYTHON=$2 ;;
            esac
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知参数: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

RUNTIME_ROOT=$(readlink -m "$RUNTIME_ROOT")
CONFIG_PATH=$(readlink -m "$CONFIG_PATH")
# 保留虚拟环境 Python 的最终符号链接，否则会绕过 venv 的 site-packages。
PYTHON=$(readlink -m "$(dirname "$PYTHON")")/$(basename "$PYTHON")

if [ ! -x "$PYTHON" ] || [ ! -f "$RUNTIME_ROOT/main.py" ] || [ ! -f "$CONFIG_PATH" ]; then
    echo "Runtime 不完整: python=$PYTHON root=$RUNTIME_ROOT config=$CONFIG_PATH" >&2
    exit 1
fi

resolve_workspace() {
    "$PYTHON" - "$CONFIG_PATH" "$WORKSPACE" <<'PY'
import sys
import tomllib
from pathlib import Path

config_path = Path(sys.argv[1])
explicit_workspace = sys.argv[2].strip()
if explicit_workspace:
    workspace = explicit_workspace
else:
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise SystemExit(f"配置文件 {config_path} 缺少 [runtime] table")
    workspace = runtime.get("workspace")
    if not isinstance(workspace, str) or not workspace.strip():
        raise SystemExit(f"配置文件 {config_path} 缺少 runtime.workspace")

print(Path(workspace).expanduser().resolve())
PY
}

WORKSPACE=$(resolve_workspace)
PID_FILE=$WORKSPACE/.launcher.pid
LOG_FILE=$WORKSPACE/logs/runtime.log
LEGACY_PID_FILE=$SCRIPT_ROOT/.agent.pid

mkdir -p "$WORKSPACE/logs" "$WORKSPACE/mcp/servers"
cd "$RUNTIME_ROOT"

owns_agent_pid() {
    local pid=$1
    local main_path mode index
    local -a argv
    [ -r "/proc/$pid/cmdline" ] || return 1
    mapfile -d '' -t argv < "/proc/$pid/cmdline"
    [ "${#argv[@]}" -ge 3 ] || return 1

    main_path=$(readlink -m "${argv[1]}")
    mode=${argv[2]}
    [ "$(basename "$main_path")" = "main.py" ] || return 1
    [ "$mode" = "supervise" ] || [ "$mode" = "gateway" ] || return 1

    for ((index = 3; index + 1 < ${#argv[@]}; index++)); do
        if [ "${argv[index]}" = "--workspace" ] && \
            [ "$(readlink -m "${argv[index + 1]}")" = "$WORKSPACE" ]; then
            return 0
        fi
    done
    return 1
}

stop_owned_pid() {
    local pid=$1
    if ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    if ! owns_agent_pid "$pid"; then
        echo "拒绝停止不属于当前 Akashic runtime 的 PID: $pid" >&2
        exit 1
    fi
    echo "Stopping Akashic runtime (PID: $pid)..."
    kill "$pid"
    for _ in $(seq 1 150); do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        sleep 0.1
    done
    echo "Akashic runtime 未在 15 秒内退出: $pid" >&2
    exit 1
}

# 2. 只停止所选 workspace 明确拥有的旧实例
if [ -f "$PID_FILE" ]; then
    stop_owned_pid "$(cat "$PID_FILE")"
    rm -f "$PID_FILE"
fi
if [ -f "$WORKSPACE/.supervisor.pid" ]; then
    supervisor_pid=$(cat "$WORKSPACE/.supervisor.pid")
    stop_owned_pid "$supervisor_pid"
    rm -f "$WORKSPACE/.supervisor.pid"
fi
if [ -f "$LEGACY_PID_FILE" ]; then
    legacy_pid=$(cat "$LEGACY_PID_FILE")
    if ! kill -0 "$legacy_pid" 2>/dev/null || owns_agent_pid "$legacy_pid"; then
        stop_owned_pid "$legacy_pid"
        rm -f "$LEGACY_PID_FILE"
    fi
fi

# 3. 启动固定 supervisor，并等待新 boot 真正 ready
old_boot=""
if [ -f "$WORKSPACE/.runtime-ready.json" ]; then
    old_boot=$(
        "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("bootId", ""))' \
            "$WORKSPACE/.runtime-ready.json" 2>/dev/null || true
    )
fi

# ncatbot 将通用 LOG_FORMAT 当作 logging 格式串，不能继承宿主的 json 值。
setsid env -u LOG_FORMAT "$PYTHON" "$RUNTIME_ROOT/main.py" supervise \
    --config "$CONFIG_PATH" \
    --workspace "$WORKSPACE" \
    > "$LOG_FILE" 2>&1 < /dev/null &
supervisor_pid=$!
echo "$supervisor_pid" > "$PID_FILE"

for _ in $(seq 1 300); do
    if ! kill -0 "$supervisor_pid" 2>/dev/null; then
        echo "Supervisor 启动失败，日志如下：" >&2
        tail -80 "$LOG_FILE" >&2
        rm -f "$PID_FILE"
        exit 1
    fi
    ready=$(
        "$PYTHON" -c '
import json, pathlib, sys
ready = pathlib.Path(sys.argv[1])
owner = pathlib.Path(sys.argv[2])
expected = int(sys.argv[3])
old_boot = sys.argv[4]
if not ready.exists() or not owner.exists():
    raise SystemExit(1)
payload = json.loads(ready.read_text())
if payload.get("state") != "ready" or payload.get("bootId") == old_boot:
    raise SystemExit(1)
if int(owner.read_text()) != expected:
    raise SystemExit(1)
print("{} {}".format(payload["bootId"], payload["pid"]))
' "$WORKSPACE/.runtime-ready.json" "$WORKSPACE/.supervisor.pid" \
            "$supervisor_pid" "$old_boot" 2>/dev/null || true
    )
    if [ -n "$ready" ]; then
        echo "Akashic supervisor ready: supervisor=$supervisor_pid boot_child=$ready"
        echo "Check logs with: tail -f $LOG_FILE"
        exit 0
    fi
    sleep 0.1
done

echo "Supervisor readiness 超时，日志如下：" >&2
tail -80 "$LOG_FILE" >&2
kill "$supervisor_pid" 2>/dev/null || true
rm -f "$PID_FILE"
exit 1
