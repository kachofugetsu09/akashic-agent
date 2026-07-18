#!/usr/bin/env bash
set -euo pipefail

readonly project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="$project_root/config.toml"
workspace_override=""

usage() {
    cat <<'EOF'
Usage: ./scripts/stop-runtime.sh [--config PATH] [--workspace PATH]

Gracefully stop the supervisor or runtime that owns the selected workspace.
EOF
}

die() {
    printf 'stop-runtime.sh: %s\n' "$*" >&2
    exit 1
}

while (($# > 0)); do
    case "$1" in
        --config)
            (($# >= 2)) || die "--config requires a path"
            config_path="$2"
            shift 2
            ;;
        --workspace)
            (($# >= 2)) || die "--workspace requires a path"
            workspace_override="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unsupported argument: $1"
            ;;
    esac
done

command -v flock >/dev/null 2>&1 || die "flock is required"
command -v fuser >/dev/null 2>&1 || die "fuser is required"

python_bin="${AKASHIC_PYTHON:-$project_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    python_bin="$(command -v python3 || true)"
fi
[[ -n "$python_bin" ]] || die "Python 3 is required to resolve the workspace"

cd "$project_root"
workspace="$("$python_bin" - "$config_path" "$workspace_override" <<'PY'
import os
import sys
import tomllib
from pathlib import Path

config_path = Path(sys.argv[1])
workspace_override = sys.argv[2]
workspace_value = workspace_override or os.environ.get("AKASHIC_WORKSPACE", "")
if not workspace_value.strip():
    with config_path.open("rb") as stream:
        runtime = tomllib.load(stream).get("runtime")
    if not isinstance(runtime, dict):
        raise SystemExit(f"配置文件 {config_path} 缺少 [runtime] table")
    configured = runtime.get("workspace")
    if not isinstance(configured, str) or not configured.strip():
        raise SystemExit(f"配置文件 {config_path} 缺少 runtime.workspace")
    workspace_value = configured
print(Path(workspace_value.strip()).expanduser().resolve())
PY
)"

supervisor_lock="$workspace/.supervisor.lock"
runtime_lock="$workspace/.instance.lock"

lock_is_free() {
    local lock_path="$1"
    [[ ! -e "$lock_path" ]] || flock -n "$lock_path" -c true
}

lock_owner() {
    local lock_path="$1"
    fuser "$lock_path" 2>/dev/null | tr ' ' '\n' | sed '/^$/d'
}

target_lock=""
if ! lock_is_free "$supervisor_lock"; then
    target_lock="$supervisor_lock"
elif ! lock_is_free "$runtime_lock"; then
    target_lock="$runtime_lock"
else
    printf 'workspace 未运行: %s\n' "$workspace"
    exit 0
fi

mapfile -t owner_pids < <(lock_owner "$target_lock")
((${#owner_pids[@]} == 1)) ||
    die "无法确定唯一锁 owner: $target_lock owners=${owner_pids[*]:-unknown}"
owner_pid="${owner_pids[0]}"
[[ "$owner_pid" =~ ^[0-9]+$ ]] || die "锁 owner 不是有效 PID: $owner_pid"

printf '正在停止 workspace runtime: %s pid=%s\n' "$workspace" "$owner_pid"
kill -TERM "$owner_pid"

readonly deadline=$((SECONDS + 30))
while ! lock_is_free "$supervisor_lock" || ! lock_is_free "$runtime_lock"; do
    ((SECONDS < deadline)) ||
        die "runtime 在 30 秒内未释放 workspace；未执行强制终止"
    sleep 0.2
done

printf 'workspace 已停止: %s\n' "$workspace"
