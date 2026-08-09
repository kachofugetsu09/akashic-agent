#!/usr/bin/env bash
set -euo pipefail

CONFIG="${AKASHIC_CONFIG:?AKASHIC_CONFIG is required}"
WORKSPACE="${AKASHIC_WORKSPACE:?AKASHIC_WORKSPACE is required}"

case "$CONFIG" in
    /*) ;;
    *) echo "AKASHIC_CONFIG 必须是绝对路径" >&2; exit 2 ;;
esac
case "$WORKSPACE" in
    /*) ;;
    *) echo "AKASHIC_WORKSPACE 必须是绝对路径" >&2; exit 2 ;;
esac

test -f /opt/akashic/runtime-info.json
test -r "$CONFIG"
mkdir -p "$WORKSPACE"

command="${1:-supervise}"
shift || true
exec /opt/venv/bin/python /opt/akashic/source/main.py \
    "$command" \
    --config "$CONFIG" \
    --workspace "$WORKSPACE" \
    "$@"
