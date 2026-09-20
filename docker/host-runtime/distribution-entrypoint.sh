#!/usr/bin/env bash
set -euo pipefail

CONFIG="${AKASHIC_CONFIG:?AKASHIC_CONFIG is required}"
WORKSPACE="${AKASHIC_WORKSPACE:?AKASHIC_WORKSPACE is required}"
PLUGIN_HOME="${AKASHIC_PLUGIN_HOME:?AKASHIC_PLUGIN_HOME is required}"
EXPECTED_COMMIT="${AKASHIC_RUNTIME_COMMIT:?AKASHIC_RUNTIME_COMMIT is required}"
EXPECTED_TREE="${AKASHIC_RUNTIME_TREE:?AKASHIC_RUNTIME_TREE is required}"

case "$CONFIG" in
    /*) ;;
    *) echo "AKASHIC_CONFIG 必须是绝对路径" >&2; exit 2 ;;
esac
case "$WORKSPACE" in
    /*) ;;
    *) echo "AKASHIC_WORKSPACE 必须是绝对路径" >&2; exit 2 ;;
esac
case "$PLUGIN_HOME" in
    /*) ;;
    *) echo "AKASHIC_PLUGIN_HOME 必须是绝对路径" >&2; exit 2 ;;
esac

test -r "$CONFIG"
test -r /opt/akashic/runtime-info.json
mkdir -p "$WORKSPACE" "$PLUGIN_HOME"

/opt/venv/bin/python /opt/akashic/source/scripts/distribution_runtime.py check \
    --runtime-info /opt/akashic/runtime-info.json \
    --expected-commit "$EXPECTED_COMMIT" \
    --expected-tree "$EXPECTED_TREE"

/opt/venv/bin/python /opt/akashic/source/scripts/install_plugin_distribution.py \
    --distribution /opt/akashic/distribution \
    --profile /opt/akashic/distribution/profiles/default.json \
    --workspace "$WORKSPACE" \
    --plugins-home "$PLUGIN_HOME" \
    --config "$CONFIG" \
    --ensure-profile \
    --receipt "$WORKSPACE/runtime/distribution-install.json"

command="${1:-supervise}"
shift || true
exec /opt/venv/bin/python /opt/akashic/source/main.py \
    "$command" \
    --config "$CONFIG" \
    --workspace "$WORKSPACE" \
    "$@"
