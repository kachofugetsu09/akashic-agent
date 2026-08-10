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
mkdir -p "${AKASHIC_PLUGIN_HOME:?AKASHIC_PLUGIN_HOME is required}"

/opt/venv/bin/python -m agent.runtime_identity \
    --runtime-info /opt/akashic/runtime-info.json \
    --release-manifest "${AKASHIC_RELEASE_MANIFEST:?AKASHIC_RELEASE_MANIFEST is required}" \
    --expected-commit "${AKASHIC_RUNTIME_COMMIT:?AKASHIC_RUNTIME_COMMIT is required}" \
    --host-checkout "${AKASHIC_RUNTIME_CHECKOUT:?AKASHIC_RUNTIME_CHECKOUT is required}"

if [[ "${AKASHIC_EXECUTION_MODE:-local}" == "host-bridge" ]]; then
    : "${AKASHIC_HOST_BRIDGE_SOCKET:?AKASHIC_HOST_BRIDGE_SOCKET is required}"
    : "${AKASHIC_HOST_BRIDGE_TOKEN:?AKASHIC_HOST_BRIDGE_TOKEN is required}"
    /opt/venv/bin/python -m agent.host_bridge.doctor \
        --socket "$AKASHIC_HOST_BRIDGE_SOCKET" \
        --token "$AKASHIC_HOST_BRIDGE_TOKEN" \
        --expected-release-commit "$AKASHIC_RUNTIME_COMMIT" \
        --expected-toolchain-digest "${AKASHIC_HOST_TOOLCHAIN_DIGEST:?AKASHIC_HOST_TOOLCHAIN_DIGEST is required}"
fi

command="${1:-supervise}"
shift || true
exec /opt/venv/bin/python /opt/akashic/source/main.py \
    "$command" \
    --config "$CONFIG" \
    --workspace "$WORKSPACE" \
    "$@"
