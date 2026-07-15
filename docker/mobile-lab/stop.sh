#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export AKASHIC_MOBILE_LAB_TUNNEL_ID=unused
export AKASHIC_MOBILE_LAB_TUNNEL_CREDENTIALS=/dev/null
export AKASHIC_MOBILE_LAB_PUBLIC_URL=wss://unused.invalid/ws
export AKASHIC_MOBILE_LAB_LAN_HOSTNAME=127.0.0.1

docker compose -f "$repo_dir/docker/mobile-lab/compose.yml" down
