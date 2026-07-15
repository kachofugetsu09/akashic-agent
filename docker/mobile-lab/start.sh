#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tunnel_name="${AKASHIC_MOBILE_LAB_TUNNEL_NAME:-akashic-mobile-lab}"
mapfile -t tunnel_ids < <(cloudflared tunnel list --output json | jq -r --arg name "$tunnel_name" '.[] | select(.name == $name) | .id')
test "${#tunnel_ids[@]}" -eq 1
tunnel_id="${tunnel_ids[0]}"

credentials="$HOME/.cloudflared/$tunnel_id.json"
test -r "$credentials"

export AKASHIC_MOBILE_LAB_TUNNEL_ID="$tunnel_id"
export AKASHIC_MOBILE_LAB_TUNNEL_CREDENTIALS="$credentials"
export AKASHIC_MOBILE_LAB_PUBLIC_URL="${AKASHIC_MOBILE_LAB_PUBLIC_URL:-wss://mobile-lab.wangyuanzhe28.site/ws}"
export AKASHIC_MOBILE_LAB_LAN_HOSTNAME="${AKASHIC_MOBILE_LAB_LAN_HOSTNAME:-$(ip -4 route get 1.1.1.1 | awk '{print $7; exit}')}"
export AKASHIC_HOST_UID="$(id -u)"
export AKASHIC_HOST_GID="$(id -g)"

docker build \
    --file "$repo_dir/docker/debug/Dockerfile" \
    --tag akashic-agent-debug:latest \
    "$repo_dir"
docker compose -f "$repo_dir/docker/mobile-lab/compose.yml" up -d --build
