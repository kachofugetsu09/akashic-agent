#!/bin/sh
set -eu

mkdir -p \
  /data/cache \
  /data/config \
  /data/home \
  /data/profile \
  /data/state

# The Workload owner proves the old container and mount are gone before start.
# Chromium leaves host-named singleton links after an abrupt stop; only these
# ephemeral locks may be removed. Cookies and the rest of the profile stay.
rm -f \
  /data/profile/SingletonCookie \
  /data/profile/SingletonLock \
  /data/profile/SingletonSocket

cleanup() {
  trap - TERM INT EXIT
  kill -TERM "${refresh_pid:-}" "${gateway_pid:-}" "${browser_pid:-}" \
    "${daemon_pid:-}" "${xvfb_pid:-}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup TERM INT EXIT

Xvfb :99 -screen 0 1280x800x24 -nolisten tcp &
xvfb_pid=$!

node /usr/local/lib/node_modules/@jackwener/opencli/dist/src/daemon.js &
daemon_pid=$!

chromium \
  --user-data-dir=/data/profile \
  --load-extension=/opt/opencli-extension \
  --remote-debugging-address=127.0.0.1 \
  --remote-debugging-port=9222 \
  --window-size=1280,800 \
  --no-sandbox \
  --disable-dev-shm-usage \
  --disable-gpu \
  --disable-software-rasterizer \
  --no-first-run \
  --no-default-browser-check \
  about:blank &
browser_pid=$!

node /opt/computer/gateway.mjs &
gateway_pid=$!

(
  sleep 900
  while :; do
    if opencli auth refresh \
      --site "${OPENCLI_AUTH_REFRESH_SITES}" \
      --concurrency 2 \
      --timeout 45 \
      --format json; then
      touch /data/state/auth-refresh.ok
      delay=43200
    else
      echo "OpenCLI login refresh failed; retrying in 15 minutes" >&2
      delay=900
    fi
    sleep "$delay"
  done
) &
refresh_pid=$!

wait "$gateway_pid"
