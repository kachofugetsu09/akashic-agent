#!/bin/sh
set -eu

# Every graphical process shares one session bus. This keeps Chromium, XFCE,
# clipboard ownership, and desktop helpers on the same real user session.
if [ "${1:-}" != "--desktop-session" ]; then
  exec dbus-run-session -- "$0" --desktop-session
fi
shift

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
    "${desktop_pid:-}" "${display_pid:-}" "${daemon_pid:-}" \
    "${xvnc_pid:-}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup TERM INT EXIT

rm -f /tmp/.X11-unix/X99 /tmp/.X99-lock
Xvnc :99 \
  -geometry 1280x800 \
  -depth 24 \
  -SecurityTypes None \
  -localhost \
  -rfbport 5999 \
  -AlwaysShared &
xvnc_pid=$!

attempt=0
while [ ! -S /tmp/.X11-unix/X99 ]; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 100 ]; then
    echo "Computer display did not create its X socket" >&2
    exit 1
  fi
  sleep 0.1
done

websockify 0.0.0.0:6080 127.0.0.1:5999 &
display_pid=$!

startxfce4 &
desktop_pid=$!

attempt=0
until xprop -root _NET_SUPPORTING_WM_CHECK 2>/dev/null | grep -q "window id # 0x"; do
  attempt=$((attempt + 1))
  if ! kill -0 "$desktop_pid" 2>/dev/null || [ "$attempt" -ge 100 ]; then
    echo "Computer desktop did not start its window manager" >&2
    exit 1
  fi
  sleep 0.1
done

node /usr/local/lib/node_modules/@jackwener/opencli/dist/src/daemon.js &
daemon_pid=$!

chromium \
  --user-data-dir=/data/profile \
  --load-extension=/opt/opencli-extension \
  --remote-debugging-address=127.0.0.1 \
  --remote-debugging-port=9222 \
  --window-size=1280,800 \
  --start-maximized \
  --disable-setuid-sandbox \
  --test-type \
  --disable-dev-shm-usage \
  --disable-gpu \
  --hide-crash-restore-bubble \
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
