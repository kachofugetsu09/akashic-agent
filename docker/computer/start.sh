#!/bin/sh
set -eu

# Every graphical process shares one session bus. This keeps Chromium, XFCE,
# clipboard ownership, and desktop helpers on the same real user session.
if [ "${1:-}" != "--desktop-session" ]; then
  exec dbus-run-session -- "$0" --desktop-session
fi
shift

if [ "${1:-}" != "--runtime" ]; then
  mkdir -p /data/cache /data/config /data/home /data/profile /data/state
  # The Workload owner proves the old container and mount are gone before start.
  # Chromium leaves host-named singleton links after an abrupt stop; only these
  # ephemeral locks may be removed. Cookies and the rest of the profile stay.
  rm -f \
    /data/profile/SingletonCookie \
    /data/profile/SingletonLock \
    /data/profile/SingletonSocket

  node /usr/local/lib/node_modules/@jackwener/opencli/dist/src/daemon.js &
  daemon_pid=$!
  node /opt/computer/gateway.mjs &
  gateway_pid=$!
  cleanup_control() {
    trap - TERM INT EXIT
    kill -TERM "$gateway_pid" 2>/dev/null || true
    wait "$gateway_pid" 2>/dev/null || true
    kill -TERM "$daemon_pid" 2>/dev/null || true
    wait "$daemon_pid" 2>/dev/null || true
  }
  trap cleanup_control TERM INT EXIT
  wait "$gateway_pid"
  exit
fi

mkdir -p \
  /data/cache \
  /data/config \
  /data/home \
  /data/profile \
  /data/state


cleanup() {
  trap - TERM INT EXIT
  kill -TERM "${browser_pid:-}" 2>/dev/null || true
  wait "${browser_pid:-}" 2>/dev/null || true
  kill -TERM "${desktop_pid:-}" "${display_pid:-}" "${stream_pid:-}" "${xvnc_pid:-}" 2>/dev/null || true
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

# 视频与输入仍在同一 X11 / IPC namespace，随唯一桌面 owner 回收。
/opt/computer/stream-venv/bin/python -m selkies \
  --addr 0.0.0.0 --port 6081 --mode websockets \
  --web-root /opt/computer/stream-web \
  --encoder h264enc --gpu-id=-1 \
  --framerate 30-30 --video-bitrate 4000-4000 --rate-control-mode cbr \
  --audio-enabled false --microphone-enabled false --webcam-enabled false \
  --gamepad-enabled false --publish-input-devices false \
  --command-enabled false --file-transfers "" --printing-enabled false \
  --enable-sharing false --enable-basic-auth false --enable-clipboard true \
  --manual-resolution true --manual-width 1280 --manual-height 800 &
stream_pid=$!

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
  --disable-quic \
  --hide-crash-restore-bubble \
  --no-first-run \
  --no-default-browser-check \
  about:blank &
browser_pid=$!

wait "$browser_pid"
