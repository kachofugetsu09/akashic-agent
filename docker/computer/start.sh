#!/bin/sh
set -eu

if [ "${1:-}" != "--runtime" ]; then
  mkdir -p /data/cache /data/config /data/home /data/profile /data/state
  # The Workload owner proves the old container and mount are gone before start.
  # Chromium leaves host-named singleton links after an abrupt stop; only these
  # ephemeral locks may be removed. Cookies and the rest of the profile stay.
  rm -f \
    /data/profile/SingletonCookie \
    /data/profile/SingletonLock \
    /data/profile/SingletonSocket

  # The shell owns shutdown order. Keep D-Bus alive until Chromium flushes its profile.
  export DBUS_SESSION_BUS_ADDRESS=unix:path=/tmp/computer-session-bus
  rm -f /tmp/computer-session-bus
  dbus-daemon --session --nofork --address="$DBUS_SESSION_BUS_ADDRESS" &
  bus_pid=$!
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
    kill -TERM "$bus_pid" 2>/dev/null || true
    wait "$bus_pid" 2>/dev/null || true
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
  # 编码服务先释放 XShm 和输入连接，X server 仍在时才能完成清理。
  kill -TERM "${stream_pid:-}" 2>/dev/null || true
  wait "${stream_pid:-}" 2>/dev/null || true
  kill -TERM "${wm_pid:-}" 2>/dev/null || true
  kill -TERM "${xvnc_pid:-}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup TERM INT EXIT

# 屏幕尺寸跟随观看面板（selkies 按 RandR 调整），这里只是还没人看时的初始值。
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
    echo "Browser display did not create its X socket" >&2
    exit 1
  fi
  sleep 0.1
done

# 有人看才编码：60fps、恒定质量，静止时补一帧高质量画面让文字清晰；光标由观看端本地绘制。
/opt/computer/stream-venv/bin/python -m selkies \
  --addr 0.0.0.0 --port 6081 --mode websockets \
  --web-root /opt/computer/stream-web \
  --encoder h264enc --gpu-id=-1 \
  --framerate 60,15-60 --rate-control-mode crf --video-crf 25,18-35 \
  --video-paintover-crf 18 --video-fullcolor true \
  --manual-resolution false --enable-resize true --use-css-scaling false \
  --use-browser-cursors true \
  --audio-enabled false --microphone-enabled false --webcam-enabled false \
  --gamepad-enabled false --publish-input-devices false \
  --command-enabled false --file-transfers "" --printing-enabled false \
  --enable-sharing false --enable-basic-auth false --enable-clipboard true &
stream_pid=$!

# matchbox 让每个窗口按当前屏幕尺寸铺满并跟随 RandR；没有标题栏，也不画光标。
matchbox-window-manager -use_titlebar no -use_cursor no &
wm_pid=$!

# kiosk 不画标签条和地址栏，那些由观看面板本地绘制。
# 每个 Agent 标签是独立窗口，被挡住的窗口和后台标签也照常渲染、计时。
# 关掉最后一个窗口时浏览器不退出，由 driver 补开空白页；OpenCLI 扩展调试时不弹提示条占画面。
chromium \
  --user-data-dir=/data/profile \
  --password-store=basic \
  --load-extension=/opt/opencli-extension \
  --remote-debugging-address=127.0.0.1 \
  --remote-debugging-port=9222 \
  --kiosk \
  --keep-alive-for-test \
  --silent-debugger-extension-api \
  --disable-backgrounding-occluded-windows \
  --disable-background-timer-throttling \
  --disable-renderer-backgrounding \
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
