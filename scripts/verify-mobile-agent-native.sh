#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${AKASHIC_PYTHON:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python"
fi

npm run typecheck
npm run lint
npm run build:mobile-web
"$python_bin" -m pytest -q tests/test_plugin_mobile_ui.py \
  tests/mobile_realtime/test_channel.py -k "plugin_ui"

flock /tmp/akashic-gradle.lock bash -lc \
  "cd '$repo_root/clients/android' && ANDROID_HOME='${ANDROID_HOME:-$HOME/Android/Sdk}' ./gradlew testDebugUnitTest --no-daemon --max-workers=1"
