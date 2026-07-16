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
npm run test:mobile-web-state
npm run build:mobile-web
"$python_bin" -m pytest -q tests/test_plugin_mobile_ui.py \
  tests/mobile_realtime/test_channel.py -k "plugin_ui"
"$python_bin" -m pytest -q \
  tests/test_agent_core_p2_reasoner.py::test_request_confirmation_tool_produces_explicit_runtime_attention \
  tests/test_agent_core_p5_agent_core.py::test_agent_core_process_runs_prepare_prompt_run_commit_in_order \
  tests/test_agent_core_p5_agent_core.py::test_agent_core_process_coerces_empty_reply_before_commit \
  tests/mobile_realtime/test_channel.py::test_stream_deltas_batch_at_50ms_and_flush_before_tool_and_final

flock /tmp/akashic-gradle.lock bash -lc \
  "cd '$repo_root/clients/android' && ANDROID_HOME='${ANDROID_HOME:-$HOME/Android/Sdk}' ./gradlew testDebugUnitTest assembleDebugAndroidTest --no-daemon --max-workers=1"
