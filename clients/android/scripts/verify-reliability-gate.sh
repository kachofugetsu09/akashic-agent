#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
android_root="$repo_root/clients/android"

# 1. 验证 WebView 的乱序、未读与投影恢复规则
cd "$repo_root"
npm run test:mobile-web-state
npm run typecheck

# 2. 串行验证 Room 迁移、通知桥和本地可靠性投影
cd "$android_root"
flock -w 300 /tmp/akashic-gradle.lock \
  env ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}" \
  ./gradlew --no-daemon --max-workers=1 testDebugUnitTest assembleDebugAndroidTest

echo "reliability gate passed: web state, Room schema, notification bridge"
