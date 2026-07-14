#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
keystore="${AKASHIC_ANDROID_KEYSTORE_PATH:-$HOME/.config/akashic-mobile/release.p12}"
store_password="${AKASHIC_ANDROID_STORE_PASSWORD:-$(secret-tool lookup application akashic-mobile secret store-password)}"
key_password="${AKASHIC_ANDROID_KEY_PASSWORD:-$(secret-tool lookup application akashic-mobile secret key-password)}"

test -f "$keystore"
test -n "$store_password"
test -n "$key_password"

export AKASHIC_ANDROID_KEYSTORE_PATH="$keystore"
export AKASHIC_ANDROID_STORE_PASSWORD="$store_password"
export AKASHIC_ANDROID_KEY_ALIAS="${AKASHIC_ANDROID_KEY_ALIAS:-akashic-mobile}"
export AKASHIC_ANDROID_KEY_PASSWORD="$key_password"
export ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"

cd "$repo_dir"
./gradlew testReleaseUnitTest lintRelease assembleRelease --no-daemon
apksigner="$(find "$ANDROID_HOME/build-tools" -maxdepth 2 -type f -name apksigner -print | sort -V | tail -n 1)"
test -x "$apksigner"
"$apksigner" verify --verbose --print-certs \
    app/build/outputs/apk/release/app-release.apk
