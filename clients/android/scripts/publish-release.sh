#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
release_repo="${AKASHIC_ANDROID_RELEASE_REPO:-kachofugetsu09/akashic-mobile-releases}"
version="${1:?usage: publish-release.sh VERSION}"
tag="v$version"
asset="Akashic-Mobile-$tag.apk"

"$repo_dir/scripts/build-release.sh"
cp "$repo_dir/app/build/outputs/apk/release/app-release.apk" "/tmp/$asset"
gh release create "$tag" "/tmp/$asset" \
    --repo "$release_repo" \
    --title "Akashic Mobile $tag" \
    --notes "Akashic Android 客户端 $tag"
rm "/tmp/$asset"
