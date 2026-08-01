#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
package_dir="${AKASHIC_MOBILE_WEB_PACKAGE_DIR:-$repo_root/dist/mobile-web-package}"
archive_path="$package_dir/akashic-mobile-web.zip"
digest_path="$package_dir/akashic-mobile-web.zip.sha256"
stage_dir="$(mktemp -d)"
trap 'rm -rf "$stage_dir"' EXIT

cd "$repo_root"
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
  echo "package:mobile-web requires a clean tracked tree" >&2
  exit 1
fi
if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  echo "package:mobile-web requires no untracked source files" >&2
  exit 1
fi

source_repository="$(git config --get remote.origin.url)"
if [[ "$source_repository" =~ ^git@github.com:(.+)\.git$ ]]; then
  source_repository="https://github.com/${BASH_REMATCH[1]}"
fi
source_commit="$(git rev-parse HEAD)"
source_tree="$(git rev-parse 'HEAD^{tree}')"
source_epoch="$(git show -s --format=%ct HEAD)"

AKASHIC_MOBILE_WEB_OUT_DIR="$stage_dir/web" npm run build:mobile-web
asset_digest="$(
  cd "$stage_dir/web"
  find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d ' ' -f 1
)"

SOURCE_REPOSITORY="$source_repository" \
SOURCE_COMMIT="$source_commit" \
SOURCE_TREE="$source_tree" \
ASSET_DIGEST="$asset_digest" \
node --input-type=module - "$stage_dir/web/akashic-webui-manifest.json" <<'NODE'
import { writeFileSync } from "node:fs";

const manifestPath = process.argv[2];
const manifest = {
  schema_version: 1,
  source_repository: process.env.SOURCE_REPOSITORY,
  source_commit: process.env.SOURCE_COMMIT,
  source_tree: process.env.SOURCE_TREE,
  entrypoint: "mobile.html",
  asset_digest: process.env.ASSET_DIGEST,
};
writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
NODE

find "$stage_dir/web" -exec touch -d "@$source_epoch" {} +
mkdir -p "$package_dir"
rm -f "$archive_path" "$digest_path"
(
  cd "$stage_dir/web"
  find . -type f -print | LC_ALL=C sort | zip -X -q "$archive_path" -@
)
archive_digest="$(sha256sum "$archive_path" | cut -d ' ' -f 1)"
printf '%s  %s\n' "$archive_digest" "$(basename "$archive_path")" > "$digest_path"

echo "$archive_path"
echo "$digest_path"
