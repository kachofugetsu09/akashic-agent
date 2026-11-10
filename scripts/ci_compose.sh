#!/bin/sh
# CI：把 Core 检出切换到合成提交，并把基线的合成提交写进 COMPOSED_BASE。
# PLUGINS_COMMIT 非空时（插件仓库 PR 调用）用这个插件提交替代 plugins.lock。
# BASE 为空时不合成基线（浅检出的作业不需要基线）；PLUGINS_BASE 非空时基线用这个插件提交。
set -eu
plugins_source="${PLUGINS_SOURCE:-https://github.com/kachofugetsu09/akashic-plugins.git}"
base="${BASE:-}"
if [ ! -f plugins.lock ]; then
  [ -n "${GITHUB_ENV:-}" ] && echo "COMPOSED_BASE=$base" >> "$GITHUB_ENV"
  exit 0
fi
if [ -n "${PLUGINS_COMMIT:-}" ]; then
  head="$(python3 scripts/compose_source.py --repository . --commit HEAD \
    --plugins-source "$plugins_source" --plugins-commit "$PLUGINS_COMMIT")"
else
  head="$(python3 scripts/compose_source.py --repository . --commit HEAD --require-on-branch main)"
fi
if [ -n "$base" ] && [ -n "${PLUGINS_BASE:-}" ]; then
  base="$(python3 scripts/compose_source.py --repository . --commit "$base" \
    --plugins-source "$plugins_source" --plugins-commit "$PLUGINS_BASE")"
elif [ -n "$base" ]; then
  base="$(python3 scripts/compose_source.py --repository . --commit "$base")"
fi
git checkout --quiet --detach "$head"
[ -n "${GITHUB_ENV:-}" ] && echo "COMPOSED_BASE=$base" >> "$GITHUB_ENV"
echo "composed $head (base ${base:-none})"
