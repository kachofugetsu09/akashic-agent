#!/usr/bin/env python3
"""Build bounded WebUI font shards from the authoritative LXGW source font."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import fontTools
from fontTools.ttLib import TTFont


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FONT = REPOSITORY_ROOT / "frontend/theme/assets/fonts/LXGWWenKaiGBScreen.woff2"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "frontend/theme/src/fonts"
SOURCE_SHA256 = "a792b1dcab65066de0a7996d738ae5d0bcca0371fa72dafb34efc33e1123c2c0"
FONTTOOLS_VERSION = "4.63.0"
MAX_MOBILE_WEBUI_FILE_BYTES = 8 * 1024 * 1024
SHARDS = (
    ("lxgw-wenkai-gb-screen-0.woff2", "U+0000-4DFF"),
    ("lxgw-wenkai-gb-screen-1.woff2", "U+4E00-7FFF"),
    ("lxgw-wenkai-gb-screen-2.woff2", "U+8000-ABFF"),
    ("lxgw-wenkai-gb-screen-3.woff2", "U+AC00-10FFFF"),
)


def _font_codepoints(path: Path) -> set[int]:
    """Return every Unicode codepoint mapped by a font."""

    with TTFont(path, lazy=True) as font:
        return set(font.getBestCmap())


def _build_css() -> str:
    """Build the composite font CSS from the single shard contract."""

    blocks = [
        "/* Generated from LXGW WenKai GB Screen v1.522 with scripts/split-paper-font.py. */",
        "",
    ]
    for filename, unicode_range in SHARDS:
        blocks.extend(
            [
                "@font-face {",
                '  font-family: "LXGW WenKai GB Screen";',
                f'  src: local("LXGW WenKai GB Screen"), url("./{filename}") format("woff2");',
                "  font-weight: 400;",
                "  font-style: normal;",
                "  font-display: swap;",
                f"  unicode-range: {unicode_range};",
                "}",
                "",
            ]
        )
    return "\n".join(blocks)


def main() -> int:
    """Generate and verify before replacing each checked-in runtime file."""

    # 1. Refuse a different tool or source instead of silently changing glyph bytes.
    if fontTools.__version__ != FONTTOOLS_VERSION:
        raise RuntimeError(
            f"FontTools {FONTTOOLS_VERSION} required, found {fontTools.__version__}"
        )
    source_digest = hashlib.sha256(SOURCE_FONT.read_bytes()).hexdigest()
    if source_digest != SOURCE_SHA256:
        raise RuntimeError(f"unexpected LXGW source digest: {source_digest}")

    # 2. Generate every subset in staging so generation failures cannot mix files.
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="lxgw-wenkai-shards-", dir=OUTPUT_DIRECTORY.parent
    ) as temporary:
        staging = Path(temporary)
        for filename, unicode_range in SHARDS:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fontTools",
                    "subset",
                    str(SOURCE_FONT),
                    f"--unicodes={unicode_range}",
                    "--flavor=woff2",
                    f"--output-file={staging / filename}",
                ],
                check=True,
            )

        # 3. Prove size and cmap invariants before replacing any checked-in file.
        shard_codepoints: set[int] = set()
        for filename, _ in SHARDS:
            shard = staging / filename
            if shard.stat().st_size > MAX_MOBILE_WEBUI_FILE_BYTES:
                raise RuntimeError(f"Mobile WebUI font shard is too large: {filename}")
            shard_codepoints.update(_font_codepoints(shard))
        source_codepoints = _font_codepoints(SOURCE_FONT)
        if shard_codepoints != source_codepoints:
            missing = len(source_codepoints - shard_codepoints)
            unexpected = len(shard_codepoints - source_codepoints)
            raise RuntimeError(
                f"font shard cmap mismatch: missing={missing} unexpected={unexpected}"
            )

        css_path = staging / "lxgw-wenkai-gb-screen.css"
        css_path.write_text(_build_css(), encoding="utf-8")
        for filename, _ in SHARDS:
            os.replace(staging / filename, OUTPUT_DIRECTORY / filename)
        os.replace(css_path, OUTPUT_DIRECTORY / css_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
