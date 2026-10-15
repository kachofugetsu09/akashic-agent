#!/usr/bin/env python3
# 把 Noto Serif SC 可变字体按字频分层切成 WOFF2 分片，并生成 reading-serif.css。
#
# 用法：python scripts/split-reading-font.py <NotoSerifSC[wght].ttf>
# 源字体取自 https://github.com/google/fonts/tree/main/ofl/notoserifsc（摘要见 SOURCE_SHA256）。

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import fontTools
from fontTools.ttLib import TTFont


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "frontend/theme/src/fonts"
CSS_NAME = "reading-serif.css"
SOURCE_SHA256 = "050080d9255a86808f2945bffac582b31ef32bc36411ce29563b4961670c66f9"
FONTTOOLS_VERSION = "4.66.1"
# 生僻区每片的字数：一个偶然出现的生僻字只多下载一片（约 330KB），而不是整个码段。
RARE_CHUNK = 800
# 二级字每片的字数：阈、渚这类现代常见字在二级字里，一页常有几个；按部首顺序细切，几个字只多下载几片（约 200KB）。
LEVEL2_CHUNK = 500

SOURCE_SERIF_CSS = """@font-face {
  font-family: "Source Serif 4";
  src: url("./source-serif.woff2") format("woff2");
  font-weight: 200 900;
  font-style: normal;
  font-display: swap;
}

@font-face {
  font-family: "Source Serif 4";
  src: url("./source-serif-italic.woff2") format("woff2");
  font-weight: 200 900;
  font-style: italic;
  font-display: swap;
}
"""


@dataclass(frozen=True)
class Shard:
    filename: str
    # 分片实际包含的码位（互不相交，并集等于源字体 cmap）。
    codepoints: frozenset[int]
    # 写进 CSS 的 unicode-range：宽分片用连续区间，可以盖住后声明分片的码位。
    css_ranges: tuple[tuple[int, int], ...]


# GB2312 一级字（3755 个常用字）与二级字（3008 个次常用字），用编码表本身作字频分层。
def gb2312_level(first: int, last: int) -> list[int]:
    codepoints: list[int] = []
    for high in range(first, last + 1):
        for low in range(0xA1, 0xFF):
            try:
                codepoints.append(ord(bytes([high, low]).decode("gb2312")))
            except UnicodeDecodeError:
                continue
    return codepoints


# 均分成每片约 size 个：不留只有几个字的尾片。
def balanced(items: list[int], size: int) -> list[list[int]]:
    count = max(1, round(len(items) / size))
    step = -(-len(items) // count)
    return [items[index:index + step] for index in range(0, len(items), step)]


def runs(codepoints: set[int] | frozenset[int]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(codepoints)
    result: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for codepoint in ordered[1:]:
        if codepoint != previous + 1:
            result.append((start, previous))
            start = codepoint
        previous = codepoint
    result.append((start, previous))
    return tuple(result)


def css_range(ranges: tuple[tuple[int, int], ...]) -> str:
    return ", ".join(f"U+{a:X}" if a == b else f"U+{a:X}-{b:X}" for a, b in ranges)


# 分层：生僻汉字、扩展 A、增补平面、其余符号先声明，常用层后声明。每片只声明实际含有的码位，区间互不重叠：
# 不能依赖"重叠时后声明的规则先匹配"（CSS Fonts 4）来省流量，Firefox 冷启动整页时会把与页面文字区间相交的
# 宽分片一并下载（实测 7–8MB，精确区间后与 Chromium 同为 1.7MB）。
def plan(source: set[int]) -> list[Shard]:
    # 1. 首屏层：拉丁、通用标点（“”—…）、箭头、CJK 标点与假名、全角形式。
    def in_core(c: int) -> bool:
        return c < 0x2070 or 0x2190 <= c <= 0x21FF or 0x3000 <= c <= 0x30FF or 0xFF00 <= c <= 0xFFEF
    core = frozenset(c for c in source if in_core(c))
    level1 = frozenset(c for c in gb2312_level(0xB0, 0xD7) if c in source)
    level2 = frozenset(c for c in gb2312_level(0xD8, 0xF7) if c in source)
    taken = core | level1 | level2
    shards: list[Shard] = []
    # 2. 生僻区按码位顺序每 RARE_CHUNK 字一片。
    def rare(prefix: str, low: int, high: int) -> None:
        chunks = balanced(sorted(c for c in source if low <= c <= high and c not in taken), RARE_CHUNK)
        for index, chunk in enumerate(chunks):
            codepoints = frozenset(chunk)
            shards.append(Shard(f"{prefix}-{index + 1:02d}.woff2", codepoints, runs(codepoints)))
    rare("noto-serif-sc-han", 0x4E00, 0x9FFF)
    rare("noto-serif-sc-ext-a", 0x3400, 0x4DFF)
    # 3. 其余符号与兼容字形、增补平面各一片。
    other = frozenset(c for c in source if c not in taken and not (0x3400 <= c <= 0x9FFF))
    symbols = frozenset(c for c in other if c < 0x10000)
    astral = frozenset(c for c in other if c >= 0x10000)
    shards.append(Shard("noto-serif-sc-symbols.woff2", symbols, runs(symbols)))
    shards.append(Shard("noto-serif-sc-astral.woff2", astral, runs(astral)))
    # 4. 常用层最后声明：二级字（按 GB2312 部首顺序分片）、一级字、首屏层。
    for index, items in enumerate(balanced([c for c in gb2312_level(0xD8, 0xF7) if c in level2], LEVEL2_CHUNK)):
        chunk = frozenset(items)
        shards.append(Shard(f"noto-serif-sc-common-2-{index + 1:02d}.woff2", chunk, runs(chunk)))
    shards.append(Shard("noto-serif-sc-common-1.woff2", level1, runs(level1)))
    shards.append(Shard("noto-serif-sc-core.woff2", core, runs(core)))
    return shards


# 校验：分片与 CSS 区间都互不相交、并集等于源 cmap，并按 CSS 的逆序匹配规则模拟，每个码位都落到实际含它的分片。
def verify(source: set[int], shards: list[Shard]) -> None:
    declared = sorted((a, b, shard.filename) for shard in shards for a, b in shard.css_ranges)
    for (_, end, first), (start, _, second) in zip(declared, declared[1:]):
        if start <= end:
            raise RuntimeError(f"unicode-range of {first} overlaps {second} at U+{start:X}")
    owner: dict[int, str] = {}
    for shard in shards:
        for codepoint in shard.codepoints:
            if codepoint in owner:
                raise RuntimeError(f"U+{codepoint:X} in both {owner[codepoint]} and {shard.filename}")
            owner[codepoint] = shard.filename
    if set(owner) != source:
        raise RuntimeError(f"cmap mismatch: missing={len(source - set(owner))} unexpected={len(set(owner) - source)}")
    for codepoint in source:
        matched = next(shard for shard in reversed(shards)
                       if any(a <= codepoint <= b for a, b in shard.css_ranges))
        if matched.filename != owner[codepoint]:
            raise RuntimeError(f"U+{codepoint:X} resolves to {matched.filename}, glyph is in {owner[codepoint]}")


def build_css(shards: list[Shard]) -> str:
    blocks = [
        "/* Noto Serif SC 分片由 scripts/split-reading-font.py 生成，不要手改。 */",
        "",
        SOURCE_SERIF_CSS,
    ]
    for shard in shards:
        blocks.extend([
            "@font-face {",
            '  font-family: "Noto Serif SC";',
            f'  src: url("./{shard.filename}") format("woff2");',
            "  font-weight: 200 900;",
            "  font-style: normal;",
            "  font-display: swap;",
            f"  unicode-range: {css_range(shard.css_ranges)};",
            "}",
            "",
        ])
    return "\n".join(blocks)


def main() -> int:
    # 1. 固定工具版本与源字体，避免悄悄换掉字形。
    if fontTools.version != FONTTOOLS_VERSION:
        raise RuntimeError(f"FontTools {FONTTOOLS_VERSION} required, found {fontTools.version}")
    if len(sys.argv) != 2:
        raise SystemExit("usage: split-reading-font.py <NotoSerifSC[wght].ttf>")
    source_font = Path(sys.argv[1])
    digest = hashlib.sha256(source_font.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(f"unexpected Noto Serif SC source digest: {digest}")
    with TTFont(source_font, lazy=True) as font:
        source = set(font.getBestCmap())
    shards = plan(source)
    verify(source, shards)

    # 2. 在临时目录生成全部分片与 CSS，全部成功后才替换仓库文件。
    with tempfile.TemporaryDirectory(prefix="noto-serif-sc-", dir=OUTPUT_DIRECTORY.parent) as temporary:
        staging = Path(temporary)
        for shard in shards:
            unicodes = staging / "unicodes.txt"
            unicodes.write_text(",".join(f"{c:X}" for c in sorted(shard.codepoints)), encoding="utf-8")
            subprocess.run([sys.executable, "-m", "fontTools", "subset", str(source_font),
                            f"--unicodes-file={unicodes}", "--flavor=woff2",
                            f"--output-file={staging / shard.filename}"], check=True)
            with TTFont(staging / shard.filename, lazy=True) as built:
                if set(built.getBestCmap()) != shard.codepoints:
                    raise RuntimeError(f"{shard.filename} cmap differs from plan")
        (staging / CSS_NAME).write_text(build_css(shards), encoding="utf-8")

        # 3. 移走旧分片，换上新分片与 CSS。
        for old in OUTPUT_DIRECTORY.glob("noto-serif-sc-*.woff2"):
            old.unlink()
        for shard in shards:
            os.replace(staging / shard.filename, OUTPUT_DIRECTORY / shard.filename)
        os.replace(staging / CSS_NAME, OUTPUT_DIRECTORY / CSS_NAME)
    for shard in shards:
        size = (OUTPUT_DIRECTORY / shard.filename).stat().st_size
        print(f"{shard.filename}: {len(shard.codepoints)} codepoints, {size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
