"""把 workspace 里的一个 HTML 和它引用的同目录资源拼成单文件快照；外部网址原样保留。"""
from __future__ import annotations

import base64
import html
import mimetypes
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

MAX_BYTES = 10 * 1024 * 1024
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
_CSS_URL = re.compile(r"""url\(\s*(?:"([^"]*)"|'([^']*)'|([^)"'\s]*))\s*\)""")
_CSS_IMPORT = re.compile(r"""@import\s+(?:"([^"]*)"|'([^']*)')""")
_ATTR = re.compile(r"""\s([^\s"'>/=]+)(?:\s*=\s*("[^"]*"|'[^']*'|[^\s"'=<>`]+))?""")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_TYPES = {".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf", ".otf": "font/otf",
          ".svg": "image/svg+xml", ".webp": "image/webp", ".avif": "image/avif", ".mjs": "text/javascript"}
# 这些标签的属性指向要内联的资源；a、form、iframe 的地址是导航，不内联。CSS 里的相对引用递归内联。
_RESOURCE_ATTRS: dict[str, tuple[str, ...]] = {
    "img": ("src", "srcset"), "source": ("src", "srcset"), "video": ("src", "poster"), "audio": ("src",),
    "track": ("src",), "input": ("src",), "image": ("href", "xlink:href"), "script": ("src",), "link": ("href",),
}


# 发布失败的原因，原样作为工具错误交还模型。
class SnapshotError(ValueError):
    pass


@dataclass(slots=True)
class Snapshot:
    html: str
    title: str
    inlined: list[str] = field(default_factory=list[str])
    missing: list[str] = field(default_factory=list[str])


@dataclass(slots=True)
class _Budget:
    root: Path
    used: int = 0
    inlined: list[str] = field(default_factory=list[str])
    missing: list[str] = field(default_factory=list[str])

    # 解析一个相对引用：必须落在入口所在目录里（跟随符号链接后再判断），不存在的记下来、保留原引用。
    def locate(self, url: str, base: Path) -> Path | None:
        target = (base / url).resolve()
        if not target.is_relative_to(self.root):
            raise SnapshotError(f"引用越界：{url} 不在 {self.root} 里；只内联入口同目录或子目录的文件")
        if not target.is_file():
            self.missing.append(url)
            return None
        return target

    def read(self, target: Path) -> bytes:
        self.used += target.stat().st_size
        if self.used > MAX_BYTES:
            raise SnapshotError(f"快照超过 {MAX_BYTES // 1024 // 1024} MB 上限（读到 {target.name} 时）；请压缩图片或改用外部地址")
        self.inlined.append(str(target.relative_to(self.root)))
        return target.read_bytes()


# 读入入口 HTML 并内联本地资源；越界、超限直接报错，缺失的资源保留原引用。
def build(entry: Path, title: str | None) -> Snapshot:
    budget = _Budget(entry.parent)
    source = budget.read(entry).decode("utf-8", errors="replace")
    budget.inlined.clear()
    document = _Inliner(source, entry.parent, budget).run()
    if len(document.encode("utf-8")) > MAX_BYTES:
        raise SnapshotError(f"快照超过 {MAX_BYTES // 1024 // 1024} MB 上限；请压缩图片或改用外部地址")
    found = _TITLE.search(source)
    name = title or (html.unescape(found.group(1)).strip() if found else "") or entry.stem
    return Snapshot(document, name[:120], budget.inlined, budget.missing)


def _local(url: str) -> str | None:
    url = url.strip()
    if not url or url.startswith(("#", "//")) or _SCHEME.match(url):
        return None
    return unquote(url.split("#", 1)[0].split("?", 1)[0]) or None


def _data_uri(budget: _Budget, url: str, base: Path, depth: int) -> str | None:
    path = _local(url)
    target = None if path is None else budget.locate(path, base)
    if target is None:
        return None
    data = budget.read(target)
    if target.suffix.lower() == ".css":
        data = _css(data.decode("utf-8", errors="replace"), target.parent, budget, depth + 1).encode("utf-8")
    kind = _TYPES.get(target.suffix.lower()) or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return f"data:{kind};base64,{base64.b64encode(data).decode('ascii')}"


# CSS 里的 url() 与 @import 换成 data URI；被引用的 CSS 递归处理，深度有上限以防循环引用。
def _css(text: str, base: Path, budget: _Budget, depth: int = 0) -> str:
    if depth > 8:
        raise SnapshotError("CSS @import 嵌套过深或循环引用")

    def replace(match: re.Match[str]) -> str:
        url = next(group for group in match.groups() if group is not None)
        uri = _data_uri(budget, url, base, depth)
        return match.group(0) if uri is None else (f"url({uri})" if match.re is _CSS_URL else f"@import url({uri})")

    return _CSS_URL.sub(replace, _CSS_IMPORT.sub(replace, text))


def _srcset(value: str, base: Path, budget: _Budget) -> str:
    if "data:" in value:
        return value
    items: list[str] = []
    for item in value.split(","):
        url, _, rest = item.strip().partition(" ")
        items.append(" ".join(part for part in ((_data_uri(budget, url, base, 0) or url), rest) if part))
    return ", ".join(items)


# 只改标签文本与 style 内容，其余字节原样保留；脚本正文按 HTML 规则跳过，不会误改字符串里的标签。
class _Inliner(HTMLParser):
    def __init__(self, source: str, base: Path, budget: _Budget) -> None:
        super().__init__(convert_charrefs=False)
        self._source = source
        self._base = base
        self._budget = budget
        self._lines = [0] + [index + 1 for index, character in enumerate(source) if character == "\n"]
        self._edits: list[tuple[int, int, str]] = []
        self._in_style = False

    def run(self) -> str:
        self.feed(self._source)
        self.close()
        parts: list[str] = []
        position = 0
        for start, end, text in sorted(self._edits):
            parts += [self._source[position:start], text]
            position = end
        return "".join(parts) + self._source[position:]

    def _offset(self) -> int:
        line, column = self.getpos()
        return self._lines[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._in_style = tag == "style"
        raw = self.get_starttag_text() or ""
        start = self._offset()
        replacement = self._rewrite(tag, dict(attrs), raw)
        if replacement != raw:
            self._edits.append((start, start + len(raw), replacement))

    handle_startendtag = handle_starttag

    def handle_endtag(self, tag: str) -> None:
        self._in_style = False

    def handle_data(self, data: str) -> None:
        if self._in_style:
            start = self._offset()
            self._edits.append((start, start + len(data), _css(data, self._base, self._budget)))

    # 资源属性换成 data URI；外链脚本与样式也走 data URI，defer、async、media、type 的语义保持不变。
    def _rewrite(self, tag: str, attrs: dict[str, str | None], raw: str) -> str:
        changes: dict[str, str | None] = {}
        for name in _RESOURCE_ATTRS.get(tag, ()):
            value = attrs.get(name)
            if value is None:
                continue
            new = _srcset(value, self._base, self._budget) if name == "srcset" \
                else (_data_uri(self._budget, value, self._base, 0) or value)
            if new != value:
                changes[name] = new
        if (style := attrs.get("style")) and (new := _css(style, self._base, self._budget)) != style:
            changes["style"] = new
        return _edit_attrs(raw, changes) if changes else raw


# 在原标签文本上逐个改属性：值为 None 删除该属性，其余写成双引号转义后的新值；属性名大小写与其他属性不动。
def _edit_attrs(raw: str, changes: dict[str, str | None]) -> str:
    head = re.match(r"<[^\s/>]+", raw)
    if head is None:
        return raw
    parts = [raw[:head.end()]]
    position = head.end()
    for match in _ATTR.finditer(raw, head.end()):
        if match.start() != position:
            break
        name = match.group(1).lower()
        if name in changes:
            value = changes[name]
            parts.append("" if value is None else f' {match.group(1)}="{html.escape(value)}"')
        else:
            parts.append(match.group(0))
        position = match.end()
    return "".join(parts) + raw[position:]
