from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def _empty_aliases() -> list[str]:
    return []


def _empty_media() -> list[str]:
    return []


@dataclass
class MemeCategory:
    name: str
    desc: str
    aliases: list[str] = field(default_factory=_empty_aliases)
    enabled: bool = True


@dataclass
class DecorateResult:
    content: str
    media: list[str] = field(default_factory=_empty_media)
    tag: str | None = None


class MemeCatalog:
    def __init__(self, memes_dir: Path) -> None:
        self._dir = memes_dir
        self._categories: dict[str, MemeCategory] = {}
        self._manifest_mtime: float = -1.0

    def _load(self) -> None:
        manifest = self._dir / "manifest.json"
        if not manifest.exists():
            self._categories = {}
            self._manifest_mtime = -1.0
            return
        mtime = manifest.stat().st_mtime
        if mtime == self._manifest_mtime:
            return
        self._manifest_mtime = mtime
        self._categories = {}
        try:
            raw: object = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        data = cast(dict[str, object], raw)
        categories = data.get("categories")
        if not isinstance(categories, dict):
            return
        categories_map = cast(dict[object, object], categories)
        for raw_name, raw_info in categories_map.items():
            if not isinstance(raw_name, str) or not isinstance(raw_info, dict):
                continue
            info = cast(dict[str, object], raw_info)
            aliases = info.get("aliases", [])
            alias_items = cast(list[object], aliases) if isinstance(aliases, list) else []
            self._categories[raw_name] = MemeCategory(
                name=raw_name,
                desc=str(info.get("desc", "") or ""),
                aliases=[str(item) for item in alias_items],
                enabled=bool(info.get("enabled", True)),
            )

    def get_enabled_categories(self) -> list[MemeCategory]:
        self._load()
        return [c for c in self._categories.values() if c.enabled]

    def pick_image(self, tag: str) -> str | None:
        self._load()
        tag = tag.lower()
        cat = self._categories.get(tag)
        if cat is None or not cat.enabled:
            return None
        cat_dir = self._dir / tag
        if not cat_dir.is_dir():
            return None
        images = [f for f in cat_dir.iterdir() if f.suffix.lower() in _IMAGE_SUFFIXES]
        if not images:
            return None
        return str(random.choice(images))

    def build_prompt_block(self) -> str | None:
        cats = self.get_enabled_categories()
        if not cats:
            return None
        lines = [
            '【表情协议】`<meme:tag>` 是系统内置回复格式标记，不是 emoji（Unicode 表情符号），不受【禁止 emoji】规则限制。',
            "",
            "可用表情类别：",
        ]
        for cat in cats:
            lines.append(f"- {cat.name}: {cat.desc}")
        lines += [
            "",
            "这是内置表情协议，不是工具能力。",
            '需要发表情时，直接在回复末尾插入 <meme:category>；不要调用任何工具去"生成表情""搜索表情包""发送图片"。',
            "每条回复最多 1 个 <meme:category>，放在整条回复的最末尾（颜文字之后也算末尾，可以紧跟颜文字后面加）。",
            '用户明确说"发个表情""用表情表达你的心情""来个表情包""给我一个表情"时，优先使用 <meme:category> 响应。',
            "用户直球表达喜欢、夸你、气氛暧昧或明显害羞时，也优先在结尾加 <meme:category>，即使已经用了颜文字也要加。",
            "严肃任务、代码解释、工具结果、查资料、执行指令时不使用。",
            "注意：历史会话中助手未使用 <meme:> 不代表本轮不需要用，以上规则优先于历史回复模式。",
            "",
            "<example>",
            "对方说：最喜欢你了 → 回复结尾加 <meme:shy>",
            "对方说：我好喜欢你 → 回复结尾加 <meme:shy>",
            "对方说：akashic你真好 → 回复结尾加 <meme:shy>",
            "对方说：你真好 → 回复结尾加 <meme:shy>",
            "对方说：你今天好棒 → 回复结尾加 <meme:shy>",
            "对方说：谢谢你今天帮了我好多 → 回复结尾加 <meme:shy> 或 <meme:happy>",
            "对方说：你好可爱 → 回复结尾加 <meme:shy>",
            "已经用了颜文字、对方直球说喜欢 → 还是加 <meme:shy>",
            "对方说：给我发个表情表达你的心情 → 正文后直接加 <meme:shy>",
            "对方说：来个表情包 → 不找工具，直接回复并加 <meme:happy> 或 <meme:shy>",
            "任务完成、对方说谢谢 → 回复结尾加 <meme:happy>",
            "轻松聊天、说了个小笑话 → 回复结尾加 <meme:clever>",
            "被夸、被顺毛、被直球关心 → 回复结尾加 <meme:shy>",
            "被戳穿、说错话后 → 回复结尾加 <meme:awkward>",
            "帮忙查资料、执行了指令 → 不加",
            "用户要表情 → 不调用 tool_search，不调用任何工具",
            "</example>",
        ]
        return "\n".join(lines)


class MemeDecorator:
    def __init__(self, catalog: MemeCatalog) -> None:
        self._catalog = catalog

    def decorate(self, content: str, *, meme_tag: str | None = None) -> DecorateResult:
        cleaned = content.strip()
        if meme_tag is None:
            return DecorateResult(content=cleaned)
        tag = meme_tag.lower()
        image = self._catalog.pick_image(tag)
        media = [image] if image else []
        return DecorateResult(content=cleaned, media=media, tag=tag)
