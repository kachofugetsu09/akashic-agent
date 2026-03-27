from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent.memes.catalog import MemeCatalog

_MEME_RE = re.compile(r"<meme:([a-zA-Z0-9_-]+)>", re.IGNORECASE)


@dataclass
class DecorateResult:
    content: str
    media: list[str] = field(default_factory=list)
    tag: str | None = None


class MemeDecorator:
    def __init__(self, catalog: MemeCatalog) -> None:
        self._catalog = catalog

    def decorate(self, content: str) -> DecorateResult:
        """Extract first <meme:tag>, strip all tags from text, pick one image."""
        first = _MEME_RE.search(content)
        # Remove all meme tags from the text regardless
        cleaned = _MEME_RE.sub("", content).strip()
        if first is None:
            return DecorateResult(content=cleaned)
        tag = first.group(1).lower()
        image = self._catalog.pick_image(tag)
        media = [image] if image else []
        return DecorateResult(content=cleaned, media=media, tag=tag)
