from __future__ import annotations

from dataclasses import dataclass, field

from agent.memes.catalog import MemeCatalog


def _empty_media() -> list[str]:
    return []


@dataclass
class DecorateResult:
    content: str
    media: list[str] = field(default_factory=_empty_media)
    tag: str | None = None


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
