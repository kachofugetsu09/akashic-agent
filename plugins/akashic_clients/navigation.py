"""Private Web navigation preferences; Session and scope dimension owners remain the fact owners."""
from __future__ import annotations

from collections.abc import Callable, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from plugins.ledger.contract import OwnerStore, OwnerTransaction
from plugins.ledger.contract import ContentPart, Control
from .services import MessageCatalogPort

_KEY = "navigation:pins"


# kind 为 "session" 时 id 是会话；否则 kind 是会话维度名、id 是该维度的值，由维度 owner 判定是否可用。
class PinReference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    kind: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    id: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def check_identity(self) -> PinReference:
        if self.kind != "session":
            if self.id != self.id.strip() or self.id == "default" or len(self.id) > 128:
                raise ValueError("维度值无效")
        elif not self.id.startswith("akashic:") or not self.id.removeprefix("akashic:").strip() or self.id != self.id.strip():
            raise ValueError("需要完整的 Akashic 会话 ID")
        return self


class PinUpdate(PinReference):
    pinned: bool

    def reference(self) -> PinReference:
        return PinReference(kind=self.kind, id=self.id)


def _references(value: Mapping[str, object] | None) -> list[PinReference]:
    if value is None:
        return []
    items = value.get("pins")
    if set(value) != {"pins"} or not isinstance(items, (tuple, list)):
        raise ValueError("置顶偏好记录损坏")
    result = [PinReference.model_validate(dict(item) if isinstance(item, Mapping) else item) for item in items]
    if len(set(result)) != len(result):
        raise ValueError("置顶偏好包含重复引用")
    return result


class NavigationPreferences:
    """One ordered owner record; only explicit operations add/remove references."""

    def __init__(self, open_store: Callable[[], OwnerStore]):
        self._open_store = open_store

    def read(self) -> list[PinReference]:
        record = self._open_store().read(_KEY)
        return _references(None if record is None else record.value)

    def update(self, reference: PinReference, *, pinned: bool) -> None:
        def save(transaction: OwnerTransaction) -> None:
            current = transaction.read(_KEY)
            refs = _references(None if current is None else current.value)
            if (reference in refs) == pinned:
                return
            if pinned:
                refs.append(reference)
            else:
                refs.remove(reference)
            transaction.save(_KEY, {"pins": [ref.model_dump() for ref in refs]},
                             expected_version=None if current is None else current.version)
        self._open_store().transact(save)


async def session_pin_row(catalog: MessageCatalogPort, session_id: str) -> dict[str, object] | None:
    """Point-read the real scope and first Message, independent of recent paging.

    四项点读合并进一次可排空 worker 行程，不占用事件循环；结果与同快照的
    同步逐项读取一致。
    """
    reader = catalog.reader(session_id)
    try:
        first_page, attributes, deleted, title = await reader.read_async(
            lambda row: (row.read_page(limit=1), row.attributes, row.deleted, row.title))
    except KeyError:
        return None
    # 带会话维度（如项目）的会话也可置顶：前端把它排在所在分组内，scope 随行返回。
    if deleted or attributes.visibility != "listed":
        return None
    first = first_page.messages[0] if first_page.messages else None
    text = "" if first is None or isinstance(first.body, Control) else "\n".join(
        str(part.value) for part in first.body.parts if isinstance(part, ContentPart) and part.kind == "text"
    )
    return {"key": session_id, "first_message_content": text, "scope": dict(attributes.scope),
            "title": title}
