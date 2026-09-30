"""Private Web navigation preferences; Session and Projects remain the fact owners."""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.plugin_composition.messages import OwnerStore, OwnerTransaction
from agent.plugin_contracts.message import ContentPart, Control
from .services import MessageCatalogPort, PluginUiProvider

_KEY = "navigation:pins"


class PinReference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    kind: Literal["project", "session"]
    id: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def check_identity(self) -> PinReference:
        if self.kind == "project":
            if re.fullmatch(r"p_[0-9a-f]{32}", self.id) is None:
                raise ValueError("项目 ID 无效")
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


def session_pin_row(catalog: MessageCatalogPort, session_id: str) -> dict[str, object] | None:
    """Point-read the real scope and first Message, independent of recent paging."""
    reader = catalog.reader(session_id)
    try:
        first_page = reader.read_page(limit=1)
    except KeyError:
        return None
    attributes = reader.attributes
    if attributes.visibility != "listed" or any(name == "project" for name, _ in attributes.scope):
        return None
    first = first_page.messages[0] if first_page.messages else None
    text = "" if first is None or isinstance(first.body, Control) else "\n".join(
        str(part.value) for part in first.body.parts if isinstance(part, ContentPart) and part.kind == "text"
    )
    return {"key": session_id, "first_message_content": text, "scope": dict(attributes.scope)}


async def check_project_pin(provider: PluginUiProvider, project_id: str) -> None:
    """Ask the current Projects owner, without importing or copying its storage."""
    catalog = await provider.catalog()
    items = catalog.get("items")
    if not isinstance(items, list):
        raise ValueError("插件目录无效")
    projects = [item for item in items if isinstance(item, dict)
                and isinstance(item.get("id"), str) and item["id"].split("@")[0] == "projects"]
    if len(projects) != 1:
        raise ValueError("项目服务暂不可用")
    project = projects[0]
    result = await provider.query(str(project["id"]), str(project["revision"]), "project.list", {},
                                  session_id=None, turn_id=None)
    rows = result.get("items")
    if not isinstance(rows, list):
        raise ValueError("项目目录无效")
    if not any(isinstance(row, dict) and row.get("id") == project_id and row.get("archived") is False for row in rows):
        raise ValueError("只能置顶当前可用的项目")
