"""文档 owner 发布有界读取口的合同。"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from agent.plugin_composition import Context, Effect, ServiceKey


@dataclass(frozen=True, slots=True)
class Document:
    """owner 发布的只读展示元数据与有界读取口；不授予任意文件访问。"""

    id: str
    title: str
    relative_path: str
    group: str
    description: str
    read: Callable[[int], bytes]
    order: int = 0


class Documents(Protocol):
    async def register(self, ctx: Context, document: Document) -> Effect: ...


DOCUMENTS = ServiceKey[Documents]("inspection.documents.v1")
