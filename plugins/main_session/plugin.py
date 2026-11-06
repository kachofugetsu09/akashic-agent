"""侧栏顶部的对话：只记一个指向会话的指针，经 navigation.destinations 让宿主把品牌行显示成它；不解释记忆与投递。"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal

from agent.plugin_composition import Context
from plugins.ledger.contract import (
    MESSAGE_CATALOG,
    OWNER_STATE,
    ContentPart,
    Input,
    MessageCatalog,
    MessageConflict,
    Output,
    OwnerStore,
    OwnerTransaction,
    is_user_input,
)
from plugins.ui.contract import UI_SLOTS, PluginUiDefinition, PluginUiRpcInvalidRequest, session_preview

api_version = 3
name = "main_session"
version = "1.0.0"
desc = "把一段对话固定在侧栏顶部，名字是 Akashic"
inject = (OWNER_STATE, MESSAGE_CATALOG, UI_SLOTS)

LABEL = "Akashic"
_KEY = "pointer"
# 侧栏只列 Web 聊天目录；宿主为新对话生成的会话 ID 也是这个前缀。
_WEB_PREFIX = "akashic:"
_RECENT = timedelta(days=60)
_CANDIDATES = 8
_TAIL = 30
_PREVIEW_CHARS = 120

type PointerState = Literal["pending", "set", "dismissed"]


class Pointer:
    """指针只有三种已定状态；没有记录表示还没问过。换一段对话只改指针，不迁移任何消息。"""

    def __init__(self, store: OwnerStore, catalog: MessageCatalog):
        self._store = store
        self._catalog = catalog
        # 句数只随会话 head 变化；按 (会话, head) 记住，选择面板反复打开不重读历史。
        self._said: dict[tuple[str, int], int] = {}

    def read(self) -> tuple[PointerState | None, str | None]:
        record = self._store.read(_KEY)
        if record is None:
            return None, None
        state, session_id = record.value.get("state"), record.value.get("session_id")
        if state not in ("pending", "set", "dismissed"):
            raise ValueError("侧栏顶部的对话记录损坏")
        return state, session_id if isinstance(session_id, str) else None

    def write(self, state: PointerState, session_id: str | None = None, *, only_from: tuple[PointerState | None, ...] = ()) -> None:
        def save(transaction: OwnerTransaction) -> None:
            current = transaction.read(_KEY)
            if only_from and (current.value.get("state") if current else None) not in only_from:
                raise PluginUiRpcInvalidRequest("侧栏顶部的对话已经变了，请刷新后再试")
            value: dict[str, object] = {"state": state, "updated_at": datetime.now(UTC).isoformat()}
            if session_id is not None:
                value["session_id"] = session_id
            _ = transaction.save(_KEY, value, expected_version=None if current is None else current.version)
        try:
            self._store.transact(save)
        except MessageConflict as error:
            raise PluginUiRpcInvalidRequest("侧栏顶部的对话正在被修改，请重试") from error

    def usable(self, session_id: str) -> bool:
        return session_id.startswith(_WEB_PREFIX) and self._catalog.exists(session_id) \
            and not self._catalog.reader(session_id).deleted

    # 读出目的地：新用户第一次读时先占一个空位，首条消息落下才有会话。
    def destination(self) -> dict[str, object] | None:
        state, session_id = self.read()
        if state is None and self._catalog.sessions(prefix=_WEB_PREFIX, visibility="listed", limit=1).total == 0:
            try:
                self.write("pending", only_from=(None,))
            except PluginUiRpcInvalidRequest:
                pass  # 另一个页面同时占了位：按它写下的状态读。
            state, session_id = self.read()
        if state == "pending":
            return {"label": LABEL, "hint": "还没聊过，点这里打个招呼"}
        if state != "set" or session_id is None or not self.usable(session_id):
            return None
        preview = self._preview(session_id)
        return {"label": LABEL, "sessionId": session_id, **({"preview": preview} if preview else {})}

    def _preview(self, session_id: str) -> str:
        reader = self._catalog.reader(session_id)
        head = self._catalog.snapshot_heads(prefix=session_id).get(session_id, -1)
        for message in reversed(reader.read(after_seq=max(-1, head - _TAIL), limit=_TAIL)):
            if not isinstance(message.body, (Input, Output)):
                continue
            text = " ".join(" ".join(str(part.value) for part in message.body.parts
                                     if isinstance(part, ContentPart) and part.kind == "text").split())
            if text:
                return ("你：" if is_user_input(message) else "") + text[:_PREVIEW_CHARS]
        return ""

    # 候选：最近的 Web 会话里消息最多的几段，按近 60 天你亲口说的句数排；其余按最近活动列在后面。
    def candidates(self) -> list[dict[str, object]]:
        page = self._catalog.sessions(prefix=_WEB_PREFIX, visibility="listed", limit=200)
        entries = page.items  # 目录已排除软删的会话。
        busiest = sorted(entries, key=lambda entry: -entry.message_count)[:_CANDIDATES]
        said = {entry.session_id: self._said_recently(entry.session_id, entry.head_seq) for entry in busiest}
        counted = sorted((entry for entry in busiest if said[entry.session_id]), key=lambda entry: -said[entry.session_id])
        rest = [entry for entry in entries if not said.get(entry.session_id)]
        state, current = self.read()
        return [{
            "id": entry.session_id, "title": entry.title or session_preview(entry.first_message) or "未命名对话",
            "updated_at": entry.updated_at.isoformat(), **({"said": said[entry.session_id]} if said.get(entry.session_id) else {}),
            **({"current": True} if state == "set" and entry.session_id == current else {}),
        } for entry in counted + rest]

    def _said_recently(self, session_id: str, head: int) -> int:
        cached = self._said.get((session_id, head))
        if cached is not None:
            return cached
        # 1. 从末尾按页往前读，读到 60 天以前就停；只数用户亲口说的输入。
        reader, cutoff, said, through = self._catalog.reader(session_id), datetime.now(UTC) - _RECENT, 0, head
        while through >= 0:
            page = reader.read(after_seq=max(-1, through - 1000), through_seq=through, limit=1000)
            said += sum(1 for message in page if message.recorded_at >= cutoff and is_user_input(message))
            if not page or page[0].recorded_at < cutoff:
                break
            through = page[0].seq - 1
        self._said[(session_id, head)] = said
        return said


def _session_id(payload: dict[str, object]) -> str:
    value = payload.get("session_id")
    if set(payload) != {"session_id"} or not isinstance(value, str) or not value:
        raise PluginUiRpcInvalidRequest("请选择一段对话")
    return value


# 插件界面的查询：读目的地、引导与候选，改指针；同步执行在有界 worker 里。
def _query(pointer: Pointer, method: str, payload: dict[str, object]) -> dict[str, object]:
    if method == "destination.load" and not payload:
        return {"destination": pointer.destination()}
    if method == "guide.read" and not payload:
        state, _ = pointer.read()
        if state is not None:
            return {"show": False}
        best = next(iter(pointer.candidates()), None)
        return {"show": True, "suggestion": best if best and best.get("said") else None}
    if method == "candidates.list" and not payload:
        return {"items": pointer.candidates()}
    if method == "pointer.new" and not payload:
        pointer.write("pending")
        return {}
    if method == "pointer.dismiss" and not payload:
        pointer.write("dismissed", only_from=(None,))
        return {}
    if method in ("pointer.set", "pointer.adopt"):
        session_id = _session_id(payload)
        if not session_id.startswith(_WEB_PREFIX):
            raise PluginUiRpcInvalidRequest("只能放 Web 聊天里的对话")
        if method == "pointer.set" and not pointer.usable(session_id):
            raise PluginUiRpcInvalidRequest("这段对话不存在或已删除")
        # 新对话的首条消息可能还没落账：只认空位状态，不校验会话已存在。
        pointer.write("set", session_id, only_from=("pending",) if method == "pointer.adopt" else ())
        return {}
    raise PluginUiRpcInvalidRequest(f"不支持的查询：{method}")


async def apply(ctx: Context) -> None:
    pointer = Pointer(ctx.require(OWNER_STATE).open(ctx), ctx.require(MESSAGE_CATALOG))

    def query(method: str, payload: dict[str, object], *, session_id: str | None, turn_id: str | None) -> dict[str, object]:
        return _query(pointer, method, payload)

    _ = await ctx.require(UI_SLOTS).register_plugin_ui(
        ctx, PluginUiDefinition(module="plugin_ui.js", stylesheet="plugin_ui.css", slots=("navigation.destinations",)),
        query=query,
    )
