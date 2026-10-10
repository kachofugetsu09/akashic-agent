"""查询事实与原始 Message 的只读投影；不重算实际召回。"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import cast

from plugins.ui.contract import PluginUiRpcInvalidRequest
from plugins.ledger.contract import MessageCatalog
from plugins.ledger.contract import CallRef, ContentPart, Input, Message, Output, ToolCall, ToolResult, json_value
from plugins.tools.contract import durable_call_key
from ._boundaries import Turn, TurnProjection
from .recalls import ContextSource, ProgramSource, Recall, ToolSource, context_identity
from .recall_tool import RecallReference


@dataclass(frozen=True, slots=True)
class _MessageRef:
    identity: str
    seq: int
    user_input: bool
    calls: tuple[CallRef, ...]
    result: CallRef | None


@dataclass(frozen=True, slots=True)
class _SourceTurns:
    key: tuple[str, str, int]
    turns: tuple[Turn, ...]
    messages: tuple[_MessageRef, ...]


class RecallInspector:
    """只读查询事实与原消息，不把学习读出重算成实际召回。"""

    def __init__(self, *, read: Callable[[str], Recall | None],
                 list_records: Callable[[], tuple[tuple[str, Recall], ...]],
                 legacy_page: Callable[[str], tuple[tuple[tuple[str, Recall], ...], str | None]],
                 catalog: MessageCatalog):
        self._read = read
        self._list = list_records
        self._legacy_page = legacy_page
        self._catalog = catalog
        self._turns: _SourceTurns | None = None
        self._lock = Lock()
        self._legacy_before: str | None = "g"
        self._legacy: dict[tuple[str, str], list[tuple[datetime, str, int]]] = {}

    def _source_turns(self, session_id: str, source: str, projection: TurnProjection) -> _SourceTurns:
        """Reuse closed references; only the open tail needs another projection."""
        reader = self._catalog.reader(session_id)
        with self._lock:
            head = reader.head(source=source)
            key = (session_id, source, head)
            cached = self._turns
            if cached is not None and cached.key == key:
                return cached
            closed: tuple[Turn, ...] = ()
            prefix: tuple[_MessageRef, ...] = ()
            if cached is not None and cached.key[:2] == key[:2] and cached.key[2] < head:
                closed = tuple(turn for turn in cached.turns if turn.status != "open")
                if closed:
                    prefix = tuple(item for item in cached.messages if item.seq <= closed[-1].through_seq)
            boundary = closed[-1].through_seq if closed else -1
            def project(messages: Iterable[Message]) -> _SourceTurns:
                refs: list[_MessageRef] = []
                def observed() -> Iterator[Message]:
                    for message in messages:
                        body = message.body
                        refs.append(_MessageRef(
                            message.message_id, message.seq,
                            isinstance(body, Input) and message.author == "user",
                            tuple(CallRef(message.message_id, index) for index, part in enumerate(body.parts)
                                  if isinstance(part, ToolCall)) if isinstance(body, Output) else (),
                            body.call_ref if isinstance(body, ToolResult) else None,
                        ))
                        yield message
                turns = projection.project(observed(), source, after_seq=boundary)
                return _SourceTurns(key, closed + turns, prefix + tuple(refs))
            self._turns = reader.scan(project, after_seq=boundary, through_seq=head, source=source)
            return self._turns

    def _legacy_automatic(self, session_id: str, source: str, after: int, through: int) -> tuple[str | None, bool]:
        """Advance at most 64 retired rows per RPC, retaining references only.

        Random IDs were written only by the retired context producer. Local
        replacement drains its calls before starting this Inspector; Bindings
        opens current providers, not archived execution graphs. This directory
        is discarded with the Inspector on restart/replacement, never persisted.
        """
        with self._lock:
            if self._legacy_before is not None:
                page, before = self._legacy_page(self._legacy_before)
                for identity, recall in page:
                    origin = recall.source
                    if isinstance(origin, ContextSource):
                        self._legacy.setdefault((origin.session_id, origin.source), []).append(
                            (recall.timestamp, identity, origin.through_seq))
                self._legacy_before = before
            if self._legacy_before is not None:
                return None, True
            earliest = min((item for item in self._legacy.get((session_id, source), ())
                            if after <= item[2] <= through), default=None)
            return (None if earliest is None else earliest[1]), False

    def recent(self, *, page: int = 1, page_size: int = 30, session_id: str = "") -> dict[str, object]:
        rows = tuple((identity, recall) for identity, recall in self._list()
                     if not session_id or (not isinstance(recall.source, ProgramSource)
                                           and recall.source.session_id == session_id))
        start = (page - 1) * page_size
        return self._ui_result({"items": [self._summary(identity, recall) for identity, recall in rows[start:start + page_size]],
                                    "total": len(rows), "page": page, "page_size": page_size})

    def for_turn(self, session_id: str, message_id: str, source: str,
                 projection: TurnProjection, *, offset: int = 0) -> dict[str, object]:
        """Read immutable identities belonging to the actual user Input/calls."""
        reader = self._catalog.reader(session_id)
        message = reader.get(message_id)
        if message is not None:
            source = message.source
        if not source:
            return {"items": [], "pending": False}
        cached = self._source_turns(session_id, source, projection)
        turn = next((item for item in cached.turns if message_id in item.message_ids), None)
        if message is None:
            turn = next((item for item in reversed(cached.turns) if item.status == "open"), None)
        if turn is None:
            return {"items": [], "pending": message is None}
        members = set(turn.message_ids)
        inputs = [item for item in cached.messages if item.identity in members and item.user_input]
        anchor_seq = message.seq if message is not None else cached.key[2]
        target = next((item for item in reversed(inputs) if item.seq <= anchor_seq), None)
        if target is None:
            return {"items": [], "pending": False}
        following = next((item for item in inputs if item.seq > target.seq), None)
        through_seq = reader.head() if turn.status == "open" else turn.through_seq
        if turn.status in {"complete", "quiet"}:
            through_seq -= 1
        if following is not None:
            through_seq = following.seq - 1
        calls = {ref for item in cached.messages
                 if item.identity in members and target.seq <= item.seq <= through_seq for ref in item.calls}
        # Results may arrive after another Input or even after an abandon boundary.
        # Their original CallRef, rather than arrival position, owns the card.
        results = {item.result: item.identity for item in cached.messages if item.result in calls}
        # Earlier Inputs remain live until the Turn closes. Closure ends event
        # following; it does not claim that canceled tool effects have drained.
        pending = turn.status == "open"
        legacy, legacy_pending = self._legacy_automatic(session_id, source, target.seq, through_seq)
        if legacy_pending:
            # An unfinished compatibility page cannot prove absence or earliest.
            return {"items": [], "pending": pending, "legacy_pending": True,
                    "input_message_id": target.identity, "next_offset": None}

        records: dict[str, Recall] = {}
        automatic: list[tuple[str, Recall]] = []
        for identity in dict.fromkeys((context_identity(session_id, source, target.identity), legacy)):
            if identity is None:
                continue
            recall = self._read(identity)
            if recall is None:
                if identity == legacy:
                    raise ValueError(f"召回记录缺失: {identity}")
                continue
            origin = recall.source
            if (not isinstance(origin, ContextSource) or origin.session_id != session_id
                or origin.source != source or not target.seq <= origin.through_seq <= through_seq):
                raise ValueError("自动召回记录不属于实际用户输入")
            automatic.append((identity, recall))
        if automatic:
            identity, recall = min(automatic, key=lambda item: (item[1].timestamp, item[0]))
            records[identity] = recall

        for ref in calls:
            identity = "tool:" + durable_call_key(ref)
            markers: set[str] = set()
            result_id = results.get(ref)
            if result_id is not None:
                result_message = reader.get(result_id)
                if result_message is None or not isinstance(result_message.body, ToolResult):
                    raise ValueError("工具结果消息缺失")
                result = result_message.body
                if result.call_ref != ref or result_message.source != source:
                    raise ValueError("工具结果不属于实际调用")
                if result.outcome == "success":
                    for part in result.parts:
                        if part.kind == "akasha.recall":
                            marker = RecallReference.model_validate_json(json.dumps(json_value(part.value)))
                            markers.add(marker.retrieval_ref)
            for identity in {identity} | markers:
                recall = self._read(identity)
                if recall is None:
                    if identity in markers:
                        raise ValueError(f"召回记录缺失: {identity}")
                    continue
                origin = recall.source
                if (not isinstance(origin, ToolSource) or origin.session_id != session_id
                    or origin.call_ref != ref):
                    raise ValueError("召回记录不属于实际工具调用")
                records[identity] = recall

        ordered = sorted(records.items(), key=lambda item: (item[1].timestamp, item[0]))
        items: list[dict[str, object]] = []
        payload = {"items": items, "pending": pending, "legacy_pending": False,
                   "input_message_id": target.identity, "next_offset": len(ordered)}
        size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()) + 4
        for identity, recall in ordered[offset:]:
            detail = self._detail(identity, recall)
            encoded_size = len(json.dumps(detail, ensure_ascii=False, separators=(",", ":")).encode())
            if size + encoded_size > 192 * 1024:
                if not items:
                    raise PluginUiRpcInvalidRequest("本条检索出处过多，超出插件界面容量；查询记录仍完整保留")
                break
            items.append(detail)
            size += encoded_size + 1
        end = offset + len(items)
        payload["next_offset"] = end if end < len(ordered) else None
        return payload

    @staticmethod
    def _summary(identity: str, recall: Recall) -> dict[str, object]:
        source = recall.source
        if isinstance(source, ContextSource):
            title = f"上下文查询 · {source.source} · #{source.through_seq}"
        elif isinstance(source, ToolSource):
            title = f"工具查询 · {source.call_ref.message_id}:{source.call_ref.part_index}"
        else:
            title = source.query
        origin = source.model_dump(mode="json", exclude={"query"})
        return {"query_id": identity, "query_text": title[:180], "query_text_truncated": len(title) > 180,
                "ts": recall.timestamp.isoformat(), "source": origin, "graph_version": recall.graph_version,
                "hit_count": len(recall.hits), "presented_count": len(recall.presented_message_ids)}

    def plugin_detail(self, identity: str) -> dict[str, object] | None:
        """命中和实际呈现分开显示；正文只从查询记录指向的原消息读取。"""
        recall = self._read(identity)
        if recall is None:
            return None
        return self._detail(identity, recall)

    def _detail(self, identity: str, recall: Recall) -> dict[str, object]:
        hits: list[dict[str, object]] = []
        for hit in recall.hits:
            messages: list[dict[str, object]] = []
            for message_id in hit.message_ids:
                message = self._catalog.reader(hit.session_id).get(message_id)
                if message is None:
                    raise ValueError(f"召回出处消息缺失: {message_id}")
                if not isinstance(message.body, (Input, Output)):
                    raise ValueError(f"召回成员不是输入或输出: {message_id}")
                text = "\n".join(cast(str, part.value) for part in message.body.parts
                                 if isinstance(part, ContentPart) and part.kind == "text")
                messages.append({"message_id": message_id, "preview": text[:240],
                                 "truncated": len(text) > 240,
                                 "presented": message_id in recall.presented_message_ids})
            hits.append({"score": hit.score, "lane": hit.lane, "sources": list(hit.sources), "messages": messages})
        return self._ui_result({**self._summary(identity, recall), "hits": hits, "pushes": recall.pushes,
                                    "residual_l1": recall.residual_l1})

    @staticmethod
    def _ui_result(payload: dict[str, object]) -> dict[str, object]:
        """插件界面投影保留全部成员；完整编码超限时明确失败，不裁掉尾部。"""
        result: dict[str, object] = {"schema": "akasha.queries.v1", **payload}
        if len(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()) > 192 * 1024:
            raise PluginUiRpcInvalidRequest("本条检索出处过多，超出插件界面容量；查询记录仍完整保留")
        return result
