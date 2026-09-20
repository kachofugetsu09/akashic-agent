"""查询事实与原始 Message 的只读投影；不重算实际召回。"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast

from agent.plugin_composition import MobileUiRpcInvalidRequest
from agent.plugin_composition.messages import MessageCatalog
from agent.plugin_contracts import ContentPart, Input, Output
from ._boundaries import Turn, TurnProjection
from .recalls import ContextSource, ProgramSource, Recall, ToolSource


class RecallInspector:
    """只读查询事实与原消息，不把学习读出重算成实际召回。"""

    def __init__(self, *, read: Callable[[str], Recall | None],
                 list_records: Callable[[], tuple[tuple[str, Recall], ...]], catalog: MessageCatalog):
        self._read = read
        self._list = list_records
        self._catalog = catalog
        self._turns: tuple[tuple[str, str, int], tuple[Turn, ...], tuple[tuple[str, int, bool], ...]] | None = None

    def recent(self, *, page: int = 1, page_size: int = 30, session_id: str = "") -> dict[str, object]:
        rows = tuple((identity, recall) for identity, recall in self._list()
                     if not session_id or (not isinstance(recall.source, ProgramSource)
                                           and recall.source.session_id == session_id))
        start = (page - 1) * page_size
        return self._mobile_result({"items": [self._summary(identity, recall) for identity, recall in rows[start:start + page_size]],
                                    "total": len(rows), "page": page, "page_size": page_size})

    def for_turn(self, session_id: str, message_id: str, source: str,
                 projection: TurnProjection, *, offset: int = 0) -> dict[str, object]:
        """按真实用户输入展示首次自动召回和主动查询，并分批返回完整记录。"""
        # 1. 已提交消息从日志取得来源；草稿只能读取该来源仍未闭合的 Turn。
        reader = self._catalog.reader(session_id)
        message = reader.get(message_id)
        if message is not None:
            source = message.source
        if not source:
            return {"items": [], "pending": False}
        # 只缓存最近一个来源的 Turn 引用；同一前缀轮询不再读取消息正文。
        source_head = reader.head(source=source)
        key = (session_id, source, source_head)
        cached = self._turns
        if cached is None or cached[0] != key:
            messages = []
            cursor = -1
            while cursor < source_head:
                page = reader.read(after_seq=cursor, through_seq=source_head, source=source)
                messages.extend(page)
                cursor = page[-1].seq
            cached = (key, projection.project(messages, source),
                      tuple((item.message_id, item.seq, isinstance(item.body, Input) and item.author == "user")
                            for item in messages))
            self._turns = cached
        turns = cached[1]
        turn = next((item for item in turns if message_id in item.message_ids), None)
        if message is None:
            turn = next((item for item in reversed(turns) if item.status == "open"), None)
        if turn is None:
            return {"items": [], "pending": message is None}
        # 2. 回复归属它之前最近的真实输入；下一条输入划开同 Turn 的卡片。
        member_ids = set(turn.message_ids)
        inputs = [(identity, seq) for identity, seq, user_input in cached[2]
                  if identity in member_ids and user_input]
        anchor_seq = message.seq if message is not None else source_head
        target = next((item for item in reversed(inputs) if item[1] <= anchor_seq), None)
        if target is None:
            return {"items": [], "pending": False}
        following = next((item for item in inputs if item[1] > target[1]), None)
        through_seq = reader.head() if turn.status == "open" else turn.through_seq
        if turn.status in {"complete", "quiet"}:
            through_seq -= 1
        if following is not None:
            through_seq = following[1] - 1
        call_ids = {identity for identity, seq, _ in cached[2]
                    if identity in member_ids and target[1] <= seq <= through_seq}
        identities: list[str] = []
        automatic_found = False
        for identity, recall in reversed(self._list()):
            origin = recall.source
            if isinstance(origin, ContextSource):
                matches = (not automatic_found and origin.session_id == session_id
                           and origin.source == source and target[1] <= origin.through_seq <= through_seq)
                automatic_found = automatic_found or matches
            elif isinstance(origin, ToolSource):
                matches = (origin.session_id == session_id and origin.call_ref.message_id in call_ids)
            else:
                matches = False
            if matches:
                identities.append(identity)
        # 3. 历史重复自动查询留在 Inspector；主动查询分页，不突破 RPC 字节边界。
        items: list[dict[str, object]] = []
        pending = turn.status == "open" and following is None
        size = len(json.dumps({"items": [], "pending": pending,
            "input_message_id": target[0], "next_offset": len(identities)},
            ensure_ascii=False, separators=(",", ":")).encode()) + 4
        for identity in identities[offset:]:
            detail = self.mobile_detail(identity)
            if detail is None:
                raise ValueError(f"召回记录缺失: {identity}")
            encoded_size = len(json.dumps(detail, ensure_ascii=False, separators=(",", ":")).encode())
            if size + encoded_size > 192 * 1024:
                if not items:
                    raise MobileUiRpcInvalidRequest("本条检索出处过多，超出移动页面容量；查询记录仍完整保留")
                break
            items.append(detail)
            size += encoded_size + 1
        end = offset + len(items)
        return {"items": items, "pending": pending,
                "input_message_id": target[0],
                "next_offset": end if end < len(identities) else None}

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

    def mobile_detail(self, identity: str) -> dict[str, object] | None:
        """命中和实际呈现分开显示；正文只从查询记录指向的原消息读取。"""
        recall = self._read(identity)
        if recall is None:
            return None
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
        return self._mobile_result({**self._summary(identity, recall), "hits": hits, "pushes": recall.pushes,
                                    "residual_l1": recall.residual_l1})

    @staticmethod
    def _mobile_result(payload: dict[str, object]) -> dict[str, object]:
        """移动投影保留全部成员；完整编码超限时明确失败，不裁掉尾部。"""
        result: dict[str, object] = {"schema": "akasha.queries.v1", **payload}
        if len(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()) > 192 * 1024:
            raise MobileUiRpcInvalidRequest("本条检索出处过多，超出移动页面容量；查询记录仍完整保留")
        return result
