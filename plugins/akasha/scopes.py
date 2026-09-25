"""宽键路由：Session scope 与 Akasha 学习策略共同决定每条消息进入哪张图。

图是日志上的物化视图，不是日志分区：`sessions.db/messages` 仍是唯一事实来源，
每张图只持有自己的选择器与消费进度。策略由 Akasha 拥有，Core 不解释维度含义。
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Literal, cast, get_args

from agent.plugin_composition.messages import (
    MessageCatalog,
    MessageConflict,
    OwnerStore,
    OwnerTransaction,
    SessionAttributes,
)

LearnPolicy = Literal["global", "isolated", "off"]
LEARN_POLICIES: tuple[LearnPolicy, ...] = get_args(LearnPolicy)
DEFAULT_GRAPH = "default"
GRAPHS_DIRECTORY = "akasha-graphs"


class PolicyLocked(ValueError):
    """已有 Session 按旧策略路由，改写需要显式重建协议。"""


@dataclass(frozen=True, slots=True)
class Route:
    """write 为 None 表示不学习；read 始终指向一张可召回的图。"""

    write: str | None
    read: str


def graph_key(isolated: tuple[tuple[str, str], ...]) -> str:
    """显式偏键即图身份；不做 hash 取模，新增维度不重排已有图。"""
    return "&".join(f"{name}={value}" for name, value in isolated) or DEFAULT_GRAPH


def graph_path(memory_path: Path, key: str) -> Path:
    """default 图沿用原文件；独立图按规范键摘要放在同一 memory root 下。"""
    if key == DEFAULT_GRAPH:
        return memory_path
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return memory_path.parent / GRAPHS_DIRECTORY / digest / "akasha.db"


def ensure_graph_directory(memory_path: Path, key: str) -> Path:
    """首次打开独立图时写下可审阅的键清单；default 图不增加任何文件。"""
    path = graph_path(memory_path, key)
    if key == DEFAULT_GRAPH:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = path.parent / "manifest.json"
    if not manifest.exists():
        payload = json.dumps({"schema_version": 1, "graph": key}, ensure_ascii=False, sort_keys=True)
        descriptor, temporary = tempfile.mkstemp(prefix="manifest.", suffix=".tmp", dir=path.parent)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            _ = handle.write(payload + "\n")
        os.replace(temporary, manifest)
    return path


class ScopePolicies:
    """每个 (维度, 取值) 一条 set-once 策略；缺失即 global，与旧行为一致。"""

    def __init__(self, store: OwnerStore, catalog: MessageCatalog):
        self._store = store
        self._catalog = catalog
        self._routes: dict[str, Route] = {}

    def read(self, dimension: str, value: str) -> LearnPolicy:
        record = self._store.read(_policy_key(dimension, value))
        if record is None:
            return "global"
        learn = record.value.get("learn")
        if learn not in LEARN_POLICIES:
            raise ValueError(f"Akasha scope 策略损坏: {dimension}={value}")
        return cast(LearnPolicy, learn)

    # 策略只在该取值还没有任何 Session 时写入一次，已有路由因此永远确定。
    def set(self, dimension: str, value: str, learn: LearnPolicy) -> LearnPolicy:
        if learn not in LEARN_POLICIES:
            raise ValueError("Akasha 学习策略无效")
        _ = SessionAttributes(scope=((dimension, value),))
        key = _policy_key(dimension, value)
        def save(transaction: OwnerTransaction) -> LearnPolicy:
            # 1. 同值重放幂等；不同值说明策略已固定。
            current = transaction.read(key)
            if current is not None:
                if current.value.get("learn") == learn:
                    return learn
                raise PolicyLocked("该范围的记忆策略已固定")
            # 2. 同一写事务内确认尚无 Session 路由到该取值，接纳无法插入其间。
            if any(attributes.dimension(dimension) == value
                   for attributes in self._catalog.snapshot_attributes().values()):
                raise PolicyLocked("该范围已有对话，记忆策略不能再改变")
            _ = transaction.save(key, {"learn": learn}, expected_version=None)
            return learn
        try:
            return self._store.transact(save)
        except MessageConflict as error:
            raise PolicyLocked("该范围的记忆策略正在被并发写入") from error

    def route(self, session_id: str) -> Route:
        """isolated 跨维度传染；任一维度 off 则不学习但仍可读取所属图。"""
        cached = self._routes.get(session_id)
        if cached is not None:
            return cached
        attributes = self._catalog.attributes(session_id)
        policies = {(name, value): self.read(name, value) for name, value in attributes.scope}
        isolated = tuple(pair for pair, learn in policies.items() if learn == "isolated")
        read = graph_key(isolated)
        write = None if attributes.learning == "excluded" or "off" in policies.values() else read
        route = Route(write=write, read=read)
        self._routes[session_id] = route
        return route

    def snapshot(self) -> Mapping[str, LearnPolicy]:
        return {key: cast(LearnPolicy, record.value["learn"]) for key, record in self._store.list()}


def _policy_key(dimension: str, value: str) -> str:
    return f"{dimension}={value}"
