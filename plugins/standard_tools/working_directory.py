"""Session directory state, with create-once defaults and durable switch receipts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from agent.plugin_composition import Context, Effect
from agent.plugin_composition.messages import OwnerStore, OwnerTransaction, SessionAttributes, MessageConflict
from agent.plugin_contracts.directories import DirectorySnapshot

from .path_access import PathAccess, check_directory
from .agents import read_agents

_SESSION = "directory:"
_SWITCH = "directory-switch:"


class WorkingDirectories:
    def __init__(self, store: OwnerStore):
        self._store = store
        self._defaults: dict[str, Callable[[str], str | None]] = {}

    async def register_default(
        self, ctx: Context, *, dimension: str, read: Callable[[str], str | None],
    ) -> Effect:
        """Register one default owner for a scope dimension, without copying its data."""
        _ = SessionAttributes(scope=((dimension, "probe"),))

        def setup():
            if dimension in self._defaults:
                raise ValueError(f"目录默认值已有 owner: {dimension}")
            self._defaults[dimension] = read
            return lambda: self._defaults.pop(dimension)

        return await ctx.effect(setup, label="directory-default:" + dimension)

    def initialize(self, session_id: str, attributes: SessionAttributes, tx: OwnerTransaction) -> None:
        """Snapshot defaults only inside the actual new Session transaction."""
        readers = dict(self._defaults)
        paths = [read(value) for dimension, value in attributes.scope
                 if (read := readers.get(dimension)) is not None]
        selected = [path for path in paths if path is not None]
        if len(set(selected)) > 1:
            raise MessageConflict("多个 scope 维度提供了不同默认目录")
        _ = tx.save(_SESSION + session_id, {"path": selected[0] if selected else None}, expected_version=None)

    def snapshot(self, session_id: str) -> DirectorySnapshot:
        record = self._store.read(_SESSION + session_id)
        # Absence is the old unset state, never permission to read new defaults.
        if record is None:
            return DirectorySnapshot(None, None)
        path = record.value["path"]
        if path is not None and (not isinstance(path, str) or not Path(path).is_absolute()):
            raise ValueError("Session 目录记录损坏")
        return DirectorySnapshot(path, record.version)

    async def check_directory(self, path: str, *, base_dir: str | None = None) -> str:
        async with PathAccess() as access:
            return check_directory(await access.read("inspect", path, base_dir=base_dir))

    async def inspect(self, path: str | None) -> Mapping[str, object]:
        if path is None:
            return {"path": None, "status": "unset"}
        async with PathAccess() as access:
            info = await access.read("inspect", path)
        if info.status == "available" and info.kind != "directory":
            return {"path": path, "status": "not_directory"}
        return {**info.model_dump(exclude_none=True), "path": path}

    async def browse(self, path: str, *, after: str | None = None) -> Mapping[str, object]:
        async with PathAccess() as access:
            return (await access.read("browse", path, after=after)).model_dump(exclude_none=True)

    async def resolve_target(self, session_id: str | None, path: str, *, legacy_base: str | None = None) -> str:
        """Fix relative targets using the live Session base on the execution host."""
        current = DirectorySnapshot(None, None) if session_id is None else self.snapshot(session_id)
        base = current.path if current.path is not None else legacy_base
        explicit = Path(path).is_absolute() or path.startswith("~")
        async with PathAccess() as access:
            if not explicit and current.path is not None:
                _ = check_directory(await access.read("inspect", current.path))
            info = await access.read("resolve", path, base_dir=base)
        if info.status != "available":
            raise ValueError(f"路径解析失败 ({info.status}): {info.error}")
        return info.path

    async def prepare_switch(self, session_id: str, path: str) -> Mapping[str, object]:
        """Validate a target and freeze the Session revision before execution."""
        current = self.snapshot(session_id)
        if not Path(path).is_absolute() and not path.startswith("~") and current.path is None:
            raise ValueError("当前目录未指定；请使用绝对目录")
        if current.path is not None and not Path(path).is_absolute() and not path.startswith("~"):
            _ = await self.check_directory(current.path)
        target = await self.check_directory(path, base_dir=current.path)
        return {"session_id": session_id, "path": target, "expected_version": current.revision}

    async def switch(self, key: str, arguments: Mapping[str, object]) -> Mapping[str, object]:
        """Commit the path and original effect receipt together, guarded by revision."""
        # 1. Replay the same durable effect before probing a possibly missing path.
        prior = self.receipt(key)
        if prior is not None:
            return prior
        target = cast(str, arguments["path"])
        _ = await self.check_directory(target)
        session_id = cast(str, arguments["session_id"])

        def save(tx: OwnerTransaction) -> Mapping[str, object]:
            previous = tx.read(_SWITCH + key)
            if previous is not None:
                return previous.value
            current = tx.read(_SESSION + session_id)
            version = None if current is None else current.version
            if version != arguments["expected_version"]:
                raise MessageConflict("Session 目录已改变；请重新读取后再切换")
            old_path = None if current is None else current.value["path"]
            # 2. Same-path switches keep the revision; different paths use CAS.
            if current is None or old_path != target:
                current = tx.save(_SESSION + session_id, {"path": target}, expected_version=version)
            result = {"session_id": session_id, "old_path": old_path, "path": target,
                      "revision": current.version}
            _ = tx.save(_SWITCH + key, result, expected_version=None)
            return result

        return await self._store.transact_async(save)

    def receipt(self, key: str) -> Mapping[str, object] | None:
        record = self._store.read(_SWITCH + key)
        return None if record is None else record.value

    async def current_info(self, session_id: str) -> dict[str, object]:
        current = self.snapshot(session_id)
        info = await self.inspect(current.path)
        rules = await read_agents(current.path)
        return {"path": current.path, "revision": current.revision, "status": info["status"],
                "agents": {key: value for key, value in rules.items() if key != "files"}}
