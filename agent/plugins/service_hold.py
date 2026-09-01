from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, NewType, Protocol, TypeVar, cast

from agent.plugin_composition.model import CompositionError, ServiceKey
from agent.plugins.artifact_pins import _ArtifactPins, _artifact_value
from agent.plugins.reload_journal import ReloadJournal, _HoldRecord
from agent.plugins.service_call import _release_critical
from agent.plugins.snapshot import (
    RuntimeSnapshot,
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    get_current_runtime_lease,
    get_lifecycle_runtime_snapshot,
    reset_runtime_snapshot,
)

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
R = TypeVar("R")
HoldId = NewType("HoldId", str)


class ServiceHold(Protocol, Generic[T_co]):
    """Keep calls to one fixed service on one exact Root across reboot."""

    async def reserve(self) -> HoldId: ...

    async def activate(self, hold_id: HoldId) -> None: ...

    async def pending(self) -> tuple[HoldId, ...]: ...

    async def call(
        self,
        hold_id: HoldId,
        action: Callable[[T_co], Awaitable[R]],
    ) -> R: ...

    async def drop(self, hold_id: HoldId) -> None: ...


@dataclass(frozen=True, slots=True)
class _HoldKey:
    value: str


@dataclass(frozen=True, slots=True)
class _PluginRef:
    owner: str
    generation: str
    artifact: str


@dataclass(frozen=True, slots=True)
class _RootRef:
    snapshot: str
    plugins: tuple[_PluginRef, ...]


class _HoldLoader(Protocol):
    """Rebuild one fresh exact Root from durable plugin pins."""

    async def load(self, ref: _RootRef) -> RuntimeSnapshot: ...


class _HoldRun:
    """Own durable exact Roots without knowing why callers keep them."""

    def __init__(
        self,
        store: RuntimeSnapshotStore,
        journal: ReloadJournal,
    ) -> None:
        self._store = store
        self._journal = journal
        self._pins = _ArtifactPins(journal.path.parent.parent, journal)
        self._held: dict[str, RuntimeSnapshot] = {}
        self._roots: dict[str, RuntimeSnapshot] = {}

    async def recover(self, loader: _HoldLoader) -> None:
        """Reopen all pending Roots from pins before callers use them."""

        errors: list[BaseException] = []
        for record in self._journal._pending_holds():
            if record.hold_id in self._held:
                continue
            try:
                ref = _decode_root(record.root_json)
                if ref.snapshot != record.snapshot_id:
                    raise RuntimeError(
                        "ServiceHold RootRef snapshot 与 journal 不一致"
                    )
                snapshot = self._roots.get(record.root_json)
                if snapshot is None:
                    snapshot = await loader.load(ref)
                    _check_root(snapshot, ref)
                    self._roots[record.root_json] = snapshot
                root = snapshot.composition_root
                if root is None:
                    raise RuntimeError("ServiceHold rebuilt snapshot 缺少 Root")
                _ = root.context.require(ServiceKey[object](record.service_key))
                retained = self._store._restore_hold(snapshot)
                self._held[record.hold_id] = retained
                self._journal._recover_hold(record.hold_id)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as error:
                self._journal._degrade_hold(
                    record.hold_id,
                    f"{type(error).__name__}: exact Root reopen failed",
                )
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("ServiceHold Root 恢复失败", errors)

    def bind(self, key: ServiceKey[T], hold_key: _HoldKey) -> ServiceHold[T]:
        """Bind one Core-minted holder to one sealed ServiceKey."""

        return _BoundHold(self, key, hold_key)

    def reserve(self, key: ServiceKey[object], hold_key: _HoldKey) -> HoldId:
        """Pin the live exact Root and create one global durable HoldId."""

        lease = get_current_runtime_lease()
        if lease is None:
            raise CompositionError(
                "HOLD_SCOPE",
                "ServiceHold.reserve 需要 live exact Root lease",
            )
        snapshot = lease.snapshot
        _ = self._store._check_hold(lease)
        root = snapshot.composition_root
        if root is None:
            raise CompositionError(
                "ROOT_MISSING",
                "ServiceHold snapshot 缺少 Root",
            )
        _ = root.context.require(key)
        ref = _root_ref(snapshot)
        with self._pins.lock():
            pinned = _pin_root(self._pins, ref)
            hold_id = self._journal._reserve_hold(
                hold_key=hold_key.value,
                service_key=key.name,
                snapshot_id=snapshot.snapshot_id,
                root_json=_encode_root(pinned),
            )
            try:
                retained = self._store._retain_hold(lease)
            except BaseException:
                _ = self._journal._drop_hold(
                    hold_id,
                    hold_key=hold_key.value,
                    service_key=key.name,
                )
                self._pins.clean()
                raise
            self._held[hold_id] = retained
            self._pins.clean()
        return HoldId(hold_id)

    def activate(
        self,
        hold_id: HoldId,
        key: ServiceKey[object],
        hold_key: _HoldKey,
    ) -> None:
        self._journal._activate_hold(
            str(hold_id),
            hold_key=hold_key.value,
            service_key=key.name,
        )

    def pending(
        self,
        key: ServiceKey[object],
        hold_key: _HoldKey,
    ) -> tuple[HoldId, ...]:
        return tuple(
            HoldId(record.hold_id)
            for record in self._journal._pending_holds(
                hold_key=hold_key.value,
                service_key=key.name,
            )
        )

    async def call(
        self,
        hold_id: HoldId,
        key: ServiceKey[T],
        hold_key: _HoldKey,
        action: Callable[[T], Awaitable[R]],
    ) -> R:
        if get_lifecycle_runtime_snapshot() is not None:
            raise CompositionError(
                "TASK_BOUND",
                "ServiceHold.call 只能从未绑定的 host task 调用",
            )
        record = self._record(hold_id, key, hold_key)
        if record.state != "active":
            raise RuntimeError("ServiceHold call 只接受 active hold")
        if record.error:
            raise RuntimeError(f"ServiceHold Root 已 degraded: {record.error}")
        snapshot = self._held.get(record.hold_id)
        if snapshot is None:
            raise RuntimeError("ServiceHold exact Root 尚未恢复")
        lease = self._store._lease_hold(snapshot.snapshot_id)
        token = bind_runtime_snapshot(lease)
        call_error: BaseException | None = None
        result: object | None = None
        try:
            root = lease.snapshot.composition_root
            if root is None:
                raise RuntimeError("ServiceHold exact Root 缺少 composition Root")
            result = await action(root.context.require(key))
        except BaseException as error:
            call_error = error
        finally:
            reset_runtime_snapshot(token)

        release_error: BaseException | None = None
        try:
            await _release_critical(lease)
        except BaseException as error:
            release_error = error
        if call_error is not None and release_error is not None:
            raise BaseExceptionGroup(
                "ServiceHold call 与 lease release 同时失败",
                [call_error, release_error],
            )
        if call_error is not None:
            raise call_error
        if release_error is not None:
            raise release_error
        return cast(R, result)

    async def drop(
        self,
        hold_id: HoldId,
        key: ServiceKey[object],
        hold_key: _HoldKey,
    ) -> None:
        """Release one durable pin even when caller cancellation races it."""

        task = asyncio.create_task(
            self._drop(hold_id, key, hold_key),
            name="service-hold-drop",
        )
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        await task
        if cancelled:
            raise asyncio.CancelledError

    async def _drop(
        self,
        hold_id: HoldId,
        key: ServiceKey[object],
        hold_key: _HoldKey,
    ) -> None:
        with self._pins.lock():
            _ = self._journal._drop_hold(
                str(hold_id),
                hold_key=hold_key.value,
                service_key=key.name,
            )
            snapshot = self._held.pop(str(hold_id), None)
        if snapshot is not None:
            await self._store._release_hold(snapshot)
        with self._pins.lock():
            self._pins.clean()

    def _record(
        self,
        hold_id: HoldId,
        key: ServiceKey[object],
        hold_key: _HoldKey,
    ) -> _HoldRecord:
        record = self._journal._hold_record(str(hold_id))
        if record.hold_key != hold_key.value:
            raise PermissionError("ServiceHold 不属于这只 holder")
        if record.service_key != key.name:
            raise PermissionError("ServiceHold 不属于这只 Service")
        return record


class _BoundHold(Generic[T]):
    __slots__ = ("_key", "_hold_key", "_run")

    def __init__(
        self,
        run: _HoldRun,
        key: ServiceKey[T],
        hold_key: _HoldKey,
    ) -> None:
        self._run = run
        self._key = key
        self._hold_key = hold_key

    async def reserve(self) -> HoldId:
        return self._run.reserve(self._key, self._hold_key)

    async def activate(self, hold_id: HoldId) -> None:
        self._run.activate(hold_id, self._key, self._hold_key)

    async def pending(self) -> tuple[HoldId, ...]:
        return self._run.pending(self._key, self._hold_key)

    async def call(
        self,
        hold_id: HoldId,
        action: Callable[[T], Awaitable[R]],
    ) -> R:
        return await self._run.call(
            hold_id,
            self._key,
            self._hold_key,
            action,
        )

    async def drop(self, hold_id: HoldId) -> None:
        await self._run.drop(hold_id, self._key, self._hold_key)


def _hold_key(identity: str) -> _HoldKey:
    """Mint one stable hidden namespace from a Core-owned capability identity."""

    if not identity or identity.strip() != identity:
        raise ValueError("ServiceHold capability identity 无效")
    digest = hashlib.sha256(
        f"akashic-service-hold-v1:{identity}".encode("utf-8")
    ).hexdigest()
    return _HoldKey(digest)


def _root_ref(snapshot: RuntimeSnapshot) -> _RootRef:
    if snapshot.composition_root is None:
        raise RuntimeError("ServiceHold snapshot 缺少 composition Root")
    if not snapshot.generations:
        raise RuntimeError("ServiceHold Root 没有可重建的 plugin artifact")
    return _RootRef(
        snapshot=snapshot.snapshot_id,
        plugins=tuple(
            _PluginRef(
                owner=owner,
                generation=generation.generation_id,
                artifact=_artifact_value(generation),
            )
            for owner, generation in sorted(snapshot.generations.items())
        ),
    )


def _pin_root(pins: _ArtifactPins, ref: _RootRef) -> _RootRef:
    return _RootRef(
        snapshot=ref.snapshot,
        plugins=tuple(
            _PluginRef(
                owner=item.owner,
                generation=item.generation,
                artifact=pins.pin(
                    item.artifact,
                    owner=item.owner,
                    name="service-hold",
                ),
            )
            for item in ref.plugins
        ),
    )


def _check_root(snapshot: RuntimeSnapshot, expected: _RootRef) -> None:
    actual = _root_ref(snapshot)
    if actual != expected:
        raise RuntimeError(
            "ServiceHold rebuilt Root identity 不一致: "
            f"snapshot={actual.snapshot == expected.snapshot}, "
            f"plugins={actual.plugins == expected.plugins}"
        )


def _encode_root(ref: _RootRef) -> str:
    return json.dumps(
        {
            "plugins": [
                {
                    "artifact": item.artifact,
                    "generation": item.generation,
                    "owner": item.owner,
                }
                for item in ref.plugins
            ],
            "snapshot": ref.snapshot,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _decode_root(value: str) -> _RootRef:
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise RuntimeError("ServiceHold RootRef 结构无效")
    root = cast(dict[str, object], loaded)
    if set(root) != {"plugins", "snapshot"}:
        raise RuntimeError("ServiceHold RootRef 结构无效")
    plugins = root["plugins"]
    if not isinstance(plugins, list) or not plugins:
        raise RuntimeError("ServiceHold RootRef 缺少 plugins")
    items: list[_PluginRef] = []
    for raw in cast(list[object], plugins):
        if not isinstance(raw, dict):
            raise RuntimeError("ServiceHold plugin ref 结构无效")
        item = cast(dict[str, object], raw)
        if set(item) != {
            "artifact",
            "generation",
            "owner",
        }:
            raise RuntimeError("ServiceHold plugin ref 结构无效")
        values = (item["owner"], item["generation"], item["artifact"])
        if not all(isinstance(part, str) and part for part in values):
            raise RuntimeError("ServiceHold plugin ref 字段无效")
        items.append(
            _PluginRef(
                owner=cast(str, item["owner"]),
                generation=cast(str, item["generation"]),
                artifact=cast(str, item["artifact"]),
            )
        )
    if tuple(item.owner for item in items) != tuple(
        sorted({item.owner for item in items})
    ):
        raise RuntimeError("ServiceHold plugins 必须唯一且按 owner 排序")
    snapshot = root["snapshot"]
    if not isinstance(snapshot, str) or not snapshot:
        raise RuntimeError("ServiceHold snapshot identity 无效")
    return _RootRef(snapshot=snapshot, plugins=tuple(items))


__all__ = ["HoldId", "ServiceHold"]
