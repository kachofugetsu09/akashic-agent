from __future__ import annotations

import asyncio
import secrets
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol

from agent.plugin_composition.background_jobs import BackgroundJobCatalog
from agent.plugins.snapshot import RuntimeSnapshotLease


@dataclass(frozen=True, slots=True)
class ActivityCatalog:
    """Freeze the public activity descriptors participating in one snapshot."""

    background_jobs: BackgroundJobCatalog | None

    @property
    def identity(self) -> str:
        return "|".join(
            (
                "jobs:"
                + (
                    ""
                    if self.background_jobs is None
                    else self.background_jobs.identity
                ),
            )
        )


class ActivityChildAdapter(Protocol):
    """Materialize one activity family under the shared publication owner."""

    name: str

    def prepare_components(
        self,
        transaction_id: str,
        target_lease: RuntimeSnapshotLease,
        target_catalog: ActivityCatalog,
    ) -> Any: ...

    def discard_plan(self, transaction_id: str, plan: Any) -> None: ...

    async def stop_components(
        self,
        transaction_id: str,
        old_binding: Any,
    ) -> None: ...

    async def materialize_closed(
        self,
        transaction_id: str,
        plan: Any,
    ) -> Any: ...

    def finalize_components(
        self,
        transaction_id: str,
        binding: Any,
    ) -> None: ...

    def pause_components(self, binding: Any) -> None: ...

    async def restore_components(
        self,
        transaction_id: str,
        old_binding: Any,
    ) -> None: ...

    async def close_components(
        self,
        transaction_id: str,
        binding: Any,
    ) -> None: ...


@dataclass(slots=True)
class ActivityBinding:
    """Own admission and in-flight work for one finalized activity catalog."""

    snapshot_id: str
    catalog_identity: str
    child_bindings: Mapping[str, object]
    admission_open: bool = False
    in_flight: int = 0


@dataclass(slots=True)
class ActivityTransaction:
    """Retain both publication sides until finalize or rollback settles."""

    transaction_id: str
    previous: ActivityBinding | None
    target_snapshot_id: str
    target_catalog: ActivityCatalog
    target_lease: RuntimeSnapshotLease
    plans: Mapping[str, object]
    staged: ActivityBinding | None = None
    stopped_children: list[str] = field(default_factory=list)
    partial_bindings: dict[str, object] = field(default_factory=dict)
    finalized: bool = False
    settled: bool = False
    recovery_mode: Literal["commit", "rollback"] | None = None


class ActivityAdmissionLease:
    """Keep one exact activity binding in-flight until request cleanup finishes."""

    __slots__ = ("_host", "_snapshot_lease", "binding", "_released")

    def __init__(
        self,
        host: ActivityHost,
        binding: ActivityBinding,
        snapshot_lease: RuntimeSnapshotLease,
    ) -> None:
        self._host = host
        self._snapshot_lease = snapshot_lease
        self.binding = binding
        self._released = False

    @property
    def active(self) -> bool:
        return not self._released

    async def release(self) -> None:
        if self._released:
            return
        cancelled = await _complete_critical(self._release_owned())
        if cancelled:
            raise asyncio.CancelledError

    async def _release_owned(self) -> None:
        await self._snapshot_lease.release()
        await self._host._release(self.binding)
        self._released = True

    async def __aenter__(self) -> ActivityBinding:
        return self.binding

    async def __aexit__(self, *exc_info: object) -> None:
        await self.release()


class ActivityHost:
    """Publish all activity families through one admission and drain boundary."""

    def __init__(self, children: tuple[ActivityChildAdapter, ...]) -> None:
        names = tuple(child.name for child in children)
        if any(not name or name.strip() != name for name in names):
            raise ValueError("Activity child name 必须是非空且无首尾空白的字符串")
        if len(set(names)) != len(names):
            raise ValueError("Activity child name 重复")
        self._children = MappingProxyType(dict(zip(names, children, strict=True)))
        self._active: ActivityBinding | None = None
        self._transaction: ActivityTransaction | None = None
        self._condition = asyncio.Condition()

    @property
    def active(self) -> ActivityBinding | None:
        return self._active

    def acquire(self, snapshot_lease: RuntimeSnapshotLease) -> ActivityAdmissionLease:
        """Acquire the finalized exact binding before the caller's first await."""

        binding = self._active
        if (
            not snapshot_lease.active
            or binding is None
            or not binding.admission_open
            or binding.snapshot_id != snapshot_lease.snapshot.snapshot_id
        ):
            raise RuntimeError("Activity admission 已关闭或 snapshot identity 不匹配")
        owned_snapshot_lease = snapshot_lease.fork()
        binding.in_flight += 1
        return ActivityAdmissionLease(self, binding, owned_snapshot_lease)

    async def prepare_transaction(
        self,
        target_lease: RuntimeSnapshotLease,
    ) -> ActivityTransaction:
        """Build immutable child plans without starting activity resources."""

        if self._transaction is not None:
            raise RuntimeError("已有 Activity publication transaction")
        if not target_lease.active:
            raise RuntimeError("Activity target snapshot lease 已失效")
        snapshot = target_lease.snapshot
        catalog = ActivityCatalog(
            background_jobs=snapshot.background_job_catalog,
        )
        transaction_id = secrets.token_hex(12)
        plans: dict[str, object] = {}
        try:
            for name, child in self._children.items():
                plans[name] = child.prepare_components(
                    transaction_id,
                    target_lease,
                    catalog,
                )
        except BaseException:
            try:
                for name, plan in plans.items():
                    self._children[name].discard_plan(transaction_id, plan)
            finally:
                await target_lease.release()
            raise
        transaction = ActivityTransaction(
            transaction_id=transaction_id,
            previous=self._active,
            target_snapshot_id=snapshot.snapshot_id,
            target_catalog=catalog,
            target_lease=target_lease,
            plans=MappingProxyType(plans),
        )
        self._transaction = transaction
        return transaction

    async def pause_and_drain(self, transaction: ActivityTransaction) -> None:
        """Close old admission, drain accepted work, and stop every old child."""

        self._require(transaction)
        previous = transaction.previous
        if previous is None:
            return
        previous.admission_open = False
        async with self._condition:
            await self._condition.wait_for(lambda: previous.in_flight == 0)
        for name, child in reversed(tuple(self._children.items())):
            if name in transaction.stopped_children:
                continue
            transaction.stopped_children.append(name)
            await child.stop_components(
                transaction.transaction_id,
                previous.child_bindings[name],
            )

    async def materialize_closed(
        self,
        transaction: ActivityTransaction,
    ) -> ActivityBinding:
        """Create target child resources while keeping target admission closed."""

        self._require(transaction)
        if transaction.previous is not None and len(
            transaction.stopped_children
        ) != len(self._children):
            raise RuntimeError("Activity old binding 尚未 drain")
        if transaction.staged is not None:
            raise RuntimeError("Activity target 已 materialize")
        try:
            for name, child in self._children.items():
                transaction.partial_bindings[name] = await child.materialize_closed(
                    transaction.transaction_id,
                    transaction.plans[name],
                )
        except BaseException:
            await self._close_partial(transaction)
            raise
        staged = ActivityBinding(
            snapshot_id=transaction.target_snapshot_id,
            catalog_identity=transaction.target_catalog.identity,
            child_bindings=MappingProxyType(dict(transaction.partial_bindings)),
        )
        transaction.staged = staged
        return staged

    def finalize(self, transaction: ActivityTransaction) -> None:
        """Switch the private host pointer while target admission remains closed."""

        self._require(transaction)
        if transaction.staged is None:
            raise RuntimeError("Activity target 尚未 materialize")
        if transaction.finalized:
            raise RuntimeError("Activity transaction 已 finalize")
        for name, child in self._children.items():
            child.finalize_components(
                transaction.transaction_id,
                transaction.staged.child_bindings[name],
            )
        self._active = transaction.staged
        transaction.staged.admission_open = True
        transaction.finalized = True

    async def open(self, transaction: ActivityTransaction) -> None:
        """Finish old-child cleanup and release the planning lease critically."""

        if transaction.recovery_mode == "rollback":
            raise RuntimeError("Activity rollback transaction 不能按 commit 恢复")
        transaction.recovery_mode = "commit"
        cancelled = await _complete_critical(self._complete_commit(transaction))
        if cancelled:
            raise asyncio.CancelledError

    async def rollback(
        self,
        transaction: ActivityTransaction,
    ) -> None:
        """Close staged resources, restore old children, and reopen the old binding."""

        if transaction.recovery_mode == "commit":
            raise RuntimeError("Activity commit transaction 不能按 rollback 恢复")
        transaction.recovery_mode = "rollback"
        cancelled = await _complete_critical(self._rollback(transaction))
        if cancelled:
            raise asyncio.CancelledError

    async def retry_recovery(self) -> None:
        """Retry the one retained publication rollback without consulting current state."""

        transaction = self._transaction
        if transaction is None:
            raise RuntimeError("ActivityHost 没有待恢复 transaction")
        if transaction.recovery_mode == "commit":
            await self.open(transaction)
        elif transaction.recovery_mode == "rollback":
            await self.rollback(transaction)
        else:
            raise RuntimeError("Activity transaction 尚未选择 recovery direction")

    async def _complete_commit(self, transaction: ActivityTransaction) -> None:
        """Close the stopped old child owners after the public pointer moved."""

        self._require(transaction)
        if not transaction.finalized or self._active is not transaction.staged:
            raise RuntimeError("Activity target 尚未 finalize")
        previous = transaction.previous
        try:
            if previous is not None:
                for name in reversed(tuple(transaction.stopped_children)):
                    await self._children[name].close_components(
                        transaction.transaction_id,
                        previous.child_bindings[name],
                    )
                    transaction.stopped_children.remove(name)
        except BaseException:
            assert transaction.staged is not None
            for name, child in self._children.items():
                child.pause_components(transaction.staged.child_bindings[name])
            transaction.staged.admission_open = False
            raise
        assert transaction.staged is not None
        if not transaction.staged.admission_open:
            for name, child in self._children.items():
                child.finalize_components(
                    transaction.transaction_id,
                    transaction.staged.child_bindings[name],
                )
            transaction.staged.admission_open = True
        if transaction.target_lease.active:
            await transaction.target_lease.release()
        transaction.settled = True
        self._transaction = None

    async def _rollback(self, transaction: ActivityTransaction) -> None:
        """Complete the rollback after caller cancellation without leaking owners."""

        self._require(transaction)
        staged = transaction.staged
        if staged is not None:
            staged.admission_open = False
            async with self._condition:
                await self._condition.wait_for(lambda: staged.in_flight == 0)
            await self._close_partial(transaction)
        elif transaction.partial_bindings:
            await self._close_partial(transaction)
        previous = transaction.previous
        if previous is not None and transaction.stopped_children:
            for name in reversed(tuple(transaction.stopped_children)):
                child = self._children[name]
                await child.restore_components(
                    transaction.transaction_id,
                    previous.child_bindings[name],
                )
                transaction.stopped_children.remove(name)
        if previous is not None:
            previous.admission_open = True
        self._active = previous
        if transaction.target_lease.active:
            await transaction.target_lease.release()
        transaction.settled = True
        self._transaction = None

    async def close(self) -> None:
        """Drain and close the current binding without inventing a publication target."""

        if self._transaction is not None:
            raise RuntimeError("Activity publication transaction 尚未收束")
        binding = self._active
        if binding is None:
            return
        binding.admission_open = False
        async with self._condition:
            await self._condition.wait_for(lambda: binding.in_flight == 0)
        transaction_id = "shutdown:" + secrets.token_hex(8)
        for name, child in reversed(tuple(self._children.items())):
            await child.close_components(
                transaction_id,
                binding.child_bindings[name],
            )
        self._active = None

    async def _release(self, binding: ActivityBinding) -> None:
        if binding.in_flight <= 0:
            raise RuntimeError("Activity in-flight 计数下溢")
        binding.in_flight -= 1
        async with self._condition:
            self._condition.notify_all()

    async def _close_partial(self, transaction: ActivityTransaction) -> None:
        for name in reversed(tuple(transaction.partial_bindings)):
            await self._children[name].close_components(
                transaction.transaction_id,
                transaction.partial_bindings[name],
            )
            transaction.partial_bindings.pop(name)

    def _require(self, transaction: ActivityTransaction) -> None:
        if self._transaction is not transaction or transaction.settled:
            raise RuntimeError("Activity publication transaction 已失效")


async def _complete_critical(awaitable: Awaitable[object]) -> bool:
    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while not task.done():
        try:
            _ = await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    await task
    return cancelled


__all__ = [
    "ActivityAdmissionLease",
    "ActivityBinding",
    "ActivityCatalog",
    "ActivityChildAdapter",
    "ActivityHost",
    "ActivityTransaction",
]
