from __future__ import annotations

# pyright: reportPrivateUsage=false

import asyncio
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Generator, Protocol, cast
from uuid import uuid4

from agent.plugin_composition.root_switch import (
    _PartEntry,
    _PartNeed,
    _PartRef,
    _PartSet,
)
from agent.plugins.reload_journal import ReloadJournal, _StoredChoice, _SwitchRecord
from agent.plugins.source_hash import file_hash, file_revision, source_revision


class _PartLoader(Protocol):
    """Load one exact recovery part from its pinned artifact."""

    async def load(self, ref: _PartRef) -> _PartEntry: ...


@dataclass(frozen=True, slots=True)
class _Artifact:
    source_type: str
    path: str
    source_revision: str
    config_revision: str
    config_hash: str
    config_path: str
    data_path: str
    entrypoint: str


@dataclass(frozen=True, slots=True)
class _Move:
    name: str
    old: _PartRef | None
    new: _PartRef | None


@dataclass(frozen=True, slots=True)
class _Plan:
    old_snapshot: str | None
    new_snapshot: str
    moves: tuple[_Move, ...]


@dataclass(frozen=True, slots=True)
class _Step:
    action: str
    entry: _PartEntry

    @property
    def resource(self) -> str:
        return f"root-switch:{self.entry.ref.name}:{self.action}"


@dataclass(frozen=True, slots=True)
class _SwitchTarget:
    tx_id: str
    use_new: bool
    snapshot: str | None
    moves: tuple[_Move, ...]


@dataclass(frozen=True, slots=True)
class _SwitchChoice:
    """Hold one durable part choice after its action journal is cleared."""

    name: str
    ref: _PartRef | None
    other: _PartRef | None
    snapshot: str | None


class _SwitchError(RuntimeError):
    def __init__(self, message: str, *, resources: tuple[str, ...]) -> None:
        super().__init__(message)
        self.resources = resources


class _SwitchWork:
    """Own one in-process run of a journaled closed switch plan."""

    def __init__(
        self,
        journal: ReloadJournal,
        workspace: Path,
        tx_id: str,
        steps: tuple[_Step, ...],
    ) -> None:
        self._journal = journal
        self._workspace = workspace
        self._tx_id = tx_id
        self._steps = steps
        self._completed: list[_Step] = []
        self._applied = False

    async def apply(self) -> None:
        """Run the fixed forward steps while caller cancellation waits."""

        if self._applied or self._completed:
            raise RuntimeError("RootSwitch work 只能 apply 一次")
        task = asyncio.create_task(self._apply(), name="root-switch-apply")
        await _wait_critical(task)
        self._applied = True

    async def _apply(self) -> None:
        try:
            for index, step in enumerate(self._steps, start=1):
                self._completed.append(step)
                await _run_step(step)
                self._journal.advance_switch(self._tx_id, index)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as forward_error:
            await self._rollback(forward_error)
            raise

    def commit(self) -> None:
        """Select the new side after pointer work and before admission."""

        if not self._applied or len(self._completed) != len(self._steps):
            raise RuntimeError("RootSwitch work 尚未完成全部 forward step")
        self._journal.commit_switch(self._tx_id)

    async def rollback(self, cause: BaseException) -> None:
        """Restore the old side when the enclosing publication rejects."""

        record = self._journal.switch_record(self._tx_id)
        if record is None or record.cleared:
            return
        if record.use_new:
            raise RuntimeError("RootSwitch 已选新边，不能回滚到旧边")
        task = asyncio.create_task(
            self._rollback(cause),
            name="root-switch-rollback",
        )
        await _wait_critical(task)

    async def _rollback(self, cause: BaseException) -> None:
        errors: list[BaseException] = []
        resources: list[str] = []
        for step in reversed(self._completed):
            inverse = _inverse(step)
            try:
                await _run_step(inverse)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as error:
                errors.append(error)
                resources.append(inverse.resource)
        if not errors:
            _finish_switch(
                self._journal,
                self._workspace,
                self._tx_id,
                use_new=False,
            )
            self._completed.clear()
            return
        resource = ",".join(dict.fromkeys(resources))
        message = "; ".join(
            str(error) or type(error).__name__ for error in errors
        )
        self._journal.degrade_switch(
            self._tx_id,
            resource=resource,
            error=message,
        )
        raise BaseExceptionGroup(
            "RootSwitch 与旧 side 恢复同时失败",
            [cause, *errors],
        )


class _SwitchRun:
    """Build, run, and recover RootSwitch inside the one reload owner."""

    def __init__(self, journal: ReloadJournal) -> None:
        self._journal = journal
        self._workspace = journal.path.parent.parent

    def pins(self) -> tuple[str, ...]:
        """List exact artifacts that cleanup must keep for boot recovery."""

        artifacts = {
            item.artifact
            for ref in self._kept_refs()
            for item in (ref, *ref.needs)
        }
        return tuple(sorted(artifacts))

    def choices(self) -> tuple[_SwitchChoice, ...]:
        """Read every durable selected part or absence tombstone."""

        return tuple(_decode_choice(item) for item in self._journal.switch_choices())

    def save(self, snapshot: str, parts: _PartSet | None) -> None:
        """Save first-seen selected parts before public admission opens."""

        with _pin_lock(self._workspace):
            try:
                pinned = (
                    None
                    if parts is None
                    else _pin_parts_locked(self._workspace, parts)
                )
                entries = () if pinned is None else tuple(pinned.values())
                self._journal.save_parts(
                    snapshot,
                    tuple(
                        (
                            entry.ref.name,
                            _encode_move(_Move(entry.ref.name, None, entry.ref)),
                        )
                        for entry in entries
                    ),
                )
            except BaseException as error:
                try:
                    _clean_pins(self._workspace, self._kept_refs())
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        "RootSwitch save 与 pin cleanup 同时失败",
                        [error, cleanup_error],
                    ) from error
                raise
            _clean_pins(self._workspace, self._kept_refs())

    @staticmethod
    def old_refs(
        old_parts: _PartSet | None,
        new_parts: _PartSet | None,
    ) -> tuple[_PartRef, ...]:
        """List only changed old owners that must become quiet."""

        old = {} if old_parts is None else dict(old_parts.items())
        new = {} if new_parts is None else dict(new_parts.items())
        return tuple(
            old_entry.ref
            for name in sorted(set(old) | set(new))
            if (old_entry := old.get(name)) is not None
            and (
                (new_entry := new.get(name)) is None
                or old_entry.ref != new_entry.ref
            )
        )

    @classmethod
    def changed(
        cls,
        old_parts: _PartSet | None,
        new_parts: _PartSet | None,
    ) -> bool:
        old = {} if old_parts is None else dict(old_parts.items())
        new = {} if new_parts is None else dict(new_parts.items())
        return bool(cls.old_refs(old_parts, new_parts)) or any(
            name not in old for name in new
        )

    def prepare(
        self,
        tx_id: str,
        *,
        old_snapshot: str | None,
        old_parts: _PartSet | None,
        new_snapshot: str,
        new_parts: _PartSet | None,
    ) -> _SwitchWork | None:
        """Journal one closed plan without calling any plugin action."""

        plan, steps = _build_plan(
            old_snapshot,
            old_parts,
            new_snapshot,
            new_parts,
        )
        if not steps:
            return None
        with _pin_lock(self._workspace):
            try:
                pinned = _pin_plan(self._workspace, plan)
                self._journal.begin_switch(
                    tx_id,
                    old_snapshot_id=old_snapshot,
                    new_snapshot_id=new_snapshot,
                    plan_json=_encode_plan(pinned),
                    step_count=len(steps),
                    choices=tuple(
                        (move.name, _encode_move(move)) for move in pinned.moves
                    ),
                )
            except BaseException as error:
                try:
                    _clean_pins(self._workspace, self._kept_refs())
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        "RootSwitch prepare 与 pin cleanup 同时失败",
                        [error, cleanup_error],
                    ) from error
                raise
        return _SwitchWork(self._journal, self._workspace, tx_id, steps)

    async def recover(self, loader: _PartLoader) -> tuple[_SwitchTarget, ...]:
        """Converge each pending plan before its target snapshot may open."""

        targets: list[_SwitchTarget] = []
        for record in self._journal.pending_switches():
            plan = _decode_record(record)
            await self._recover_record(record, plan, loader)
            targets.append(_target(record, plan))
        return tuple(targets)

    def targets(self) -> tuple[_SwitchTarget, ...]:
        """Read pending target sides without running plugin code."""

        return tuple(
            _target(record, _decode_record(record))
            for record in self._journal.pending_switches()
        )

    def finish_recovery(
        self,
        target: _SwitchTarget,
        parts: _PartSet | None,
        source_snapshot: str | None = None,
    ) -> None:
        """Release pins only after the outer reload owner proves its pointer."""

        self.check_recovery(target, parts, source_snapshot)
        record = self._journal.switch_record(target.tx_id)
        if record is None:
            raise RuntimeError("RootSwitch recovery record 不存在")
        _finish_switch(
            self._journal,
            self._workspace,
            target.tx_id,
            use_new=record.use_new,
        )

    @staticmethod
    def check_recovery(
        target: _SwitchTarget,
        parts: _PartSet | None,
        source_snapshot: str | None = None,
    ) -> None:
        """Prove a rebuilt stable registry matches one journal side."""

        if (
            source_snapshot is not None
            and target.snapshot is not None
            and source_snapshot != target.snapshot
        ):
            raise RuntimeError("RootSwitch recovery snapshot 不一致")
        selected = {} if parts is None else dict(parts.items())
        for move in target.moves:
            expected = move.new if target.use_new else move.old
            actual = selected.get(move.name)
            if expected is None:
                if actual is not None:
                    raise RuntimeError(
                        "RootSwitch recovery target 与 stable part 不一致"
                    )
                continue
            if actual is None or not _same_part(expected, actual.ref):
                raise RuntimeError(
                    "RootSwitch recovery target 与 stable part 不一致"
                )

    def finish(self, tx_id: str, snapshot_id: str) -> None:
        """Release pins after the exact new snapshot is public."""

        record = self._journal.switch_record(tx_id)
        if record is None:
            raise RuntimeError("RootSwitch record 不存在")
        plan = _decode_record(record)
        if not record.use_new or plan.new_snapshot != snapshot_id:
            raise RuntimeError("RootSwitch stable snapshot 尚未证明")
        _finish_switch(
            self._journal,
            self._workspace,
            tx_id,
            use_new=True,
        )

    async def _recover_record(
        self,
        record: _SwitchRecord,
        plan: _Plan,
        loader: _PartLoader,
    ) -> None:
        target_new = record.use_new
        inactive = tuple(
            ref
            for move in plan.moves
            for ref in ((move.old,) if target_new else (move.new,))
            if ref is not None
        )
        active = tuple(
            ref
            for move in plan.moves
            for ref in ((move.new,) if target_new else (move.old,))
            if ref is not None
        )
        errors: list[BaseException] = []
        resources: list[str] = []
        for ref, should_run in (
            *((ref, False) for ref in inactive),
            *((ref, True) for ref in active),
        ):
            try:
                entry = await loader.load(ref)
                if entry.ref != ref:
                    raise RuntimeError(
                        "RootSwitch loader 返回了错误 artifact identity"
                    )
                await entry.part.recover(should_run)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as error:
                errors.append(error)
                resources.append(f"root-switch:{ref.name}:recover")
        if not errors:
            return
        unique_resources = tuple(dict.fromkeys(resources))
        message = "; ".join(
            str(error) or type(error).__name__ for error in errors
        )
        self._journal.degrade_switch(
            record.tx_id,
            resource=",".join(unique_resources),
            error=message,
        )
        raise _SwitchError(
            "RootSwitch recovery 未能证明唯一 active side",
            resources=unique_resources,
        ) from errors[0]

    def _kept_refs(self) -> tuple[_PartRef, ...]:
        selected = tuple(
            choice.ref for choice in self.choices() if choice.ref is not None
        )
        pending = tuple(
            ref
            for record in self._journal.pending_switches()
            for move in _decode_record(record).moves
            for ref in _move_refs(move)
        )
        return (*selected, *pending)


def _build_plan(
    old_snapshot: str | None,
    old_parts: _PartSet | None,
    new_snapshot: str,
    new_parts: _PartSet | None,
) -> tuple[_Plan, tuple[_Step, ...]]:
    if old_snapshot is not None:
        _check_text(old_snapshot, "old snapshot")
    _check_text(new_snapshot, "new snapshot")
    old = {} if old_parts is None else dict(old_parts.items())
    new = {} if new_parts is None else dict(new_parts.items())
    moves: list[_Move] = []
    changed: list[tuple[_PartEntry | None, _PartEntry | None]] = []
    for name in sorted(set(old) | set(new)):
        old_entry = old.get(name)
        new_entry = new.get(name)
        if (
            old_entry is not None
            and new_entry is not None
            and old_entry.ref == new_entry.ref
        ):
            continue
        moves.append(
            _Move(
                name=name,
                old=None if old_entry is None else old_entry.ref,
                new=None if new_entry is None else new_entry.ref,
            )
        )
        changed.append((old_entry, new_entry))
    steps = (
        *(
            _Step("stop", old_entry)
            for old_entry, _ in changed
            if old_entry is not None
        ),
        *(
            _Step("leave", old_entry)
            for old_entry, _ in changed
            if old_entry is not None
        ),
        *(
            _Step("enter", new_entry)
            for _, new_entry in changed
            if new_entry is not None
        ),
        *(
            _Step("start", new_entry)
            for _, new_entry in changed
            if new_entry is not None
        ),
    )
    return _Plan(old_snapshot, new_snapshot, tuple(moves)), tuple(steps)


def _target(record: _SwitchRecord, plan: _Plan) -> _SwitchTarget:
    return _SwitchTarget(
        tx_id=record.tx_id,
        use_new=record.use_new,
        snapshot=(
            plan.new_snapshot if record.use_new else plan.old_snapshot
        ),
        moves=plan.moves,
    )


async def _run_step(step: _Step) -> None:
    await getattr(step.entry.part, step.action)()


def _inverse(step: _Step) -> _Step:
    return _Step(
        {
            "stop": "start",
            "leave": "enter",
            "enter": "leave",
            "start": "stop",
        }[step.action],
        step.entry,
    )


def _encode_plan(plan: _Plan) -> str:
    return json.dumps(
        {
            "old_snapshot": plan.old_snapshot,
            "new_snapshot": plan.new_snapshot,
            "moves": [
                json.loads(_encode_move(move))
                for move in plan.moves
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _encode_move(move: _Move) -> str:
    return json.dumps(
        {
            "name": move.name,
            "old": _ref_value(move.old),
            "new": _ref_value(move.new),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _decode_record(record: _SwitchRecord) -> _Plan:
    loaded = json.loads(record.plan_json)
    if not isinstance(loaded, dict):
        raise RuntimeError("RootSwitch journal plan 结构无效")
    value = cast(dict[str, object], loaded)
    if set(value) != {
        "old_snapshot",
        "new_snapshot",
        "moves",
    }:
        raise RuntimeError("RootSwitch journal plan 结构无效")
    old_value = value["old_snapshot"]
    old_snapshot = (
        None if old_value is None else _stored_text(old_value, "old snapshot")
    )
    new_snapshot = _stored_text(value["new_snapshot"], "new snapshot")
    if (
        old_snapshot != record.old_snapshot_id
        or new_snapshot != record.new_snapshot_id
    ):
        raise RuntimeError("RootSwitch plan 与 snapshot 列不一致")
    moves_value = value["moves"]
    if not isinstance(moves_value, list):
        raise RuntimeError("RootSwitch journal moves 结构无效")
    raw_moves = cast(list[object], moves_value)
    moves: list[_Move] = []
    for raw_move in raw_moves:
        if not isinstance(raw_move, dict):
            raise RuntimeError("RootSwitch journal move 结构无效")
        move_value = cast(dict[str, object], raw_move)
        if set(move_value) != {
            "name",
            "old",
            "new",
        }:
            raise RuntimeError("RootSwitch journal move 结构无效")
        name = _stored_text(move_value["name"], "part name")
        old = _decode_ref(move_value["old"])
        new = _decode_ref(move_value["new"])
        if old is None and new is None:
            raise RuntimeError("RootSwitch move 两边不能都为空")
        if old is not None and old.name != name:
            raise RuntimeError("RootSwitch old ref 名称不一致")
        if new is not None and new.name != name:
            raise RuntimeError("RootSwitch new ref 名称不一致")
        moves.append(_Move(name, old, new))
    names = tuple(move.name for move in moves)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise RuntimeError("RootSwitch journal part 必须唯一并按名称排序")
    plan = _Plan(old_snapshot, new_snapshot, tuple(moves))
    if len(_step_refs(plan)) != record.step_count:
        raise RuntimeError("RootSwitch journal step count 不一致")
    return plan


def _ref_value(ref: _PartRef | None) -> dict[str, object] | None:
    if ref is None:
        return None
    return {
        "name": ref.name,
        "owner": ref.owner,
        "generation": ref.generation,
        "artifact": ref.artifact,
        "needs": [
            {
                "owner": need.owner,
                "generation": need.generation,
                "artifact": need.artifact,
            }
            for need in ref.needs
        ],
    }


def _decode_ref(value: object) -> _PartRef | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RuntimeError("RootSwitch journal part ref 结构无效")
    ref_value = cast(dict[str, object], value)
    if set(ref_value) != {
        "name",
        "owner",
        "generation",
        "artifact",
        "needs",
    }:
        raise RuntimeError("RootSwitch journal part ref 结构无效")
    needs_value = ref_value["needs"]
    if not isinstance(needs_value, list):
        raise RuntimeError("RootSwitch journal part needs 结构无效")
    needs = tuple(_decode_need(item) for item in cast(list[object], needs_value))
    if tuple(need.owner for need in needs) != tuple(
        sorted({need.owner for need in needs})
    ):
        raise RuntimeError("RootSwitch journal part needs 必须唯一并按 owner 排序")
    return _PartRef(
        name=_stored_text(ref_value["name"], "part name"),
        owner=_stored_text(ref_value["owner"], "part owner"),
        generation=_stored_text(ref_value["generation"], "part generation"),
        artifact=_stored_text(ref_value["artifact"], "part artifact"),
        needs=needs,
    )


def _decode_need(value: object) -> _PartNeed:
    if not isinstance(value, dict):
        raise RuntimeError("RootSwitch journal part need 结构无效")
    item = cast(dict[str, object], value)
    if set(item) != {"owner", "generation", "artifact"}:
        raise RuntimeError("RootSwitch journal part need 结构无效")
    return _PartNeed(
        owner=_stored_text(item["owner"], "need owner"),
        generation=_stored_text(item["generation"], "need generation"),
        artifact=_stored_text(item["artifact"], "need artifact"),
    )


def _decode_choice(stored: _StoredChoice) -> _SwitchChoice:
    loaded = json.loads(stored.move_json)
    if not isinstance(loaded, dict):
        raise RuntimeError("RootSwitch choice move 结构无效")
    item = cast(dict[str, object], loaded)
    if set(item) != {"name", "old", "new"}:
        raise RuntimeError("RootSwitch choice move 结构无效")
    name = _stored_text(item["name"], "choice name")
    if name != stored.name:
        raise RuntimeError("RootSwitch choice name 不一致")
    old = _decode_ref(item["old"])
    new = _decode_ref(item["new"])
    selected = new if stored.use_new else old
    other = old if stored.use_new else new
    return _SwitchChoice(name, selected, other, stored.snapshot_id)


def _step_refs(plan: _Plan) -> tuple[tuple[str, _PartRef], ...]:
    return (
        *(("stop", move.old) for move in plan.moves if move.old is not None),
        *(("leave", move.old) for move in plan.moves if move.old is not None),
        *(("enter", move.new) for move in plan.moves if move.new is not None),
        *(("start", move.new) for move in plan.moves if move.new is not None),
    )


def _pin_plan(workspace: Path, plan: _Plan) -> _Plan:
    """Pin every code artifact before the first shared-owner action."""

    return _Plan(
        old_snapshot=plan.old_snapshot,
        new_snapshot=plan.new_snapshot,
        moves=tuple(
            _Move(
                name=move.name,
                old=_pin_ref(workspace, move.old),
                new=_pin_ref(workspace, move.new),
            )
            for move in plan.moves
        ),
    )


def _pin_ref(workspace: Path, ref: _PartRef | None) -> _PartRef | None:
    if ref is None:
        return None
    artifact = _decode_artifact(ref.artifact)
    source = Path(artifact.path)
    if artifact.source_type not in {"builtin", "installed"}:
        raise RuntimeError(
            f"RootSwitch artifact source type 无效: {artifact.source_type}"
        )
    needs = tuple(_pin_need(workspace, need) for need in ref.needs)
    digest = hashlib.sha256(ref.artifact.encode("utf-8")).hexdigest()
    pinned_source = _pin_source(workspace, artifact, ref, digest)
    pinned_config = _pin_config(workspace, artifact, ref, digest)
    pinned = _Artifact(
        source_type=artifact.source_type,
        path=str(pinned_source),
        source_revision=artifact.source_revision,
        config_revision=artifact.config_revision,
        config_hash=artifact.config_hash,
        config_path=str(pinned_config),
        data_path=artifact.data_path,
        entrypoint=artifact.entrypoint,
    )
    return _PartRef(
        name=ref.name,
        owner=ref.owner,
        generation=ref.generation,
        artifact=_encode_artifact(pinned),
        needs=needs,
    )


def _pin_source(
    workspace: Path,
    artifact: _Artifact,
    ref: _PartRef,
    digest: str,
) -> Path:
    """Return exact source, copying mutable builtin code only once."""

    source = Path(artifact.path)
    root = workspace / "runtime" / "root-switch" / "artifacts"
    resolved = source.resolve(strict=False)
    pinned = resolved.parent == root.resolve(strict=False)
    if artifact.source_type == "installed" or pinned:
        _check_source(source, artifact, ref)
        return source

    target = root / digest
    if target.exists():
        _check_source(target, artifact, ref)
        return target

    _check_source(source, artifact, ref)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{digest}-{uuid4().hex}.tmp"
    try:
        _copy_source(source, temporary)
        if source_revision(temporary) != artifact.source_revision:
            raise RuntimeError(
                f"RootSwitch builtin pin 内容漂移: {ref.owner}:{ref.name}"
            )
        _sync_tree(temporary)
        try:
            os.replace(temporary, target)
        except FileExistsError:
            shutil.rmtree(temporary)
        _sync_dir(root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    _check_source(target, artifact, ref)
    return target


def _check_source(source: Path, artifact: _Artifact, ref: _PartRef) -> None:
    if source_revision(source) != artifact.source_revision:
        raise RuntimeError(
            f"RootSwitch artifact source 已漂移: {ref.owner}:{ref.name}"
        )
    entrypoint = source / artifact.entrypoint
    if not entrypoint.is_file() or entrypoint.is_symlink():
        raise RuntimeError(
            f"RootSwitch artifact entrypoint 无效: {ref.owner}:{entrypoint}"
        )


def _pin_config(
    workspace: Path,
    artifact: _Artifact,
    ref: _PartRef,
    digest: str,
) -> Path:
    """Keep one private exact config file for crash recovery."""

    source = Path(artifact.config_path)
    root = workspace / "runtime" / "root-switch" / "configs"
    resolved = source.resolve(strict=False)
    pinned = (
        resolved.parent.parent == root.resolve(strict=False)
        and resolved.name == "config.local.toml"
    )
    if pinned:
        _check_config_pin(source, root, artifact, ref)
        return source

    target_dir = root / digest
    target = target_dir / "config.local.toml"
    if target_dir.exists():
        _check_config_pin(target, root, artifact, ref)
        return target

    if source.is_symlink():
        raise RuntimeError(f"RootSwitch config 不能是符号链接: {ref.owner}")
    if file_revision(source) != artifact.config_revision:
        raise RuntimeError(f"RootSwitch config 在 publication 前已漂移: {ref.owner}")
    _check_config_hash(source, artifact, ref)
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    temporary = root / f".{digest}-{uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700)
        if source.is_file():
            _ = shutil.copyfile(source, temporary / "config.local.toml")
            os.chmod(temporary / "config.local.toml", 0o600)
        _sync_tree(temporary)
        try:
            os.replace(temporary, target_dir)
        except FileExistsError:
            shutil.rmtree(temporary)
        _sync_dir(root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    _check_config_pin(target, root, artifact, ref)
    return target


def _check_config_hash(path: Path, artifact: _Artifact, ref: _PartRef) -> None:
    if file_hash(path) != artifact.config_hash:
        raise RuntimeError(f"RootSwitch config pin 已漂移: {ref.owner}:{ref.name}")


def _check_config_pin(
    path: Path,
    root: Path,
    artifact: _Artifact,
    ref: _PartRef,
) -> None:
    """Validate private pin location, mode, and exact content."""

    if root.stat().st_mode & 0o777 != 0o700:
        raise RuntimeError(f"RootSwitch config root 权限无效: {root}")
    if path.parent.stat().st_mode & 0o777 != 0o700:
        raise RuntimeError(f"RootSwitch config pin 权限无效: {path.parent}")
    if path.exists() and (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_mode & 0o777 != 0o600
    ):
        raise RuntimeError(f"RootSwitch config file 权限无效: {path}")
    _check_config_hash(path, artifact, ref)


def _pin_need(workspace: Path, need: _PartNeed) -> _PartNeed:
    pinned = _pin_ref(
        workspace,
        _PartRef("need", need.owner, need.generation, need.artifact),
    )
    if pinned is None:
        raise RuntimeError("RootSwitch dependency pin 意外为空")
    return _PartNeed(pinned.owner, pinned.generation, pinned.artifact)


def _pin_parts(workspace: Path, parts: _PartSet) -> _PartSet:
    """Pin every part before its snapshot becomes committed state."""

    with _pin_lock(workspace):
        return _pin_parts_locked(workspace, parts)


def _pin_parts_locked(workspace: Path, parts: _PartSet) -> _PartSet:
    return _PartSet(
        {
            name: _PartEntry(_pin_required(workspace, entry.ref), entry.part)
            for name, entry in parts.items()
        }
    )


def _pin_required(workspace: Path, ref: _PartRef) -> _PartRef:
    pinned = _pin_ref(workspace, ref)
    if pinned is None:
        raise RuntimeError("RootSwitch part pin 意外为空")
    return pinned


def _copy_source(source: Path, target: Path) -> None:
    excluded = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
    }
    _ = shutil.copytree(
        source,
        target,
        symlinks=True,
        ignore=lambda _path, names: sorted(excluded.intersection(names)),
    )


def _sync_tree(root: Path) -> None:
    for current, directories, filenames in os.walk(root, topdown=False):
        for name in filenames:
            path = Path(current) / name
            if path.is_symlink():
                continue
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        for name in directories:
            path = Path(current) / name
            if not path.is_symlink():
                _sync_dir(path)
    _sync_dir(root)


def _sync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _finish_switch(
    journal: ReloadJournal,
    workspace: Path,
    tx_id: str,
    *,
    use_new: bool,
) -> None:
    with _pin_lock(workspace):
        record = journal.switch_record(tx_id)
        if record is None:
            raise RuntimeError("RootSwitch record 不存在")
        journal.finish_switch(tx_id, use_new=use_new)
        kept = tuple(
            choice.ref
            for choice in (
                _decode_choice(item) for item in journal.switch_choices()
            )
            if choice.ref is not None
        )
        pending = tuple(
            ref
            for item in journal.pending_switches()
            for move in _decode_record(item).moves
            for ref in _move_refs(move)
        )
        _clean_pins(workspace, (*kept, *pending))


def _clean_pins(workspace: Path, refs: tuple[_PartRef, ...]) -> None:
    artifacts = (
        workspace / "runtime" / "root-switch" / "artifacts"
    ).resolve(strict=False)
    configs = (workspace / "runtime" / "root-switch" / "configs").resolve(
        strict=False
    )
    artifact_paths = tuple(
        Path(_decode_artifact(ref.artifact).path).resolve(strict=False)
        for part in refs
        for ref in (part, *part.needs)
    )
    config_paths = tuple(
        Path(_decode_artifact(ref.artifact).config_path).resolve(strict=False).parent
        for part in refs
        for ref in (part, *part.needs)
    )
    _clean_pin_root(artifacts, artifact_paths)
    _clean_pin_root(configs, config_paths)


def _clean_pin_root(root: Path, paths: tuple[Path, ...]) -> None:
    """Remove unreferenced direct children from one private pin root."""

    if not root.exists():
        return
    kept = {path for path in paths if path.parent == root}
    for path in root.iterdir():
        resolved = path.resolve(strict=False)
        if resolved.parent != root:
            raise RuntimeError(f"RootSwitch pin path 越界: {resolved}")
        if resolved not in kept:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()


def _path_pinned(journal: ReloadJournal, path: Path) -> bool:
    """Check an exact artifact path while the caller owns the pin lock."""

    target = path.resolve(strict=False)
    refs: list[_PartRef] = []
    for stored in journal.switch_choices():
        choice = _decode_choice(stored)
        if choice.ref is not None:
            refs.append(choice.ref)
    refs.extend(
        ref
        for record in journal.pending_switches()
        for move in _decode_record(record).moves
        for ref in _move_refs(move)
    )
    for part in refs:
        for ref in (part, *part.needs):
                artifact = _decode_artifact(ref.artifact)
                pinned = Path(artifact.path).resolve(strict=False)
                if (
                    target == pinned
                    or target.is_relative_to(pinned)
                    or pinned.is_relative_to(target)
                ):
                    return True
    return False


def _path_pending(journal: ReloadJournal, path: Path) -> bool:
    """Check whether one unfinished switch covers this plugin path."""

    target = path.resolve(strict=False)
    return any(
        target == pinned
        or target.is_relative_to(pinned)
        or pinned.is_relative_to(target)
        for record in journal.pending_switches()
        for move in _decode_record(record).moves
        for ref in _move_refs(move)
        for pinned in (Path(_decode_artifact(ref.artifact).path).resolve(strict=False),)
    )


def _move_refs(move: _Move) -> tuple[_PartRef, ...]:
    return tuple(ref for ref in (move.old, move.new) if ref is not None)


@contextmanager
def _pin_lock(workspace: Path) -> Generator[None]:
    """Serialize durable pin creation with every artifact delete check."""

    path = workspace / "runtime" / "root-switch" / ".lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    stream: IO[str] = path.open("a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _decode_artifact(value: str) -> _Artifact:
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as error:
        raise RuntimeError("RootSwitch artifact identity 不是 JSON") from error
    if not isinstance(loaded, dict):
        raise RuntimeError("RootSwitch artifact identity 结构无效")
    item = cast(dict[str, object], loaded)
    keys = {
        "source_type",
        "path",
        "source_revision",
        "config_revision",
        "config_hash",
        "config_path",
        "data_path",
        "entrypoint",
    }
    if set(item) != keys:
        raise RuntimeError("RootSwitch artifact identity 结构无效")
    return _Artifact(
        source_type=_stored_text(item["source_type"], "artifact source type"),
        path=_stored_text(item["path"], "artifact path"),
        source_revision=_stored_text(
            item["source_revision"], "artifact source revision"
        ),
        config_revision=_stored_text(
            item["config_revision"], "artifact config revision"
        ),
        config_hash=_stored_text(item["config_hash"], "artifact config hash"),
        config_path=_stored_text(item["config_path"], "artifact config path"),
        data_path=_stored_text(item["data_path"], "artifact data path"),
        entrypoint=_stored_text(item["entrypoint"], "artifact entrypoint"),
    )


def _encode_artifact(value: _Artifact) -> str:
    return json.dumps(
        {
            "config_hash": value.config_hash,
            "config_path": value.config_path,
            "config_revision": value.config_revision,
            "data_path": value.data_path,
            "entrypoint": value.entrypoint,
            "path": value.path,
            "source_revision": value.source_revision,
            "source_type": value.source_type,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _same_part(expected: _PartRef, actual: _PartRef) -> bool:
    return expected == actual


def _check_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"RootSwitch {label} 必须非空且无首尾空白")


def _stored_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RuntimeError(f"RootSwitch journal {label} 无效")
    return value


async def _wait_critical(task: asyncio.Task[None]) -> None:
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    await task
    if cancelled:
        raise asyncio.CancelledError
