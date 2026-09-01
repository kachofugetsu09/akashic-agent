from __future__ import annotations

# pyright: reportPrivateUsage=false

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from agent.plugin_composition.root_switch import (
    _PartEntry,
    _PartNeed,
    _PartRef,
    _PartSet,
    _SwitchInputRef,
)
from agent.plugins.artifact_pins import _ArtifactPins
from agent.plugins.reload_journal import ReloadJournal, _StoredChoice, _SwitchRecord


class _PartLoader(Protocol):
    """Load one exact recovery part from its pinned artifact."""

    async def load(self, ref: _PartRef) -> _PartEntry: ...


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
    snapshot: str

    @property
    def resource(self) -> str:
        return f"root-switch:{self.entry.ref.name}:{self.action}"


@dataclass(frozen=True, slots=True)
class _SwitchTarget:
    tx_id: str
    use_new: bool
    snapshot: str | None
    other_snapshot: str | None
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
                _ArtifactPins(self._workspace, self._journal),
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


class _BootSwitch:
    """Choose a journaled first-seen side after its recovery succeeds."""

    def __init__(self, journal: ReloadJournal, workspace: Path, tx_id: str) -> None:
        self._journal = journal
        self._pins = _ArtifactPins(workspace, journal)
        self._tx_id = tx_id

    def commit(self) -> None:
        """Record that recover converged the new side while admission is closed."""

        self._journal.commit_switch(self._tx_id, recovered=True)

    def finish(self) -> None:
        """Clear the transition while its selected pins remain in the choice."""

        _finish_switch(
            self._journal,
            self._pins,
            self._tx_id,
            use_new=True,
        )


class _SwitchRun:
    """Build, run, and recover RootSwitch inside the one reload owner."""

    def __init__(self, journal: ReloadJournal) -> None:
        self._journal = journal
        self._workspace = journal.path.parent.parent
        self._pins = _ArtifactPins(self._workspace, journal)

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
        with self._pins.lock():
            try:
                pinned = _pin_plan(self._pins, plan)
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
                    self._pins.clean()
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        "RootSwitch prepare 与 pin cleanup 同时失败",
                        [error, cleanup_error],
                    ) from error
                raise
        return _SwitchWork(self._journal, self._workspace, tx_id, steps)

    def prepare_boot(
        self,
        tx_id: str,
        *,
        new_snapshot: str,
        new_parts: _PartSet | None,
    ) -> _BootSwitch | None:
        """Journal only first-seen parts before their first recovery call."""

        current = {} if new_parts is None else dict(new_parts.items())
        choices = {choice.name: choice for choice in self.choices()}
        for name, choice in choices.items():
            actual = current.get(name)
            if choice.ref is None:
                if actual is not None:
                    raise RuntimeError(
                        "RootSwitch boot target 与 absence choice 不一致"
                    )
            elif actual is None or actual.ref != choice.ref:
                raise RuntimeError(
                    "RootSwitch boot target 与 selected choice 不一致"
                )
        first = {
            name: entry for name, entry in current.items() if name not in choices
        }
        if not first:
            return None
        if self.targets():
            raise RuntimeError("RootSwitch recovery 期间不能增加 first part")
        work = self.prepare(
            tx_id,
            old_snapshot=None,
            old_parts=None,
            new_snapshot=new_snapshot,
            new_parts=_PartSet(first),
        )
        if work is None:
            raise RuntimeError("RootSwitch first plan 意外为空")
        return _BootSwitch(self._journal, self._workspace, tx_id)

    async def preflight(self, loader: _PartLoader) -> tuple[_SwitchTarget, ...]:
        """Prove each pinned side without calling a shared-owner action."""

        targets: list[_SwitchTarget] = []
        for record in self._journal.pending_switches():
            plan = _decode_record(record)
            await self._preflight_record(record, plan, loader)
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
            self._pins,
            target.tx_id,
            use_new=record.use_new,
            recovered=True,
        )

    def degrade_recovery(
        self,
        targets: tuple[_SwitchTarget, ...],
        *,
        resource: str,
        error: BaseException,
    ) -> None:
        """Keep every affected plan closed after a callback failure."""

        message = str(error) or type(error).__name__
        for target in targets:
            record = self._journal.switch_record(target.tx_id)
            if (
                record is not None
                and not record.cleared
                and record.state == "running"
            ):
                self._journal.degrade_switch(
                    target.tx_id,
                    resource=resource,
                    error=message,
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
            self._pins,
            tx_id,
            use_new=True,
        )

    async def _preflight_record(
        self,
        record: _SwitchRecord,
        plan: _Plan,
        loader: _PartLoader,
    ) -> None:
        errors: list[BaseException] = []
        resources: list[str] = []
        refs = tuple(
            dict.fromkeys(ref for move in plan.moves for ref in _move_refs(move))
        )
        for ref in refs:
            try:
                entry = await loader.load(ref)
                if entry.ref != ref:
                    raise RuntimeError(
                        "RootSwitch loader 返回了错误 artifact identity"
                    )
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as error:
                errors.append(error)
                resources.append(f"root-switch:{ref.name}:preflight")
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
            "RootSwitch preflight 未能证明 exact side",
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
    if old and old_snapshot is None:
        raise RuntimeError("RootSwitch old parts 缺少 snapshot identity")
    old_step_snapshot = new_snapshot if old_snapshot is None else old_snapshot
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
            _Step("stop", old_entry, old_step_snapshot)
            for old_entry, _ in changed
            if old_entry is not None
        ),
        *(
            _Step("leave", old_entry, old_step_snapshot)
            for old_entry, _ in changed
            if old_entry is not None
        ),
        *(
            _Step("enter", new_entry, new_snapshot)
            for _, new_entry in changed
            if new_entry is not None
        ),
        *(
            _Step("start", new_entry, new_snapshot)
            for _, new_entry in changed
            if new_entry is not None
        ),
    )
    return _Plan(old_snapshot, new_snapshot, tuple(moves)), tuple(steps)


def _target(record: _SwitchRecord, plan: _Plan) -> _SwitchTarget:
    return _SwitchTarget(
        tx_id=record.tx_id,
        use_new=record.use_new,
        snapshot=(plan.new_snapshot if record.use_new else plan.old_snapshot),
        other_snapshot=(plan.old_snapshot if record.use_new else plan.new_snapshot),
        moves=plan.moves,
    )


async def _run_step(step: _Step) -> None:
    await getattr(step.entry.part, step.action)(step.snapshot)


def _inverse(step: _Step) -> _Step:
    return _Step(
        {
            "stop": "start",
            "leave": "enter",
            "enter": "leave",
            "start": "stop",
        }[step.action],
        step.entry,
        step.snapshot,
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
        "fiber": ref.fiber,
        "inputs": [
            {
                "owner": item.owner,
                "generation": item.generation,
                "artifact": item.artifact,
                "fiber": item.fiber,
            }
            for item in ref.inputs
        ],
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
        "fiber",
        "inputs",
        "needs",
    }:
        raise RuntimeError("RootSwitch journal part ref 结构无效")
    inputs_value = ref_value["inputs"]
    if not isinstance(inputs_value, list):
        raise RuntimeError("RootSwitch journal part inputs 结构无效")
    inputs = tuple(_decode_input(item) for item in cast(list[object], inputs_value))
    if inputs != tuple(
        sorted(
            inputs,
            key=lambda item: (
                item.owner,
                item.fiber,
                item.generation,
                item.artifact,
            ),
        )
    ):
        raise RuntimeError("RootSwitch journal part inputs 必须稳定排序")
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
        fiber=_stored_text(ref_value["fiber"], "part fiber"),
        inputs=inputs,
        needs=needs,
    )


def _decode_input(value: object) -> _SwitchInputRef:
    if not isinstance(value, dict):
        raise RuntimeError("RootSwitch journal input ref 结构无效")
    item = cast(dict[str, object], value)
    if set(item) != {"owner", "generation", "artifact", "fiber"}:
        raise RuntimeError("RootSwitch journal input ref 结构无效")
    return _SwitchInputRef(
        owner=_stored_text(item["owner"], "input owner"),
        generation=_stored_text(item["generation"], "input generation"),
        artifact=_stored_text(item["artifact"], "input artifact"),
        fiber=_stored_text(item["fiber"], "input fiber"),
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


def _pin_plan(pins: _ArtifactPins, plan: _Plan) -> _Plan:
    """Pin every code artifact before the first shared-owner action."""

    return _Plan(
        old_snapshot=plan.old_snapshot,
        new_snapshot=plan.new_snapshot,
        moves=tuple(
            _Move(
                name=move.name,
                old=_pin_ref(pins, move.old),
                new=_pin_ref(pins, move.new),
            )
            for move in plan.moves
        ),
    )


def _pin_ref(pins: _ArtifactPins, ref: _PartRef | None) -> _PartRef | None:
    if ref is None:
        return None
    artifact = pins.pin(ref.artifact, owner=ref.owner, name=ref.name)
    needs = tuple(_pin_need(pins, need) for need in ref.needs)
    pinned_by_owner = {need.owner: need for need in needs}
    inputs: list[_SwitchInputRef] = []
    for item in ref.inputs:
        if item.owner == ref.owner:
            if item.generation != ref.generation or item.artifact != ref.artifact:
                raise RuntimeError("RootSwitch owner input identity 不一致")
            input_artifact = artifact
        else:
            need = pinned_by_owner.get(item.owner)
            if (
                need is None
                or need.generation != item.generation
                or next(
                    (
                        source.artifact
                        for source in ref.needs
                        if source.owner == item.owner
                    ),
                    None,
                )
                != item.artifact
            ):
                raise RuntimeError("RootSwitch input dependency identity 不一致")
            input_artifact = need.artifact
        inputs.append(
            _SwitchInputRef(
                owner=item.owner,
                generation=item.generation,
                artifact=input_artifact,
                fiber=item.fiber,
            )
        )
    return _PartRef(
        name=ref.name,
        owner=ref.owner,
        generation=ref.generation,
        artifact=artifact,
        fiber=ref.fiber,
        inputs=tuple(inputs),
        needs=needs,
    )


def _pin_need(pins: _ArtifactPins, need: _PartNeed) -> _PartNeed:
    pinned = _pin_ref(
        pins,
        _PartRef(
            "need",
            need.owner,
            need.generation,
            need.artifact,
            "need",
        ),
    )
    if pinned is None:
        raise RuntimeError("RootSwitch dependency pin 意外为空")
    return _PartNeed(pinned.owner, pinned.generation, pinned.artifact)


def _pin_parts(pins: _ArtifactPins, parts: _PartSet) -> _PartSet:
    """Pin every part before its snapshot becomes committed state."""

    with pins.lock():
        return _pin_parts_locked(pins, parts)


def _pin_parts_locked(pins: _ArtifactPins, parts: _PartSet) -> _PartSet:
    return _PartSet(
        {
            name: _PartEntry(_pin_required(pins, entry.ref), entry.part)
            for name, entry in parts.items()
        }
    )


def _pin_required(pins: _ArtifactPins, ref: _PartRef) -> _PartRef:
    pinned = _pin_ref(pins, ref)
    if pinned is None:
        raise RuntimeError("RootSwitch part pin 意外为空")
    return pinned


def _finish_switch(
    journal: ReloadJournal,
    pins: _ArtifactPins,
    tx_id: str,
    *,
    use_new: bool,
    recovered: bool = False,
) -> None:
    with pins.lock():
        record = journal.switch_record(tx_id)
        if record is None:
            raise RuntimeError("RootSwitch record 不存在")
        journal.finish_switch(tx_id, use_new=use_new, recovered=recovered)
        pins.clean()


def _move_refs(move: _Move) -> tuple[_PartRef, ...]:
    return tuple(ref for ref in (move.old, move.new) if ref is not None)


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
