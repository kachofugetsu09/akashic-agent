"""每轮改动的观测窗口：该轮开始时拍起始快照，结束时拍结束快照并保存 diff。

统计口径是窗口内的工作目录差异，不等同于“Agent 本轮做的改动”；窗口可能不准时带标记说明。
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import aclosing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import cast

from agent.plugin_composition import Context
from core.common.file_io import run_file_io
from plugins.jobs.contract import JOB_NOTICES
from plugins.ledger.contract import MESSAGE_CATALOG, OWNER_STATE, Input, Message, OwnerTransaction, Output, ToolCall
from plugins.sources.contract import SourceChangedV3
from plugins.turn_projection.contract import TURN_PROJECTION, Turn, TurnProjection

from .git import Git, GitUnavailable, Repo

SLOTS = 50
_ACTIVE = "active"


# 内存里只放进行中的窗口；记录本身在 OWNER_STATE，按会话 50 个槽位循环覆盖。
@dataclass(slots=True)
class _Window:
    session: str
    source: str
    turn_id: str
    after_seq: int
    restored_at: datetime | None
    head: int = -1


class TurnWindows:
    def __init__(self, ctx: Context, git: Git, repo: Callable[[str], Awaitable[Repo]]) -> None:
        self._ctx = ctx
        self._git = git
        self._repo = repo
        self._queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._queued: dict[tuple[str, str], int] = {}
        self._starting: dict[str, int] = {}
        self._starting_after: dict[str, int] = {}
        self._active: dict[str, _Window] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    # SOURCE_CHANGED_V3 的同步监听：只记下当时的 head（索引查询）并排队，不读历史。
    # 起点按这个 head 投影，后台任务开始时该轮即使已结束也能认出是哪一轮。
    def changed(self, event: SourceChangedV3) -> None:
        if not event.pending:
            return
        item = (event.reader.session_id, event.source)
        head = event.reader.head()
        if item not in self._queued:
            self._starting[item[0]] = self._starting.get(item[0], 0) + 1
            self._starting_after[item[0]] = min(self._starting_after.get(item[0], head), head)
            self._queue.put_nowait(item)
        self._queued[item] = head

    async def run(self) -> None:
        await self._restore()
        async with asyncio.TaskGroup() as tasks:
            _ = tasks.create_task(self._follow(tasks))
            while True:
                session, source = await self._queue.get()
                head = self._queued.pop((session, source))
                _ = tasks.create_task(self._guard(f"start:{session}", self._start(session, source, head)))

    # 单轮出错只报告，不让整个采集停下；该轮留在 started，下次唤醒或重启时再试。
    async def _guard(self, label: str, work: Awaitable[None]) -> None:
        try:
            await work
        except Exception as error:
            _ = self._ctx.report_incident("workspace-changes", f"{label}: {error!r}")

    # 回复下方的摘要按结束消息查；消息落在进行中（或正在开始）的窗口之后才返回 pending，
    # 更早的回答（被覆盖的旧槽、启用插件之前的轮次）直接说没有记录，不跟着新一轮一起“统计中”。
    async def by_message(self, session: str, message_id: str) -> dict[str, object]:
        for value in await self._records(session):
            if value["ending_message_id"] == message_id:
                return _view(value)
        async with self._ctx.runtime_scope():
            catalog = self._ctx.require(MESSAGE_CATALOG)
            message = await run_file_io(lambda: catalog.reader(session).get(message_id))
        starting = self._starting_after.get(session)
        if message is not None and (
            (starting is not None and message.seq > starting)
            or any(window.session == session and message.seq > window.after_seq for window in self._active.values())
        ):
            return {"phase": "pending"}
        return {"phase": "unavailable", "error": {"reason": "not_recorded", "message": "这一轮没有改动记录"}}

    # 1. 起点：按 (session, source, turn_id) 只拍一次；同一轮追加输入或 resume 不重置。
    async def _start(self, session: str, source: str, head: int) -> None:
        try:
            key = await self._open_window(session, source, head)
        finally:
            self._starting[session] -= 1
            if not self._starting[session]:
                del self._starting[session]
                del self._starting_after[session]
        # 拍快照期间该轮可能已经结束，那次唤醒没有看到这个窗口。
        if key is not None:
            await self._end(key)

    async def _open_window(self, session: str, source: str, head: int) -> str | None:
        async with self._ctx.runtime_scope(), self._lock(session):
            started_at = _now()
            catalog = self._ctx.require(MESSAGE_CATALOG)
            projection = self._ctx.require(TURN_PROJECTION)
            reader = catalog.reader(session)
            turns, messages = await run_file_io(lambda: _project(projection, reader.snapshot(through_seq=head), source, -1))
            turn = turns[-1] if turns and turns[-1].status == "open" else None
            turn_id = None if turn is None else _first_input(turn, messages)
            if turn is None or turn_id is None or any(
                (item.session, item.source, item.turn_id) == (session, source, turn_id) for item in self._active.values()
            ):
                return None
            record = await self._baseline(session, source, turn_id, turn.after_seq, started_at)
            if record is None:
                return None
            key = await self._save_new(session, record)
            self._active[key] = _Window(session, source, turn_id, turn.after_seq, None)
            return key

    async def _baseline(self, session: str, source: str, turn_id: str, after_seq: int, started_at: datetime) -> dict[str, object] | None:
        record: dict[str, object] = {
            "v": 1, "session": session, "source": source, "turn_id": turn_id, "after_seq": after_seq,
            "phase": "started", "status": None, "ending_message_id": None, "repo": None, "baseline": None,
            "window": {"started_at": started_at.isoformat(), "baseline_done_at": None, "ended_at": None, "end_done_at": None},
            "files": [], "truncated": False, "baseline_late": False, "end_late": False, "error": None,
        }
        try:
            repo = await self._repo(session)
            snapshot = await self._git.snapshot(repo)
        except GitUnavailable as error:
            # 没设工作目录的会话（如聊天渠道）不留记录。
            if error.reason == "no_directory":
                return None
            record["error"] = error.view()
            return record
        record.update(repo=asdict(repo), baseline=snapshot.tree)
        cast(dict[str, object], record["window"])["baseline_done_at"] = _now().isoformat()
        return record

    # 2. 终点：看到该轮 complete/quiet/abandoned 后拍结束快照，立刻算好 diff 保存。
    async def _end(self, key: str) -> None:
        window = self._active.get(key)
        if window is None:
            return
        async with self._ctx.runtime_scope(), self._lock(window.session):
            if key not in self._active:
                return
            catalog = self._ctx.require(MESSAGE_CATALOG)
            projection = self._ctx.require(TURN_PROJECTION)
            reader = catalog.reader(window.session)
            after = window.after_seq
            turns, messages = await run_file_io(lambda: _project(projection, reader.snapshot(after_seq=after), window.source, after))
            ours = next((turn for turn in turns if window.turn_id in turn.message_ids), None)
            if ours is None or ours.status == "open" or ours.ending_message_id is None:
                return
            state = self._ctx.require(OWNER_STATE).open(self._ctx)
            current = await run_file_io(lambda: state.read(key))
            if current is None:
                return
            record: dict[str, object] = {**current.value, "window": dict(cast(Mapping[str, object], current.value["window"]))}
            await self._finish(record, window, ours, turns, messages)
            await self._save_ended(key, record)
            del self._active[key]

    async def _finish(self, record: dict[str, object], window: _Window, ours: Turn, turns: Sequence[Turn], messages: Mapping[str, Message]) -> None:
        times = cast(dict[str, object], record["window"])
        ended = messages[ours.ending_message_id or ""].recorded_at
        record.update(phase="ended", status=ours.status, ending_message_id=ours.ending_message_id)
        times["ended_at"] = ended.isoformat()
        if record["error"] is not None or record["repo"] is None:
            return
        # 下一轮已经开始、会话还有后台命令、abandon 后旧工具可能还在跑、或重启前就已结束：结束点可能偏晚。
        later = any(turn.after_seq >= ours.through_seq for turn in turns if turn is not ours)
        restored = window.restored_at is not None and ended < window.restored_at
        late = later or ours.status == "abandoned" or restored or await self._jobs_running(window.session)
        fixed = cast(dict[str, object], record["repo"])
        repo = Repo(cast(str, fixed["toplevel"]), cast(str, fixed["git_dir"]), cast(bool, fixed["submodules"]))
        try:
            snapshot = await self._git.snapshot(repo)
            changes = await self._git.files(repo, cast(str, record["baseline"]), snapshot.tree)
        except GitUnavailable as error:
            record["error"] = error.view()
            return
        times["end_done_at"] = _now().isoformat()
        first_call = min((message.recorded_at for message in messages.values() if message.message_id in ours.message_ids and _calls(message)), default=None)
        baseline_done = datetime.fromisoformat(cast(str, times["baseline_done_at"]))
        record.update(
            files=[asdict(item) for item in changes.files], truncated=changes.truncated,
            baseline_late=first_call is not None and first_call < baseline_done, end_late=late,
        )

    async def _jobs_running(self, session: str) -> bool:
        with self._ctx.borrow(JOB_NOTICES) as jobs:
            if jobs is None:
                return False
            async with aclosing(jobs.follow_jobs(session)) as snapshots:
                async for current in snapshots:
                    return any(job.status in {"running", "stopping"} for job in current)
        return False

    async def _follow(self, tasks: asyncio.TaskGroup) -> None:
        async for heads in self._ctx.require(MESSAGE_CATALOG).follow():
            for key, window in list(self._active.items()):
                head = heads.get(window.session)
                if head is not None and head != window.head:
                    window.head = head
                    _ = tasks.create_task(self._guard(f"end:{key}", self._end(key)))

    # 3. 重启：接回进行中的窗口；重启期间开始的轮次没有起点，不补算。
    async def _restore(self) -> None:
        async with self._ctx.runtime_scope():
            await self._restore_windows()

    async def _restore_windows(self) -> None:
        state = self._ctx.require(OWNER_STATE).open(self._ctx)
        restored_at = _now()

        def read() -> list[tuple[str, Mapping[str, object]]]:
            active = state.read(_ACTIVE)
            keys = cast(list[str], active.value["keys"]) if active is not None else []
            return [(key, record.value) for key in keys if (record := state.read(key)) is not None and record.value["phase"] == "started"]

        for key, value in await run_file_io(read):
            self._active[key] = _Window(
                cast(str, value["session"]), cast(str, value["source"]), cast(str, value["turn_id"]),
                cast(int, value["after_seq"]), restored_at,
            )

    async def _save_new(self, session: str, record: dict[str, object]) -> str:
        state = self._ctx.require(OWNER_STATE).open(self._ctx)
        prefix = _prefix(session)

        def save(tx: OwnerTransaction) -> str:
            cursor = tx.read("cursor:" + prefix)
            slot = 0 if cursor is None else cast(int, cursor.value["next"])
            _ = tx.save("cursor:" + prefix, {"next": (slot + 1) % SLOTS}, expected_version=None if cursor is None else cursor.version)
            key = f"{prefix}{slot:02d}"
            previous = tx.read(key)
            _ = tx.save(key, record, expected_version=None if previous is None else previous.version)
            _active_keys(tx, add=key)
            return key

        return await state.transact_async(save)

    async def _save_ended(self, key: str, record: dict[str, object]) -> None:
        state = self._ctx.require(OWNER_STATE).open(self._ctx)

        def save(tx: OwnerTransaction) -> None:
            previous = tx.read(key)
            # 槽位已被更新的轮次覆盖时放弃本次结果。
            if previous is None or previous.value["turn_id"] != record["turn_id"]:
                return
            _ = tx.save(key, record, expected_version=previous.version)
            _active_keys(tx, remove=key)

        await state.transact_async(save)

    async def _records(self, session: str) -> list[Mapping[str, object]]:
        prefix = _prefix(session)
        async with self._ctx.runtime_scope():
            state = self._ctx.require(OWNER_STATE).open(self._ctx)
            rows = await run_file_io(lambda: state.scan(start=prefix, stop=prefix + "~", limit=SLOTS))
        values = [record.value for _, record in rows]
        return sorted(values, key=lambda value: cast(str, cast(Mapping[str, object], value["window"])["started_at"]), reverse=True)

    def _lock(self, session: str) -> asyncio.Lock:
        return self._locks.setdefault(session, asyncio.Lock())


def _active_keys(tx: OwnerTransaction, *, add: str | None = None, remove: str | None = None) -> None:
    current = tx.read(_ACTIVE)
    keys = [] if current is None else [key for key in cast(list[str], current.value["keys"]) if key not in {add, remove}]
    _ = tx.save(_ACTIVE, {"keys": keys + ([add] if add else [])}, expected_version=None if current is None else current.version)


def _project(projection: TurnProjection, messages: tuple[Message, ...], source: str, after_seq: int) -> tuple[tuple[Turn, ...], dict[str, Message]]:
    turns = projection.project(messages, source, after_seq=after_seq)
    return turns, {message.message_id: message for message in messages}


def _first_input(turn: Turn, messages: Mapping[str, Message]) -> str | None:
    return next((item for item in turn.message_ids if isinstance(messages[item].body, Input)), None)


def _calls(message: Message) -> bool:
    body = message.body
    return isinstance(body, Output) and any(isinstance(part, ToolCall) for part in body.parts)


def _prefix(session: str) -> str:
    return f"turn:{len(session)}:{session}:"


def _now() -> datetime:
    return datetime.now(UTC)


# 旧记录里可能还存着 patch，一并不返回。
def _view(value: Mapping[str, object]) -> dict[str, object]:
    hidden = {"v", "session", "baseline", "after_seq", "patch"}
    return {key: item for key, item in value.items() if key not in hidden}
