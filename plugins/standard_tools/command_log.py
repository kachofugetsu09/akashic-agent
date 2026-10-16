"""shell 工具自己的命令记录：只记 Agent 通过工具已经读到的事实，供本插件的终端标签只读展示。"""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

from agent.plugin_composition import Context, ServiceKey
from plugins.host_execution.contract import ExecutionResult, ExecutionState
from plugins.jobs.contract import JOB_NOTICES
from plugins.ledger.contract import Input, Message, OwnerStore, OwnerTransaction

SLOTS = 200
COMMAND_LIMIT = 4096
OUTPUT_LIMIT = 16 * 1024
PREVIEW_LIMIT = 80
_TRACKED = 512
_PREFIX = "command:"

CommandState = Literal["running", "exited", "timeout", "terminated", "unknown"]


@dataclass(frozen=True, slots=True)
class CommandStart:
    session_id: str
    call_id: str
    input_id: str
    input_preview: str
    command: str
    cwd: str
    tty: bool
    background: bool


def turn_input(messages: tuple[Message, ...], source: str) -> tuple[str, str]:
    """调用所属的最近一条同来源输入：给终端分组用，不是全局轮次身份。"""
    for message in reversed(messages):
        if message.source == source and isinstance(message.body, Input):
            text = " ".join(str(part.value) for part in message.body.parts if part.kind == "text")
            return message.message_id, " ".join(text.split())[:PREVIEW_LIMIT]
    return "", ""


def _clip(text: str, limit: int) -> tuple[str, bool]:
    return (text, False) if len(text) <= limit else (text[-limit:], True)


# 会话 ID 前加长度，任何会话 ID 的键区间都不会互相包含。
def _key(session_id: str, slot: int | str) -> str:
    return f"{_PREFIX}{len(session_id)}:{session_id}:{slot if isinstance(slot, str) else f'{slot:03d}'}"


class CommandLog:
    """每会话固定 SLOTS 个槽位循环覆盖；槽位数就是物理上限（OwnerStore 没有删除）。"""

    def __init__(self, ctx: Context, store: OwnerStore):
        self._ctx = ctx
        self._store = store
        self._boot = os.environ.get("AKASHIC_BOOT_ID", "")
        # (owner_key, execution_id) → (session_id, slot, call_id)。结算后 Agent 仍可能用 write_stdin 领取输出，
        # 所以不在结算时删除，只按插入顺序保留最近 _TRACKED 条；进程重启后丢失，读侧按 boot 判为 unknown。
        self._executions: dict[tuple[str, int], tuple[str, int, str]] = {}

    async def start(self, entry: CommandStart) -> int:
        command, clipped = _clip(entry.command, COMMAND_LIMIT)

        def write(tx: OwnerTransaction) -> int:
            head_key = _key(entry.session_id, "head")
            head = tx.read(head_key)
            serial = 0 if head is None else cast(int, head.value["next"])
            _ = tx.save(head_key, {"next": serial + 1}, expected_version=None if head is None else head.version)
            slot = serial % SLOTS
            current = tx.read(_key(entry.session_id, slot))
            _ = tx.save(_key(entry.session_id, slot), {
                "serial": serial, "call_id": entry.call_id, "input_id": entry.input_id,
                "input_preview": entry.input_preview, "command": command, "command_truncated": clipped,
                "cwd": entry.cwd, "tty": entry.tty, "background": entry.background, "boot": self._boot,
                "state": "running", "exit_code": None, "execution_id": None, "job_id": None,
                "started_at": time.time(), "ended_at": None, "output": "", "output_truncated": False,
            }, expected_version=None if current is None else current.version)
            return slot

        return await self._store.transact_async(write)

    async def result(self, session_id: str, slot: int, call_id: str, owner: str, result: ExecutionResult) -> None:
        if result.execution_id is not None:
            self._executions[(owner, result.execution_id)] = (session_id, slot, call_id)
            while len(self._executions) > _TRACKED:
                del self._executions[next(iter(self._executions))]
        await self._update(session_id, slot, call_id, lambda value: _apply_result(value, result))

    async def output(self, owner: str, result: ExecutionResult, execution_id: int) -> None:
        located = self._executions.get((owner, execution_id))
        if located is not None:
            await self._update(*located, lambda value: _apply_result(value, result))

    async def settled(self, owner: str, state: ExecutionState) -> None:
        located = self._executions.get((owner, state.execution_id))
        if located is not None:
            await self._update(*located, lambda value: {**value, **_final(state.exit_code, state.finish_reason)})

    # 已自然结束的命令再被 task_stop 释放时保留原结局。
    async def stopped(self, owner: str, execution_id: int) -> None:
        located = self._executions.pop((owner, execution_id), None)
        if located is not None:
            await self._update(*located, lambda value: value if value["state"] != "running"
                               else {**value, "state": "terminated", "ended_at": time.time()})

    # 槽位已被新命令覆盖（call_id 不同）时放弃更新，旧命令不会在新槽位里复活。
    async def _update(self, session_id: str, slot: int, call_id: str,
                      change: Callable[[dict[str, object]], dict[str, object]]) -> None:
        def write(tx: OwnerTransaction) -> None:
            record = tx.read(_key(session_id, slot))
            if record is None or record.value.get("call_id") != call_id:
                return
            _ = tx.save(_key(session_id, slot), change(dict(record.value)), expected_version=record.version)

        # 记录是展示用的副本：写失败只报告，不让命令本身失败。
        try:
            await self._store.transact_async(write)
        except Exception as error:
            _ = self._ctx.report_incident("shell_command_log_failed", f"{type(error).__name__}: {error}")

    def list(self, session_id: str) -> list[dict[str, object]]:
        """按开始顺序返回本会话的记录；不是本次启动写下且未结束的标为 unknown，不猜结局。"""
        rows = self._store.scan(start=_key(session_id, 0), stop=_key(session_id, SLOTS), limit=SLOTS)
        records: list[dict[str, object]] = []
        for key, record in rows:
            value = dict(record.value)
            if value.get("state") == "running" and value.get("boot") != self._boot:
                value["state"] = "unknown"
            value["slot"] = int(key.rsplit(":", 1)[1])
            records.append(value)
        return sorted(records, key=lambda value: cast(int, value["serial"]))

    async def stop(self, session_id: str, slot: int) -> bool:
        """只停后台命令：走 jobs 的同一个停止入口；前台命令由对话的中断处理。"""
        record = self._store.read(_key(session_id, slot))
        job = None if record is None else record.value.get("job_id")
        if not isinstance(job, str) or record is None or record.value.get("state") != "running":
            return False
        async with self._ctx.runtime_scope():
            with self._ctx.borrow(JOB_NOTICES) as notices:
                return False if notices is None else await notices.stop_job(session_id, job)


def _final(exit_code: int | None, finish_reason: str) -> dict[str, object]:
    state = "exited" if finish_reason == "natural" else "timeout" if finish_reason == "timeout" else "terminated"
    return {"state": state, "exit_code": exit_code, "ended_at": time.time()}


def _apply_result(value: dict[str, object], result: ExecutionResult) -> dict[str, object]:
    text = cast(str, value["output"]) + result.output.decode("utf-8", errors="replace")
    output, clipped = _clip(text, OUTPUT_LIMIT)
    updated: dict[str, object] = {**value, "output": output, "output_truncated": bool(value["output_truncated"])
                                  or clipped or result.output_omitted_bytes > 0}
    if result.execution_id is not None:
        updated["execution_id"] = result.execution_id
        if value["background"]:
            updated["job_id"] = f"shell-{result.execution_id}"
        return updated
    return {**updated, **_final(result.exit_code, result.finish_reason)}


COMMAND_LOG_VIEW = ServiceKey[CommandLog]("standard_tools.command-log-view.v1")


def record_view(value: Mapping[str, object], *, with_output: bool) -> dict[str, object]:
    """对外只给展示需要的字段；列表默认不带输出，展开时单独取。"""
    keys = ("slot", "serial", "call_id", "input_id", "input_preview", "command", "command_truncated", "cwd", "tty",
            "background", "state", "exit_code", "started_at", "ended_at", "output_truncated")
    view = {key: value.get(key) for key in keys}
    view["stoppable"] = value.get("state") == "running" and isinstance(value.get("job_id"), str)
    if with_output:
        view["output"] = value.get("output", "")
    return view
