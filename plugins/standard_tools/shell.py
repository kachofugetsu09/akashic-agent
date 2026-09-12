from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from functools import partial
import hashlib
import json
from pathlib import Path
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agent.plugin_composition import Context, PROCESSES, ServiceKey
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.tasks import TASKS, Task, TaskSlot
from .shell_backend import _log_shell_execution, _shell_env
from agent.tools.shell_command import resolve_shell
from agent.tools.shell_security import validate_command
from agent.tools.unified_exec import (
    DEFAULT_HARD_TIMEOUT_S, DEFAULT_INITIAL_YIELD_TIME_MS, DEFAULT_MAX_OUTPUT_TOKENS,
    MAX_HARD_TIMEOUT_S, ExecutionCleanupReport, UnknownExecutionError,
    clamp_initial_yield_time, clamp_write_stdin_yield_time, format_execution_result,
)
from plugins.tools.api import InvalidArguments
from agent.plugin_composition.messages import MessageReader
from agent.plugin_contracts import CallRef, ContentPart, Control, Message, Output, ToolCall
from agent.plugin_contracts import json_value

from ._tool_boundary import CallSource, TOOLS, ToolRef, ToolResultValue


class ShellSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    owner_key: str | None = None
    owner_boundary: Literal["abandon"] = "abandon"
    working_dir: str | None = None
    restricted_dir: str | None = None
    allow_network: bool = True

    @field_validator("working_dir", "restricted_dir")
    @classmethod
    def absolute_path(cls, value: str | None) -> str | None:
        if value is not None and not Path(value).is_absolute():
            raise ValueError("工具目录必须是绝对路径")
        return value

    @field_validator("owner_key")
    @classmethod
    def nonempty_owner(cls, value: str | None) -> str | None:
        if value is not None and not value:
            raise ValueError("owner_key 不能为空")
        return value


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    command: str = Field(min_length=1, description="要执行的命令；需要观察长任务时避免等待 EOF 的过滤器。")
    description: str = Field(description="简短说明命令作用。")
    cwd: str | None = Field(default=None, description="可选工作目录，覆盖默认目录。")
    shell: str | None = Field(default=None, description="shell binary；默认使用当前用户的默认 shell。")
    login: bool = True
    tty: bool = Field(default=False, description="需要交互输入时启用 PTY。")
    yield_time_ms: int = DEFAULT_INITIAL_YIELD_TIME_MS
    max_output_tokens: int = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, ge=0)
    timeout: int = Field(default=DEFAULT_HARD_TIMEOUT_S, ge=1, le=MAX_HARD_TIMEOUT_S)


class Stdin(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    execution_id: int
    chars: str = ""
    yield_time_ms: int | None = None
    max_output_tokens: int = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, ge=0)


class Stop(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    execution_id: int


class PreparedCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    owner_key: str
    command: str
    description: str
    argv: list[str]
    shell_kind: str
    login: bool
    cwd: str | None
    tty: bool
    yield_time_ms: int
    max_output_tokens: int
    timeout: int


class PreparedStdin(Stdin):
    owner_key: str


class PreparedStop(Stop):
    owner_key: str


def _owner_key(settings: ShellSettings, session_id: str, source: str, abandon_id: str | None = None) -> str:
    key = settings.owner_key or json.dumps((session_id, source), ensure_ascii=False, separators=(",", ":"))
    return key if abandon_id is None else json.dumps((key, "abandon", abandon_id), ensure_ascii=False, separators=(",", ":"))


def _abandon_before(messages: tuple[Message, ...], source: str, through_seq: int) -> str | None:
    """只从已持久化边界区分新旧进程集合，暂停和新输入不改变 owner。"""
    return next((message.message_id for message in reversed(messages)
                 if message.source == source and message.seq <= through_seq
                 and isinstance(message.body, Control) and message.body.action == "abandon"), None)


class ShellOwners:
    """调用方只释放一个 Shell 作业；实际插件 owner 由此端口固定。"""

    def __init__(self, ctx: Context):
        self._ctx = ctx

    async def release_tool(self, state: Mapping[str, object], session_id: str, source: str) -> ExecutionCleanupReport:
        """由原插件解释原工具配置；调用方不重复保存或推断作业 owner。"""
        settings = ShellSettings.model_validate(json_value(state))
        return await self.release(_owner_key(settings, session_id, source))

    async def release(self, owner_key: str) -> ExecutionCleanupReport:
        return await self._ctx.require(PROCESSES).terminate_owner(self._ctx, owner_key)


SHELL_OWNERS = ServiceKey[ShellOwners]("shell.owners.v1")


class ShellCleanup(Protocol):
    """标准 Shell 提供给程序组合的真实收尾边界。"""

    def __call__(
        self,
        ctx: Context,
        reader: MessageReader,
        source: str,
        from_seq: int,
        *,
        task: Task | None = None,
        drain: Callable[[tuple[CallRef, ...]], Awaitable[None]] | None = None,
    ) -> AbstractAsyncContextManager[None]: ...


TOOL_CLEANUP = ServiceKey[ShellCleanup]("tools.cleanup.v1")


class ShellTool:
    idempotent = False

    def __init__(self, ctx: Context, name: Literal["shell", "write_stdin", "task_stop"], settings: ShellSettings):
        self._ctx = ctx
        self._name = name
        self._settings = settings

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object]:
        """校验最终命令并固定进程 owner；恢复不重选目录、shell 或默认参数。"""
        owner = (
            self._settings.owner_key or "standalone" if source is None
            else _owner_key(
                self._settings, source.messages[-1].session_id, source.messages[-1].source,
                _abandon_before(source.messages, source.messages[-1].source, source.messages[-1].seq),
            )
        )
        raw = json_value(arguments)
        try:
            if self._name == "task_stop":
                return PreparedStop(**Stop.model_validate(raw).model_dump(), owner_key=owner).model_dump()
            if self._name == "write_stdin":
                args = Stdin.model_validate(raw)
                default = 5000 if not args.chars else 250
                args.yield_time_ms = clamp_write_stdin_yield_time(
                    default if args.yield_time_ms is None else args.yield_time_ms, has_input=bool(args.chars),
                )
                return PreparedStdin(**args.model_dump(), owner_key=owner).model_dump()
            command = Command.model_validate(raw)
        except ValidationError as error:
            raise InvalidArguments(str(error)) from error
        text = command.command.strip()
        if not text:
            raise InvalidArguments("命令不能为空")
        try:
            shell = resolve_shell(command.shell)
        except ValueError as error:
            raise InvalidArguments(str(error)) from error
        cwd = command.cwd or self._settings.working_dir or self._settings.restricted_dir
        directory = None if cwd is None else Path(cwd).expanduser().absolute()
        denied = validate_command(
            text, allow_network=self._settings.allow_network,
            restricted_dir=None if self._settings.restricted_dir is None else Path(self._settings.restricted_dir),
            cwd=directory,
        )
        if denied:
            raise InvalidArguments(denied)
        return PreparedCommand(
            owner_key=owner, command=text, description=command.description,
            argv=shell.derive_argv(text, login=command.login), shell_kind=shell.kind.value,
            login=command.login, cwd=None if directory is None else str(directory), tty=command.tty,
            yield_time_ms=clamp_initial_yield_time(command.yield_time_ms),
            max_output_tokens=command.max_output_tokens, timeout=command.timeout,
        ).model_dump()

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> ToolResultValue:
        """执行与续接只访问同一个物理进程 owner，失败不伪装为成功。"""
        processes = self._ctx.require(PROCESSES)
        raw = json_value(arguments)
        command_text: str | None = None
        if self._name == "task_stop":
            stop = PreparedStop.model_validate(raw)
            stopped = await processes.terminate_execution(self._ctx, stop.owner_key, stop.execution_id)
            return ToolResultValue("success" if stopped else "error", (ContentPart("text", json.dumps({
                "execution_id": stop.execution_id, **({"process_status": "stopped"} if stopped else {}),
                "status": "stopped" if stopped else "not_found",
            })),))
        if self._name == "write_stdin":
            stdin = PreparedStdin.model_validate(raw)
            assert stdin.yield_time_ms is not None
            try:
                result = await processes.write_stdin(
                    self._ctx, stdin.owner_key, execution_id=stdin.execution_id, chars=stdin.chars,
                    yield_time_ms=stdin.yield_time_ms, max_output_tokens=stdin.max_output_tokens,
                )
            except UnknownExecutionError as error:
                return ToolResultValue("error", (ContentPart("text", str(error)),))
        else:
            command = PreparedCommand.model_validate(raw)
            command_text = command.command
            log = partial(
                _log_shell_execution, operation_id=key, description=command.description,
                command_fp=hashlib.sha256(command.command.encode()).hexdigest()[:16],
                command_bytes=len(command.command.encode()), cwd=command.cwd or "",
                shell_kind=command.shell_kind, login=command.login, tty=command.tty, session=command.owner_key,
            )
            log("shell.execution_admitted")
            result = await processes.exec_command(
                self._ctx, command.owner_key, command=command.command, argv=command.argv,
                cwd=None if command.cwd is None else Path(command.cwd), env=_shell_env(), tty=command.tty,
                yield_time_ms=command.yield_time_ms, max_output_tokens=command.max_output_tokens,
                hard_timeout_s=command.timeout,
            )
            log("shell.execution_result", result=result)
        outcome = "success" if result.execution_id is not None or result.exit_code == 0 else "error"
        return ToolResultValue(outcome, (ContentPart("text", format_execution_result(result, command=command_text)),))

    async def query(self, key: str) -> ToolResultValue | None:
        return None


async def register_shell(ctx: Context) -> tuple[ToolRef, ...]:
    """配置由 Shell owner 校验，所有操作与作业释放共用此插件身份。"""
    _ = await ctx.provide(SHELL_OWNERS, ShellOwners(ctx))
    _ = await ctx.provide(TOOL_CLEANUP, shell_cleanup)
    definitions: tuple[tuple[Literal["shell", "write_stdin", "task_stop"], type[BaseModel], str], ...] = (
        ("shell", Command, "执行 shell 命令；返回终态或可供 write_stdin/task_stop 使用的 execution_id。"),
        ("write_stdin", Stdin, "续接命令，等待新增输出或输入 PTY 字符；仅返回上次读取后的新增内容。"),
        ("task_stop", Stop, "确认终止命令的进程组并释放 execution_id。"),
    )
    refs: list[ToolRef] = []
    for name, schema, description in definitions:
        refs.append(await _register(ctx, name, schema, description))
    return tuple(refs)


async def _register(
    ctx: Context,
    name: Literal["shell", "write_stdin", "task_stop"],
    schema: type[BaseModel],
    description: str,
) -> ToolRef:
    def capture(configuration: Mapping[str, object]) -> Mapping[str, object]:
        return ShellSettings.model_validate(json_value(configuration)).model_dump()

    @asynccontextmanager
    async def open_tool(state: Mapping[str, object]) -> AsyncGenerator[ShellTool]:
        yield ShellTool(ctx, name, ShellSettings.model_validate(json_value(state)))

    return cast(ToolRef, await ctx.require(TOOLS).register(
        ctx,
        name=name,
        description=description,
        parameters=schema.model_json_schema(),
        open=open_tool,
        capture=capture,
        risk="external-side-effect",
    ))


@asynccontextmanager
async def shell_cleanup(
    ctx: Context, reader: MessageReader, source: str, from_seq: int, *,
    task: Task | None = None, drain: Callable[[tuple[CallRef, ...]], Awaitable[None]] | None = None,
) -> AsyncGenerator[None]:
    """放弃可停止等待；独立 Task 保留旧进程清理、generation 与重启许可。"""
    try:
        yield
    finally:
        # 1. 放弃后到达的新输入与调用不属于旧程序的清理范围。
        messages = reader.snapshot()
        abandoned = next((message for message in messages
                          if message.source == source and isinstance(message.body, Control)
                          and message.body.action == "abandon" and message.body.through_seq >= from_seq), None)
        end = reader.head() if abandoned is None else cast(Control, abandoned.body).through_seq
        calls = tuple(
            (CallRef(message.message_id, index), part.binding_id,
             _abandon_before(messages, source, message.seq))
            for message in messages
            if message.source == source and from_seq <= message.seq <= end and isinstance(message.body, Output)
            for index, part in enumerate(message.body.parts) if isinstance(part, ToolCall)
        )
        if calls:
            async def cleanup() -> None:
                try:
                    if drain is not None:
                        await drain(tuple(ref for ref, _, _ in calls))
                finally:
                    bindings = ctx.require(BINDINGS)
                    for identity, boundary in dict.fromkeys((identity, boundary) for _, identity, boundary in calls):
                        try:
                            metadata = bindings.describe(identity, TOOLS)
                            description = cast(Mapping[str, object], metadata["tool"])
                            if description["name"] not in {"shell", "write_stdin", "task_stop"}:
                                continue
                            # 2. 只装配原工具闭包，不打开或重跑工具；在其 scope 固定清理 provider。
                            async with bindings.open(identity, TOOLS):
                                owners_binding = bindings.bind(SHELL_OWNERS, {})
                            async with bindings.open(owners_binding, SHELL_OWNERS) as (owners, _):
                                state = cast(Mapping[str, object], metadata["state"])
                                # 老归档没有此标记，仍按原 owner 释放；新归档固定放弃前的分区。
                                if "owner_boundary" in state:
                                    settings = ShellSettings.model_validate(json_value(state))
                                    state = {**state, "owner_key": _owner_key(settings, reader.session_id, source, boundary)}
                                report = await owners.release_tool(
                                    state, reader.session_id, source,
                                )
                            if report.failures:
                                _ = ctx.report_incident("shell_cleanup_failed", f"{identity}: {report.failures}")
                        except Exception as error:
                            # SH-002: 此处是独立清理边界，失败保留真实 owner 并明确报告。
                            _ = ctx.report_incident("shell_cleanup_failed", f"{identity}: {type(error).__name__}: {error}")

            async def run(_task: Task | None) -> None:
                """宿主停止也先排空真实清理，不能把撤权当作进程已退出。"""
                scope = ctx.capture_runtime_scope()
                async def scoped_cleanup() -> None:
                    async with scope:
                        await cleanup()
                operation = scoped_cleanup()
                try:
                    work = asyncio.create_task(operation)
                except BaseException:
                    operation.close()
                    await scope.close()
                    raise
                try:
                    await asyncio.shield(work)
                except asyncio.CancelledError:
                    while not work.done():
                        try:
                            await asyncio.shield(work)
                        except asyncio.CancelledError:
                            continue
                    if not work.cancelled():
                        work.result()
                    raise

            # 2. Task 自己固定当前 scope；源程序结束不会释放它的资源或 permit。
            def admit(slot: TaskSlot) -> Task:
                permit = task.child_permit() if task is not None and task.has_external_permit else None
                try:
                    owned = slot.start(run)
                except BaseException:
                    if permit is not None:
                        permit.release()
                    raise
                if permit is not None:
                    owned.on_done(permit.release)
                return owned

            if task is None:
                await run(None)
            else:
                owned = await ctx.require(TASKS).open(ctx).admit(
                    ("shell-cleanup", reader.session_id, source, from_seq, end), admit,
                )
                if abandoned is None:
                    await _wait_cleanup(owned, reader, source, from_seq)


async def _wait_cleanup(task: Task, reader: MessageReader, source: str, from_seq: int) -> None:
    """普通停止排空；若随后明确放弃，只释放等待者，原清理 Task 继续持有资源。"""
    async def abandoned() -> None:
        async for message in reader.follow(after_seq=from_seq - 1):
            if (message.source == source and isinstance(message.body, Control)
                    and message.body.action == "abandon" and message.body.through_seq >= from_seq):
                return
        raise RuntimeError("Shell 清理订阅提前结束")

    joined = asyncio.create_task(task.join())
    stopped = asyncio.create_task(abandoned())
    cancellation: asyncio.CancelledError | None = None
    try:
        while True:
            try:
                done, _ = await asyncio.wait((joined, stopped), return_when=asyncio.FIRST_COMPLETED)
                break
            except asyncio.CancelledError as error:
                cancellation = error
        if joined in done:
            _ = joined.result()
        else:
            stopped.result()
        if cancellation is not None:
            raise cancellation
    finally:
        _ = joined.cancel()
        _ = stopped.cancel()
        _ = await asyncio.gather(joined, stopped, return_exceptions=True)
