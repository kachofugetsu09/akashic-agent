from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from plugins.host_execution.contract import PROCESSES
from agent.plugin_composition import CompositionError, Context
from plugins.ledger.contract import BINDINGS
from plugins.ledger.contract import MessageReader
from plugins.host_execution.contract import (
    DEFAULT_HARD_TIMEOUT_S, DEFAULT_INITIAL_YIELD_TIME_MS, DEFAULT_MAX_OUTPUT_TOKENS, MAX_HARD_TIMEOUT_S, ExecutionCleanupReport, ExecutionResult, ExecutionState, UnknownExecutionError, clamp_initial_yield_time, clamp_write_stdin_yield_time
)
from .shell_backend import format_execution_result
from agent.plugin_composition.tasks import TASKS, Task, TaskAdmission, TaskSlot
from plugins.ledger.contract import (
    CallRef,
    ContentPart,
    Control,
    Message,
    Message,
    Output,
    ToolCall,
    json_value,
)
from plugins.tools.contract import (
    TOOL_CLEANUP as TOOL_CLEANUP,
)

from ._tool_boundary import TOOLS, CallSource, ToolRef, ToolResultValue
from plugins.context.contract import MATERIALS_V4
from plugins.jobs.contract import JOB_NOTICES, JobOutcome, JobWatch
from .shell_backend import _log_shell_execution, _shell_env
from .shell_security import validate_command
from .working_directory import WorkingDirectories


_SETTLE_POLL_S = 1.0


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
    shell: str | None = Field(default=None, description="shell binary；默认值见当前命令环境材料。按 Bash 语法编写时显式传入 bash。")
    login: bool = True
    tty: bool = Field(default=False, description="需要交互输入时启用 PTY。")
    yield_time_ms: int = DEFAULT_INITIAL_YIELD_TIME_MS
    max_output_tokens: int = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, ge=0)
    timeout: int = Field(default=DEFAULT_HARD_TIMEOUT_S, ge=1, le=MAX_HARD_TIMEOUT_S)
    run_in_background: bool = Field(
        default=False,
        description=(
            "后台运行并在结束时回调：命令退出后自动通知当前会话（本轮进行中注入下一步，空闲时唤醒新一轮），"
            "期间可以继续对话或结束本轮。\n"
            "任何需要等待结果的事都能变成回调，只要把它写成一条阻塞到完成才退出的命令："
            "编译、测试、下载、渲染、部署、长脚本；其他 agent 的非交互模式（如 codex exec、devin -p、claude -p）；"
            "只提供异步接口的服务，写一个轮询到终态再退出的脚本。结果写入文件，通知到达后再读取。\n"
            "不要在本轮轮询或用 write_stdin 长等待，也不要承诺稍后再看：你不会自己回来，只有回调会唤醒你。"
            "需要时用 write_stdin 读取输出、task_stop 终止。"
        ),
    )


class Stdin(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    execution_id: int
    chars: str = ""
    yield_time_ms: int | None = None
    max_output_tokens: int = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, ge=0)


class Stop(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    execution_id: int


class BackgroundTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    session_id: str
    source: str


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
    call_context: dict[str, object] = Field(default_factory=dict)
    background: BackgroundTarget | None = None


class PreparedStdin(Stdin):
    owner_key: str
    background_owner_key: str | None = None


class PreparedStop(Stop):
    owner_key: str
    background_owner_key: str | None = None


def _owner_key(settings: ShellSettings, session_id: str, source: str, abandon_id: str | None = None) -> str:
    key = settings.owner_key or json.dumps((session_id, source), ensure_ascii=False, separators=(",", ":"))
    return key if abandon_id is None else json.dumps((key, "abandon", abandon_id), ensure_ascii=False, separators=(",", ":"))


# 后台进程不随回合放弃或回复结束清理；它只按会话来源分区，由 task_stop、超时或宿主停止回收。
def _background_owner(settings: ShellSettings, session_id: str, source: str) -> str:
    return json.dumps((_owner_key(settings, session_id, source), "background"), ensure_ascii=False, separators=(",", ":"))


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
        async with self._ctx.runtime_scope():
            return await self._ctx.require(PROCESSES).terminate_owner(self._ctx, owner_key)


class ShellTool:
    idempotent = False

    def __init__(self, ctx: Context, name: Literal["shell", "write_stdin", "task_stop"], settings: ShellSettings,
                 directories: WorkingDirectories | None = None):
        self._ctx = ctx
        self._name = name
        self._settings = settings
        self._directories = directories

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object] | str:
        """校验最终命令并固定进程 owner；恢复不重选目录、shell 或默认参数。"""
        owner = (
            self._settings.owner_key or "standalone" if source is None
            else _owner_key(
                self._settings, source.messages[-1].session_id, source.messages[-1].source,
                _abandon_before(source.messages, source.messages[-1].source, source.messages[-1].seq),
            )
        )
        background_owner = None
        background_source = None
        if source is not None and self._settings.owner_key is None:
            latest = source.messages[-1]
            with self._ctx.borrow(JOB_NOTICES) as notices:
                background_source = (latest.source if notices is None
                                     else notices.origin_source(latest.session_id, latest.source))
            background_owner = _background_owner(self._settings, latest.session_id, background_source)
        raw = json_value(arguments)
        try:
            if self._name == "task_stop":
                return PreparedStop(**Stop.model_validate(raw).model_dump(), owner_key=owner,
                                    background_owner_key=background_owner).model_dump()
            if self._name == "write_stdin":
                args = Stdin.model_validate(raw)
                default = 5000 if not args.chars else 250
                args.yield_time_ms = clamp_write_stdin_yield_time(
                    default if args.yield_time_ms is None else args.yield_time_ms, has_input=bool(args.chars),
                )
                return PreparedStdin(**args.model_dump(), owner_key=owner,
                                     background_owner_key=background_owner).model_dump()
            command = Command.model_validate(raw)
        except ValidationError as error:
            return str(error)
        text = command.command.strip()
        if not text:
            return '命令不能为空'
        try:
            shell = self._ctx.require(PROCESSES).resolve_shell(command.shell)
        except ValueError as error:
            return str(error)
        cwd = command.cwd or self._settings.working_dir or self._settings.restricted_dir
        if self._directories is not None:
            session_id = None if source is None else source.messages[-1].session_id
            if command.cwd is None and self._settings.working_dir is None and session_id is not None:
                current = self._directories.snapshot(session_id)
                if current.path is not None:
                    cwd = current.path
            try:
                cwd, _ = await self._directories.resolve_target(session_id, cwd or ".")
                _ = await self._directories.check_directory(cwd)
            except ValueError as error:
                return str(error)
        directory = None if cwd is None else Path(cwd).expanduser().absolute()
        denied = validate_command(
            text, allow_network=self._settings.allow_network,
            restricted_dir=None if self._settings.restricted_dir is None else Path(self._settings.restricted_dir),
            cwd=directory,
        )
        if denied:
            return denied
        background = None
        if command.run_in_background:
            if background_owner is None or background_source is None or source is None:
                return '当前调用上下文不支持后台运行'
            with self._ctx.borrow(JOB_NOTICES) as notices:
                if notices is None:
                    return '后台运行需要启用 jobs 插件'
            owner = background_owner
            background = BackgroundTarget(session_id=source.messages[-1].session_id, source=background_source)
        return PreparedCommand(
            background=background, owner_key=owner, command=text, description=command.description,
            argv=shell.derive_argv(text, login=command.login), shell_kind=shell.kind.value,
            login=command.login, cwd=None if directory is None else str(directory), tty=command.tty,
            yield_time_ms=clamp_initial_yield_time(command.yield_time_ms),
            max_output_tokens=command.max_output_tokens, timeout=command.timeout,
            call_context={} if source is None else {
                "session_id": source.messages[-1].session_id,
                "call_message_id": source.call_ref.message_id,
                "call_part_index": source.call_ref.part_index,
                "boot_id": os.environ.get("AKASHIC_BOOT_ID", ""),
            },
        ).model_dump()

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> ToolResultValue:
        """执行与续接只访问同一个物理进程 owner，失败不伪装为成功。"""
        raw = json_value(arguments)
        if self._name == "task_stop":
            return await self._stop(PreparedStop.model_validate(raw))
        if self._name == "write_stdin":
            return await self._continue(PreparedStdin.model_validate(raw))
        return await self._execute(key, PreparedCommand.model_validate(raw))

    # 回合分区找不到时再查后台分区；唤醒只继承本会话原来源的后台分区。
    async def _stop(self, stop: PreparedStop) -> ToolResultValue:
        processes = self._ctx.require(PROCESSES)
        stopped = await processes.terminate_execution(self._ctx, stop.owner_key, stop.execution_id)
        if not stopped and stop.background_owner_key is not None:
            stopped = await processes.terminate_execution(self._ctx, stop.background_owner_key, stop.execution_id)
        return ToolResultValue("success" if stopped else "error", (ContentPart("text", json.dumps({
            "execution_id": stop.execution_id, **({"process_status": "stopped"} if stopped else {}),
            "status": "stopped" if stopped else "not_found",
        })),))

    async def _continue(self, stdin: PreparedStdin) -> ToolResultValue:
        processes = self._ctx.require(PROCESSES)
        assert stdin.yield_time_ms is not None
        error: UnknownExecutionError | None = None
        for owner in (stdin.owner_key, stdin.background_owner_key):
            if owner is None:
                continue
            try:
                return _result(await processes.write_stdin(
                    self._ctx, owner, execution_id=stdin.execution_id, chars=stdin.chars,
                    yield_time_ms=stdin.yield_time_ms, max_output_tokens=stdin.max_output_tokens,
                ), None)
            except UnknownExecutionError as caught:
                error = caught
        assert error is not None
        return ToolResultValue("error", (ContentPart("text", str(error)),))

    async def _execute(self, key: str, command: PreparedCommand) -> ToolResultValue:
        processes = self._ctx.require(PROCESSES)
        log = partial(
            _log_shell_execution, operation_id=key, description=command.description,
            command_fp=hashlib.sha256(command.command.encode()).hexdigest()[:16],
            command_bytes=len(command.command.encode()), cwd=command.cwd or "",
            shell_kind=command.shell_kind, login=command.login, tty=command.tty, session=command.owner_key,
        )
        log("shell.execution_admitted")
        env = _shell_env()
        env["AKASHIC_CALL_CONTEXT"] = json.dumps(command.call_context)
        started_at_ms = time.time_ns() // 1_000_000
        result = await processes.exec_command(
            self._ctx, command.owner_key, command=command.command, argv=command.argv,
            cwd=None if command.cwd is None else Path(command.cwd), env=env, tty=command.tty,
            yield_time_ms=command.yield_time_ms, max_output_tokens=command.max_output_tokens,
            hard_timeout_s=command.timeout,
            shell_snapshot=command.login and command.shell_kind in {"bash", "zsh"},
        )
        log("shell.execution_result", result=result)
        value = _result(result, command.command)
        if command.background is None or result.execution_id is None:
            return value
        watched = self._watch(command, command.background, result.execution_id, started_at_ms)
        notice = (
            f"已转入后台（execution_id={result.execution_id}）。完成时会通知当前会话，不要轮询；"
            "可以继续处理其他事情，需要时用 write_stdin 读取输出、task_stop 终止。"
            if watched else "jobs 插件已停用，后台完成将不会通知；需要时用 write_stdin 读取输出。"
        )
        return ToolResultValue(value.outcome, (*value.parts, ContentPart("text", notice)))

    # 运行中的后台命令交给 jobs 等待结算；插件缺席时命令照常运行，只是没有完成通知。
    def _watch(self, command: PreparedCommand, target: BackgroundTarget, execution_id: int, started_at_ms: int) -> bool:
        with self._ctx.borrow(JOB_NOTICES) as notices:
            if notices is None:
                return False
            notices.watch(JobWatch(
                session_id=target.session_id, source=target.source, job_id=f"shell-{execution_id}",
                kind="shell", label=command.description or command.command[:60],
                collect_hint=f"需要结果时用 write_stdin(execution_id={execution_id}) 读取输出。",
                settled=partial(_settled, self._ctx, command.owner_key, execution_id),
                started_at_ms=started_at_ms,
                stop=partial(_stop_background, self._ctx, command.owner_key, execution_id),
            ))
        return True

    async def query(self, key: str) -> ToolResultValue | None:
        return None


def _result(result: ExecutionResult, command: str | None) -> ToolResultValue:
    outcome = "success" if result.execution_id is not None or result.exit_code == 0 else "error"
    return ToolResultValue(outcome, (ContentPart("text", format_execution_result(result, command=command)),))


def _outcome(state: ExecutionState) -> JobOutcome:
    detail = f"exit_code={state.exit_code}"
    if state.finish_reason == "timeout":
        return JobOutcome("timeout", detail)
    if state.finish_reason != "natural":
        return JobOutcome("killed", f"{detail}，{state.finish_reason}")
    return JobOutcome("completed" if state.exit_code == 0 else "failed", detail)


# 用户在界面上停止后台命令；与 task_stop 一样终止进程组，结算由 _settled 照常观察。
async def _stop_background(ctx: Context, owner_key: str, execution_id: int) -> bool:
    async with ctx.runtime_scope():
        return await ctx.require(PROCESSES).terminate_execution(ctx, owner_key, execution_id)


# 后台执行退出且无人等待时结算；条目消失说明终态已交给模型或执行已被停止。
async def _settled(ctx: Context, owner_key: str, execution_id: int) -> JobOutcome | None:
    while True:
        try:
            async with ctx.runtime_scope():
                states = await ctx.require(PROCESSES).execution_states(ctx, owner_key)
        except CompositionError:
            return None
        state = next((item for item in states if item.execution_id == execution_id), None)
        if state is None:
            return None
        if state.settled:
            return _outcome(state)
        await asyncio.sleep(_SETTLE_POLL_S)


async def register_shell(ctx: Context, directories: WorkingDirectories | None = None) -> tuple[ToolRef, ...]:
    """配置由 Shell owner 校验，所有操作与作业释放共用此插件身份。"""
    owners = ShellOwners(ctx)
    shell_tasks = ctx.require(TASKS).open(ctx)

    def cleanup_entry(
        reader: MessageReader,
        source: str,
        from_seq: int,
        *,
        task: Task | None = None,
        drain: Callable[[tuple[CallRef, ...]], Awaitable[None]] | None = None,
    ) -> AbstractAsyncContextManager[None]:
        return shell_cleanup(
            ctx, owners, shell_tasks, reader, source, from_seq, task=task, drain=drain,
        )

    _ = await ctx.provide(TOOL_CLEANUP, cleanup_entry)

    async def environment(_snapshot: tuple[Message, ...], _source: str) -> Mapping[str, object]:
        """请求准备时公开实际默认 shell；安装和归档装配不探测执行环境。"""
        shell = ctx.require(PROCESSES).resolve_shell()
        return {"reminders": ({"name": "shell-environment", "priority": 350, "replay": False,
                               "text": f"默认命令 shell：{shell.path}（{shell.kind.value}）。"
                               "未指定 shell 参数时使用此环境，命令须遵守该 shell 的语法；"
                               "需要 Bash 或其他 shell 时显式传入 shell 参数，不要假定默认是 Bash。"},)}

    _ = await ctx.require(MATERIALS_V4).register(
        ctx, name="shell-environment", kind="context", prepare=environment,
    )
    definitions: tuple[tuple[Literal["shell", "write_stdin", "task_stop"], type[BaseModel], str], ...] = (
        ("shell", Command, "执行 shell 命令；返回终态或可供 write_stdin/task_stop 使用的 execution_id。"
                           "run_in_background=true 时命令结束会回调通知本会话，任何要等待结果的工作都可借此异步完成。"),
        ("write_stdin", Stdin, "续接命令，等待新增输出或输入 PTY 字符；仅返回上次读取后的新增内容。"),
        ("task_stop", Stop, "确认终止命令的进程组并释放 execution_id。"),
    )
    refs: list[ToolRef] = []
    for name, schema, description in definitions:
        refs.append(await _register(ctx, name, schema, description, directories))
    return tuple(refs)


async def _register(
    ctx: Context,
    name: Literal["shell", "write_stdin", "task_stop"],
    schema: type[BaseModel],
    description: str,
    directories: WorkingDirectories | None,
) -> ToolRef:
    def capture(configuration: Mapping[str, object]) -> Mapping[str, object]:
        return ShellSettings.model_validate(json_value(configuration)).model_dump()

    @asynccontextmanager
    async def open_tool(state: Mapping[str, object]) -> AsyncGenerator[ShellTool]:
        yield ShellTool(ctx, name, ShellSettings.model_validate(json_value(state)), directories)

    return cast(ToolRef, await ctx.require(TOOLS).register(
        ctx,
        name=name,
        description=description,
        parameters=schema.model_json_schema(),
        open=open_tool,
        capture=capture,
    ))


@asynccontextmanager
async def shell_cleanup(
    ctx: Context, owners: ShellOwners, shell_tasks: TaskAdmission,
    reader: MessageReader, source: str, from_seq: int, *,
    task: Task | None = None, drain: Callable[[tuple[CallRef, ...]], Awaitable[None]] | None = None,
) -> AsyncGenerator[None]:
    """放弃可停止等待；独立 Task 保留旧进程清理、generation 与重启许可。"""
    async with ctx.runtime_scope():
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
                                # 2. 清理按 PluginProcesses 的稳定 owner key 进行；它不是外部效果重试，
                                # 不因插件换版跳过同一进程集合的终止。原 Shell Context 与 owner 已在入口固定。
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

                # 3. Task 自己固定当前 scope；源程序结束不会释放它的资源或 permit。
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
                    owned = await shell_tasks.admit(
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
