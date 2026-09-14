"""命令注册、目录封存、执行和恢复由当前 provider 独占。"""
from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.context import Context
from agent.plugin_composition.diagnostics import plugin_entrypoint
from agent.plugin_composition.model import CompositionError
from agent.plugin_composition.commands import (
    COMMANDS, CommandDefinition, CommandDescriptor, CommandExecution,
    CommandInvocation, CommandRecoveryRequired, CommandResult,
)

_COMMAND_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_RESERVED_COMMAND_NAMES = frozenset({"stop"})
_MAX_COMMAND_DESCRIPTION_LENGTH = 256


@dataclass(frozen=True, slots=True)
class _ParsedCommand:
    name: str
    raw_input: str


@dataclass(frozen=True, slots=True)
class _RegisteredCommand:
    token: int
    plugin_id: str
    generation_id: str
    fiber: str
    definition: CommandDefinition
    context: Context


class CommandRegistry:
    """提供一个不可变命令目录及其执行入口。"""

    def __init__(
        self,
        commands: Mapping[str, CommandDefinition],
        owners: Mapping[str, str],
        descriptors: tuple[CommandDescriptor, ...],
        generations: Mapping[str, str],
        fibers: Mapping[str, str],
        contexts: Mapping[str, Context],
    ) -> None:
        self._commands = MappingProxyType(dict(commands))
        self._owners = MappingProxyType(dict(owners))
        self._generations = MappingProxyType(dict(generations))
        self._fibers = MappingProxyType(dict(fibers))
        self._contexts = MappingProxyType(dict(contexts))
        self._descriptors = descriptors

    @property
    def descriptors(self) -> tuple[CommandDescriptor, ...]:
        return self._descriptors

    def bind(self, bindings: Bindings, line: str) -> str | None:
        """匹配后固定真正的 handler owner；恢复不重新选择当前注册。"""
        parsed = _parse_command(line)
        if parsed is None or parsed.name not in self._commands:
            return None
        definition = self._commands[parsed.name]
        return bindings.bind(
            COMMANDS, {"name": definition.name},
            contributors=(self._contexts[parsed.name],),
        )

    async def execute(
        self,
        line: str,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        sender: str,
        message_id: str = "",
        recover: bool = False,
    ) -> CommandExecution | None:
        """执行已知斜杠命令，未命中时交回普通回复。"""

        # 1. 未命中不调用 handler，也不创建命令状态。
        parsed = _parse_command(line)
        if parsed is None:
            return None
        definition = self._commands.get(parsed.name)
        if definition is None:
            return None

        # 2. handler 拥有业务效果；命令 provider 校验公开结果。
        invocation = CommandInvocation(
            name=definition.name,
            raw_input=parsed.raw_input,
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            sender=sender,
            message_id=message_id,
        )
        handler = definition.handler
        if recover and not definition.read_only:
            if definition.recover is None:
                raise CommandRecoveryRequired(f"命令 {definition.name} 没有领域恢复回执；禁止自动重跑")
            handler = definition.recover
        with plugin_entrypoint(
            plugin_id=self._owners[parsed.name],
            generation_id=self._generations[parsed.name],
            fiber=self._fibers[parsed.name],
            operation="command.call",
            entrypoint=definition.name,
        ):
            result = handler(invocation)
            if inspect.isawaitable(result):
                result = await result
            settled = _validate_result(definition.name, result)
        return CommandExecution(name=definition.name, result=settled)


class PluginCommands:
    """收集同一 Root 中各贡献方的人类命令。"""

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx
        self._next_token = 1
        self._registrations: dict[int, _RegisteredCommand] = {}
        self._names: dict[str, int] = {}
        self._frozen: CommandRegistry | None = None

    async def register(
        self,
        ctx: Context,
        definition: CommandDefinition,
    ) -> None:
        """用调用方的 Effect 注册命令，退出时由同一 owner 清理。"""

        if ctx.root_instance_token is not self._ctx.root_instance_token or ctx.require(COMMANDS) is not self:
            raise ValueError("Command 注册必须使用同一 Root 实际选中的 provider")
        normalized = _validate_definition(definition)
        _ = await ctx.effect(
            lambda: self._register(
                ctx.runtime.plugin_id,
                ctx.runtime.generation_id,
                ctx.fiber.name,
                normalized,
                ctx,
            ),
            label=f"command:{normalized.name}",
        )

    def freeze(self) -> CommandRegistry:
        """将当前注册封存为不可变目录。"""

        if self._frozen is not None:
            return self._frozen
        ordered = sorted(
            self._registrations.values(),
            key=lambda item: item.token,
        )
        commands: dict[str, CommandDefinition] = {}
        owners: dict[str, str] = {}
        generations: dict[str, str] = {}
        fibers: dict[str, str] = {}
        contexts: dict[str, Context] = {}
        for registration in ordered:
            definition = registration.definition
            for name in (definition.name, *definition.aliases):
                commands[name] = definition
                owners[name] = registration.plugin_id
                generations[name] = registration.generation_id
                fibers[name] = registration.fiber
                contexts[name] = registration.context
        descriptors = tuple(
            sorted(
                (
                    CommandDescriptor(
                        name=item.definition.name,
                        description=item.definition.description,
                        aliases=item.definition.aliases,
                        input_hint=item.definition.input_hint,
                        owner=item.plugin_id,
                    )
                    for item in ordered
                ),
                key=lambda item: item.name,
            )
        )
        self._frozen = CommandRegistry(
            commands,
            owners,
            descriptors,
            generations,
            fibers,
            contexts,
        )
        return self._frozen

    def _register(
        self,
        plugin_id: str,
        generation_id: str,
        fiber: str,
        definition: CommandDefinition,
        context: Context,
    ) -> Callable[[], None]:
        """登记命令并返回只清理该注册的关闭操作。"""

        # 1. 命令名和别名共享同一命名空间。
        if self._frozen is not None:
            raise CompositionError(
                "PLUGIN_COMMANDS_FROZEN",
                "插件 Command 目录已封存，不能新增注册",
            )
        claimed = (definition.name, *definition.aliases)
        duplicate = next((name for name in claimed if name in self._names), None)
        if duplicate is not None:
            raise CompositionError(
                "DUPLICATE_PLUGIN_COMMAND",
                f"插件 Command 名称重复: {duplicate}",
            )

        # 2. 关闭只移除本次注册拥有的名字。
        token = self._next_token
        self._next_token += 1
        self._registrations[token] = _RegisteredCommand(
            token=token,
            plugin_id=plugin_id,
            generation_id=generation_id,
            fiber=fiber,
            definition=definition,
            context=context,
        )
        for name in claimed:
            self._names[name] = token

        def cleanup() -> None:
            _ = self._registrations.pop(token, None)
            for name in claimed:
                if self._names.get(name) == token:
                    _ = self._names.pop(name)

        return cleanup


def _parse_command(line: str) -> _ParsedCommand | None:
    """解析斜杠语法并保留原始参数。"""

    if not isinstance(line, str):
        raise TypeError("Command 输入必须是字符串")
    stripped = line.strip()
    if not stripped:
        return None
    match = re.match(r"^(\S+)(.*)$", stripped, re.DOTALL)
    assert match is not None
    head, raw_input = match.groups()
    if not head.startswith("/"):
        return None
    name = head[1:].split("@", 1)[0].lower()
    if not _COMMAND_NAME.fullmatch(name):
        return None
    return _ParsedCommand(name=name, raw_input=raw_input)


def _validate_definition(definition: CommandDefinition) -> CommandDefinition:
    """校验并复制插件贡献的命令定义。"""

    # 1. 校验目录元数据和保留名称。
    if not isinstance(definition, CommandDefinition):
        raise TypeError("PluginCommands.register 只接受 CommandDefinition")
    if not isinstance(definition.name, str) or not _COMMAND_NAME.fullmatch(
        definition.name
    ):
        raise ValueError(f"Command name 无效: {definition.name}")
    if definition.name in _RESERVED_COMMAND_NAMES:
        raise CompositionError(
            "RESERVED_PLUGIN_COMMAND",
            f"Plugin Command 名称由会话控制保留: {definition.name}",
        )
    if not isinstance(definition.description, str):
        raise TypeError(f"Command description 必须是字符串: {definition.name}")
    if not definition.description.strip():
        raise ValueError(f"Command description 不能为空: {definition.name}")
    if len(definition.description) > _MAX_COMMAND_DESCRIPTION_LENGTH:
        raise ValueError(
            f"Command description 超过 256 字符: {definition.name}"
        )
    if not callable(definition.handler):
        raise TypeError(f"Command handler 必须可调用: {definition.name}")
    if not isinstance(definition.aliases, tuple):
        raise TypeError(f"Command aliases 必须是 tuple: {definition.name}")
    aliases = definition.aliases
    if len(set(aliases)) != len(aliases) or definition.name in aliases:
        raise ValueError(f"Command aliases 重复: {definition.name}")
    for alias in aliases:
        if not isinstance(alias, str) or not _COMMAND_NAME.fullmatch(alias):
            raise ValueError(f"Command alias 无效: {alias}")
        if alias in _RESERVED_COMMAND_NAMES:
            raise CompositionError(
                "RESERVED_PLUGIN_COMMAND",
                f"Plugin Command 别名由会话控制保留: {alias}",
            )
    if definition.input_hint is not None:
        if not isinstance(definition.input_hint, str):
            raise TypeError(f"Command input_hint 必须是字符串: {definition.name}")
        if not definition.input_hint.strip():
            raise ValueError(f"Command input_hint 不能为空: {definition.name}")

    if type(definition.read_only) is not bool:
        raise TypeError("Command read_only 必须是 bool")
    if definition.recover is not None and not callable(definition.recover):
        raise TypeError("Command recover 必须是 callable")

    # 2. 发布前复制定义中的元数据。
    return CommandDefinition(
        name=definition.name,
        description=definition.description,
        handler=definition.handler,
        aliases=tuple(aliases),
        input_hint=definition.input_hint,
        read_only=definition.read_only,
        recover=definition.recover,
    )


def _validate_result(command: str, value: object) -> CommandResult:
    """要求 handler 返回符合公开合同的命令结果。"""

    if not isinstance(value, CommandResult):
        raise TypeError(f'Command "{command}" handler 必须返回 CommandResult')
    if value.kind not in {"success", "error"}:
        raise ValueError(f'Command "{command}" result kind 无效: {value.kind}')
    if not isinstance(value.text, str):
        raise TypeError(f'Command "{command}" result text 必须是字符串')
    if value.text and not value.text.strip() or not value.text and value.kind != "success":
        raise ValueError(f'Command "{command}" result text 不能为空')
    return cast(CommandResult, value)
