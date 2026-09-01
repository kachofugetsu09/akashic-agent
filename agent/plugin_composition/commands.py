from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

from agent.plugin_composition.context import Context
from agent.plugin_composition.diagnostics import plugin_entrypoint
from agent.plugin_composition.model import CompositionError, ServiceKey

CommandHandler = Callable[
    ["CommandInvocation"],
    "CommandResult | Awaitable[CommandResult]",
]
CommandResultKind = Literal["success", "error"]

_COMMAND_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_RESERVED_COMMAND_NAMES = frozenset({"stop"})
_MAX_COMMAND_DESCRIPTION_LENGTH = 256


@dataclass(frozen=True, slots=True)
class CommandInvocation:
    name: str
    raw_input: str
    session_key: str
    channel: str
    chat_id: str
    sender: str


@dataclass(frozen=True, slots=True)
class CommandResult:
    kind: CommandResultKind
    text: str


@dataclass(frozen=True, slots=True)
class CommandDefinition:
    name: str
    description: str
    handler: CommandHandler
    aliases: tuple[str, ...] = ()
    input_hint: str | None = None


@dataclass(frozen=True, slots=True)
class CommandDescriptor:
    name: str
    description: str
    aliases: tuple[str, ...]
    input_hint: str | None
    owner: str


@dataclass(frozen=True, slots=True)
class CommandExecution:
    name: str
    result: CommandResult


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


COMMANDS = ServiceKey["PluginCommands"]("core.commands")


class CommandRegistry:
    """Expose one immutable stable command catalog and execution seam."""

    def __init__(
        self,
        commands: Mapping[str, CommandDefinition],
        owners: Mapping[str, str],
        descriptors: tuple[CommandDescriptor, ...],
        generations: Mapping[str, str] | None = None,
        fibers: Mapping[str, str] | None = None,
    ) -> None:
        self._commands = MappingProxyType(dict(commands))
        self._owners = MappingProxyType(dict(owners))
        self._generations = MappingProxyType(dict(generations or {}))
        self._fibers = MappingProxyType(dict(fibers or {}))
        self._descriptors = descriptors
        payload = [
            {
                "name": item.name,
                "description": item.description,
                "aliases": list(item.aliases),
                "input_hint": item.input_hint,
                "owner": item.owner,
            }
            for item in descriptors
        ]
        self._catalog_digest = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @property
    def descriptors(self) -> tuple[CommandDescriptor, ...]:
        return self._descriptors

    @property
    def catalog_digest(self) -> str:
        return self._catalog_digest

    async def execute(
        self,
        line: str,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        sender: str,
    ) -> CommandExecution | None:
        """Execute a known slash command and leave misses to the normal turn."""

        # 1. An admission miss does not invoke a handler or create command state.
        parsed = _parse_command(line)
        if parsed is None:
            return None
        definition = self._commands.get(parsed.name)
        if definition is None:
            return None

        # 2. The plugin owns behavior; Core validates the settled public result.
        invocation = CommandInvocation(
            name=definition.name,
            raw_input=parsed.raw_input,
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            sender=sender,
        )
        generation_id = self._generations.get(parsed.name)
        if generation_id is None:
            result = definition.handler(invocation)
            if inspect.isawaitable(result):
                result = await result
            settled = _validate_result(definition.name, result)
        else:
            with plugin_entrypoint(
                plugin_id=self._owners[parsed.name],
                generation_id=generation_id,
                fiber=self._fibers[parsed.name],
                operation="command.call",
                entrypoint=definition.name,
            ):
                result = definition.handler(invocation)
                if inspect.isawaitable(result):
                    result = await result
                settled = _validate_result(definition.name, result)
        return CommandExecution(name=definition.name, result=settled)


def command_discovery_catalog(
    registry: CommandRegistry | None,
) -> tuple[tuple[str, str], ...]:
    """Project the universal channel discovery catalog from one v3 registry."""

    if registry is None:
        return ()
    return tuple(
        (descriptor.name, descriptor.description)
        for descriptor in registry.descriptors
    )


class PluginCommands:
    """Collect Fiber-owned human command definitions for one Root."""

    def __init__(self) -> None:
        self._next_token = 1
        self._registrations: dict[int, _RegisteredCommand] = {}
        self._names: dict[str, int] = {}
        self._frozen: CommandRegistry | None = None

    async def register(
        self,
        ctx: Context,
        definition: CommandDefinition,
    ) -> None:
        """Register a definition as an Effect of the calling Fiber."""

        normalized = _validate_definition(definition)
        _ = await ctx.effect(
            lambda: self._register(
                ctx.runtime.plugin_id,
                ctx.runtime.generation_id,
                ctx.fiber.name,
                normalized,
            ),
            label=f"command:{normalized.name}",
        )

    def freeze(self) -> CommandRegistry:
        """Freeze registrations into an immutable snapshot catalog."""

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
        for registration in ordered:
            definition = registration.definition
            for name in (definition.name, *definition.aliases):
                commands[name] = definition
                owners[name] = registration.plugin_id
                generations[name] = registration.generation_id
                fibers[name] = registration.fiber
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
        )
        return self._frozen

    def _register(
        self,
        plugin_id: str,
        generation_id: str,
        fiber: str,
        definition: CommandDefinition,
    ) -> Callable[[], None]:
        """Add one candidate definition and return its exact inverse."""

        # 1. Canonical names and compatibility aliases share one namespace.
        if self._frozen is not None:
            raise CompositionError(
                "PLUGIN_COMMANDS_FROZEN",
                "插件 Command 声明已冻结，不能在 snapshot 发布后新增",
            )
        claimed = (definition.name, *definition.aliases)
        duplicate = next((name for name in claimed if name in self._names), None)
        if duplicate is not None:
            raise CompositionError(
                "DUPLICATE_PLUGIN_COMMAND",
                f"插件 Command 名称重复: {duplicate}",
            )

        # 2. Cleanup removes only names owned by this registration.
        token = self._next_token
        self._next_token += 1
        self._registrations[token] = _RegisteredCommand(
            token=token,
            plugin_id=plugin_id,
            generation_id=generation_id,
            fiber=fiber,
            definition=definition,
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
    """Parse Akashic slash syntax while preserving raw arguments."""

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
    """Validate and detach one plugin-owned command definition."""

    # 1. Validate discovery metadata and the whole reserved namespace.
    if not isinstance(definition, CommandDefinition):
        raise TypeError("PluginCommands.register 只接受 CommandDefinition")
    if not isinstance(definition.name, str) or not _COMMAND_NAME.fullmatch(
        definition.name
    ):
        raise ValueError(f"Command name 无效: {definition.name}")
    if definition.name in _RESERVED_COMMAND_NAMES:
        raise CompositionError(
            "RESERVED_PLUGIN_COMMAND",
            f"Plugin Command 名称由 Core 保留: {definition.name}",
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
                f"Plugin Command 别名由 Core 保留: {alias}",
            )
    if definition.input_hint is not None:
        if not isinstance(definition.input_hint, str):
            raise TypeError(f"Command input_hint 必须是字符串: {definition.name}")
        if not definition.input_hint.strip():
            raise ValueError(f"Command input_hint 不能为空: {definition.name}")

    # 2. Copy tuple metadata before the candidate can publish.
    return CommandDefinition(
        name=definition.name,
        description=definition.description,
        handler=definition.handler,
        aliases=tuple(aliases),
        input_hint=definition.input_hint,
    )


def _validate_result(command: str, value: object) -> CommandResult:
    """Require handlers to return the stable command result contract."""

    if not isinstance(value, CommandResult):
        raise TypeError(f'Command "{command}" handler 必须返回 CommandResult')
    if value.kind not in {"success", "error"}:
        raise ValueError(f'Command "{command}" result kind 无效: {value.kind}')
    if not isinstance(value.text, str):
        raise TypeError(f'Command "{command}" result text 必须是字符串')
    if not value.text.strip():
        raise ValueError(f'Command "{command}" result text 不能为空')
    return cast(CommandResult, value)
