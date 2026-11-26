"""OperationRegistry：统一注册、正交扩展解析、动态自省与两阶段执行。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import re
from typing import cast

from agent.plugin_composition.context import Context
from agent.plugin_composition.effect import Effect
from plugins.ledger.contract import OwnerWrite

from .contract import (
    ActorKind,
    CliNodeHandle,
    ExtensionSpec,
    OperationRequest,
    OperationResult,
    OperationSpec,
    OptionSpec,
    cli_node_key,
)

_IDENT = re.compile(r"^[a-z][a-z0-9_-]{0,47}$")


@dataclass(frozen=True, slots=True)
class _RegisteredOp:
    owner: str
    spec: OperationSpec


@dataclass(frozen=True, slots=True)
class _RegisteredExt:
    owner: str
    spec: ExtensionSpec


def _coerce_value(opt: OptionSpec, raw: str) -> object:
    if opt.value_type == "json":
        try:
            return json.loads(raw)
        except json.JSONDecodeError as err:
            raise ValueError(f"选项 --{opt.name} 需要 JSON 值，收到 {raw!r}") from err
    if opt.value_type == "bool":
        if raw.lower() in ("true", "1", "yes"):
            return True
        if raw.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"选项 --{opt.name} 需要布尔值 (true/false)，收到 {raw!r}")
    if opt.value_type == "int":
        try:
            return int(raw)
        except ValueError as err:
            raise ValueError(f"选项 --{opt.name} 需要整数，收到 {raw!r}") from err
    if opt.choices and raw not in opt.choices:
        raise ValueError(f"选项 --{opt.name} 仅允许 [{', '.join(opt.choices)}]，收到 {raw!r}")
    return raw


def _parse_flags(tokens: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if not token.startswith("--") or len(token) <= 2:
            raise ValueError(f"无法识别的参数: {token!r}（请使用 --key value 格式）")
        body = token[2:]
        if "=" in body:
            key, val, idx = *body.split("=", 1), idx + 1
        elif idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
            key, val, idx = body, tokens[idx + 1], idx + 2
        else:
            key, val, idx = body, "true", idx + 1
        if key in parsed:
            raise ValueError(f"选项重复: --{key}")
        parsed[key] = val
    return parsed


def _opt_view(opt: OptionSpec, prefix: str = "") -> dict[str, object]:
    return {
        "flag": f"--{prefix}{opt.name}",
        "type": opt.value_type,
        "required": opt.required,
        "choices": list(opt.choices),
        "set_once": opt.set_once,
        "description": opt.description,
    }


class OperationRegistry:
    """进程内 Operation 总线；所有注册经 Context.effect 托管，卸载即自动清理。"""

    def __init__(self) -> None:
        self._ops: dict[str, _RegisteredOp] = {}
        self._by_path: dict[tuple[str, ...], str] = {}
        self._exts: list[_RegisteredExt] = []
        self._receipts: dict[str, OperationResult] = {}

    async def register(self, ctx: Context, spec: OperationSpec) -> Effect:
        if not spec.path or any(_IDENT.fullmatch(seg) is None for seg in spec.path):
            raise ValueError(f"无效的 CLI 路径: {spec.path!r}")
        if spec.operation_id in self._ops or spec.path in self._by_path:
            raise ValueError(f"重复注册 operation 或路径: {spec.operation_id}")
        entry = _RegisteredOp(owner=ctx.runtime.plugin_id, spec=spec)

        def attach() -> object:
            self._ops[spec.operation_id] = entry
            self._by_path[spec.path] = spec.operation_id

            def detach() -> None:
                if self._ops.get(spec.operation_id) is entry:
                    self._ops.pop(spec.operation_id, None)
                if self._by_path.get(spec.path) == spec.operation_id:
                    self._by_path.pop(spec.path, None)

            return detach

        effect = await ctx.effect(attach, label=f"operation:{spec.operation_id}")
        _ = await ctx.provide(
            cli_node_key(".".join(spec.path)),
            CliNodeHandle(operation_id=spec.operation_id, path=spec.path),
        )
        return effect

    async def extend(self, ctx: Context, spec: ExtensionSpec) -> Effect:
        if _IDENT.fullmatch(spec.namespace) is None:
            raise ValueError(f"无效的扩展命名空间: {spec.namespace!r}")
        if (spec.target_operation is None) == (spec.extension_point is None):
            raise ValueError("扩展必须且只能指定 target_operation 或 extension_point 之一")
        for existing in self._exts:
            if (
                existing.spec.target_operation == spec.target_operation
                and existing.spec.extension_point == spec.extension_point
                and existing.spec.namespace == spec.namespace
            ):
                raise ValueError(f"扩展命名空间冲突: {spec.namespace}")
        entry = _RegisteredExt(owner=ctx.runtime.plugin_id, spec=spec)

        def attach() -> object:
            self._exts.append(entry)
            return lambda: self._exts.remove(entry) if entry in self._exts else None

        target = spec.target_operation or f"point:{spec.extension_point}"
        return await ctx.effect(attach, label=f"operation-ext:{spec.namespace}->{target}")

    def _extensions_for(self, spec: OperationSpec) -> tuple[_RegisteredExt, ...]:
        points = set(spec.extension_points)
        return tuple(
            ext for ext in self._exts
            if ext.spec.target_operation == spec.operation_id
            or (ext.spec.extension_point is not None and ext.spec.extension_point in points)
        )

    def _visible(self, spec: OperationSpec, actor: ActorKind) -> bool:
        return actor == "user" or spec.audience != "user_only"

    def describe(self, path: tuple[str, ...] = (), *, actor: ActorKind = "agent") -> dict[str, object]:
        rows = []
        for op in sorted(self._ops.values(), key=lambda item: item.spec.path):
            if not self._visible(op.spec, actor) or (path and op.spec.path[: len(path)] != path):
                continue
            rows.append({
                "operation_id": op.spec.operation_id,
                "path": list(op.spec.path),
                "summary": op.spec.summary,
                "owner": op.owner,
                "options": [_opt_view(o) for o in op.spec.options],
                "extensions": [
                    {
                        "namespace": ext.spec.namespace,
                        "owner": ext.owner,
                        "options": [_opt_view(o, f"{ext.spec.namespace}.") for o in ext.spec.options],
                    }
                    for ext in self._extensions_for(op.spec)
                ],
            })
        return {"operations": rows}

    def render_help(self, path: tuple[str, ...] = (), *, actor: ActorKind = "agent") -> str:
        ops = cast(list[dict[str, object]], self.describe(path, actor=actor)["operations"])
        if not ops:
            return f"未找到匹配命令: {' '.join(path)}" if path else "当前没有可用命令。"
        lines = ["可用命令（使用 `help <group>` 或 `describe <cmd>` 查看详情）："]
        for item in ops:
            lines.append(f"  {' '.join(cast(list[str], item['path']))} — {item['summary']}")
            all_opts = list(cast(list[dict[str, object]], item["options"]))
            for ext in cast(list[dict[str, object]], item["extensions"]):
                all_opts.extend(cast(list[dict[str, object]], ext["options"]))
            for opt in all_opts:
                tag = " [set-once]" if opt["set_once"] else ""
                lines.append(f"    {opt['flag']} ({opt['type']}){tag}: {opt['description']}")
        return "\n".join(lines)

    def receipt(self, request_id: str) -> OperationResult | None:
        return self._receipts.get(request_id)

    async def invoke(
        self, argv: list[str], *, actor: ActorKind,
        session_id: str | None = None, request_id: str | None = None,
    ) -> OperationResult:
        if request_id and request_id in self._receipts:
            return self._receipts[request_id]
        if not argv or argv[0] in ("help", "--help", "-h"):
            return OperationResult(True, self.render_help(tuple(argv[1:]), actor=actor))
        if argv[0] == "describe":
            data = self.describe(tuple(argv[1:]), actor=actor)
            return OperationResult(True, json.dumps(data, ensure_ascii=False, indent=2), data=data)
        if argv[0] == "receipt" and len(argv) == 2:
            saved = self.receipt(argv[1])
            return saved or OperationResult(False, f"未找到回执: {argv[1]}", error_code="not_found")

        split = next((i for i, tok in enumerate(argv) if tok.startswith("--")), len(argv))
        path, flag_tokens = tuple(argv[:split]), argv[split:]
        if "--help" in flag_tokens:
            return OperationResult(True, self.render_help(path, actor=actor))
        op_id = self._by_path.get(path)
        if op_id is None:
            return OperationResult(False, f"未知命令: {' '.join(path)}", error_code="unknown_operation")
        spec = self._ops[op_id].spec
        if not self._visible(spec, actor):
            return OperationResult(False, f"命令仅限用户操作: {' '.join(path)}", error_code="forbidden")

        try:
            raw_flags = _parse_flags(flag_tokens)
            exts = self._extensions_for(spec)
            ext_map = {ext.spec.namespace: ext for ext in exts}
            main_opts = {o.name: o for o in spec.options}
            args: dict[str, object] = {}
            ext_args: dict[str, dict[str, object]] = {ns: {} for ns in ext_map}
            for key, raw_val in raw_flags.items():
                if "." in key:
                    ns, opt_name = key.split(".", 1)
                    if ns not in ext_map:
                        raise ValueError(f"未知扩展命名空间: --{key}")
                    ext_opts = {o.name: o for o in ext_map[ns].spec.options}
                    if opt_name not in ext_opts:
                        raise ValueError(f"扩展 {ns} 不支持选项: --{key}")
                    ext_args[ns][opt_name] = _coerce_value(ext_opts[opt_name], raw_val)
                else:
                    if key not in main_opts:
                        raise ValueError(f"未知选项: --{key}")
                    args[key] = _coerce_value(main_opts[key], raw_val)
            for opt in spec.options:
                if opt.name not in args:
                    if opt.required:
                        raise ValueError(f"缺少必填选项: --{opt.name}")
                    if opt.default is not None:
                        args[opt.name] = opt.default
            for ns, ext in ext_map.items():
                for opt in ext.spec.options:
                    if opt.name not in ext_args[ns]:
                        if opt.required:
                            raise ValueError(f"缺少必填扩展选项: --{ns}.{opt.name}")
                        if opt.default is not None:
                            ext_args[ns][opt.name] = opt.default
        except ValueError as err:
            return OperationResult(False, str(err), error_code="invalid_arguments")

        active = tuple((ext, ext_args[ext.spec.namespace]) for ext in exts if ext_args.get(ext.spec.namespace))

        # 主操作确定目标后，仅收集带值扩展的联合写入，由主 owner 一次提交。
        async def stage(target: Mapping[str, object]) -> tuple[OwnerWrite, ...]:
            writes: list[OwnerWrite] = []
            for ext, vals in active:
                if ext.spec.stage is not None:
                    write = await ext.spec.stage(req, vals, target)
                    if write is not None:
                        writes.append(write)
            return tuple(writes)

        req = OperationRequest(
            operation_id=spec.operation_id, actor=actor, session_id=session_id,
            request_id=request_id, args=args,
            extensions={ext.spec.namespace: vals for ext, vals in active}, stage=stage,
        )
        for ext, vals in active:
            if ext.spec.validate is None:
                continue
            try:
                await ext.spec.validate(req, vals)
            except Exception as err:
                return OperationResult(False, f"扩展 {ext.spec.namespace} 校验失败: {err}", error_code="validation_failed")
        try:
            result = await spec.handler(req)
        except (ValueError, LookupError, PermissionError) as err:
            return OperationResult(False, str(err) or type(err).__name__, error_code="rejected")
        if not result.ok:
            return result
        ext_outputs: dict[str, object] = {}
        for ext, vals in active:
            if ext.spec.apply is not None:
                applied = await ext.spec.apply(req, result, vals)
                if applied is not None:
                    ext_outputs[ext.spec.namespace] = applied
        final = (
            OperationResult(True, result.summary, data={**dict(result.data), "extensions": ext_outputs}, parts=result.parts)
            if ext_outputs else result
        )
        if request_id:
            self._receipts[request_id] = final
        return final
