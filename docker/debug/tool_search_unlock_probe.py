#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.config_models import Config
from bootstrap.tools import build_core_runtime
from core.net.http import (
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
)


def _as_mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return cast(Mapping[str, object], value)


def _profile_dir(profile: str) -> Path:
    return Path(__file__).resolve().parent / "profiles" / profile


def _tool_names(tools: object) -> list[str]:
    if not isinstance(tools, list):
        return []
    names: list[str] = []
    for raw_tool in cast(list[object], tools):
        tool = _as_mapping(raw_tool)
        if tool is None:
            continue
        function = tool.get("function")
        function_map = _as_mapping(function)
        if function_map is None:
            continue
        name = function_map.get("name")
        if isinstance(name, str):
            names.append(name)
    return names


def _tool_chain_names(messages: list[dict[str, object]]) -> list[str]:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        chain = message.get("tool_chain")
        if not isinstance(chain, list):
            continue
        names: list[str] = []
        for raw_group in cast(list[object], chain):
            group = _as_mapping(raw_group)
            if group is None:
                continue
            calls = group.get("calls")
            if not isinstance(calls, list):
                continue
            for raw_call in cast(list[object], calls):
                call = _as_mapping(raw_call)
                if call is None:
                    continue
                name = call.get("name")
                if isinstance(name, str):
                    names.append(name)
        return names
    return []


async def _run(args: argparse.Namespace) -> None:
    profile_dir = _profile_dir(args.profile)
    config_path = args.config or profile_dir / "config.toml"
    workspace = args.workspace or profile_dir / "workspace"
    config = Config.load(config_path)
    if config.model != "mimo-v2.5":
        raise SystemExit(f"主模型必须是 mimo-v2.5，当前是 {config.model!r}")
    if config.light_model != "qwen-flash":
        raise SystemExit(f"轻量模型必须是 qwen-flash，当前是 {config.light_model!r}")

    _ = workspace.mkdir(parents=True, exist_ok=True)
    resources = SharedHttpResources()
    configure_default_shared_http_resources(resources)
    schema_calls: list[list[str]] = []
    llm_calls: list[dict[str, Any]] = []
    try:
        core = build_core_runtime(config, workspace, resources)
        provider = cast(Any, core.loop)._llm_services.provider
        original_chat = provider.chat

        async def capture_chat(*chat_args: Any, **chat_kwargs: Any):
            names = _tool_names(chat_kwargs.get("tools"))
            schema_calls.append(names)
            response = await original_chat(*chat_args, **chat_kwargs)
            prompt_tokens = response.cache_prompt_tokens
            hit_tokens = response.cache_hit_tokens
            llm_calls.append(
                {
                    "tools": names,
                    "cache_prompt_tokens": prompt_tokens,
                    "cache_hit_tokens": hit_tokens,
                    "cache_hit_rate": (
                        round(hit_tokens / prompt_tokens, 4)
                        if prompt_tokens and hit_tokens is not None
                        else None
                    ),
                }
            )
            return response

        provider.chat = capture_chat
        reply = await core.loop.process_direct(
            args.prompt,
            session_key=args.session_key,
            channel="cli",
            chat_id="tool-search-probe",
            skip_post_memory=True,
        )
        session = core.session_manager.get_or_create(args.session_key)
        tool_calls = _tool_chain_names(cast(list[dict[str, object]], session.messages))

        if tool_calls[:2] != ["tool_search", "list_schedules"]:
            raise SystemExit(f"工具调用顺序异常: {tool_calls}")
        first_schema = schema_calls[0] if schema_calls else []
        unlocked_schema = next(
            (names for names in schema_calls[1:] if "list_schedules" in names),
            list[str](),
        )
        if not first_schema or not unlocked_schema:
            raise SystemExit(f"未捕获到解锁前后 schema: {schema_calls}")
        if unlocked_schema[: len(first_schema)] != first_schema:
            raise SystemExit(
                "解锁后 schema 没有保留原前缀: "
                + json.dumps(schema_calls, ensure_ascii=False)
            )
        if unlocked_schema[len(first_schema)] != "list_schedules":
            raise SystemExit(
                "list_schedules 没有追加在原 schema 前缀之后: "
                + json.dumps(schema_calls, ensure_ascii=False)
            )

        print(json.dumps(
            {
                "ok": True,
                "model": config.model,
                "light_model": config.light_model,
                "tool_calls": tool_calls,
                "llm_calls": llm_calls,
                "reply": reply,
            },
            ensure_ascii=False,
            indent=2,
        ))
    finally:
        clear_default_shared_http_resources(resources)
        await resources.aclose()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="真实配置验证 tool_search 解锁顺序。")
    _ = parser.add_argument("--profile", default="scheduler-soft-real")
    _ = parser.add_argument("--config", type=Path)
    _ = parser.add_argument("--workspace", type=Path)
    _ = parser.add_argument("--session-key", default="probe:tool_search_unlock")
    _ = parser.add_argument(
        "--prompt",
        default=(
            "请严格按顺序执行：先调用 tool_search(query=\"select:list_schedules\") "
            "加载工具，再调用 list_schedules 查看当前定时任务，最后用一句中文概括结果。"
            "不要跳过工具。"
        ),
    )
    return parser.parse_args()


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
