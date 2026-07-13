from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path

from agent.config import Config
from agent.provider import LLMProvider, LLMResponse
from bootstrap.providers import build_providers


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "probe_add",
                "description": "Add two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "probe_echo",
                "description": "Echo text.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        },
    ]


async def _call(
    provider: LLMProvider,
    config: Config,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    tool_choice: str | dict = "auto",
    disable_thinking: bool = False,
    reasoning_effort: str | None = None,
    stream: bool = False,
) -> tuple[LLMResponse, list[str]]:
    """发送单次真实请求并记录 delta 类型。"""
    delta_types: list[str] = []

    async def on_delta(delta: dict[str, str]) -> None:
        delta_types.extend(delta)

    response = await provider.chat(
        messages=messages,
        tools=tools or [],
        model=config.model,
        max_tokens=min(config.max_tokens, 1024),
        tool_choice=tool_choice,
        extra_body=(
            {"reasoning_effort": reasoning_effort}
            if reasoning_effort is not None
            else None
        ),
        disable_thinking=disable_thinking,
        on_content_delta=on_delta if stream else None,
        cache_namespace="model-runtime-live-probe",
    )
    return response, delta_types


def _assistant_tool_message(response: LLMResponse) -> dict:
    message: dict[str, object] = {
        "role": "assistant",
        "content": response.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call in response.tool_calls
        ],
    }
    message.update(response.provider_fields)
    return message


def _usage(response: LLMResponse) -> dict[str, int | str | None]:
    usage = response.usage
    return {
        "input": usage.input_tokens if usage else None,
        "cached": usage.cached_input_tokens if usage else None,
        "output": usage.output_tokens if usage else None,
        "reasoning": usage.reasoning_output_tokens if usage else None,
        "coverage": usage.coverage.value if usage else None,
    }


async def run_probe(config_path: Path, expected_provider: str) -> dict[str, object]:
    """覆盖文本、推理、流式、工具、续接、缓存和图片请求。"""

    # 1. 从真实配置装配统一 provider。
    config = Config.load(config_path)
    if config.provider != expected_provider:
        raise ValueError(
            f"provider 不匹配: expected={expected_provider} actual={config.provider}"
        )
    provider, _, _ = build_providers(config)
    base_messages = [{"role": "user", "content": "Reply with exactly: probe-ok"}]

    # 2. 覆盖普通、流式、推理强度和关闭推理。
    plain, _ = await _call(provider, config, base_messages)
    streamed, delta_types = await _call(
        provider,
        config,
        base_messages,
        reasoning_effort="low",
        stream=True,
    )
    disabled, _ = await _call(
        provider,
        config,
        base_messages,
        disable_thinking=True,
    )

    # 3. 强制命名工具并回放 tool output，验证 continuation。
    tool_messages = [
        {
            "role": "user",
            "content": (
                "First call probe_add with a=2 and b=3. After receiving the tool result, "
                "reply with exactly: result=5"
            ),
        }
    ]
    tool_response, _ = await _call(
        provider,
        config,
        tool_messages,
        tools=_tools(),
        tool_choice={"type": "function", "function": {"name": "probe_add"}},
    )
    if len(tool_response.tool_calls) != 1:
        raise RuntimeError("命名 tool_choice 未返回单个工具调用")
    call = tool_response.tool_calls[0]
    continuation_messages = [
        *tool_messages,
        _assistant_tool_message(tool_response),
        {
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.name,
            "content": "5",
        },
    ]
    continued, _ = await _call(
        provider,
        config,
        continuation_messages,
        tools=_tools(),
        tool_choice="none",
    )

    # 4. 重复稳定前缀以读取厂商真实缓存 usage。
    cache_messages = [
        {
            "role": "user",
            "content": f"{'stable-cache-prefix ' * 800}\nReply with exactly: cache-ok",
        }
    ]
    cache_first, _ = await _call(provider, config, cache_messages)
    cache_second, _ = await _call(provider, config, cache_messages)

    # 5. 仅对声明支持图片的 runtime 发送一张 1x1 PNG。
    image_ok: bool | None = None
    if config.multimodal:
        png = base64.b64encode(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
            )
        ).decode()
        image_response, _ = await _call(
            provider,
            config,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in one word."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{png}"},
                        },
                    ],
                }
            ],
        )
        image_ok = bool(image_response.content)

    return {
        "provider": config.provider,
        "model": config.model,
        "plain": bool(plain.content),
        "stream": bool(streamed.content),
        "delta_types": sorted(set(delta_types)),
        "reasoning_visible": bool(
            streamed.thinking or tool_response.thinking or continued.thinking
        ),
        "disable_thinking_reply": bool(disabled.content),
        "tool": call.name,
        "tool_arguments": call.arguments,
        "continuation": bool(continued.content),
        "continuation_content": continued.content,
        "continuation_thinking": continued.thinking,
        "continuation_tools": [item.name for item in continued.tool_calls],
        "continuation_state": bool(tool_response.provider_fields.get("model_state")),
        "cache_first": _usage(cache_first),
        "cache_second": _usage(cache_second),
        "image": image_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(run_probe(args.config, args.provider)),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
