from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agent.config import Config
from agent.provider import LLMProvider, LLMResponse
from bootstrap.providers import build_providers

_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
_TOOL = {
    "type": "function",
    "function": {
        "name": "probe_add",
        "description": "Add two integers.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
}


async def _call(
    provider: LLMProvider,
    config: Config,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    tool_choice: str | dict = "auto",
    disable_thinking: bool = False,
    effort: str | None = None,
    stream: bool = False,
) -> tuple[LLMResponse, set[str]]:
    """发送真实请求并记录流式 delta 类型。"""
    deltas: set[str] = set()

    async def on_delta(delta: dict[str, str]) -> None:
        deltas.update(delta)

    response = await provider.chat(
        messages=messages,
        tools=tools or [],
        model=config.model,
        max_tokens=min(config.max_tokens, 1024),
        tool_choice=tool_choice,
        extra_body={"reasoning_effort": effort} if effort else None,
        disable_thinking=disable_thinking,
        on_content_delta=on_delta if stream else None,
        cache_namespace="model-runtime-live-probe",
    )
    return response, deltas


def _assistant(response: LLMResponse) -> dict[str, object]:
    message: dict[str, object] = {
        "role": "assistant",
        "content": response.content or "",
        "tool_calls": [{
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
        } for call in response.tool_calls],
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


async def run_probe(
    config_path: Path,
    workspace: Path,
    expected_provider: str,
) -> dict[str, object]:
    """实测文本、推理、流式、工具续接、缓存和图片矩阵。"""

    # 1. 由真实配置构建统一 provider。
    config = Config.load(config_path, workspace=workspace)
    if config.provider != expected_provider:
        raise ValueError(
            f"provider 不匹配: expected={expected_provider} actual={config.provider}"
        )
    provider, _, _ = build_providers(config)
    prompt = [{"role": "user", "content": "Reply with exactly: probe-ok"}]

    # 2. 普通、流式和关闭推理共用同一调用边界。
    plain, _ = await _call(provider, config, prompt)
    streamed, delta_types = await _call(
        provider, config, prompt, effort="low", stream=True
    )
    disabled, _ = await _call(provider, config, prompt, disable_thinking=True)

    # 3. 强制命名工具并回放结果，验证 opaque continuation。
    tool_prompt = [{
        "role": "user",
        "content": "Call probe_add with a=2 and b=3, then reply exactly: result=5",
    }]
    tool_response, _ = await _call(
        provider,
        config,
        tool_prompt,
        tools=[_TOOL],
        tool_choice={"type": "function", "function": {"name": "probe_add"}},
    )
    if len(tool_response.tool_calls) != 1:
        raise RuntimeError("命名 tool_choice 未返回单个工具调用")
    call = tool_response.tool_calls[0]
    continued, _ = await _call(
        provider,
        config,
        [
            *tool_prompt,
            _assistant(tool_response),
            {"role": "tool", "tool_call_id": call.id, "content": "5"},
        ],
        tools=[_TOOL],
        tool_choice="none",
    )

    # 4. 重复长前缀读取真实缓存 usage；仅按能力声明发送图片。
    cache_prompt = [{
        "role": "user",
        "content": f"{'stable-cache-prefix ' * 800}\nReply exactly: cache-ok",
    }]
    cache_first, _ = await _call(provider, config, cache_prompt)
    cache_second, _ = await _call(provider, config, cache_prompt)
    image_ok: bool | None = None
    if config.multimodal:
        image, _ = await _call(provider, config, [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one word."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PNG}"}},
            ],
        }])
        image_ok = bool(image.content)

    return {
        "provider": config.provider,
        "model": config.model,
        "plain": bool(plain.content),
        "stream": bool(streamed.content),
        "delta_types": sorted(delta_types),
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
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    args = parser.parse_args()
    print(json.dumps(
        asyncio.run(run_probe(args.config, args.workspace, args.provider)),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
