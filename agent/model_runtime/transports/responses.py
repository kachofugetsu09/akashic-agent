from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, cast

import httpx
import openai
from openai import AsyncOpenAI

from agent.model_runtime.auth.codex import CODEX_API_BASE, CodexAuthDriver
from agent.model_runtime.errors import (
    AuthenticationError,
    ContextWindowError,
    RateLimitError,
    TransportError,
)
from agent.model_runtime.types import (
    ContinuationState,
    LLMResponse,
    ModelRequest,
    ModelUsage,
    ToolCall,
    UsageCoverage,
)


class CodexResponsesTransport:
    """把规范化模型请求映射到 Codex Responses 流。"""

    def __init__(
        self,
        auth: CodexAuthDriver,
        *,
        runtime_id: str,
        base_url: str = CODEX_API_BASE,
        connect_timeout_s: float = 30,
        read_timeout_s: float = 120,
        write_timeout_s: float = 30,
        pool_timeout_s: float = 30,
    ) -> None:
        self.auth = auth
        self.runtime_id = runtime_id
        self.base_url = base_url
        self.network_timeout = httpx.Timeout(
            connect=connect_timeout_s,
            read=read_timeout_s,
            write=write_timeout_s,
            pool=pool_timeout_s,
        )

    async def send(self, request: ModelRequest) -> LLMResponse:
        try:
            return await self._send_once(request, force_refresh=False)
        except AuthenticationError:
            return await self._send_once(request, force_refresh=True)

    async def _send_once(
        self, request: ModelRequest, *, force_refresh: bool
    ) -> LLMResponse:
        headers = await asyncio.to_thread(self.auth.headers, force_refresh=force_refresh)
        client = AsyncOpenAI(
            api_key=headers["Authorization"].removeprefix("Bearer "),
            base_url=self.base_url,
            default_headers={
                "ChatGPT-Account-ID": headers.get("ChatGPT-Account-ID", ""),
                "OpenAI-Beta": "responses=experimental",
            },
            timeout=self.network_timeout,
            max_retries=0,
        )
        try:
            stream = await client.responses.create(**self._build_payload(request))
            return await self._consume_stream(cast(Any, stream), request)
        except openai.APIStatusError as exc:
            status_code = exc.status_code
            error_text = str(exc).lower()
            if status_code == 401:
                raise AuthenticationError("Codex 请求认证失败") from exc
            if status_code == 429:
                raise RateLimitError("Codex 请求被限流") from exc
            if status_code == 400 and any(
                marker in error_text
                for marker in ("context_length", "context window", "too many tokens")
            ):
                raise ContextWindowError("Codex 请求超过上下文窗口") from exc
            raise
        except (openai.APIConnectionError, openai.APITimeoutError) as exc:
            raise TransportError("Codex Responses 连接失败") from exc
        finally:
            await client.close()

    def _build_payload(self, request: ModelRequest) -> dict[str, Any]:
        messages, instructions = _responses_input(
            request.messages,
            request.system_prompt,
            runtime_id=self.runtime_id,
            model=request.model,
        )
        payload: dict[str, Any] = {
            "model": request.model,
            "instructions": instructions,
            "input": messages,
            "max_output_tokens": request.max_output_tokens,
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
        }
        if request.reasoning_effort:
            payload["reasoning"] = {
                "effort": _normalize_effort(request.reasoning_effort),
                "summary": "auto",
            }
        tools = _responses_tools(request.tools)
        if tools:
            payload.update(tools=tools, tool_choice=request.tool_choice, parallel_tool_calls=True)
        cache_key = request.prompt_cache_key or _prompt_cache_key(instructions, tools)
        if cache_key:
            payload["prompt_cache_key"] = cache_key
        return payload

    async def _consume_stream(self, stream: Any, request: ModelRequest) -> LLMResponse:
        """消费 SSE 事件并保留后续重放必需的 output item。"""
        content: list[str] = []
        thinking: list[str] = []
        tool_args: dict[str, dict[str, str]] = {}
        output_items: list[dict[str, Any]] = []
        usage: ModelUsage | None = None
        iterator = aiter(stream)
        while True:
            try:
                event = await anext(iterator)
            except StopAsyncIteration:
                break
            event_type = str(_field(event, "type") or "")
            delta = _field(event, "delta")
            if event_type == "response.output_text.delta" and isinstance(delta, str):
                content.append(delta)
                if request.on_delta:
                    await request.on_delta({"content_delta": delta})
            elif event_type == "response.reasoning_summary_text.delta" and isinstance(delta, str):
                thinking.append(delta)
                if request.on_delta:
                    await request.on_delta({"thinking_delta": delta})
            elif event_type == "response.function_call_arguments.delta":
                item_id = str(_field(event, "item_id") or _field(event, "output_index") or "")
                slot = tool_args.setdefault(item_id, {"arguments": ""})
                slot["arguments"] += str(delta or "")
            elif event_type == "response.output_item.done":
                item = _dump(_field(event, "item"))
                if item:
                    if item.get("type") == "reasoning":
                        output_items.append(item)
                    if item.get("type") == "function_call":
                        item_id = str(item.get("id") or item.get("call_id") or "")
                        tool_args[item_id] = {
                            "id": str(item.get("call_id") or item_id),
                            "name": str(item.get("name") or ""),
                            "arguments": str(item.get("arguments") or "{}"),
                        }
            elif event_type == "response.completed":
                response = _field(event, "response")
                usage = _parse_usage(_field(response, "usage"))
            elif event_type in {"response.failed", "response.incomplete"}:
                response = _field(event, "response")
                error = _field(response, "error") or _field(response, "incomplete_details")
                raise TransportError(f"Codex Responses 未完成: {error}")
        calls = [_tool_call(value) for value in tool_args.values() if value.get("name")]
        continuation = ContinuationState(
            runtime_id=self.runtime_id,
            transport="responses",
            model=request.model,
            items=tuple(output_items),
        )
        return LLMResponse(
            content="".join(content).strip() or None,
            tool_calls=calls,
            thinking="".join(thinking).strip() or None,
            provider_fields={"model_state": continuation.to_dict()},
            cache_prompt_tokens=usage.input_tokens if usage else None,
            cache_hit_tokens=usage.cached_input_tokens if usage else None,
            usage=usage,
            continuation=continuation,
        )


def _responses_input(
    messages: list[dict],
    system_prompt: str,
    *,
    runtime_id: str = "",
    model: str = "",
) -> tuple[list[dict], str]:
    """转换 Chat 历史，并原样重放同 transport 的 opaque item。"""
    result: list[dict] = []
    instructions = system_prompt
    for message in messages:
        role = message.get("role")
        if role == "system":
            instructions = f"{instructions}\n\n{message.get('content', '')}".strip()
            continue
        state = message.get("model_state")
        if _matches_continuation(state, runtime_id=runtime_id, model=model):
            items = state.get("items")
            if isinstance(items, list):
                result.extend(items)
        if role == "tool":
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.get("tool_call_id") or ""),
                    "output": str(message.get("content") or ""),
                }
            )
            continue
        tool_calls = message.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list):
            for call in tool_calls:
                function = call.get("function") or {}
                result.append(
                    {
                        "type": "function_call",
                        "call_id": str(call.get("id") or ""),
                        "name": str(function.get("name") or ""),
                        "arguments": str(function.get("arguments") or "{}"),
                    }
                )
        content = message.get("content")
        if content not in (None, ""):
            result.append({"role": role, "content": _responses_content(role, content)})
    return result, instructions


def _matches_continuation(value: object, *, runtime_id: str, model: str) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        value.get("schema_version") == 1
        and value.get("runtime_id") == runtime_id
        and value.get("transport") == "responses"
        and value.get("model") == model
    )


def _responses_content(role: object, content: object) -> object:
    """按消息角色把 Chat content blocks 转为 Responses blocks。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TransportError("消息 content 必须是字符串或数组")
    converted: list[dict[str, Any]] = []
    for raw in content:
        if not isinstance(raw, dict):
            raise TransportError("消息 content block 必须是对象")
        block_type = raw.get("type")
        if block_type in {"input_text", "output_text", "input_image"}:
            converted.append(raw)
        elif block_type == "text":
            target = "output_text" if role == "assistant" else "input_text"
            converted.append({"type": target, "text": str(raw.get("text") or "")})
        elif block_type == "image_url" and role == "user":
            image = raw.get("image_url")
            image_url = image.get("url") if isinstance(image, dict) else image
            if not isinstance(image_url, str) or not image_url:
                raise TransportError("image_url block 缺少 URL")
            item: dict[str, Any] = {"type": "input_image", "image_url": image_url}
            if isinstance(image, dict) and image.get("detail"):
                item["detail"] = image["detail"]
            converted.append(item)
        else:
            raise TransportError(f"Responses 不支持的 content block: {block_type}")
    return converted


def _responses_tools(tools: list[dict]) -> list[dict]:
    result: list[dict] = []
    for tool in tools:
        function = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(function, dict) or not function.get("name"):
            raise TransportError("工具 schema 缺少函数名")
        result.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {"type": "object", "properties": {}}),
                "strict": bool(function.get("strict", False)),
            }
        )
    return result


def _prompt_cache_key(instructions: str, tools: list[dict]) -> str | None:
    if not instructions and not tools:
        return None
    static = instructions + "\0" + json.dumps(tools, sort_keys=True, ensure_ascii=False)
    return "pck_" + hashlib.sha256(static.encode()).hexdigest()[:24]


def _normalize_effort(value: str) -> str:
    return {"ultra": "max"}.get(value, value)


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _dump(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    dumped = value.model_dump(mode="json")
    return cast(dict[str, Any], dumped)


def _tool_call(raw: dict[str, str]) -> ToolCall:
    try:
        arguments = json.loads(raw.get("arguments") or "{}")
    except json.JSONDecodeError as exc:
        raise TransportError("Codex 工具调用参数不是有效 JSON") from exc
    if not isinstance(arguments, dict):
        raise TransportError("Codex 工具调用参数必须是 JSON 对象")
    return ToolCall(id=raw.get("id", ""), name=raw["name"], arguments=arguments)


def _parse_usage(raw: Any) -> ModelUsage | None:
    if raw is None:
        return None
    input_tokens = _field(raw, "input_tokens")
    output_tokens = _field(raw, "output_tokens")
    input_details = _field(raw, "input_tokens_details")
    output_details = _field(raw, "output_tokens_details")
    return ModelUsage(
        input_tokens=int(input_tokens) if input_tokens is not None else None,
        cached_input_tokens=_optional_int(_field(input_details, "cached_tokens")),
        output_tokens=int(output_tokens) if output_tokens is not None else None,
        reasoning_output_tokens=_optional_int(_field(output_details, "reasoning_tokens")),
        covered_request_count=1 if input_tokens is not None and output_tokens is not None else 0,
        coverage=(
            UsageCoverage.EXACT
            if input_tokens is not None and output_tokens is not None
            else UsageCoverage.PARTIAL
            if input_tokens is not None or output_tokens is not None
            else UsageCoverage.UNAVAILABLE
        ),
        raw_usage=_dump(raw),
    )


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None
