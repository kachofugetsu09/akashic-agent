from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, cast

import httpx
import openai
from openai import AsyncOpenAI

from agent.model_runtime.auth.codex import (
    CODEX_API_BASE,
    CODEX_CLIENT_VERSION,
    CodexAuthDriver,
)
from agent.model_runtime.errors import (
    AuthenticationError,
    ContextWindowError,
    QuotaError,
    RateLimitError,
    RetryableTransportError,
    TransportError,
)
from agent.model_runtime.types import (
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
        read_timeout_s: float = 120,
        use_responses_lite: bool = False,
        supports_parallel_tool_calls: bool = True,
        reasoning_summary: str = "none",
    ) -> None:
        self.auth = auth
        self.runtime_id = runtime_id
        self.base_url = base_url
        self.use_responses_lite = use_responses_lite
        self.supports_parallel_tool_calls = supports_parallel_tool_calls
        self.reasoning_summary = reasoning_summary
        self.installation_id = str(uuid.uuid4())
        self.session_id = str(uuid.uuid4())
        self.thread_id = str(uuid.uuid4())
        self.window_id = str(uuid.uuid4())
        self.network_timeout = httpx.Timeout(
            connect=30,
            read=read_timeout_s,
            write=30,
            pool=30,
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
        default_headers = {
            "ChatGPT-Account-ID": headers.get("ChatGPT-Account-ID", ""),
            "originator": "codex_cli_rs",
            "User-Agent": f"codex_cli_rs/{CODEX_CLIENT_VERSION}",
            "x-codex-installation-id": self.installation_id,
            "session-id": self.session_id,
            "thread-id": self.thread_id,
            "x-codex-window-id": self.window_id,
        }
        if self.use_responses_lite:
            default_headers["x-openai-internal-codex-responses-lite"] = "true"
        client = AsyncOpenAI(
            api_key=headers["Authorization"].removeprefix("Bearer "),
            base_url=self.base_url,
            default_headers=default_headers,
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
                if any(
                    marker in error_text
                    for marker in ("insufficient_quota", "quota exceeded", "billing")
                ):
                    raise QuotaError("Codex 账号额度不足") from exc
                raise RateLimitError("Codex 请求被限流") from exc
            if status_code == 400 and any(
                marker in error_text
                for marker in ("context_length", "context window", "too many tokens")
            ):
                raise ContextWindowError("Codex 请求超过上下文窗口") from exc
            raise
        except (openai.APIConnectionError, openai.APITimeoutError) as exc:
            raise RetryableTransportError("Codex Responses 连接失败") from exc
        finally:
            await client.close()

    def _build_payload(self, request: ModelRequest) -> dict[str, Any]:
        messages, instructions = _responses_input(
            request.messages,
            request.system_prompt,
            runtime_id=self.runtime_id,
            model=request.model,
        )
        tools = _responses_tools(request.tools)
        tool_choice, tools = _normalize_tool_choice(request.tool_choice, tools)
        if self.use_responses_lite:
            messages = _responses_lite_input(messages, instructions, tools)
            instructions = ""
        payload: dict[str, Any] = {
            "model": request.model,
            "instructions": instructions,
            "input": messages,
            "extra_body": {
                "client_metadata": {
                    "x-codex-installation-id": self.installation_id,
                    "session-id": self.session_id,
                    "thread-id": self.thread_id,
                    "x-codex-window-id": self.window_id,
                }
            },
            "tool_choice": tool_choice,
            "parallel_tool_calls": (
                self.supports_parallel_tool_calls and not self.use_responses_lite
            ),
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
        }
        reasoning: dict[str, str] = {}
        if request.reasoning_effort:
            reasoning["effort"] = _normalize_effort(request.reasoning_effort)
        if self.reasoning_summary != "none":
            reasoning["summary"] = self.reasoning_summary
        if self.use_responses_lite:
            reasoning["context"] = "all_turns"
        if reasoning:
            payload["reasoning"] = reasoning
        if tools and not self.use_responses_lite:
            payload["tools"] = tools
        cache_key = request.prompt_cache_key or self.thread_id
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
        completed = False
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
            elif event_type == "response.output_text.done":
                done_text = _field(event, "text")
                current = "".join(content)
                if isinstance(done_text, str) and done_text.startswith(current):
                    suffix = done_text[len(current) :]
                    if suffix:
                        content.append(suffix)
                        if request.on_delta:
                            await request.on_delta({"content_delta": suffix})
            elif event_type == "response.reasoning_summary_text.delta" and isinstance(delta, str):
                thinking.append(delta)
                if request.on_delta:
                    await request.on_delta({"thinking_delta": delta})
            elif event_type == "response.reasoning_text.delta" and isinstance(delta, str):
                thinking.append(delta)
                if request.on_delta:
                    await request.on_delta({"thinking_delta": delta})
            elif event_type == "response.reasoning_summary_text.done":
                done_text = _field(event, "text")
                current = "".join(thinking)
                if isinstance(done_text, str) and done_text.startswith(current):
                    suffix = done_text[len(current) :]
                    if suffix:
                        thinking.append(suffix)
                        if request.on_delta:
                            await request.on_delta({"thinking_delta": suffix})
            elif event_type == "response.function_call_arguments.delta":
                item_id = str(_field(event, "item_id") or _field(event, "output_index") or "")
                slot = tool_args.setdefault(item_id, {"arguments": ""})
                slot["arguments"] += str(delta or "")
            elif event_type == "response.output_item.done":
                item = _dump(_field(event, "item"))
                if item:
                    if item.get("type") == "reasoning":
                        output_items.append(_sanitize_replay_item(item))
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
                completed = True
                break
            elif event_type in {"response.failed", "response.incomplete"}:
                response = _field(event, "response")
                error = _field(response, "error") or _field(response, "incomplete_details")
                _raise_stream_error(error)
        if not completed:
            raise RetryableTransportError("Codex Responses 在 completed 事件前断流")
        calls = [_tool_call(value) for value in tool_args.values() if value.get("name")]
        model_state = {
            "schema_version": 1,
            "runtime_id": self.runtime_id,
            "transport": "responses",
            "model": request.model,
            "items": output_items,
        }
        return LLMResponse(
            content="".join(content).strip() or None,
            tool_calls=calls,
            thinking="".join(thinking).strip() or None,
            provider_fields={"model_state": model_state},
            cache_prompt_tokens=usage.input_tokens if usage else None,
            cache_hit_tokens=usage.cached_input_tokens if usage else None,
            usage=usage,
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
                result.extend(
                    _sanitize_replay_item(item)
                    for item in items
                    if isinstance(item, dict)
                )
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


def _normalize_tool_choice(
    tool_choice: str | dict[str, Any], tools: list[dict]
) -> tuple[str, list[dict]]:
    """把 Chat Completions 工具选择收敛为 Codex Responses 字符串契约。"""
    if isinstance(tool_choice, str):
        if tool_choice not in {"auto", "none", "required"}:
            raise TransportError(f"Responses 不支持的 tool_choice: {tool_choice}")
        return tool_choice, tools
    function = tool_choice.get("function")
    name = function.get("name") if isinstance(function, dict) else tool_choice.get("name")
    if tool_choice.get("type") != "function" or not isinstance(name, str) or not name:
        raise TransportError("Responses 命名 tool_choice 结构无效")
    selected = [tool for tool in tools if tool.get("name") == name]
    if not selected:
        raise TransportError(f"Responses 命名 tool_choice 引用了未知工具: {name}")
    return "required", selected


def _responses_lite_input(
    messages: list[dict], instructions: str, tools: list[dict]
) -> list[dict]:
    """按 Codex Responses Lite 契约内嵌工具和开发者指令。"""
    prefix: list[dict] = [
        {"type": "additional_tools", "role": "developer", "tools": tools}
    ]
    if instructions:
        prefix.append(
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": instructions}],
            }
        )
    return [*prefix, *_strip_image_details(messages)]


def _strip_image_details(items: list[dict]) -> list[dict]:
    """复制 Lite input，并移除后端不接受的图片 detail。"""
    copied = json.loads(json.dumps(items, ensure_ascii=False))
    for item in copied:
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "input_image":
                _ = block.pop("detail", None)
    return cast(list[dict], copied)


def _sanitize_replay_item(item: dict[str, Any]) -> dict[str, Any]:
    """只保留 reasoning 重放契约允许的字段。"""
    if item.get("type") != "reasoning":
        raise TransportError(f"Responses continuation 包含不支持的 item: {item.get('type')}")
    allowed = {"type", "summary", "content", "encrypted_content"}
    return {key: value for key, value in item.items() if key in allowed}


def _raise_stream_error(error: Any) -> None:
    code = str(_field(error, "code") or "").lower()
    message = str(_field(error, "message") or error or "未知错误")
    if code in {"context_length_exceeded", "context_window_exceeded"}:
        raise ContextWindowError(f"Codex 请求超过上下文窗口: {message}")
    if code in {"insufficient_quota", "usage_not_included"}:
        raise QuotaError(f"Codex 账号额度不足: {message}")
    if code in {"rate_limit_exceeded", "rate_limit_error"}:
        raise RateLimitError(f"Codex 请求受限: {message}")
    if code in {"invalid_prompt", "bio_policy", "cyber_policy", "policy_violation"}:
        raise TransportError(f"Codex Responses 请求被拒绝 code={code}: {message}")
    raise RetryableTransportError(
        f"Codex Responses 暂时失败 code={code or '-'}: {message}"
    )


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
    )


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None
