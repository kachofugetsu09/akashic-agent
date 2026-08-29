from __future__ import annotations

# Provider JSON is validated at this boundary. Pyright cannot retain all narrowing.
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, cast

import httpx

from agent.plugin_composition import (
    AuthenticationError,
    BoundModelDescriptor,
    ContentSafetyError,
    ContextLengthError,
    CredentialHandle,
    InvalidRequestError,
    LLMResponse,
    ModelContinuation,
    ModelRequest,
    ModelTimeoutError,
    ModelUsage,
    QuotaError,
    RateLimitError,
    ToolCall,
    TransportError,
    UsageCoverage,
)

from .auth import CODEX_CLIENT_VERSION, headers


class CodexResponses:
    """Map one exact model binding to the Codex Responses SSE protocol."""

    def __init__(
        self,
        *,
        endpoint: str,
        connect_timeout: float,
        read_timeout: float,
        credential: CredentialHandle,
        descriptor: BoundModelDescriptor,
        config: Mapping[str, Any],
    ) -> None:
        self._endpoint = endpoint
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._credential = credential
        self._descriptor = descriptor
        self._lite = bool(config.get("use_responses_lite", False))
        self._summary = str(config.get("reasoning_summary", "none"))
        raw_limit = config.get("max_tool_schemas")
        self._max_tool_schemas = cast(int | None, raw_limit)
        self._installation_id = str(uuid.uuid4())
        self._session_id = str(uuid.uuid4())
        self._thread_id = str(uuid.uuid4())
        self._window_id = str(uuid.uuid4())

    @property
    def max_tool_schemas(self) -> int | None:
        return self._max_tool_schemas

    async def complete(self, request: ModelRequest) -> LLMResponse:
        """Send one stream; refresh once after a rejected access token."""

        payload, previous_items = self._build_payload(request)
        rejected: str | None = None
        for attempt in range(2):
            token, auth_headers = await headers(
                self._credential,
                rejected_access_token=rejected,
            )
            request_headers = {
                **auth_headers,
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
                "originator": "codex_cli_rs",
                "User-Agent": f"codex_cli_rs/{CODEX_CLIENT_VERSION}",
                "x-codex-installation-id": self._installation_id,
                "session-id": self._session_id,
                "thread-id": self._thread_id,
                "x-codex-window-id": self._window_id,
            }
            if self._lite:
                request_headers["x-openai-internal-codex-responses-lite"] = "true"
            try:
                timeout = httpx.Timeout(
                    connect=self._connect_timeout,
                    read=self._read_timeout,
                    write=self._connect_timeout,
                    pool=self._connect_timeout,
                )
                async with httpx.AsyncClient(
                    base_url=self._endpoint,
                    timeout=timeout,
                    follow_redirects=False,
                    headers=request_headers,
                ) as client:
                    async with client.stream("POST", "/responses", json=payload) as response:
                        if response.status_code >= 400:
                            _ = await response.aread()
                        if response.status_code == 401 and attempt == 0:
                            rejected = token
                            continue
                        _raise_status(response, token)
                        return await _consume_stream(
                            response,
                            request,
                            self._descriptor.binding_id,
                            previous_items,
                        )
            except asyncio.CancelledError:
                raise
            except _CallbackError as exc:
                raise exc.error from exc
            except (httpx.TimeoutException, TimeoutError) as exc:
                error = ModelTimeoutError("Codex Responses 请求超时")
                if getattr(exc, "response_delta_seen", False):
                    setattr(error, "retryable", False)
                raise error from exc
            except httpx.TransportError as exc:
                error = TransportError("Codex Responses 连接失败")
                if getattr(exc, "response_delta_seen", False):
                    setattr(error, "retryable", False)
                raise error from exc
        raise AuthenticationError("Codex 请求认证失败，请重新登录")

    def _build_payload(
        self,
        request: ModelRequest,
    ) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
        previous_items = _continuation_items(request.continuation)
        messages, instructions = _responses_input(
            request.messages,
            request.system_prompt,
            previous_items,
        )
        tools = _responses_tools(request.tools)
        tool_choice, tools = _normalize_tool_choice(request.tool_choice, tools)
        if self._max_tool_schemas is not None and len(tools) > self._max_tool_schemas:
            raise InvalidRequestError(
                f"Codex model accepts at most {self._max_tool_schemas} tools"
            )
        if self._lite:
            messages = _responses_lite_input(messages, instructions, tools)
            instructions = ""
        payload: dict[str, Any] = {
            "model": self._descriptor.model,
            "instructions": instructions,
            "input": messages,
            "extra_body": {
                "client_metadata": {
                    "x-codex-installation-id": self._installation_id,
                    "session-id": self._session_id,
                    "thread-id": self._thread_id,
                    "x-codex-window-id": self._window_id,
                }
            },
            "tool_choice": tool_choice,
            "parallel_tool_calls": bool(
                self._descriptor.capabilities.supports_parallel_tool_calls
                and not self._lite
            ),
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
        }
        if request.max_output_tokens > 0:
            payload["max_output_tokens"] = request.max_output_tokens
        reasoning: dict[str, str] = {}
        if self._descriptor.reasoning_effort and not request.disable_reasoning:
            reasoning["effort"] = _normalize_effort(
                self._descriptor.reasoning_effort
            )
        if self._summary != "none" and not request.disable_reasoning:
            reasoning["summary"] = self._summary
        if self._lite and not request.disable_reasoning:
            reasoning["context"] = "all_turns"
        if reasoning:
            payload["reasoning"] = reasoning
        if tools and not self._lite:
            payload["tools"] = tools
        if request.prompt_cache_key or self._thread_id:
            payload["prompt_cache_key"] = request.prompt_cache_key or self._thread_id
        return payload, previous_items

    def estimate_context_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> int:
        return max(1, len(json.dumps([messages, tools], ensure_ascii=False)) // 4)

    def estimate_appended_message_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> int:
        return max(1, len(json.dumps(list(messages), ensure_ascii=False)) // 4)


class _CallbackError(RuntimeError):
    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


async def _consume_stream(
    response: httpx.Response,
    request: ModelRequest,
    binding_id: str,
    previous_items: tuple[dict[str, Any], ...],
) -> LLMResponse:
    content: list[str] = []
    thinking: list[str] = []
    tool_args: dict[str, dict[str, str]] = {}
    new_items: list[dict[str, Any]] = []
    usage: ModelUsage | None = None
    completed = False
    delta_seen = False
    try:
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            try:
                event: Any = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise TransportError("Codex Responses stream 包含无效 JSON") from exc
            if not isinstance(event, Mapping):
                raise TransportError("Codex Responses event 必须是对象")
            event_type = str(event.get("type") or "")
            delta = event.get("delta")
            if event_type == "response.output_text.delta" and isinstance(delta, str):
                content.append(delta)
                delta_seen = True
                await _emit(request.on_delta, {"content_delta": delta})
            elif event_type == "response.output_text.done":
                delta_seen = (
                    await _append_done(
                        content,
                        event.get("text"),
                        request.on_delta,
                        "content_delta",
                    )
                    or delta_seen
                )
            elif event_type in {
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.delta",
            } and isinstance(delta, str):
                thinking.append(delta)
                delta_seen = True
                await _emit(request.on_delta, {"thinking_delta": delta})
            elif event_type == "response.reasoning_summary_text.done":
                delta_seen = (
                    await _append_done(
                        thinking,
                        event.get("text"),
                        request.on_delta,
                        "thinking_delta",
                    )
                    or delta_seen
                )
            elif event_type == "response.function_call_arguments.delta":
                key = str(event.get("item_id") or event.get("output_index") or "")
                slot = tool_args.setdefault(key, {"arguments": ""})
                slot["arguments"] += str(delta or "")
                delta_seen = True
            elif event_type == "response.output_item.done":
                item = event.get("item")
                if not isinstance(item, Mapping):
                    raise TransportError("Codex output item 必须是对象")
                if item.get("type") == "reasoning":
                    new_items.append(_sanitize_replay_item(item))
                elif item.get("type") == "function_call":
                    key = str(item.get("id") or item.get("call_id") or "")
                    tool_args[key] = {
                        "id": str(item.get("call_id") or key),
                        "name": str(item.get("name") or ""),
                        "arguments": str(item.get("arguments") or "{}"),
                    }
                    delta_seen = True
            elif event_type == "response.completed":
                response_payload = event.get("response")
                usage = _parse_usage(
                    response_payload.get("usage")
                    if isinstance(response_payload, Mapping)
                    else None
                )
                completed = True
                break
            elif event_type in {"response.failed", "response.incomplete"}:
                response_payload = event.get("response")
                error = (
                    response_payload.get("error")
                    or response_payload.get("incomplete_details")
                    if isinstance(response_payload, Mapping)
                    else response_payload
                )
                _raise_stream_error(error)
    except asyncio.CancelledError:
        raise
    except _CallbackError:
        raise
    except Exception as exc:
        if delta_seen:
            setattr(exc, "response_delta_seen", True)
            if hasattr(exc, "retryable"):
                setattr(exc, "retryable", False)
        raise
    if not completed:
        error = TransportError("Codex Responses 在 completed 事件前断流")
        if delta_seen:
            setattr(error, "retryable", False)
        raise error
    try:
        calls = [_tool_call(item) for item in tool_args.values() if item.get("name")]
    except Exception as exc:
        if delta_seen and hasattr(exc, "retryable"):
            setattr(exc, "retryable", False)
        raise
    continuation_items = (*previous_items, *new_items)
    return LLMResponse(
        content="".join(content).strip() or None,
        tool_calls=calls,
        thinking="".join(thinking).strip() or None,
        continuation=ModelContinuation(
            binding_id=binding_id,
            payload={"format_version": 1, "items": continuation_items},
        ),
        usage=usage,
    )


async def _emit(
    callback: Callable[[dict[str, str]], Awaitable[None]] | None,
    delta: dict[str, str],
) -> None:
    if callback is None:
        return
    try:
        await callback(delta)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise _CallbackError(exc) from exc


async def _append_done(
    target: list[str],
    done: object,
    callback: Callable[[dict[str, str]], Awaitable[None]] | None,
    key: str,
) -> bool:
    current = "".join(target)
    if isinstance(done, str):
        if not done.startswith(current):
            raise TransportError("Codex Responses done text 与已接收 delta 冲突")
        suffix = done[len(current) :]
        if suffix:
            target.append(suffix)
            await _emit(callback, {key: suffix})
            return True
    return False


def _continuation_items(
    continuation: ModelContinuation | None,
) -> tuple[dict[str, Any], ...]:
    if continuation is None:
        return ()
    payload = continuation.payload
    if payload.get("format_version") != 1:
        raise InvalidRequestError("unsupported Codex continuation format")
    raw_items = payload.get("items")
    if not isinstance(raw_items, tuple):
        raise InvalidRequestError("Codex continuation items must be an array")
    return tuple(_sanitize_replay_item(item) for item in raw_items)


def _responses_input(
    messages: Sequence[Mapping[str, Any]],
    system_prompt: str,
    continuation_items: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    result = [dict(item) for item in continuation_items]
    instructions = system_prompt
    for message in messages:
        role = message.get("role")
        if role == "system":
            instructions = f"{instructions}\n\n{message.get('content', '')}".strip()
            continue
        if role == "tool":
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.get("tool_call_id") or ""),
                    "output": str(message.get("content") or ""),
                }
            )
            continue
        calls = message.get("tool_calls")
        if role == "assistant" and isinstance(calls, (list, tuple)):
            for call in calls:
                if not isinstance(call, Mapping):
                    raise InvalidRequestError("assistant tool call must be an object")
                function = call.get("function")
                if not isinstance(function, Mapping):
                    raise InvalidRequestError("assistant tool call misses function")
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


def _responses_content(role: object, content: object) -> object:
    if isinstance(content, str):
        return content
    if not isinstance(content, (list, tuple)):
        raise InvalidRequestError("message content must be a string or array")
    converted: list[dict[str, Any]] = []
    for raw in content:
        if not isinstance(raw, Mapping):
            raise InvalidRequestError("message content block must be an object")
        block_type = raw.get("type")
        if block_type in {"input_text", "output_text", "input_image"}:
            converted.append(dict(raw))
        elif block_type == "text":
            target = "output_text" if role == "assistant" else "input_text"
            converted.append({"type": target, "text": str(raw.get("text") or "")})
        elif block_type == "image_url" and role == "user":
            image = raw.get("image_url")
            url = image.get("url") if isinstance(image, Mapping) else image
            if not isinstance(url, str) or not url:
                raise InvalidRequestError("image_url block misses URL")
            item: dict[str, Any] = {"type": "input_image", "image_url": url}
            if isinstance(image, Mapping) and image.get("detail"):
                item["detail"] = image["detail"]
            converted.append(item)
        else:
            raise InvalidRequestError(f"unsupported Responses content block: {block_type}")
    return converted


def _responses_tools(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(function, Mapping) or not function.get("name"):
            raise InvalidRequestError("tool schema misses function name")
        result.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function.get("description", ""),
                "parameters": function.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
                "strict": bool(function.get("strict", False)),
            }
        )
    return result


def _normalize_tool_choice(
    choice: str | Mapping[str, Any],
    tools: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(choice, str):
        if choice not in {"auto", "none", "required"}:
            raise InvalidRequestError(f"unsupported tool_choice: {choice}")
        return choice, tools
    function = choice.get("function")
    name = function.get("name") if isinstance(function, Mapping) else choice.get("name")
    if choice.get("type") != "function" or not isinstance(name, str) or not name:
        raise InvalidRequestError("named tool_choice is invalid")
    selected = [tool for tool in tools if tool.get("name") == name]
    if not selected:
        raise InvalidRequestError(f"named tool_choice references unknown tool: {name}")
    return "required", selected


def _responses_lite_input(
    messages: list[dict[str, Any]],
    instructions: str,
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prefix: list[dict[str, Any]] = [
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
    copied = cast(list[dict[str, Any]], json.loads(json.dumps(messages)))
    for item in copied:
        content = item.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "input_image":
                    _ = block.pop("detail", None)
    return [*prefix, *copied]


def _sanitize_replay_item(item: object) -> dict[str, Any]:
    if not isinstance(item, Mapping) or item.get("type") != "reasoning":
        raise InvalidRequestError("Codex continuation only accepts reasoning items")
    allowed = {"type", "summary", "content", "encrypted_content"}
    return {key: _thaw(value) for key, value in item.items() if key in allowed}


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _tool_call(raw: Mapping[str, str]) -> ToolCall:
    try:
        arguments: Any = json.loads(raw.get("arguments") or "{}")
    except json.JSONDecodeError as exc:
        raise TransportError("Codex tool arguments are invalid JSON") from exc
    if not isinstance(arguments, Mapping):
        raise TransportError("Codex tool arguments must decode to an object")
    return ToolCall(
        id=raw.get("id", ""),
        name=raw["name"],
        arguments=dict(arguments),
    )


def _parse_usage(raw: object) -> ModelUsage | None:
    if not isinstance(raw, Mapping):
        return None
    input_tokens = _optional_int(raw.get("input_tokens"))
    output_tokens = _optional_int(raw.get("output_tokens"))
    input_details = raw.get("input_tokens_details")
    output_details = raw.get("output_tokens_details")
    cached = (
        _optional_int(input_details.get("cached_tokens"))
        if isinstance(input_details, Mapping)
        else None
    )
    cache_write = (
        _optional_int(input_details.get("cache_write_tokens"))
        if isinstance(input_details, Mapping)
        else None
    )
    reasoning = (
        _optional_int(output_details.get("reasoning_tokens"))
        if isinstance(output_details, Mapping)
        else None
    )
    covered = int(input_tokens is not None and output_tokens is not None)
    coverage = (
        UsageCoverage.EXACT
        if covered
        else UsageCoverage.PARTIAL
        if input_tokens is not None or output_tokens is not None
        else UsageCoverage.UNAVAILABLE
    )
    return ModelUsage(
        input_tokens=input_tokens,
        cache_write_input_tokens=cache_write,
        cached_input_tokens=cached,
        output_tokens=output_tokens,
        reasoning_output_tokens=reasoning,
        covered_request_count=covered,
        coverage=coverage,
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TransportError("Codex usage token count is invalid")
    return value


def _raise_status(response: httpx.Response, secret: str) -> None:
    if response.status_code < 400:
        return
    text = response.text.replace(secret, "[REDACTED]") if secret else response.text
    lowered = text.lower()
    if response.status_code in {401, 403}:
        raise AuthenticationError("Codex 请求认证失败，请重新登录")
    if "context_length" in lowered or "context window" in lowered:
        raise ContextLengthError("Codex 请求超过上下文窗口")
    if any(marker in lowered for marker in ("bio_policy", "cyber_policy", "policy_violation")):
        raise ContentSafetyError("Codex 请求被安全策略拒绝")
    if response.status_code == 402 or (
        response.status_code == 429
        and any(marker in lowered for marker in ("quota", "billing", "usage limit"))
    ):
        raise QuotaError("Codex 账号额度不足")
    if response.status_code == 429:
        raise RateLimitError("Codex 请求被限流")
    if 400 <= response.status_code < 500:
        raise InvalidRequestError(f"Codex 请求失败 (HTTP {response.status_code})")
    raise TransportError(f"Codex 服务失败 (HTTP {response.status_code})")


def _raise_stream_error(error: object) -> None:
    code = ""
    if isinstance(error, Mapping):
        code = str(error.get("code") or "").lower()
    if code in {"context_length_exceeded", "context_window_exceeded"}:
        raise ContextLengthError("Codex 请求超过上下文窗口")
    if code in {"insufficient_quota", "usage_not_included"}:
        raise QuotaError("Codex 账号额度不足")
    if code in {"rate_limit_exceeded", "rate_limit_error"}:
        raise RateLimitError("Codex 请求被限流")
    if code in {"invalid_prompt", "bio_policy", "cyber_policy", "policy_violation"}:
        raise ContentSafetyError(f"Codex 请求被安全策略拒绝: {code}")
    raise TransportError(f"Codex Responses 暂时失败: {code or 'unknown'}")


def _normalize_effort(value: str) -> str:
    return "max" if value == "ultra" else value
