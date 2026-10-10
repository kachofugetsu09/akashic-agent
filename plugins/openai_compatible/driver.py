from __future__ import annotations

from core.common.diagnostic_log import log_timing

# External JSON is validated field by field below; pyright cannot preserve the
# narrowed key/value types of arbitrary Mapping and list payloads.
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

import asyncio
import json
import zlib
import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

import httpx

from core.net.http import HttpClient, StreamProgress, describe_transport_error, finish_response, retry_after_time

from agent.plugin_contracts import freeze_json

from agent.plugin_composition import (
    BoundModelDescriptor,
    CapabilitySources,
    CredentialHandle,
    DiscoveredModel,
    DriverConnection,
    DriverConnectionDescriptor,
    EmbeddingResult,
    EmbeddingSpaceDescriptor,
    LLMResponse,
    ModelCapabilities,
    ModelRequest,
    ModelUsage,
    ToolCall,
)
from plugins.models.contract import (
    AuthenticationError,
    ContentSafetyError,
    ContextLengthError,
    InvalidRequestError,
    ModelError,
    ModelTimeoutError,
    QuotaError,
    RateLimitError,
    TransportError,
)
from plugins.models.contract import ModelDriverDefinition

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


_DRIVER_ID = "openai-compatible"
_CONTRACT_VERSION = "1"
_DISCOVERY_CONNECT_TIMEOUT_SECONDS = 5.0
_DISCOVERY_READ_TIMEOUT_SECONDS = 10.0
_DISCOVERY_TOTAL_TIMEOUT_SECONDS = 15.0
_DISCOVERY_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_DISCOVERY_MAX_MODELS = 10_000
_SAFETY_CODES = (
    "content_filter",
    "content_policy_violation",
    "data_inspection_failed",
)
_CONTEXT_CODES = (
    "context_length_exceeded",
    "maximum context length",
    "context window exceeds limit",
    "range of input length",
    "reduce the length",
    "string too long",
    "too many tokens",
)


@dataclass(frozen=True, slots=True)
class _ConnectionConfig:
    base_url: str
    connect_timeout: float
    read_timeout: float
    max_retries: int
    allow_unverified_manual: bool
    thinking_format: str = "none"
    progress_timeout: float = 300.0


@dataclass(frozen=True, slots=True)
class _ModelConfig:
    max_tool_schemas: int | None
    embedding_batch_size: int


@dataclass(frozen=True, slots=True)
class _ChatRow:
    source: Mapping[str, Any]
    fill_reasoning: bool
    encoded: bytes
    images: tuple[bytes, ...]


class _BoundChat:
    def __init__(
        self,
        connection: _ConnectionConfig,
        credential: CredentialHandle,
        descriptor: BoundModelDescriptor,
        config: _ModelConfig,
        http: HttpClient,
    ) -> None:
        self._connection = connection
        self._credential = credential
        self._descriptor = descriptor
        self._config = config
        self._http = http
        self._message_sizes: dict[int, tuple[Mapping[str, Any], int, int]] = {}
        self._tool_size: tuple[object, int] | None = None
        # 原冻结行只编译一次；图片发射位置仍由本轮工具配对决定。
        self._chat_rows: dict[int, _ChatRow] = {}
        self._tools_json: tuple[object, bytes] | None = None

    @property
    def max_tool_schemas(self) -> int | None:
        return self._config.max_tool_schemas

    async def complete(self, request: ModelRequest) -> LLMResponse:
        """Send one exact bound model request through Chat Completions."""

        if request.continuation is not None:
            # 发送前本地校验失败：可证明请求未发出。
            raise _unsent(InvalidRequestError(
                "OpenAI-compatible Chat Completions does not support continuation state"
            ).exception())
        # 生成调用恒为一次物理 attempt：重试预算唯一 owner 是 Models；
        # 未带 request_key 的直调同样不得隐式重发（§6.3）。max_retries
        # 连接配置只留给 embeddings/discovery 等非生成路径。
        connection = replace(self._connection, max_retries=0)
        body = _chat_body(
            self._descriptor, connection, request,
        )
        if request.on_delta is None and connection.thinking_format != "deepseek":
            payload = await _request_json(
                connection,
                self._credential,
                "POST",
                "/chat/completions",
                body_bytes=self._encode_body(body, request),
                http=self._http,
            )
            return _parse_chat_response(payload)
        # V4 长生成可能超过网关非流式等待窗口；无观察者时也完整聚合 SSE。
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        return await _stream_chat(
            connection,
            self._credential,
            self._encode_body(body, request),
            request.on_delta,
            self._http,
        )

    def _encode_body(self, body: Mapping[str, Any], request: ModelRequest) -> bytes:
        """逐字段复用身份缓存字节；与 httpx encode_json 的字节格式一致。"""
        frags: list[bytes] = []
        for key, value in body.items():
            if key == "messages":
                encoded = self._encode_rows(
                    value, request.system_prompt,
                    self._connection.thinking_format == "deepseek" and not request.disable_reasoning,
                )
            elif key == "tools":
                encoded = self._encode_tools(value)
            else:
                encoded = _json_dumps_bytes(value)
            frags.append(b'"' + key.encode("utf-8") + b'":' + encoded)
        return b"{" + b",".join(frags) + b"}"

    def _encode_rows(
        self, messages: Sequence[Mapping[str, Any]], system_prompt: str,
        fill_reasoning: bool,
    ) -> bytes:
        """复用逐行协议字节，并在工具组闭合处发射图片。"""
        # 1. 连续 system 只合并非空文本；全空且没有后续行时保留原行。
        parts: list[bytes] = []
        index = 0
        system_contents: list[str] = []
        while index < len(messages) and messages[index].get("role") == "system":
            content = messages[index].get("content")
            if isinstance(content, str) and content:
                system_contents.append(content)
            index += 1
        if index == 0 and system_prompt:
            system_contents.append(system_prompt)
        if system_contents:
            parts.append(_json_dumps_bytes({"role": "system", "content": "\n\n".join(system_contents)}))
        elif index == len(messages):
            index = 0

        # 2. 缓存只拥有单行变换；跨行配对状态每次从当前窗口重建。
        cache: dict[int, _ChatRow] = {}
        images: list[bytes] = []
        pending_calls: set[str] = set()
        for row in messages[index:]:
            identity = id(row)
            entry = self._chat_rows.get(identity)
            if entry is None or entry.source is not row or entry.fill_reasoning != fill_reasoning:
                entry = _compile_chat_row(row, fill_reasoning)
            cache[identity] = entry
            images.extend(entry.images)
            role = str(row.get("role") or "")
            if role == "assistant" and row.get("tool_calls"):
                if images and pending_calls:
                    raise InvalidRequestError("图片所在工具请求尚未完成配对").exception()
                pending_calls = {call["id"] for call in row["tool_calls"]}
            elif role == "tool":
                pending_calls.discard(row["tool_call_id"])
            parts.append(entry.encoded)
            if images and not pending_calls:
                parts.append(b'{"role":"user","content":[' + b",".join(images) + b"]}")
                images = []
        if images:
            raise InvalidRequestError("图片所在工具请求缺少结果，不能重排为有效历史").exception()
        self._chat_rows = cache
        return b"[" + b",".join(parts) + b"]"

    def _encode_tools(self, tools: Sequence[Mapping[str, Any]]) -> bytes:
        saved = self._tools_json
        if saved is not None and saved[0] is tools:
            return saved[1]
        encoded = _json_dumps_bytes(tools)
        self._tools_json = (tools, encoded)
        return encoded

    def estimate_context_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> int:
        """固定 schema 只编码一次；消息仍按原公式合计后取整。"""
        frozen_tools = freeze_json(tools)
        saved = self._tool_size
        if saved is not None and saved[0] is frozen_tools:
            chars = saved[1]
        else:
            chars = len(json.dumps(frozen_tools, ensure_ascii=False, separators=(",", ":"), default=dict))
            self._tool_size = (frozen_tools, chars)
        return max(1, chars // 3 + self.estimate_appended_message_tokens(messages))

    def estimate_appended_message_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> int:
        """仅复用同一个深冻结消息的长度；动态内容和普通可变输入重新计算。"""
        sizes: dict[int, tuple[Mapping[str, Any], int, int]] = {}
        text_chars = image_tokens = 0
        for message in messages:
            frozen = cast(Mapping[str, Any], freeze_json(message))
            saved = self._message_sizes.get(id(frozen))
            if saved is None or saved[0] is not frozen:
                chars, images = _message_size(frozen)
                saved = (frozen, chars, images)
            sizes[id(frozen)] = saved
            text_chars += saved[1]
            image_tokens += saved[2]
        # 只保留本次输入，切窗或结束 scope 后不保留被排除的历史。
        self._message_sizes = sizes
        return max(1, text_chars // 3 + image_tokens) if messages else 0


class _BoundEmbedding:
    def __init__(
        self,
        connection: _ConnectionConfig,
        credential: CredentialHandle,
        descriptor: EmbeddingSpaceDescriptor,
        config: _ModelConfig,
        http: HttpClient,
    ) -> None:
        self._connection = connection
        self._credential = credential
        self._descriptor = descriptor
        self._config = config
        self._http = http

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed a non-empty text batch and preserve response ordering."""

        if not texts or any(not isinstance(text, str) for text in texts):
            raise ValueError("embedding texts must be a non-empty string sequence")
        vectors: list[tuple[float, ...]] = []
        usages: list[ModelUsage | None] = []
        for start in range(0, len(texts), self._config.embedding_batch_size):
            batch = texts[start : start + self._config.embedding_batch_size]
            payload = await _request_json(
                self._connection,
                self._credential,
                "POST",
                "/embeddings",
                body={"model": self._descriptor.model, "input": list(batch)},
                http=self._http,
            )
            result = _parse_embedding_response(payload, expected_count=len(batch))
            vectors.extend(result.vectors)
            usages.append(result.usage)
        return EmbeddingResult(vectors=tuple(vectors), usage=_merge_usage(usages))


def definition() -> ModelDriverDefinition:
    """Build this artifact's immutable driver contribution."""

    return ModelDriverDefinition(
        driver_id=_DRIVER_ID,
        contract_version=_CONTRACT_VERSION,
        open=_open,
        discover=_discover,
        probe=_probe,
        probe_embedding=_probe_embedding,
    )


async def _open(
    descriptor: DriverConnectionDescriptor,
    credential: CredentialHandle,
) -> DriverConnection:
    connection = _connection_config(descriptor)
    if credential.connection_id != descriptor.connection_id:
        raise AuthenticationError("credential connection scope does not match").exception()
    if credential.auth_identity != descriptor.auth_identity:
        raise AuthenticationError("credential auth identity does not match").exception()

    http = HttpClient(lambda: _client(connection))

    def bind_chat(
        model: BoundModelDescriptor,
        raw_config: Mapping[str, Any],
    ) -> _BoundChat:
        _check_bound_model(descriptor, model.connection_id, model.driver_id)
        return _BoundChat(
            connection,
            credential,
            model,
            _model_config(raw_config),
            http,
        )

    def bind_embedding(
        model: EmbeddingSpaceDescriptor,
        raw_config: Mapping[str, Any],
    ) -> _BoundEmbedding:
        _check_bound_model(descriptor, model.connection_id, model.driver_id)
        return _BoundEmbedding(
            connection,
            credential,
            model,
            _model_config(raw_config),
            http,
        )

    return DriverConnection(bind_chat=bind_chat, bind_embedding=bind_embedding, close=http.aclose)


async def _probe_embedding(
    descriptor: DriverConnectionDescriptor, credential: CredentialHandle, model: str,
) -> DiscoveredModel:
    """用两段固定文本测量实际维度，不创建虚假的绑定空间。"""
    _check_credential_scope(descriptor, credential)
    connection = replace(_connection_config(descriptor), max_retries=0)
    try:
        async with asyncio.timeout(30):
            payload = await _request_limited_json(
                connection, credential, "POST", "/embeddings",
                body={"model": model, "input": ["Akashic embedding setup check", "Akashic embedding order check"]},
                max_bytes=4 * 1024 * 1024,
            )
    except TimeoutError as error:
        raise ModelTimeoutError("向量试算超时，请检查服务地址或稍后重试；配置未保存。").exception() from error
    result = _parse_embedding_response(payload, expected_count=2)
    return DiscoveredModel(
        kind='embedding', model=model,
        capabilities=ModelCapabilities(embedding_dimensions=len(result.vectors[0]), embedding_normalization="none"),
        capability_sources=CapabilitySources(embedding_dimensions="probe", embedding_normalization="driver"),
        driver_config={"format_version": 1},
    )


async def _probe(
    descriptor: DriverConnectionDescriptor,
    credential: CredentialHandle,
) -> None:
    connection = _connection_config(descriptor)
    _check_credential_scope(descriptor, credential)
    token = _credential_token(await credential.read())
    try:
        async with _client(connection, token) as client:
            response = await client.get("/models")
    except asyncio.CancelledError:
        raise
    except Exception as error:
        mapped = _map_error(error)
        if mapped is error and not ModelError.matches(error):
            raise
        raise mapped from error
    if response.status_code == 404 and connection.allow_unverified_manual:
        return
    _raise_status(response, secret=token)
    _ = _json_object(response)


async def _discover(
    descriptor: DriverConnectionDescriptor,
    credential: CredentialHandle,
) -> tuple[DiscoveredModel, ...]:
    connection = _connection_config(descriptor)
    _check_credential_scope(descriptor, credential)
    discovery_connection = _ConnectionConfig(
        base_url=connection.base_url,
        connect_timeout=_DISCOVERY_CONNECT_TIMEOUT_SECONDS,
        read_timeout=_DISCOVERY_READ_TIMEOUT_SECONDS,
        max_retries=0,
        allow_unverified_manual=connection.allow_unverified_manual,
    )
    try:
        async with asyncio.timeout(_DISCOVERY_TOTAL_TIMEOUT_SECONDS):
            payload = await _request_limited_json(
                discovery_connection,
                credential,
                "GET",
                "/models",
                max_bytes=_DISCOVERY_MAX_RESPONSE_BYTES,
            )
    except asyncio.CancelledError:
        raise
    except TimeoutError as error:
        raise ModelTimeoutError("model discovery timed out").exception() from error
    raw_models = payload.get("data")
    if not isinstance(raw_models, list):
        raise TransportError("models response is missing data array").exception()
    if len(raw_models) > _DISCOVERY_MAX_MODELS:
        raise TransportError(f"models response exceeds {_DISCOVERY_MAX_MODELS} entries").exception()
    result: list[DiscoveredModel] = []
    seen_models: set[str] = set()
    for raw in raw_models:
        if not isinstance(raw, Mapping):
            raise TransportError("models response contains a non-object item").exception()
        model = raw.get("id")
        if not isinstance(model, str) or not model.strip():
            raise TransportError("models response contains an invalid id").exception()
        if model != model.strip():
            raise TransportError("models response contains an id with outer whitespace").exception()
        if len(model) > 256:
            raise TransportError(
                "models response contains an id longer than 256 characters"
            ).exception()
        if model in seen_models:
            raise TransportError(f"models response contains duplicate id: {model}").exception()
        seen_models.add(model)
        result.append(
            DiscoveredModel(
                kind=None,
                model=model,
                default_reasoning_effort=None,
                capabilities=ModelCapabilities(),
                capability_sources=CapabilitySources(),
            )
        )
    return tuple(result)


def _connection_config(descriptor: DriverConnectionDescriptor) -> _ConnectionConfig:
    if descriptor.driver_id != _DRIVER_ID:
        raise ValueError(f"unexpected driver id: {descriptor.driver_id}")
    config = descriptor.config
    allowed = {
        "format_version",
        "connect_timeout",
        "read_timeout",
        "progress_timeout",
        "max_retries",
        "max_attempts",
        "allow_unverified_manual",
        "catalog_provider_id",
        "thinking_format",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"unsupported connection config fields: {', '.join(unknown)}")
    format_version = config.get("format_version", 1)
    if format_version != 1:
        raise ValueError(f"unsupported connection config format: {format_version}")
    connect_timeout = _positive_float(
        config.get("connect_timeout", 30.0), "connect_timeout"
    )
    read_timeout = _positive_float(config.get("read_timeout", 90.0), "read_timeout")
    max_retries = config.get("max_retries", 3)
    if (
        not isinstance(max_retries, int)
        or isinstance(max_retries, bool)
        or max_retries < 0
    ):
        raise ValueError("max_retries must be a non-negative integer")
    # max_attempts 是 Models 独占的重试预算字段，driver 只校验不消费；
    # accounted 调用的 driver 重试恒为 0。
    max_attempts = config.get("max_attempts")
    if max_attempts is not None and (
        not isinstance(max_attempts, int)
        or isinstance(max_attempts, bool)
        or max_attempts < 1
    ):
        raise ValueError("max_attempts must be a positive integer")
    allow_unverified_manual = config.get("allow_unverified_manual", False)
    if not isinstance(allow_unverified_manual, bool):
        raise ValueError("allow_unverified_manual must be boolean")
    thinking_format = config.get("thinking_format", "none")
    if not isinstance(thinking_format, str) or thinking_format not in {"none", "deepseek"}:
        raise ValueError("thinking_format must be none or deepseek")
    return _ConnectionConfig(
        thinking_format=thinking_format,
        base_url=_normalize_base_url(descriptor.endpoint),
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        progress_timeout=_positive_float(config.get("progress_timeout", 300.0), "progress_timeout"),
        max_retries=max_retries,
        allow_unverified_manual=allow_unverified_manual,
    )


def _model_config(config: Mapping[str, Any]) -> _ModelConfig:
    allowed = {
        "format_version",
        "embedding_batch_size",
        "max_tool_schemas",
        "use_responses_lite",
        "reasoning_summary",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"unsupported model config fields: {', '.join(unknown)}")
    format_version = config.get("format_version", 1)
    if format_version != 1:
        raise ValueError(f"unsupported model config format: {format_version}")
    max_tool_schemas = config.get("max_tool_schemas")
    if max_tool_schemas is not None and (
        not isinstance(max_tool_schemas, int)
        or isinstance(max_tool_schemas, bool)
        or max_tool_schemas <= 0
    ):
        raise ValueError("max_tool_schemas must be a positive integer or null")
    if config.get("use_responses_lite", False) not in {False, 0}:
        raise ValueError("use_responses_lite belongs to a different driver")
    if config.get("reasoning_summary", "none") not in {"", "none"}:
        raise ValueError("reasoning_summary belongs to a different driver")
    embedding_batch_size = config.get("embedding_batch_size", 10)
    if (
        not isinstance(embedding_batch_size, int)
        or isinstance(embedding_batch_size, bool)
        or embedding_batch_size <= 0
    ):
        raise ValueError("embedding_batch_size must be a positive integer")
    return _ModelConfig(
        max_tool_schemas=max_tool_schemas,
        embedding_batch_size=embedding_batch_size,
    )


def _json_dumps_bytes(value: object) -> bytes:
    """与 httpx encode_json 相同的字节格式，供请求体增量拼装复用。"""
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _chat_body(
    descriptor: BoundModelDescriptor,
    connection: _ConnectionConfig,
    request: ModelRequest,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": descriptor.model,
        "messages": request.messages,
    }
    if request.max_output_tokens > 0:
        body["max_tokens"] = request.max_output_tokens
    if request.tools:
        body["tools"] = request.tools
        body["tool_choice"] = request.tool_choice
    if descriptor.reasoning_effort and not request.disable_reasoning:
        body["reasoning_effort"] = descriptor.reasoning_effort
    if request.disable_reasoning:
        for key in ("enable_thinking", "thinking", "reasoning_effort"):
            body.pop(key, None)
        if connection.thinking_format == "deepseek":
            body["thinking"] = {"type": "disabled"}
    return body


async def _request_json(
    connection: _ConnectionConfig,
    credential: CredentialHandle,
    method: str,
    path: str,
    *,
    body: Mapping[str, Any] | None = None,
    body_bytes: bytes | None = None,
    http: HttpClient,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(connection.max_retries + 1):
        try:
            token = _credential_token(await credential.read())
            log_timing("model.http.credential")
            client = http.client()
            # 只复用传输连接，不继承旧凭据请求产生的 Cookie。
            client.cookies.clear()
            if body_bytes is not None:
                response = await client.request(
                    method, path, content=body_bytes,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                )
            else:
                response = await client.request(
                    method, path, json=body, headers={"Authorization": f"Bearer {token}"}
                )
            _raise_status(response, secret=token)
            return _json_object(response)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            mapped = _map_error(error)
            if mapped is error and not ModelError.matches(error):
                raise
            if not _retryable(mapped) or attempt >= connection.max_retries:
                raise mapped from error
            last_error = mapped
            await asyncio.sleep(min(8.0, float(2**attempt)))
    raise TransportError("request failed without result").exception() from last_error


async def _request_limited_json(
    connection: _ConnectionConfig,
    credential: CredentialHandle,
    method: str,
    path: str,
    *,
    max_bytes: int,
    body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read one discovery response with a hard byte limit and no retries."""

    try:
        token = _credential_token(await credential.read())
        async with _client(connection, token) as client:
            async with client.stream(
                method,
                path,
                headers={"Accept-Encoding": "gzip, identity"},
                json=body,
            ) as response:
                content = await _read_limited_response(response, max_bytes=max_bytes)
        bounded = httpx.Response(
            status_code=response.status_code,
            content=content,
            request=response.request,
        )
        _raise_status(bounded, secret=token)
        return _json_bytes_object(content)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        mapped = _map_error(error)
        if mapped is error and not ModelError.matches(error):
            raise
        raise mapped from error


async def _read_limited_response(
    response: httpx.Response,
    *,
    max_bytes: int,
) -> bytes:
    """Stop reading as soon as Content-Length or streamed bytes exceed the cap."""

    encoding = response.headers.get("content-encoding", "").strip().lower()
    if encoding in {"", "identity"}:
        decoder: Any | None = None
    elif encoding == "gzip":
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    else:
        raise TransportError(
            f"provider returned unsupported Content-Encoding: {encoding}"
        ).exception()
    raw_length = response.headers.get("content-length")
    if raw_length is not None:
        try:
            content_length = int(raw_length)
        except ValueError as error:
            raise TransportError(
                "provider returned an invalid Content-Length"
            ).exception() from error
        if content_length < 0 or content_length > max_bytes:
            raise TransportError(f"provider response exceeds {max_bytes} bytes").exception()
    content = bytearray()
    raw_bytes = 0
    try:
        async for chunk in response.aiter_raw(chunk_size=64 * 1024):
            raw_bytes += len(chunk)
            if raw_bytes > max_bytes:
                raise TransportError(f"provider response exceeds {max_bytes} bytes").exception()
            if decoder is None:
                decoded = chunk
            else:
                remaining = max_bytes - len(content)
                decoded = decoder.decompress(chunk, remaining + 1)
                if decoder.unconsumed_tail:
                    raise TransportError(
                        f"provider response exceeds {max_bytes} decoded bytes"
                    ).exception()
            if len(content) + len(decoded) > max_bytes:
                raise TransportError(
                    f"provider response exceeds {max_bytes} decoded bytes"
                ).exception()
            content.extend(decoded)
        if decoder is not None:
            remaining = max_bytes - len(content)
            decoded = decoder.flush(remaining + 1)
            if len(content) + len(decoded) > max_bytes:
                raise TransportError(
                    f"provider response exceeds {max_bytes} decoded bytes"
                ).exception()
            content.extend(decoded)
            if not decoder.eof or decoder.unused_data:
                raise TransportError("provider returned an invalid gzip response").exception()
    except zlib.error as error:
        raise TransportError("provider returned an invalid gzip response").exception() from error
    return bytes(content)


async def _stream_chat(
    connection: _ConnectionConfig,
    credential: CredentialHandle,
    body_bytes: bytes,
    on_delta: Callable[[dict[str, str]], Awaitable[None]] | None,
    http: HttpClient,
) -> LLMResponse:
    last_error: Exception | None = None
    for attempt in range(connection.max_retries + 1):
        response_delta_seen = False
        try:
            token = _credential_token(await credential.read())
            log_timing("model.http.credential")
            client = http.client()
            # 只复用传输连接，不继承旧凭据请求产生的 Cookie。
            client.cookies.clear()
            async with client.stream(
                "POST", "/chat/completions", content=body_bytes,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            ) as response:
                log_timing("model.http.headers")
                if response.status_code >= 400:
                    _ = await response.aread()
                _raise_status(response, secret=token)
                return await _consume_stream(response, on_delta, progress_timeout=connection.progress_timeout)
        except asyncio.CancelledError:
            raise
        except _CallbackError as error:
            raise error.error from error
        except Exception as error:
            mapped = _map_error(error)
            if mapped is error and not ModelError.matches(error):
                raise
            # 证据以协议层观察为准：on_delta 回调缺席时 provider 仍可能
            # 已吐出部分输出，不能凭"没有预览观察者"主张安全。已进入
            # HTTP 200 流的任何失败都不携带 send_evidence，_retryable
            # 不会授予重发——无论是否观察到 delta。
            response_delta_seen = (error.response_delta_seen if isinstance(error, _StreamReadError)
                                   else (value.response_delta_seen if (value := ModelError.read(error)) is not None else False))
            if (
                response_delta_seen
                or not _retryable(mapped)
                or attempt >= connection.max_retries
            ):
                raise mapped from error
            last_error = mapped
            await asyncio.sleep(min(8.0, float(2**attempt)))
    raise TransportError("stream failed without result").exception() from last_error


class _StreamReadError(RuntimeError):
    def __init__(self, error: Exception, *, response_delta_seen: bool) -> None:
        super().__init__(str(error))
        self.error = error
        self.response_delta_seen = response_delta_seen


class _CallbackError(RuntimeError):
    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


async def _consume_stream(
    response: httpx.Response,
    on_delta: Callable[[dict[str, str]], Awaitable[None]] | None,
    *,
    progress_timeout: float = 300.0,
) -> LLMResponse:
    content: list[str] = []
    thinking: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    tool_seen = False
    finish_reason: str | None = None
    usage: ModelUsage | None = None
    response_delta_seen = False
    completed = False
    native_reasoning_seen = False
    pending_content = ""
    legacy_candidate: str | None = None
    progress = StreamProgress(progress_timeout)
    try:
        lines = response.aiter_lines()
        async for line in progress.read(lines):
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            if data == "[DONE]":
                completed = True
                await finish_response(lines)
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as error:
                raise TransportError("stream contains invalid JSON").exception() from error
            if not isinstance(chunk, dict):
                raise TransportError("stream chunk must be an object").exception()
            raw_usage = chunk.get("usage")
            if isinstance(raw_usage, Mapping):
                usage = _usage(raw_usage)
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, Mapping):
                raise TransportError("stream choice must be an object").exception()
            raw_finish = choice.get("finish_reason")
            if raw_finish is not None:
                finish_reason = str(raw_finish)
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                continue
            raw_calls = delta.get("tool_calls")
            if isinstance(raw_calls, list) and raw_calls:
                response_delta_seen = True
                tool_seen = True
                if _merge_tool_deltas(calls, raw_calls):
                    progress.advance()
            reasoning = delta.get("reasoning_content")
            if reasoning is None:
                reasoning = delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                progress.advance()
                response_delta_seen = True
                if not native_reasoning_seen:
                    native_reasoning_seen = True
                    held_content = legacy_candidate or pending_content
                    if held_content and not tool_seen:
                        await _emit_delta(
                            on_delta,
                            {"content_delta": held_content},
                        )
                    pending_content = ""
                    legacy_candidate = None
                thinking.append(reasoning)
                if not tool_seen:
                    await _emit_delta(on_delta, {"thinking_delta": reasoning})
            piece = delta.get("content")
            if isinstance(piece, str) and piece:
                progress.advance()
                response_delta_seen = True
                content.append(piece)
                if native_reasoning_seen:
                    if not tool_seen:
                        await _emit_delta(on_delta, {"content_delta": piece})
                elif legacy_candidate is not None:
                    legacy_candidate += piece
                else:
                    ready, pending_content, legacy_candidate = (
                        _hold_legacy_thinking_candidate(pending_content + piece)
                    )
                    if ready and not tool_seen:
                        await _emit_delta(on_delta, {"content_delta": ready})
    except asyncio.CancelledError as error:
        if response_delta_seen:
            setattr(error, "response_delta_seen", True)
        raise
    except _CallbackError:
        raise
    except TimeoutError as error:
        failure = ModelTimeoutError(f"模型流超过 {progress_timeout:g} 秒没有有效进展").exception()
        raise _StreamReadError(failure, response_delta_seen=response_delta_seen) from error
    except Exception as error:
        raise _StreamReadError(
            error, response_delta_seen=response_delta_seen
        ) from error
    if not completed:
        error = TransportError("stream ended before its terminal marker").exception()
        raise _StreamReadError(error, response_delta_seen=response_delta_seen)
    parsed_content, parsed_thinking = _split_tagged_thinking(
        "".join(content).strip() or None,
        "".join(thinking).strip() or None,
    )
    if not native_reasoning_seen and not tool_seen:
        if legacy_candidate is not None:
            candidate_content, candidate_thinking = _split_tagged_thinking(
                legacy_candidate,
                None,
            )
            if candidate_thinking:
                await _emit_delta(on_delta, {"thinking_delta": candidate_thinking})
            if candidate_content:
                await _emit_delta(on_delta, {"content_delta": candidate_content})
        elif pending_content:
            await _emit_delta(on_delta, {"content_delta": pending_content})
    return LLMResponse(
        content=parsed_content,
        thinking=parsed_thinking,
        tool_calls=_tool_calls(calls),
        finish_reason=finish_reason,
        usage=usage,
    )


async def _emit_delta(
    on_delta: Callable[[dict[str, str]], Awaitable[None]] | None,
    delta: dict[str, str],
) -> None:
    if on_delta is None:
        return
    try:
        await on_delta(delta)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        raise _CallbackError(error) from error


def _client(connection: _ConnectionConfig, token: str | None = None) -> httpx.AsyncClient:
    headers = {} if token is None else {"Authorization": f"Bearer {token}"}
    timeout = httpx.Timeout(
        connect=connection.connect_timeout,
        read=connection.read_timeout,
        write=connection.connect_timeout,
        pool=connection.connect_timeout,
    )
    return httpx.AsyncClient(
        base_url=connection.base_url,
        headers=headers,
        timeout=timeout,
        follow_redirects=False,
    )


def _parse_chat_response(payload: Mapping[str, Any]) -> LLMResponse:
    choices = payload.get("choices")
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], Mapping)
    ):
        raise TransportError("chat response is missing first choice").exception()
    choice = cast(Mapping[str, Any], choices[0])
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise TransportError("chat response is missing message").exception()
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise TransportError("chat message content must be string or null").exception()
    thinking = message.get("reasoning_content")
    if thinking is None:
        thinking = message.get("reasoning")
    if thinking is not None and not isinstance(thinking, str):
        raise TransportError("chat reasoning content must be string or null").exception()
    raw_calls = message.get("tool_calls", [])
    calls: list[ToolCall] = []
    if not isinstance(raw_calls, list):
        raise TransportError("chat tool_calls must be an array").exception()
    for raw in raw_calls:
        if not isinstance(raw, Mapping):
            raise TransportError("chat tool call must be an object").exception()
        function = raw.get("function")
        if not isinstance(function, Mapping):
            raise TransportError("chat tool call is missing function").exception()
        calls.append(
            ToolCall(
                id=_required_string(raw.get("id"), "tool call id"),
                name=_required_string(function.get("name"), "tool call name"),
                arguments=_tool_arguments(function.get("arguments")),
            )
        )
    raw_usage = payload.get("usage")
    usage = _usage(raw_usage) if isinstance(raw_usage, Mapping) else None
    finish = choice.get("finish_reason")
    content, thinking = _split_tagged_thinking(content, thinking)
    return LLMResponse(
        content=content,
        thinking=thinking,
        tool_calls=calls,
        finish_reason=None if finish is None else str(finish),
        usage=usage,
    )


def _split_tagged_thinking(
    content: str | None,
    thinking: str | None,
) -> tuple[str | None, str | None]:
    """Read legacy think tags only when the provider has no reasoning field."""

    if thinking is not None or not content:
        return content, thinking
    match = _THINK_RE.search(content)
    if match is None:
        return content, None
    answer = _THINK_RE.sub("", content).strip() or None
    return answer, match.group(1).strip() or None


def _hold_legacy_thinking_candidate(
    buffer: str,
) -> tuple[str, str, str | None]:
    """Stream plain text while holding only a possible legacy think section."""

    token = "<think>"
    marker = buffer.find(token)
    if marker >= 0:
        return buffer[:marker], "", buffer[marker:]
    pending_size = next(
        (
            size
            for size in range(min(len(buffer), len(token) - 1), 0, -1)
            if buffer.endswith(token[:size])
        ),
        0,
    )
    if pending_size:
        return buffer[:-pending_size], buffer[-pending_size:], None
    return buffer, "", None


def _parse_embedding_response(
    payload: Mapping[str, Any], *, expected_count: int
) -> EmbeddingResult:
    raw_data = payload.get("data")
    if not isinstance(raw_data, list) or len(raw_data) != expected_count:
        raise TransportError("embedding response count does not match input").exception()
    ordered: list[tuple[int, tuple[float, ...]]] = []
    for raw in raw_data:
        if not isinstance(raw, Mapping):
            raise TransportError("embedding item must be an object").exception()
        index = raw.get("index")
        vector = raw.get("embedding")
        if not isinstance(index, int) or isinstance(index, bool):
            raise TransportError("embedding index must be an integer").exception()
        if not isinstance(vector, list) or not vector:
            raise TransportError("embedding vector must be non-empty").exception()
        values: list[float] = []
        for value in vector:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TransportError("embedding vector must contain numbers").exception()
            number = float(value)
            if not math.isfinite(number):
                raise TransportError("embedding vector must contain finite numbers").exception()
            values.append(number)
        ordered.append((index, tuple(values)))
    if sorted(index for index, _vector in ordered) != list(range(expected_count)):
        raise TransportError("embedding indexes must cover the input batch exactly").exception()
    ordered.sort(key=lambda item: item[0])
    if len({len(vector) for _index, vector in ordered}) != 1:
        raise TransportError("服务返回了不一致的向量维度；请联系服务提供方，配置未保存。").exception()
    raw_usage = payload.get("usage")
    return EmbeddingResult(
        vectors=tuple(vector for _index, vector in ordered),
        usage=_usage(raw_usage) if isinstance(raw_usage, Mapping) else None,
    )


def _merge_tool_deltas(calls: dict[int, dict[str, str]], raw_calls: list[Any]) -> bool:
    """合并合法工具片段，并报告是否增加了调用内容。"""
    advanced = False
    for raw in raw_calls:
        if not isinstance(raw, Mapping):
            raise TransportError("stream tool call delta must be an object").exception()
        index = raw.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise TransportError("stream tool call index must be non-negative").exception()
        slot = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
        raw_id = raw.get("id")
        if isinstance(raw_id, str):
            slot["id"] += raw_id
            advanced = advanced or bool(raw_id)
        function = raw.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            arguments = function.get("arguments")
            if isinstance(name, str):
                slot["name"] += name
                advanced = advanced or bool(name)
            if isinstance(arguments, str):
                slot["arguments"] += arguments
                advanced = advanced or bool(arguments)

    return advanced


def _tool_calls(calls: Mapping[int, Mapping[str, str]]) -> list[ToolCall]:
    result: list[ToolCall] = []
    for index in sorted(calls):
        raw = calls[index]
        result.append(
            ToolCall(
                id=_required_string(raw.get("id"), "tool call id"),
                name=_required_string(raw.get("name"), "tool call name"),
                arguments=_tool_arguments(raw.get("arguments") or "{}"),
            )
        )
    return result


def _tool_arguments(value: object) -> Mapping[str, Any]:
    if not isinstance(value, str):
        raise TransportError("tool call arguments must be a JSON string").exception()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise TransportError("tool call arguments are invalid JSON").exception() from error
    if not isinstance(parsed, dict):
        raise TransportError("tool call arguments must decode to an object").exception()
    return cast(dict[str, Any], parsed)


def _usage(raw: Mapping[str, Any]) -> ModelUsage:
    input_tokens = _optional_int(raw.get("prompt_tokens"))
    output_tokens = _optional_int(raw.get("completion_tokens"))
    prompt_details = raw.get("prompt_tokens_details")
    completion_details = raw.get("completion_tokens_details")
    cached = (
        _optional_int(prompt_details.get("cached_tokens"))
        if isinstance(prompt_details, Mapping)
        else None
    )
    cache_hit = _optional_int(raw.get("prompt_cache_hit_tokens"))
    cache_miss = _optional_int(raw.get("prompt_cache_miss_tokens"))
    if cache_hit is not None or cache_miss is not None:
        if input_tokens is None:
            input_tokens = (cache_hit or 0) + (cache_miss or 0)
        if cached is None:
            cached = cache_hit or 0
    if cached is None:
        # Kimi 使用顶层命中字段；不覆盖既有字段中的明确零值。
        cached = _optional_int(raw.get("cached_tokens"))
    cache_write = (
        _optional_int(prompt_details.get("cache_write_tokens"))
        if isinstance(prompt_details, Mapping)
        else None
    )
    reasoning = (
        _optional_int(completion_details.get("reasoning_tokens"))
        if isinstance(completion_details, Mapping)
        else None
    )
    covered = int(input_tokens is not None and output_tokens is not None)
    coverage = (
        'exact'
        if covered
        else (
            'partial'
            if input_tokens is not None or output_tokens is not None
            else 'unavailable'
        )
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


def _merge_usage(items: Sequence[ModelUsage | None]) -> ModelUsage | None:
    """Merge per-request usage while preserving unknown token counts."""

    if not any(item is not None for item in items):
        return None
    normalized = tuple(item or ModelUsage() for item in items)

    def total(field: str) -> int | None:
        known = [
            value
            for item in normalized
            if (value := getattr(item, field)) is not None
        ]
        return sum(known) if known else None

    request_count = sum(item.request_count for item in normalized)
    covered = sum(item.covered_request_count for item in normalized)
    coverage = (
        'unavailable'
        if all(item.coverage == 'unavailable' for item in normalized)
        else 'exact'
        if covered == request_count
        and all(item.coverage == 'exact' for item in normalized)
        else 'partial'
    )
    return ModelUsage(
        input_tokens=total("input_tokens"),
        cache_write_input_tokens=total("cache_write_input_tokens"),
        cached_input_tokens=total("cached_input_tokens"),
        output_tokens=total("output_tokens"),
        reasoning_output_tokens=total("reasoning_output_tokens"),
        request_count=request_count,
        covered_request_count=covered,
        coverage=coverage,
    )


def _unsent(error: Exception) -> Exception:
    """发送前本地校验失败：请求可证明未到达 provider，标记为允许重试的证据。"""
    error = ModelError.change(error, send_evidence="unsent")
    return error


def _raise_status(response: httpx.Response, *, secret: str) -> None:
    error = _status_error(response, secret=secret)
    if error is None:
        return
    # 4xx 是对本请求的明确拒绝应答——正面证据。5xx 只说明服务端/网关
    # 未能给出结论，不能证明后端未接收或未处理：不授证据，fail-closed。
    if response.status_code < 500:
        error = ModelError.change(error, send_evidence="rejected")
    raise error


def _status_error(response: httpx.Response, *, secret: str) -> Exception | None:
    if response.status_code < 400:
        return None
    message = _response_error_message(response, secret=secret)
    lowered = message.lower()
    if response.status_code in {401, 403}:
        return AuthenticationError(
            f"模型连接授权未通过（HTTP {response.status_code}）。"
            f"请在模型设置中核对 API Key 或账号权限后重试。服务返回：{message}"
        ).exception()
    if response.status_code >= 500:
        # status-first：5xx 只说明服务端/网关未给出结论，正文诊断文案
        # （context_length 等）不得把错误提升为可证明的容量拒绝。
        return TransportError(f"模型服务暂不可用（HTTP {response.status_code}），请稍后重试。服务返回：{message}").exception()
    if any(code in lowered for code in _CONTEXT_CODES):
        return ContextLengthError(f"模型上下文超过限制（HTTP {response.status_code}）。服务返回：{message}").exception()
    if any(code in lowered for code in _SAFETY_CODES):
        return ContentSafetyError(f"模型安全策略拒绝请求（HTTP {response.status_code}）。服务返回：{message}").exception()
    if response.status_code == 402 or (
        response.status_code == 429
        and any(value in lowered for value in ("quota", "usage limit", "credit"))
    ):
        return QuotaError(f"模型账号额度不足（HTTP {response.status_code}）。服务返回：{message}").exception()
    if response.status_code == 429:
        error = RateLimitError(f"模型服务限流（HTTP 429）。服务返回：{message}").exception()
        # Retry-After 必须随错误传给 Models，由独占重试预算决定何时再付。
        error = ModelError.change(error, retry_at=retry_after_time(response.headers.get("retry-after")))
        return error
    if 400 <= response.status_code < 500:
        return InvalidRequestError(
            f"模型服务拒绝请求（HTTP {response.status_code}）。服务返回：{message}"
        ).exception()
    error = TransportError(f"provider returned HTTP {response.status_code}: {message}").exception()
    return error


def _response_error_message(response: httpx.Response, *, secret: str = "") -> str:
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _redact_secret(response.text, secret)[:500] or f"HTTP {response.status_code}"
    if isinstance(payload, Mapping):
        raw = payload.get("error", payload.get("detail", payload))
        if isinstance(raw, Mapping):
            message, code = raw.get("message"), raw.get("code")
            details = [value for value in (code, message) if isinstance(value, str) and value]
            if details:
                return _redact_secret(": ".join(dict.fromkeys(details)), secret)[:500]
        if isinstance(raw, str) and raw:
            return _redact_secret(raw, secret)[:500]
    return f"HTTP {response.status_code}"


def _redact_secret(message: str, secret: str) -> str:
    if secret:
        return message.replace(secret, "[REDACTED]")
    return message


def _map_error(error: Exception) -> Exception:
    if ModelError.matches(error, AuthenticationError, ContentSafetyError, ContextLengthError, InvalidRequestError, ModelTimeoutError, QuotaError, RateLimitError, TransportError):
        return error
    if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout)):
        # 连接建立失败可证明请求未发出：这是允许重试的正面证据。
        mapped = TransportError(describe_transport_error(error)).exception()
        mapped = ModelError.change(mapped, send_evidence="unsent")
        mapped = ModelError.change(mapped, retry_safe=True)
        return mapped
    if isinstance(error, (httpx.TimeoutException, TimeoutError)):
        return ModelTimeoutError(describe_transport_error(error)).exception()
    if isinstance(error, httpx.TransportError):
        # 请求发出后的读/写失败不携带任何安全证据。
        return TransportError(describe_transport_error(error)).exception()
    if isinstance(error, _StreamReadError):
        # 已进入 HTTP 200 流：无论是否观察到 delta，远端效果都不可证。
        mapped = _map_error(error.error)
        if ModelError.matches(mapped):
            mapped = ModelError.change(mapped, response_delta_seen=error.response_delta_seen, send_evidence=None)
        else:
            setattr(mapped, "response_delta_seen", error.response_delta_seen)
        return mapped
    return error


def _retryable(error: Exception) -> bool:
    value = ModelError.read(error)
    return value is not None and value.send_evidence in ("rejected", "unsent") and (
        isinstance(value, (ModelTimeoutError, RateLimitError)) or value.retry_safe
    )


def _json_object(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise TransportError("provider response is not valid JSON").exception() from error
    if not isinstance(payload, dict):
        raise TransportError("provider response must be a JSON object").exception()
    return cast(dict[str, Any], payload)


def _json_bytes_object(content: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise TransportError("provider response is not valid JSON").exception() from error
    if not isinstance(payload, dict):
        raise TransportError("provider response must be a JSON object").exception()
    return cast(dict[str, Any], payload)


def _credential_token(payload: Mapping[str, str]) -> str:
    token = (
        payload.get("access_token") or payload.get("api_key") or payload.get("token")
    )
    if not token or not token.strip():
        raise AuthenticationError("credential does not contain an API token").exception()
    return token.strip()


def _check_credential_scope(
    descriptor: DriverConnectionDescriptor,
    credential: CredentialHandle,
) -> None:
    if credential.connection_id != descriptor.connection_id:
        raise AuthenticationError("credential connection scope does not match").exception()
    if credential.auth_identity != descriptor.auth_identity:
        raise AuthenticationError("credential auth identity does not match").exception()


def _check_bound_model(
    descriptor: DriverConnectionDescriptor,
    connection_id: str,
    driver_id: str,
) -> None:
    if connection_id != descriptor.connection_id or driver_id != descriptor.driver_id:
        raise ValueError("bound model does not belong to this driver connection")


def _normalize_base_url(value: str) -> str:
    text = value.strip()
    parsed = urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("endpoint must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("endpoint must not contain credentials")
    if parsed.query:
        raise ValueError("endpoint must not contain a query")
    if parsed.fragment:
        raise ValueError("endpoint must not contain a fragment")
    path = parsed.path.rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/embeddings", "/models"):
        if path.endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
            break
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _compile_chat_row(message: Mapping[str, Any], fill_reasoning: bool) -> _ChatRow:
    """将冻结行编译成协议正文与待发射图片，只在字段改变时复制。"""
    role = str(message.get("role") or "")
    content = message.get("content")
    item: dict[str, Any] | None = None
    images: tuple[bytes, ...] = ()
    # 1. assistant/tool 图片只能移到 user 行；每个来源行保留一个标签。
    if role in {"assistant", "tool"} and isinstance(content, (list, tuple)):
        pictures = [block for block in content if block.get("type") == "image_url"]
        if pictures:
            label = (
                f"工具结果 {message['tool_call_id']} 的图片"
                if role == "tool" else "Agent 输出消息中的图片"
            )
            images = tuple(_json_dumps_bytes(block) for block in (
                {"type": "text", "text": label}, *pictures,
            ))
            content = [block for block in content if block.get("type") != "image_url"]
            if not content:
                content = [{"type": "text", "text": "图片见本组消息后的图像内容。"}]
            item = dict(message)
            item["content"] = content
    # 2. 空正文兜底与 DeepSeek 补齐共享同一次外层复制。
    if role == "assistant" and message.get("tool_calls"):
        if content is None or (isinstance(content, str) and not content.strip()):
            calls = message.get("tool_calls")
            first = calls[0] if isinstance(calls, (list, tuple)) and calls else {}
            function = first.get("function") if isinstance(first, dict) else {}
            tool_name = str(function.get("name") or "") if isinstance(function, dict) else ""
            if item is None:
                item = dict(message)
            item["content"] = f"调用工具 {tool_name}" if tool_name else "调用工具"
    elif role in {"user", "assistant", "tool"} and content is None:
        if item is None:
            item = dict(message)
        item["content"] = ""
    if fill_reasoning and message.get("role") == "assistant" and "reasoning_content" not in message:
        if item is None:
            item = dict(message)
        item["reasoning_content"] = ""
    return _ChatRow(message, fill_reasoning, _json_dumps_bytes(message if item is None else item), images)


def _message_size(message: Mapping[str, Any]) -> tuple[int, int]:
    """统计一条消息的字符与图片成本；调用方统一取整，保持容量边界。"""
    text_chars = image_tokens = 0
    content = message.get("content")
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        for block in content:
            if isinstance(block, Mapping) and block.get("type") in {
                "image_url",
                "input_image",
            }:
                detail = block.get("detail")
                image = block.get("image_url")
                if isinstance(image, Mapping):
                    detail = image.get("detail", detail)
                image_tokens += 1024 if detail == "low" else 8192
                continue
            text_chars += len(
                json.dumps(block, ensure_ascii=False, separators=(",", ":"), default=dict)
            )
    elif content is not None:
        text_chars += len(str(content))
    text_chars += len(
        json.dumps(
            {
                key: value
                for key, value in message.items()
                if key != "content"
            },
            ensure_ascii=False,
            separators=(",", ":"),
            default=dict,
        )
    )
    return text_chars, image_tokens


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TransportError(f"{name} must be a non-empty string").exception()
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TransportError("usage token counts must be non-negative integers").exception()
    return value


def _positive_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


__all__ = ["definition"]
