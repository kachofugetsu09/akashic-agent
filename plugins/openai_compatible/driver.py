from __future__ import annotations

# External JSON is validated field by field below; pyright cannot preserve the
# narrowed key/value types of arbitrary Mapping and list payloads.
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

import asyncio
import json
import zlib
import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

import httpx

from agent.plugin_composition import (
    AuthenticationError,
    BoundModelDescriptor,
    CapabilitySources,
    ContentSafetyError,
    ContextLengthError,
    CredentialHandle,
    DiscoveredModel,
    DriverConnection,
    DriverConnectionDescriptor,
    EmbeddingResult,
    EmbeddingSpaceDescriptor,
    LLMResponse,
    InvalidRequestError,
    ModelCapabilities,
    ModelDriverDefinition,
    ModelError,
    ModelKind,
    ModelRequest,
    ModelTimeoutError,
    ModelUsage,
    QuotaError,
    RateLimitError,
    ToolCall,
    TransportError,
    UsageCoverage,
)

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


@dataclass(frozen=True, slots=True)
class _ModelConfig:
    max_tool_schemas: int | None


class _BoundChat:
    def __init__(
        self,
        connection: _ConnectionConfig,
        credential: CredentialHandle,
        descriptor: BoundModelDescriptor,
        config: _ModelConfig,
    ) -> None:
        self._connection = connection
        self._credential = credential
        self._descriptor = descriptor
        self._config = config

    @property
    def max_tool_schemas(self) -> int | None:
        return self._config.max_tool_schemas

    async def complete(self, request: ModelRequest) -> LLMResponse:
        """Send one exact bound model request through Chat Completions."""

        if request.continuation is not None:
            raise InvalidRequestError(
                "OpenAI-compatible Chat Completions does not support continuation state"
            )
        body = _chat_body(self._descriptor, self._connection, self._config, request)
        if request.on_delta is None:
            payload = await _request_json(
                self._connection,
                self._credential,
                "POST",
                "/chat/completions",
                body=body,
            )
            return _parse_chat_response(payload)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        return await _stream_chat(
            self._connection,
            self._credential,
            body,
            request.on_delta,
        )

    def estimate_context_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> int:
        return _estimate_context_tokens("", messages, tools)

    def estimate_appended_message_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> int:
        return _estimate_message_tokens(messages)


class _BoundEmbedding:
    def __init__(
        self,
        connection: _ConnectionConfig,
        credential: CredentialHandle,
        descriptor: EmbeddingSpaceDescriptor,
        config: _ModelConfig,
    ) -> None:
        self._connection = connection
        self._credential = credential
        self._descriptor = descriptor
        self._config = config

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed a non-empty text batch and preserve response ordering."""

        if not texts or any(not isinstance(text, str) for text in texts):
            raise ValueError("embedding texts must be a non-empty string sequence")
        body: dict[str, Any] = {
            "model": self._descriptor.model,
            "input": list(texts),
        }
        payload = await _request_json(
            self._connection,
            self._credential,
            "POST",
            "/embeddings",
            body=body,
        )
        return _parse_embedding_response(payload, expected_count=len(texts))


def definition() -> ModelDriverDefinition:
    """Build this artifact's immutable driver contribution."""

    return ModelDriverDefinition(
        driver_id=_DRIVER_ID,
        contract_version=_CONTRACT_VERSION,
        open=_open,
        discover=_discover,
        probe=_probe,
    )


async def _open(
    descriptor: DriverConnectionDescriptor,
    credential: CredentialHandle,
) -> DriverConnection:
    connection = _connection_config(descriptor)
    if credential.connection_id != descriptor.connection_id:
        raise AuthenticationError("credential connection scope does not match")
    if credential.auth_identity != descriptor.auth_identity:
        raise AuthenticationError("credential auth identity does not match")

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
        )

    return DriverConnection(bind_chat=bind_chat, bind_embedding=bind_embedding)


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
        if mapped is error and not isinstance(error, ModelError):
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
        raise ModelTimeoutError("model discovery timed out") from error
    raw_models = payload.get("data")
    if not isinstance(raw_models, list):
        raise TransportError("models response is missing data array")
    if len(raw_models) > _DISCOVERY_MAX_MODELS:
        raise TransportError(f"models response exceeds {_DISCOVERY_MAX_MODELS} entries")
    result: list[DiscoveredModel] = []
    seen_models: set[str] = set()
    for raw in raw_models:
        if not isinstance(raw, Mapping):
            raise TransportError("models response contains a non-object item")
        model = raw.get("id")
        if not isinstance(model, str) or not model.strip():
            raise TransportError("models response contains an invalid id")
        if model != model.strip():
            raise TransportError("models response contains an id with outer whitespace")
        if len(model) > 256:
            raise TransportError(
                "models response contains an id longer than 256 characters"
            )
        if model in seen_models:
            raise TransportError(f"models response contains duplicate id: {model}")
        seen_models.add(model)
        result.append(
            DiscoveredModel(
                kind=ModelKind.CHAT,
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
        "max_retries",
        "allow_unverified_manual",
        "catalog_provider_id",
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
    allow_unverified_manual = config.get("allow_unverified_manual", False)
    if not isinstance(allow_unverified_manual, bool):
        raise ValueError("allow_unverified_manual must be boolean")
    return _ConnectionConfig(
        base_url=_normalize_base_url(descriptor.endpoint),
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        max_retries=max_retries,
        allow_unverified_manual=allow_unverified_manual,
    )


def _model_config(config: Mapping[str, Any]) -> _ModelConfig:
    allowed = {
        "format_version",
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
    return _ModelConfig(max_tool_schemas=max_tool_schemas)


def _chat_body(
    descriptor: BoundModelDescriptor,
    connection: _ConnectionConfig,
    model: _ModelConfig,
    request: ModelRequest,
) -> dict[str, Any]:
    messages = _normalize_messages(request.messages)
    if request.system_prompt and not (messages and messages[0].get("role") == "system"):
        messages.insert(0, {"role": "system", "content": request.system_prompt})
    messages = _merge_leading_system_messages(messages)
    body: dict[str, Any] = {
        "model": descriptor.model,
        "messages": messages,
    }
    if request.max_output_tokens > 0:
        body["max_tokens"] = request.max_output_tokens
    if request.tools:
        body["tools"] = [_thaw_mapping(item) for item in request.tools]
        body["tool_choice"] = _thaw(request.tool_choice)
    if descriptor.reasoning_effort and not request.disable_reasoning:
        body["reasoning_effort"] = descriptor.reasoning_effort
    if request.disable_reasoning:
        for key in ("enable_thinking", "thinking", "reasoning_effort"):
            body.pop(key, None)
    return body


async def _request_json(
    connection: _ConnectionConfig,
    credential: CredentialHandle,
    method: str,
    path: str,
    *,
    body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(connection.max_retries + 1):
        try:
            token = _credential_token(await credential.read())
            async with _client(connection, token) as client:
                response = await client.request(method, path, json=body)
            _raise_status(response, secret=token)
            return _json_object(response)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            mapped = _map_error(error)
            if mapped is error and not isinstance(error, ModelError):
                raise
            if not _retryable(mapped) or attempt >= connection.max_retries:
                raise mapped from error
            last_error = mapped
            await asyncio.sleep(min(8.0, float(2**attempt)))
    raise TransportError("request failed without result") from last_error


async def _request_limited_json(
    connection: _ConnectionConfig,
    credential: CredentialHandle,
    method: str,
    path: str,
    *,
    max_bytes: int,
) -> dict[str, Any]:
    """Read one discovery response with a hard byte limit and no retries."""

    try:
        token = _credential_token(await credential.read())
        async with _client(connection, token) as client:
            async with client.stream(
                method,
                path,
                headers={"Accept-Encoding": "gzip, identity"},
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
        if mapped is error and not isinstance(error, ModelError):
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
        )
    raw_length = response.headers.get("content-length")
    if raw_length is not None:
        try:
            content_length = int(raw_length)
        except ValueError as error:
            raise TransportError(
                "provider returned an invalid Content-Length"
            ) from error
        if content_length < 0 or content_length > max_bytes:
            raise TransportError(f"provider response exceeds {max_bytes} bytes")
    content = bytearray()
    raw_bytes = 0
    try:
        async for chunk in response.aiter_raw(chunk_size=64 * 1024):
            raw_bytes += len(chunk)
            if raw_bytes > max_bytes:
                raise TransportError(f"provider response exceeds {max_bytes} bytes")
            if decoder is None:
                decoded = chunk
            else:
                remaining = max_bytes - len(content)
                decoded = decoder.decompress(chunk, remaining + 1)
                if decoder.unconsumed_tail:
                    raise TransportError(
                        f"provider response exceeds {max_bytes} decoded bytes"
                    )
            if len(content) + len(decoded) > max_bytes:
                raise TransportError(
                    f"provider response exceeds {max_bytes} decoded bytes"
                )
            content.extend(decoded)
        if decoder is not None:
            remaining = max_bytes - len(content)
            decoded = decoder.flush(remaining + 1)
            if len(content) + len(decoded) > max_bytes:
                raise TransportError(
                    f"provider response exceeds {max_bytes} decoded bytes"
                )
            content.extend(decoded)
            if not decoder.eof or decoder.unused_data:
                raise TransportError("provider returned an invalid gzip response")
    except zlib.error as error:
        raise TransportError("provider returned an invalid gzip response") from error
    return bytes(content)


async def _stream_chat(
    connection: _ConnectionConfig,
    credential: CredentialHandle,
    body: Mapping[str, Any],
    on_delta: Callable[[dict[str, str]], Awaitable[None]],
) -> LLMResponse:
    last_error: Exception | None = None
    for attempt in range(connection.max_retries + 1):
        response_delta_seen = False
        try:
            token = _credential_token(await credential.read())
            async with _client(connection, token) as client:
                async with client.stream(
                    "POST", "/chat/completions", json=body
                ) as response:
                    if response.status_code >= 400:
                        _ = await response.aread()
                    _raise_status(response, secret=token)
                    return await _consume_stream(response, on_delta)
        except asyncio.CancelledError:
            raise
        except _CallbackError as error:
            raise error.error from error
        except Exception as error:
            mapped = _map_error(error)
            if mapped is error and not isinstance(error, ModelError):
                raise
            response_delta_seen = bool(getattr(error, "response_delta_seen", False))
            if response_delta_seen:
                setattr(mapped, "retryable", False)
            if (
                response_delta_seen
                or not _retryable(mapped)
                or attempt >= connection.max_retries
            ):
                raise mapped from error
            last_error = mapped
            await asyncio.sleep(min(8.0, float(2**attempt)))
    raise TransportError("stream failed without result") from last_error


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
    on_delta: Callable[[dict[str, str]], Awaitable[None]],
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
    try:
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            if data == "[DONE]":
                completed = True
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as error:
                raise TransportError("stream contains invalid JSON") from error
            if not isinstance(chunk, dict):
                raise TransportError("stream chunk must be an object")
            raw_usage = chunk.get("usage")
            if isinstance(raw_usage, Mapping):
                usage = _usage(raw_usage)
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, Mapping):
                raise TransportError("stream choice must be an object")
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
                _merge_tool_deltas(calls, raw_calls)
            reasoning = delta.get("reasoning_content")
            if reasoning is None:
                reasoning = delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
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
    except asyncio.CancelledError:
        raise
    except _CallbackError:
        raise
    except Exception as error:
        raise _StreamReadError(
            error, response_delta_seen=response_delta_seen
        ) from error
    if not completed:
        error = TransportError("stream ended before its terminal marker")
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
    on_delta: Callable[[dict[str, str]], Awaitable[None]],
    delta: dict[str, str],
) -> None:
    try:
        await on_delta(delta)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        raise _CallbackError(error) from error


def _client(connection: _ConnectionConfig, token: str) -> httpx.AsyncClient:
    headers = {"Authorization": f"Bearer {token}"}
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
        raise TransportError("chat response is missing first choice")
    choice = cast(Mapping[str, Any], choices[0])
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise TransportError("chat response is missing message")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise TransportError("chat message content must be string or null")
    thinking = message.get("reasoning_content")
    if thinking is None:
        thinking = message.get("reasoning")
    if thinking is not None and not isinstance(thinking, str):
        raise TransportError("chat reasoning content must be string or null")
    raw_calls = message.get("tool_calls", [])
    calls: list[ToolCall] = []
    if not isinstance(raw_calls, list):
        raise TransportError("chat tool_calls must be an array")
    for raw in raw_calls:
        if not isinstance(raw, Mapping):
            raise TransportError("chat tool call must be an object")
        function = raw.get("function")
        if not isinstance(function, Mapping):
            raise TransportError("chat tool call is missing function")
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
        raise TransportError("embedding response count does not match input")
    ordered: list[tuple[int, tuple[float, ...]]] = []
    for position, raw in enumerate(raw_data):
        if not isinstance(raw, Mapping):
            raise TransportError("embedding item must be an object")
        index = raw.get("index", position)
        vector = raw.get("embedding")
        if not isinstance(index, int) or isinstance(index, bool):
            raise TransportError("embedding index must be an integer")
        if not isinstance(vector, list) or not vector:
            raise TransportError("embedding vector must be non-empty")
        values: list[float] = []
        for value in vector:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TransportError("embedding vector must contain numbers")
            number = float(value)
            if not math.isfinite(number):
                raise TransportError("embedding vector must contain finite numbers")
            values.append(number)
        ordered.append((index, tuple(values)))
    if sorted(index for index, _vector in ordered) != list(range(expected_count)):
        raise TransportError("embedding indexes must cover the input batch exactly")
    ordered.sort(key=lambda item: item[0])
    raw_usage = payload.get("usage")
    return EmbeddingResult(
        vectors=tuple(vector for _index, vector in ordered),
        usage=_usage(raw_usage) if isinstance(raw_usage, Mapping) else None,
    )


def _merge_tool_deltas(calls: dict[int, dict[str, str]], raw_calls: list[Any]) -> None:
    for raw in raw_calls:
        if not isinstance(raw, Mapping):
            raise TransportError("stream tool call delta must be an object")
        index = raw.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise TransportError("stream tool call index must be non-negative")
        slot = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
        raw_id = raw.get("id")
        if isinstance(raw_id, str):
            slot["id"] += raw_id
        function = raw.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            arguments = function.get("arguments")
            if isinstance(name, str):
                slot["name"] += name
            if isinstance(arguments, str):
                slot["arguments"] += arguments


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
        raise TransportError("tool call arguments must be a JSON string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise TransportError("tool call arguments are invalid JSON") from error
    if not isinstance(parsed, dict):
        raise TransportError("tool call arguments must decode to an object")
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
        UsageCoverage.EXACT
        if covered
        else (
            UsageCoverage.PARTIAL
            if input_tokens is not None or output_tokens is not None
            else UsageCoverage.UNAVAILABLE
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


def _raise_status(response: httpx.Response, *, secret: str) -> None:
    if response.status_code < 400:
        return
    message = _redact_secret(_response_error_message(response), secret)
    lowered = message.lower()
    if response.status_code in {401, 403}:
        raise AuthenticationError(message)
    if any(code in lowered for code in _CONTEXT_CODES):
        raise ContextLengthError(message)
    if any(code in lowered for code in _SAFETY_CODES):
        raise ContentSafetyError(message)
    if response.status_code == 402 or (
        response.status_code == 429
        and any(value in lowered for value in ("quota", "usage limit", "credit"))
    ):
        raise QuotaError(message)
    if response.status_code == 429:
        raise RateLimitError(message)
    if 400 <= response.status_code < 500:
        raise InvalidRequestError(
            f"provider rejected the request with HTTP {response.status_code}: {message}"
        )
    error = TransportError(f"provider returned HTTP {response.status_code}: {message}")
    if response.status_code in {500, 502, 503, 504}:
        setattr(error, "retry_safe", True)
    raise error


def _response_error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return f"HTTP {response.status_code}"
    if isinstance(payload, Mapping):
        raw = payload.get("error")
        if isinstance(raw, Mapping):
            message = raw.get("message") or raw.get("code")
            if isinstance(message, str) and message:
                return message[:500]
        if isinstance(raw, str) and raw:
            return raw[:500]
    return f"HTTP {response.status_code}"


def _redact_secret(message: str, secret: str) -> str:
    if secret:
        return message.replace(secret, "[REDACTED]")
    return message


def _map_error(error: Exception) -> Exception:
    if isinstance(
        error,
        (
            AuthenticationError,
            ContentSafetyError,
            ContextLengthError,
            InvalidRequestError,
            ModelTimeoutError,
            QuotaError,
            RateLimitError,
            TransportError,
        ),
    ):
        return error
    if isinstance(error, (httpx.TimeoutException, TimeoutError)):
        return ModelTimeoutError("model request timed out")
    if isinstance(error, httpx.TransportError):
        mapped = TransportError(f"model transport failed: {type(error).__name__}")
        setattr(mapped, "retry_safe", True)
        return mapped
    if isinstance(error, _StreamReadError):
        mapped = _map_error(error.error)
        setattr(mapped, "response_delta_seen", error.response_delta_seen)
        return mapped
    return error


def _retryable(error: Exception) -> bool:
    return isinstance(error, (ModelTimeoutError, RateLimitError)) or bool(
        getattr(error, "retry_safe", False)
    )


def _json_object(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise TransportError("provider response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise TransportError("provider response must be a JSON object")
    return cast(dict[str, Any], payload)


def _json_bytes_object(content: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise TransportError("provider response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise TransportError("provider response must be a JSON object")
    return cast(dict[str, Any], payload)


def _credential_token(payload: Mapping[str, str]) -> str:
    token = (
        payload.get("access_token") or payload.get("api_key") or payload.get("token")
    )
    if not token or not token.strip():
        raise AuthenticationError("credential does not contain an API token")
    return token.strip()


def _check_credential_scope(
    descriptor: DriverConnectionDescriptor,
    credential: CredentialHandle,
) -> None:
    if credential.connection_id != descriptor.connection_id:
        raise AuthenticationError("credential connection scope does not match")
    if credential.auth_identity != descriptor.auth_identity:
        raise AuthenticationError("credential auth identity does not match")


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


def _estimate_context_tokens(
    system_prompt: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
) -> int:
    complete = list(messages)
    if system_prompt and not (complete and complete[0].get("role") == "system"):
        complete.insert(0, {"role": "system", "content": system_prompt})
    fixed_chars = len(
        json.dumps(_thaw(tools), ensure_ascii=False, separators=(",", ":"))
    )
    return max(1, fixed_chars // 3 + _estimate_message_tokens(complete))


def _normalize_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Match the established generic Chat Completions message contract."""

    normalized: list[dict[str, Any]] = []
    for message in messages:
        item = _thaw_mapping(message)
        item.pop("reasoning_content", None)
        role = str(item.get("role") or "")
        content = item.get("content")
        if role == "assistant" and item.get("tool_calls"):
            if content is None or (isinstance(content, str) and not content.strip()):
                calls = item.get("tool_calls")
                first = calls[0] if isinstance(calls, list) and calls else {}
                function = first.get("function") if isinstance(first, dict) else {}
                tool_name = (
                    str(function.get("name") or "")
                    if isinstance(function, dict)
                    else ""
                )
                item["content"] = f"调用工具 {tool_name}" if tool_name else "调用工具"
        elif role in {"user", "assistant", "tool"} and content is None:
            item["content"] = ""
        normalized.append(item)
    return normalized


def _merge_leading_system_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    system_contents: list[str] = []
    index = 0
    while index < len(messages) and messages[index].get("role") == "system":
        content = messages[index].get("content")
        if isinstance(content, str) and content:
            system_contents.append(content)
        index += 1
    result = (
        [{"role": "system", "content": "\n\n".join(system_contents)}]
        if system_contents
        else []
    )
    result.extend(_thaw_mapping(item) for item in messages[index:])
    return result if result else [_thaw_mapping(item) for item in messages]


def _estimate_message_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    text_chars = 0
    image_tokens = 0
    for message in messages:
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
                    json.dumps(_thaw(block), ensure_ascii=False, separators=(",", ":"))
                )
        elif content is not None:
            text_chars += len(str(content))
        text_chars += len(
            json.dumps(
                {
                    key: _thaw(value)
                    for key, value in message.items()
                    if key != "content"
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    if not messages:
        return 0
    return max(1, text_chars // 3 + image_tokens)


def _thaw_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _thaw(item) for key, item in value.items()}


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_thaw(item) for item in value]
    return value


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TransportError(f"{name} must be a non-empty string")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TransportError("usage token counts must be non-negative integers")
    return value


def _positive_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


__all__ = ["definition"]
