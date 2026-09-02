from __future__ import annotations

# External JSON is validated field by field below; pyright cannot preserve the
# narrowed key/value types of arbitrary Mapping and list payloads.
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

import asyncio
import json
import math
import sqlite3
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
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
    ModelUnavailableError,
    QuotaError,
    RateLimitError,
    ToolCall,
    TransportError,
    UsageCoverage,
)


_DRIVER_ID = "opencode-go"
_CONTRACT_VERSION = "1"
_DEFAULT_ENDPOINT = "https://opencode.ai/zen/go/v1"
_MAX_TOOL_SCHEMAS = 16
_MESSAGES_PREFIXES = ("minimax-", "qwen")
_OPENCODE_DATA_DIR = Path.home() / ".local" / "share" / "opencode"
_OPENCODE_EXECUTABLE = "opencode"
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


class _BoundChat:
    def __init__(
        self,
        connection: _ConnectionConfig,
        credential: CredentialHandle,
        descriptor: BoundModelDescriptor,
    ) -> None:
        self._connection = connection
        self._credential = credential
        self._descriptor = descriptor

    @property
    def max_tool_schemas(self) -> int | None:
        return _MAX_TOOL_SCHEMAS

    async def complete(self, request: ModelRequest) -> LLMResponse:
        """Send one exact bound model request through Chat Completions."""

        if request.continuation is not None:
            raise InvalidRequestError(
                "OpenCode Go Chat Completions does not support continuation state"
            )
        body = _chat_body(self._descriptor, request)
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


def definition() -> ModelDriverDefinition:
    """Build this artifact's immutable driver contribution."""

    return ModelDriverDefinition(
        driver_id=_DRIVER_ID,
        contract_version=_CONTRACT_VERSION,
        open=_open,
        discover=_discover,
        probe=_probe,
        start_auth=_start_auth,
        finish_auth=_finish_auth,
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
        if model.model.strip().lower().startswith(_MESSAGES_PREFIXES):
            raise InvalidRequestError(
                f"OpenCode Go model {model.model} requires the Messages API"
            )
        _check_model_config(raw_config)
        return _BoundChat(connection, credential, model)

    def bind_embedding(
        model: EmbeddingSpaceDescriptor,
        raw_config: Mapping[str, Any],
    ) -> Any:
        _ = model, raw_config
        raise ModelUnavailableError("OpenCode Go is a chat-only model driver")

    return DriverConnection(bind_chat=bind_chat, bind_embedding=bind_embedding)


async def _start_auth(input: Mapping[str, Any]) -> Mapping[str, Any]:
    """Import an API key from the request or OpenCode's local auth store."""

    allowed = {
        "api_key",
        "endpoint",
        "name",
        "auth_identity",
    }
    unknown = sorted(set(input) - allowed)
    if unknown:
        raise ValueError(f"unsupported auth fields: {', '.join(unknown)}")
    raw_key = input.get("api_key")
    if raw_key is not None and not isinstance(raw_key, str):
        raise ValueError("api_key must be a string")
    key = raw_key.strip() if isinstance(raw_key, str) else ""
    source = "api_key"
    if not key:
        if "endpoint" in input:
            raise ValueError("local OpenCode import uses the official endpoint")
        key = _read_local_key(
            _OPENCODE_DATA_DIR / "opencode.db",
            _OPENCODE_DATA_DIR / "auth.json",
        )
        source = "local_opencode"
    endpoint = _auth_text(input, "endpoint", _DEFAULT_ENDPOINT)
    _ = _normalize_base_url(endpoint)
    return {
        "state": {
            "api_key": key,
            "endpoint": endpoint,
            "name": _auth_text(input, "name", "OpenCode Go"),
            "auth_identity": _auth_text(input, "auth_identity", "opencode-go"),
            "source": source,
        },
        "challenge": {"method": source},
    }


async def _finish_auth(state: Mapping[str, Any]) -> Mapping[str, Any]:
    key = _required_string(state.get("api_key"), "api_key")
    return {
        "status": "complete",
        "name": _required_string(state.get("name"), "name"),
        "endpoint": _required_string(state.get("endpoint"), "endpoint"),
        "auth_identity": _required_string(state.get("auth_identity"), "auth_identity"),
        "credential": {"driver": "api_key", "access_token": key},
        "driver_config": {"format_version": 1},
    }


def _read_local_key(database: Path, legacy: Path) -> str:
    """Read the current SQLite owner, falling back only when no DB row exists."""

    if database.exists():
        try:
            connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
            try:
                _ = connection.execute("PRAGMA query_only = ON")
                row = connection.execute(
                    "SELECT value FROM credential "
                    "WHERE integration_id = ? AND COALESCE(active, 1) != 0 "
                    "ORDER BY time_updated DESC, id DESC LIMIT 1",
                    ("opencode-go",),
                ).fetchone()
            finally:
                connection.close()
        except sqlite3.Error as error:
            raise AuthenticationError("local OpenCode credential database cannot be read") from error
        if row is not None:
            try:
                value = json.loads(row[0]) if isinstance(row[0], str) else None
            except json.JSONDecodeError as error:
                raise AuthenticationError("local OpenCode Go database credential is invalid") from error
            key = value.get("key") if isinstance(value, dict) and value.get("type") == "key" else None
            if not isinstance(key, str) or not key.strip():
                raise AuthenticationError("local OpenCode Go database credential is invalid")
            return key.strip()
    try:
        document = json.loads(legacy.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise AuthenticationError("local OpenCode login was not found") from error
    except (OSError, json.JSONDecodeError) as error:
        raise AuthenticationError("local OpenCode auth.json cannot be read") from error
    entry = document.get("opencode-go") if isinstance(document, dict) else None
    key = entry.get("key") if isinstance(entry, dict) else None
    if not isinstance(key, str) or not key.strip():
        raise AuthenticationError("local OpenCode Go login does not contain an API key")
    return key.strip()


def _auth_text(input: Mapping[str, Any], name: str, default: str) -> str:
    value = input.get(name, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


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
    _raise_status(response, secret=token)
    _ = _json_object(response)


async def _discover(
    descriptor: DriverConnectionDescriptor,
    credential: CredentialHandle,
) -> tuple[DiscoveredModel, ...]:
    connection = _connection_config(descriptor)
    _check_credential_scope(descriptor, credential)
    payload = await _request_json(connection, credential, "GET", "/models")
    cli_catalog = await _load_cli_catalog()
    raw_models = payload.get("data")
    if not isinstance(raw_models, list):
        raise TransportError("models response is missing data array")
    result: list[DiscoveredModel] = []
    for raw in raw_models:
        if not isinstance(raw, Mapping):
            raise TransportError("models response contains a non-object item")
        model = raw.get("id")
        if not isinstance(model, str) or not model.strip():
            raise TransportError("models response contains an invalid id")
        normalized = model.strip()
        if normalized.lower().startswith(_MESSAGES_PREFIXES):
            continue
        cli_owned = normalized in cli_catalog
        metadata = cli_catalog.get(normalized, raw)
        efforts = _reasoning_efforts(cli_catalog[normalized]) if cli_owned else ()
        context_window, max_output_tokens, supports_tools = _catalog_capabilities(metadata)
        result.append(
            DiscoveredModel(
                kind=ModelKind.CHAT,
                model=normalized,
                default_reasoning_effort=("high" if "high" in efforts else None),
                capabilities=ModelCapabilities(
                    context_window=context_window,
                    max_output_tokens=max_output_tokens,
                    input_modalities=("text",),
                    supports_tool_calls=supports_tools,
                    supported_reasoning_efforts=efforts,
                ),
                capability_sources=CapabilitySources(
                    context_window=(
                        ("opencode-cli" if cli_owned else "opencode-go-catalog")
                        if context_window
                        else "unknown"
                    ),
                    max_output_tokens=(
                        ("opencode-cli" if cli_owned else "opencode-go-catalog")
                        if max_output_tokens
                        else "unknown"
                    ),
                    input_modalities="unknown",
                    tool_calls=(
                        ("opencode-cli" if cli_owned else "opencode-go-catalog")
                        if supports_tools is not None
                        else "unknown"
                    ),
                    reasoning_efforts="opencode-cli" if cli_owned else "unknown",
                ),
                driver_config={"format_version": 1},
            )
        )
    return tuple(result)


def _reasoning_efforts(model: Mapping[str, Any]) -> tuple[str, ...]:
    variants = model.get("variants")
    if not isinstance(variants, Mapping):
        return ()
    result: list[str] = []
    for value in variants:
        if not isinstance(value, str) or not value.strip():
            raise TransportError("model variants contain an invalid name")
        result.append(value.strip())
    return tuple(result)


async def _load_cli_catalog() -> dict[str, Mapping[str, Any]]:
    """Read OpenCode's own provider metadata without making it a hard dependency."""

    try:
        return await _run_cli_catalog()
    except asyncio.CancelledError:
        raise
    except (OSError, TransportError):
        return {}


async def _run_cli_catalog() -> dict[str, Mapping[str, Any]]:
    """Run the optional CLI and reap its process when interrupted."""

    process = await asyncio.create_subprocess_exec(
        _OPENCODE_EXECUTABLE,
        "models",
        "opencode-go",
        "--verbose",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
    except asyncio.CancelledError:
        process.kill()
        _ = await process.wait()
        raise
    except TimeoutError as error:
        process.kill()
        _ = await process.wait()
        raise TransportError("OpenCode model catalog command timed out") from error
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()[:500]
        raise TransportError(f"OpenCode model catalog command failed: {detail}")
    try:
        output = stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise TransportError("OpenCode model catalog is not valid UTF-8") from error
    return _parse_cli_catalog(output)


def _parse_cli_catalog(output: str) -> dict[str, Mapping[str, Any]]:
    decoder = json.JSONDecoder()
    cursor = 0
    result: dict[str, Mapping[str, Any]] = {}
    prefix = "opencode-go/"
    while cursor < len(output):
        while cursor < len(output) and output[cursor].isspace():
            cursor += 1
        if cursor >= len(output):
            break
        line_end = output.find("\n", cursor)
        if line_end < 0:
            raise TransportError("OpenCode model catalog has an incomplete header")
        header = output[cursor:line_end].strip()
        if not header.startswith(prefix) or not header.removeprefix(prefix).strip():
            raise TransportError("OpenCode model catalog contains an unknown record")
        model = header.removeprefix(prefix).strip()
        json_start = line_end + 1
        while json_start < len(output) and output[json_start].isspace():
            json_start += 1
        try:
            value, cursor = decoder.raw_decode(output, json_start)
        except json.JSONDecodeError as error:
            raise TransportError("OpenCode model catalog contains invalid JSON") from error
        if not isinstance(value, dict):
            raise TransportError(f"OpenCode model {model} metadata is not an object")
        result[model] = value
    return result


def _catalog_capabilities(
    model: Mapping[str, Any],
) -> tuple[int | None, int | None, bool | None]:
    limit = model.get("limit")
    context = limit.get("context") if isinstance(limit, Mapping) else None
    output = limit.get("output") if isinstance(limit, Mapping) else None
    capabilities = model.get("capabilities")
    toolcall = capabilities.get("toolcall") if isinstance(capabilities, Mapping) else None
    return (
        context if isinstance(context, int) and context > 0 else None,
        output if isinstance(output, int) and output > 0 else None,
        toolcall if isinstance(toolcall, bool) else None,
    )


def _connection_config(descriptor: DriverConnectionDescriptor) -> _ConnectionConfig:
    if descriptor.driver_id != _DRIVER_ID:
        raise ValueError(f"unexpected driver id: {descriptor.driver_id}")
    config = descriptor.config
    allowed = {
        "format_version",
        "connect_timeout",
        "read_timeout",
        "max_retries",
        "catalog_provider_id",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"unsupported connection config fields: {', '.join(unknown)}")
    format_version = config.get("format_version", 1)
    if format_version != 1:
        raise ValueError(f"unsupported connection config format: {format_version}")
    legacy_provider = config.get("catalog_provider_id", "opencode-go")
    if legacy_provider not in {"", "opencode-go"}:
        raise ValueError("OpenCode Go catalog_provider_id must be opencode-go or empty")
    connect_timeout = _positive_float(config.get("connect_timeout", 30.0), "connect_timeout")
    read_timeout = _positive_float(config.get("read_timeout", 90.0), "read_timeout")
    max_retries = config.get("max_retries", 3)
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    return _ConnectionConfig(
        base_url=_normalize_base_url(descriptor.endpoint),
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        max_retries=max_retries,
    )


def _check_model_config(config: Mapping[str, Any]) -> None:
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
    legacy_max_tools = config.get("max_tool_schemas")
    if legacy_max_tools is not None and (
        not isinstance(legacy_max_tools, int)
        or isinstance(legacy_max_tools, bool)
        or legacy_max_tools <= 0
    ):
        raise ValueError("legacy max_tool_schemas must be a positive integer or null")
    if config.get("use_responses_lite", False) not in {False, 0}:
        raise ValueError("use_responses_lite belongs to a different driver")
    if config.get("reasoning_summary", "none") not in {"", "none"}:
        raise ValueError("reasoning_summary belongs to a different driver")


def _chat_body(
    descriptor: BoundModelDescriptor,
    request: ModelRequest,
) -> dict[str, Any]:
    messages = _normalize_messages(request.messages)
    if request.system_prompt and not (
        messages and messages[0].get("role") == "system"
    ):
        messages.insert(0, {"role": "system", "content": request.system_prompt})
    messages = _merge_leading_system_messages(messages)
    body: dict[str, Any] = {
        "model": descriptor.model,
        "messages": messages,
    }
    if request.max_output_tokens > 0:
        body["max_tokens"] = (
            min(request.max_output_tokens, 131_072)
            if descriptor.model.lower().startswith("mimo-v2.5-pro")
            else request.max_output_tokens
        )
    if request.tools:
        body["tools"] = _normalize_tools(request.tools)
        body["tool_choice"] = _thaw(request.tool_choice)
    _apply_reasoning_profile(body, descriptor, request)
    return body


def _apply_reasoning_profile(
    body: dict[str, Any],
    descriptor: BoundModelDescriptor,
    request: ModelRequest,
) -> None:
    """Apply only the model-family switches required by the Go wire endpoint."""

    model = descriptor.model.lower()
    effort = descriptor.reasoning_effort
    named_tool = isinstance(request.tool_choice, Mapping)
    disabled = request.disable_reasoning or (model.startswith("deepseek-") and named_tool)
    if model.startswith("deepseek-"):
        if disabled:
            body["thinking"] = {"type": "disabled"}
            return
        body["thinking"] = {"type": "enabled"}
        body["reasoning_effort"] = _cap_effort(effort, maximum="max") or "high"
        for message in body["messages"]:
            if message.get("role") == "assistant" and "reasoning_content" not in message:
                message["reasoning_content"] = ""
        return
    if model.startswith("glm-"):
        if not disabled and effort:
            body["reasoning_effort"] = (
                "max" if effort.lower() in {"xhigh", "max", "ultra"} else "high"
            )
        return
    if model.startswith("kimi-"):
        if disabled:
            body["thinking"] = {"type": "disabled"}
        elif effort:
            body["reasoning_effort"] = _cap_effort(effort, maximum="high")
        return
    if not disabled and effort:
        body["reasoning_effort"] = effort


def _cap_effort(effort: str | None, *, maximum: str) -> str | None:
    if effort is None:
        return None
    normalized = effort.strip().lower()
    if maximum == "max" and normalized in {"xhigh", "max", "ultra"}:
        return "max"
    if maximum == "high" and normalized in {"xhigh", "max", "ultra"}:
        return "high"
    return normalized


_SCHEMA_KEYS = frozenset(
    {"type", "description", "properties", "required", "items", "enum"}
)


def _normalize_tools(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        item = _thaw_mapping(tool)
        function = item.get("function")
        if isinstance(function, dict):
            projected = {
                key: value
                for key, value in function.items()
                if key in {"name", "description"}
            }
            parameters = function.get("parameters")
            if isinstance(parameters, dict):
                projected["parameters"] = _normalize_schema(parameters)
            item["function"] = projected
        normalized.append(item)
    return normalized


def _normalize_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(value, Mapping):
            projected[key] = {
                str(name): _normalize_schema(child)
                for name, child in value.items()
                if isinstance(child, Mapping)
            }
        elif key == "items" and isinstance(value, Mapping):
            projected[key] = _normalize_schema(value)
        elif key == "enum" and isinstance(value, Sequence) and not isinstance(value, str):
            projected[key] = [item for item in value if item is not None]
        elif key == "type" and isinstance(value, Sequence) and not isinstance(value, str):
            non_null = [item for item in value if item != "null"]
            if non_null:
                projected[key] = non_null[0]
        else:
            projected[key] = _thaw(value)
    if "type" not in projected:
        raw_branches = schema.get("anyOf") or schema.get("oneOf")
        if isinstance(raw_branches, Sequence) and not isinstance(raw_branches, str):
            branches = [
                item
                for item in raw_branches
                if isinstance(item, Mapping) and item.get("type") != "null"
            ]
            if len(branches) == 1:
                branch = _normalize_schema(branches[0])
                branch.update(projected)
                projected = branch
    return projected


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
                async with client.stream("POST", "/chat/completions", json=body) as response:
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
            if response_delta_seen or not _retryable(mapped) or attempt >= connection.max_retries:
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
                thinking.append(reasoning)
                if not tool_seen:
                    await _emit_delta(on_delta, {"thinking_delta": reasoning})
            piece = delta.get("content")
            if isinstance(piece, str) and piece:
                response_delta_seen = True
                content.append(piece)
                if not tool_seen:
                    await _emit_delta(on_delta, {"content_delta": piece})
    except asyncio.CancelledError:
        raise
    except _CallbackError:
        raise
    except Exception as error:
        raise _StreamReadError(error, response_delta_seen=response_delta_seen) from error
    if not completed:
        error = TransportError("stream ended before its terminal marker")
        raise _StreamReadError(error, response_delta_seen=response_delta_seen)
    try:
        return LLMResponse(
            content="".join(content).strip() or None,
            thinking="".join(thinking).strip() or None,
            tool_calls=_tool_calls(calls),
            finish_reason=finish_reason,
            usage=usage,
        )
    except Exception as error:
        raise _StreamReadError(error, response_delta_seen=response_delta_seen) from error


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
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
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
    return LLMResponse(
        content=content,
        thinking=thinking,
        tool_calls=calls,
        finish_reason=None if finish is None else str(finish),
        usage=usage,
    )



def _merge_tool_deltas(
    calls: dict[int, dict[str, str]], raw_calls: list[Any]
) -> None:
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


def _credential_token(payload: Mapping[str, str]) -> str:
    token = payload.get("access_token") or payload.get("api_key") or payload.get("token")
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
    fixed_chars = len(json.dumps(_thaw(tools), ensure_ascii=False, separators=(",", ":")))
    return max(1, fixed_chars // 3 + _estimate_message_tokens(complete))


def _normalize_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Match the established generic Chat Completions message contract."""

    normalized: list[dict[str, Any]] = []
    for message in messages:
        item = _thaw_mapping(message)
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
                text_chars += len(json.dumps(_thaw(block), ensure_ascii=False, separators=(",", ":")))
        elif content is not None:
            text_chars += len(str(content))
        text_chars += len(
            json.dumps(
                {key: _thaw(value) for key, value in message.items() if key != "content"},
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
