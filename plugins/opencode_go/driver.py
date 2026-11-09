from __future__ import annotations

from plugins.models.contract import stream_text
from plugins.models.contract import describe_transport_error

# External JSON is validated field by field below; pyright cannot preserve the
# narrowed key/value types of arbitrary Mapping and list payloads.
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

import asyncio
import json
import math
import sqlite3
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from core.net.http import HttpClient, StreamProgress, finish_response, retry_after_time

import httpx

from plugins.models.contract import (
    LLMResponse,
    ModelRequest,
    ModelUsage,
)
from plugins.models.contract import (
    BoundModelDescriptor,
    CapabilitySources,
    CredentialHandle,
    DiscoveredModel,
    DriverConnection,
    DriverConnectionDescriptor,
    EmbeddingSpaceDescriptor,
    ModelCapabilities,
    QuotaReading,
    QuotaWindow,
    quota_window_label,
)
from plugins.models.contract import (
    AuthenticationError,
    ContentSafetyError,
    ContextLengthError,
    InvalidRequestError,
    ModelError,
    ModelTimeoutError,
    ModelUnavailableError,
    QuotaError,
    RateLimitError,
    TransportError,
)
from plugins.models.contract import (
    ModelDriverDefinition,
    StreamCallbackError as _CallbackError,
    StreamReadError as _StreamReadError,
    build_stream_tool_calls as _tool_calls,
    check_bound_model as _check_bound_model,
    NonStreamBody,
    chat_completion_response,
    check_chat_response_error,
    check_credential_scope as _check_credential_scope,
    credential_bearer_token as _credential_token,
    describe_transport_error,
    estimate_tokens,
    http_status_error as _status_error,
    is_driver_retryable as _retryable,
    map_driver_error as _map_error,
    mark_unsent_error as _unsent,
    merge_stream_tool_deltas as _merge_tool_deltas,
    normalize_http_base_url as _normalize_base_url,
    optional_non_negative_int as _optional_int,
    parse_openai_usage as _usage,
    positive_float_config as _positive_float,
    raise_http_status as _raise_status,
    redact_secret_text as _redact_secret,
    required_non_empty_string as _required_string,
    response_error_message as _response_error_message,
    response_json_object as _json_object,
)


_DRIVER_ID = "opencode-go"
_CONTRACT_VERSION = "1"
_DEFAULT_ENDPOINT = "https://opencode.ai/zen/go/v1"
_MAX_TOOL_SCHEMAS = 16
_MESSAGES_PREFIXES = ("minimax-", "qwen")
_OPENCODE_DATA_DIR = Path.home() / ".local" / "share" / "opencode"
_OPENCODE_EXECUTABLE = "opencode"


@dataclass(frozen=True, slots=True)
class _ConnectionConfig:
    base_url: str
    connect_timeout: float
    read_timeout: float
    max_retries: int
    progress_timeout: float = 300.0


class _BoundChat:
    def __init__(
        self,
        connection: _ConnectionConfig,
        credential: CredentialHandle,
        descriptor: BoundModelDescriptor,
        http: HttpClient,
    ) -> None:
        self._connection = connection
        self._credential = credential
        self._descriptor = descriptor
        self._http = http

    @property
    def max_tool_schemas(self) -> int | None:
        return _MAX_TOOL_SCHEMAS

    async def complete(self, request: ModelRequest) -> LLMResponse:
        """Send one exact bound model request through Chat Completions."""

        if request.continuation is not None:
            # 发送前本地校验失败：可证明请求未发出。
            raise _unsent(InvalidRequestError(
                "OpenCode Go Chat Completions does not support continuation state"
            ).exception())
        body = _chat_body(self._descriptor, request)
        # 生成调用恒为一次物理 attempt：重试预算唯一 owner 是 Models；
        # 未带 request_key 的直调同样不得隐式重发（§6.3）。max_retries
        # 连接配置只留给 /models discovery 等非生成路径。
        connection = replace(self._connection, max_retries=0)
        # 摘要等无观察者的调用也聚合 SSE，让读取期限约束流中空闲。
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        return await _stream_chat(
            connection,
            self._credential,
            body,
            request.on_delta,
            self._http,
        )

    def estimate_context_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> int:
        return max(1, estimate_tokens(messages, tools, chars_per_token=3))

    def estimate_appended_message_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> int:
        return max(1, estimate_tokens(messages, chars_per_token=3)) if messages else 0


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
        icon=lambda _descriptor: (Path(__file__).parent / "icon.svg").read_text(encoding="utf-8"),
        read_quota=_read_quota,
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
        if model.model.strip().lower().startswith(_MESSAGES_PREFIXES):
            raise InvalidRequestError(
                f"OpenCode Go model {model.model} requires the Messages API"
            ).exception()
        _check_model_config(raw_config)
        return _BoundChat(connection, credential, model, http)

    def bind_embedding(
        model: EmbeddingSpaceDescriptor,
        raw_config: Mapping[str, Any],
    ) -> Any:
        _ = model, raw_config
        raise ModelUnavailableError("OpenCode Go is a chat-only model driver").exception()

    return DriverConnection(bind_chat=bind_chat, bind_embedding=bind_embedding, close=http.aclose)


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
            raise AuthenticationError("local OpenCode credential database cannot be read").exception() from error
        if row is not None:
            try:
                value = json.loads(row[0]) if isinstance(row[0], str) else None
            except json.JSONDecodeError as error:
                raise AuthenticationError("local OpenCode Go database credential is invalid").exception() from error
            key = value.get("key") if isinstance(value, dict) and value.get("type") == "key" else None
            if not isinstance(key, str) or not key.strip():
                raise AuthenticationError("local OpenCode Go database credential is invalid").exception()
            return key.strip()
    try:
        document = json.loads(legacy.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise AuthenticationError("local OpenCode login was not found").exception() from error
    except (OSError, json.JSONDecodeError) as error:
        raise AuthenticationError("local OpenCode auth.json cannot be read").exception() from error
    entry = document.get("opencode-go") if isinstance(document, dict) else None
    key = entry.get("key") if isinstance(entry, dict) else None
    if not isinstance(key, str) or not key.strip():
        raise AuthenticationError("local OpenCode Go login does not contain an API key").exception()
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
        if mapped is error and not ModelError.matches(error):
            raise
        raise mapped from error
    _raise_status(response, secret=token)
    _ = _json_object(response)


# Go 订阅的三个窗口；0% 的窗口尚未开始计时，服务商给的重置时间没有意义。
_QUOTA_WINDOWS = (("rolling", 5 * 3600), ("weekly", 7 * 86400), ("monthly", 30 * 86400))


async def _read_quota(
    descriptor: DriverConnectionDescriptor,
    credential: CredentialHandle,
) -> QuotaReading:
    connection = _connection_config(descriptor)
    _check_credential_scope(descriptor, credential)
    token = _credential_token(await credential.read())
    # 1. 用量接口挂在网关的 /zen/go/v1 下，与连接填的推理地址无关，只沿用主机。
    parts = urlsplit(connection.base_url)
    session = uuid4().hex
    try:
        async with _client(connection, token) as client:
            response = await client.get(
                urlunsplit((parts.scheme, parts.netloc, "/zen/go/v1/usage", "", "")),
                headers={"Accept": "application/json", "x-opencode-session": session,
                         "x-opencode-session-id": session},
            )
    except asyncio.CancelledError:
        raise
    except Exception as error:
        mapped = _map_error(error)
        if mapped is error and not ModelError.matches(error):
            raise
        raise mapped from error
    # 2. 403 是这把 key 不属于订阅者本人或所在工作区。
    if response.status_code == 403:
        raise AuthenticationError("这把 key 没有 OpenCode Go 订阅：请用订阅者本人在该工作区创建的 key。").exception()
    _raise_status(response, secret=token)
    usage = _json_object(response).get("usage")
    windows: list[QuotaWindow] = []
    for key, seconds in _QUOTA_WINDOWS:
        item = cast(Mapping[str, Any], usage).get(key) if isinstance(usage, Mapping) else None
        raw = cast(Mapping[str, Any], item) if isinstance(item, Mapping) else {}
        percent = raw.get("percent")
        if not isinstance(percent, (int, float)) or isinstance(percent, bool):
            continue
        resets = raw.get("resetsAt")
        windows.append(QuotaWindow(quota_window_label(seconds), float(percent),
                                   resets if isinstance(resets, str) and percent > 0 else None, seconds))
    if not windows:
        raise TransportError("OpenCode Go 用量响应里没有 rolling、weekly 或 monthly 窗口").exception()
    return QuotaReading(plan="Go", windows=tuple(windows))


async def _discover(
    descriptor: DriverConnectionDescriptor,
    credential: CredentialHandle,
) -> tuple[DiscoveredModel, ...]:
    connection = _connection_config(descriptor)
    _check_credential_scope(descriptor, credential)
    http = HttpClient(lambda: _client(connection))
    try:
        payload = await _request_json(connection, credential, "GET", "/models", http=http)
    finally:
        await http.aclose()
    cli_catalog = await _load_cli_catalog()
    raw_models = payload.get("data")
    if not isinstance(raw_models, list):
        raise TransportError("models response is missing data array").exception()
    result: list[DiscoveredModel] = []
    for raw in raw_models:
        if not isinstance(raw, Mapping):
            raise TransportError("models response contains a non-object item").exception()
        model = raw.get("id")
        if not isinstance(model, str) or not model.strip():
            raise TransportError("models response contains an invalid id").exception()
        normalized = model.strip()
        if normalized.lower().startswith(_MESSAGES_PREFIXES):
            continue
        cli_owned = normalized in cli_catalog
        metadata = cli_catalog.get(normalized, raw)
        efforts = _reasoning_efforts(cli_catalog[normalized]) if cli_owned else ()
        context_window, max_output_tokens, supports_tools = _catalog_capabilities(metadata)
        result.append(
            DiscoveredModel(
                kind='chat',
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
            raise TransportError("model variants contain an invalid name").exception()
        result.append(value.strip())
    return tuple(result)


async def _load_cli_catalog() -> dict[str, Mapping[str, Any]]:
    """Read OpenCode's own provider metadata without making it a hard dependency."""

    try:
        return await _run_cli_catalog()
    except asyncio.CancelledError:
        raise
    except (RuntimeError, TimeoutError, OSError) as _model_error:
        if not (ModelError.matches(_model_error, TransportError) or isinstance(_model_error, OSError)):
            raise
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
        raise TransportError("OpenCode model catalog command timed out").exception() from error
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()[:500]
        raise TransportError(f"OpenCode model catalog command failed: {detail}").exception()
    try:
        output = stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise TransportError("OpenCode model catalog is not valid UTF-8").exception() from error
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
            raise TransportError("OpenCode model catalog has an incomplete header").exception()
        header = output[cursor:line_end].strip()
        if not header.startswith(prefix) or not header.removeprefix(prefix).strip():
            raise TransportError("OpenCode model catalog contains an unknown record").exception()
        model = header.removeprefix(prefix).strip()
        json_start = line_end + 1
        while json_start < len(output) and output[json_start].isspace():
            json_start += 1
        try:
            value, cursor = decoder.raw_decode(output, json_start)
        except json.JSONDecodeError as error:
            raise TransportError("OpenCode model catalog contains invalid JSON").exception() from error
        if not isinstance(value, dict):
            raise TransportError(f"OpenCode model {model} metadata is not an object").exception()
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
        "progress_timeout",
        "max_retries",
        "max_attempts",
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
    # max_attempts 是 Models 独占的重试预算字段，driver 只校验不消费；
    # accounted 调用的 driver 重试恒为 0。
    max_attempts = config.get("max_attempts")
    if max_attempts is not None and (
        not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1
    ):
        raise ValueError("max_attempts must be a positive integer")
    return _ConnectionConfig(
        base_url=_normalize_base_url(descriptor.endpoint),
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        progress_timeout=_positive_float(config.get("progress_timeout", 300.0), "progress_timeout"),
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
    http: HttpClient,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(connection.max_retries + 1):
        try:
            token = _credential_token(await credential.read())
            client = http.client()
            # 只复用传输连接，不继承旧凭据请求产生的 Cookie。
            client.cookies.clear()
            response = await client.request(
                method, path, json=body, headers={"Authorization": f"Bearer {token}"}
            )
            _raise_status(response, secret=token)
            payload = _json_object(response)
            if path == "/chat/completions":
                check_chat_response_error(payload, secret=token)
            return payload
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


async def _stream_chat(
    connection: _ConnectionConfig,
    credential: CredentialHandle,
    body: Mapping[str, Any],
    on_delta: Callable[[dict[str, str]], Awaitable[None]] | None,
    http: HttpClient,
) -> LLMResponse:
    last_error: Exception | None = None
    for attempt in range(connection.max_retries + 1):
        response_delta_seen = False
        try:
            token = _credential_token(await credential.read())
            client = http.client()
            # 只复用传输连接，不继承旧凭据请求产生的 Cookie。
            client.cookies.clear()
            async with client.stream(
                "POST", "/chat/completions", json=body,
                headers={"Authorization": f"Bearer {token}"},
            ) as response:
                if response.status_code >= 400:
                    _ = await response.aread()
                _raise_status(response, secret=token)
                return await _consume_stream(response, on_delta, progress_timeout=connection.progress_timeout, secret=token)
        except asyncio.CancelledError:
            raise
        except _CallbackError as error:
            raise error.error from error
        except Exception as error:
            mapped = _map_error(error)
            if mapped is error and not ModelError.matches(error):
                raise
            response_delta_seen = (error.response_delta_seen if isinstance(error, _StreamReadError)
                                   else (value.response_delta_seen if (value := ModelError.read(error)) is not None else False))
            if response_delta_seen or not _retryable(mapped) or attempt >= connection.max_retries:
                raise mapped from error
            last_error = mapped
            await asyncio.sleep(min(8.0, float(2**attempt)))
    raise TransportError("stream failed without result").exception() from last_error


async def _consume_stream(
    response: httpx.Response,
    on_delta: Callable[[dict[str, str]], Awaitable[None]] | None,
    *,
    progress_timeout: float = 300.0,
    secret: str = "",
) -> LLMResponse:
    content: list[str] = []
    thinking: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    tool_seen = False
    finish_reason: str | None = None
    usage: ModelUsage | None = None
    response_delta_seen = False
    completed = False
    progress = StreamProgress(progress_timeout)
    body = NonStreamBody()
    try:
        lines = response.aiter_lines()
        async for line in progress.read(lines):
            if not line.startswith("data:"):
                if body.feed(line):
                    break
                continue
            body.frame()
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
            check_chat_response_error(chunk, secret=secret)
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
                thinking.append(reasoning)
                if not tool_seen:
                    await _emit_delta(on_delta, {"thinking_delta": reasoning})
            piece = delta.get("content")
            if isinstance(piece, str) and piece:
                progress.advance()
                response_delta_seen = True
                content.append(piece)
                if not tool_seen:
                    await _emit_delta(on_delta, {"content_delta": piece})
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
        raise _StreamReadError(error, response_delta_seen=response_delta_seen) from error
    if not completed and finish_reason is None:
        # 1. 未见任何 data 帧：对端其实回了非流正文，按完整响应解析或分类。
        payload = body.payload(secret=secret, success="choices")
        if payload is not None:
            return chat_completion_response(payload)
        # 2. 其余都是真断流，保持可重试的传输错误。
        error = TransportError("stream ended before its terminal marker").exception()
        raise _StreamReadError(error, response_delta_seen=response_delta_seen)
    try:
        return LLMResponse(
            # 半截工具调用的文本不算截断正文，交给 ReAct 按拒绝提示处理。
            content=None if finish_reason == "length" and calls else stream_text("".join(content), finish_reason),
            thinking="".join(thinking).strip() or None,
            tool_calls=() if finish_reason == "length" else _tool_calls(calls),
            finish_reason=finish_reason,
            usage=usage,
            output_discarded=finish_reason == "length" and bool(calls),
        )
    except Exception as error:
        raise _StreamReadError(error, response_delta_seen=response_delta_seen) from error


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
    messages: list[dict[str, Any]],
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
    # 行已由 _normalize_messages 新建，无需再次复制正文。
    result.extend(messages[index:])
    return result if result else messages


def _thaw_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _thaw(item) for key, item in value.items()}


def _thaw(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_thaw(item) for item in value]
    return value


__all__ = ["definition"]
