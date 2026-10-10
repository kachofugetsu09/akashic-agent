from __future__ import annotations

import asyncio
import base64
import binascii
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
from core.net.http import retry_after_time

from agent.plugin_composition.models import (
    LLMResponse,
    ModelRequest,
    ModelUsage,
    ToolCall,
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
)
from plugins.models.contract import (
    AuthenticationError,
    ContentSafetyError,
    ContextLengthError,
    InvalidRequestError,
    ModelError,
    ModelTimeoutError,
    RateLimitError,
    TransportError,
)
from plugins.models.contract import ModelDriverDefinition


def definition() -> ModelDriverDefinition:
    """声明独立的 Gemini 原生协议，不经过 Chat Completions 转换。"""
    return ModelDriverDefinition("gemini", "1", _open, discover=_discover)


async def _open(connection: DriverConnectionDescriptor, credential: CredentialHandle) -> DriverConnection:
    """校验连接边界，冻结传输配置并绑定其唯一凭据身份。"""
    client = _client(connection, credential)

    def bind_chat(model: BoundModelDescriptor, raw: Mapping[str, Any]) -> _Chat:
        if model.connection_id != connection.connection_id or model.driver_id != "gemini":
            raise ValueError("Gemini 模型不属于当前连接")
        if set(raw) - {"format_version"} or raw.get("format_version", 1) != 1:
            raise ValueError("Gemini 模型配置不受支持")
        return _Chat(client, credential, model)

    def bind_embedding(_model: EmbeddingSpaceDescriptor, _raw: Mapping[str, Any]):
        raise InvalidRequestError("此 Gemini 驱动仅支持 GenerateContent 对话").exception()

    return DriverConnection(bind_chat, bind_embedding, close=client.aclose)


def _client(connection: DriverConnectionDescriptor, credential: CredentialHandle) -> httpx.AsyncClient:
    """在连接边界验证协议配置，创建唯一的传输资源。"""
    url = urlsplit(connection.endpoint)
    if url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError("Gemini Base URL 必须是没有凭据、查询或片段的 HTTP(S) 地址")
    if credential.connection_id != connection.connection_id or credential.auth_identity != connection.auth_identity:
        raise AuthenticationError("Gemini 凭据不属于当前连接").exception()
    config = connection.config
    if set(config) - {"format_version", "catalog_provider_id", "connect_timeout", "read_timeout", "max_attempts"}:
        raise ValueError("Gemini 连接包含不支持的配置字段")
    if config.get("format_version", 1) != 1:
        raise ValueError("Gemini 配置版本必须为 1")
    connect = _timeout(config.get("connect_timeout", 30))
    read = _timeout(config.get("read_timeout", 90))
    attempts = config.get("max_attempts", 1)
    if type(attempts) is not int or attempts < 1:
        raise ValueError("max_attempts 必须为正整数，由 Models 消费")
    # SDK 风格的网关根路径使用默认版本；显式版本保持调用者的选择。
    endpoint = connection.endpoint.rstrip('/')
    if not re.fullmatch(r'v\d+(?:alpha|beta\d*)?', url.path.rstrip('/').rsplit('/', 1)[-1]):
        endpoint += '/v1beta'
    return httpx.AsyncClient(base_url=endpoint + '/', timeout=httpx.Timeout(read, connect=connect), follow_redirects=False)


def _timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float('inf'):
        raise ValueError("Gemini 超时必须为有限正数")
    return float(value)


async def _headers(credential: CredentialHandle) -> dict[str, str]:
    payload = await credential.read()
    if payload.get("driver") not in {None, "api_key"} or not payload.get("access_token"):
        raise AuthenticationError("Gemini 需要 API Key 凭据").exception()
    return {"x-goog-api-key": payload["access_token"]}


async def _discover(connection: DriverConnectionDescriptor, credential: CredentialHandle) -> tuple[DiscoveredModel, ...]:
    """分页读取原生模型目录；未知能力保持未知，不伪造探测结果。"""
    client = _client(connection, credential)
    try:
        headers = await _headers(credential)
        rows = []
        page = ""
        seen_pages = set()
        async with asyncio.timeout(30):
            for _ in range(100):
                response = await client.get('models', headers=headers, params={"pageToken": page} if page else {})
                _status(response)
                data = _object(response.json(), "Gemini 模型目录")
                for item in _list(data.get("models"), "models"):
                    item = _object(item, "model")
                    name = _string(item.get("name"), "model.name")
                    methods = _list(item.get("supportedGenerationMethods", []), "supportedGenerationMethods")
                    if "generateContent" not in methods:
                        continue
                    context = _count(item.get('inputTokenLimit'), 'inputTokenLimit')
                    output = _count(item.get('outputTokenLimit'), 'outputTokenLimit')
                    if context == 0 or output == 0:
                        raise TransportError("Gemini 模型容量必须为正数").exception()
                    rows.append(DiscoveredModel('chat', name.removeprefix('models/'),
                        ModelCapabilities(context_window=context, max_output_tokens=output),
                        CapabilitySources(context_window='driver' if context is not None else 'unknown',
                            max_output_tokens='driver' if output is not None else 'unknown')))
                next_page = data.get("nextPageToken")
                if not next_page:
                    return tuple(rows)
                page = _string(next_page, "nextPageToken")
                if page in seen_pages:
                    raise TransportError("Gemini 模型目录重复分页 token").exception()
                seen_pages.add(page)
            raise TransportError("Gemini 模型目录超过分页上限").exception()
    except (httpx.HTTPError, TimeoutError) as error:
        raise TransportError(f"Gemini 模型目录传输失败：{type(error).__name__}").exception() from error
    except json.JSONDecodeError as error:
        raise TransportError("Gemini 模型目录不是有效 JSON").exception() from error
    finally:
        await client.aclose()


class _Chat:
    def __init__(self, client: httpx.AsyncClient, credential: CredentialHandle, descriptor: BoundModelDescriptor):
        self.client, self.credential, self.descriptor = client, credential, descriptor

    @property
    def max_tool_schemas(self) -> int | None:
        return None

    def estimate_context_tokens(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] = ()) -> int:
        # 与通用驱动一样使用保守文本估计；原生协议状态也计入预算。
        return max(1, len(json.dumps(_json([messages, tools]), ensure_ascii=False)) // 3)

    def estimate_appended_message_tokens(self, messages: Sequence[Mapping[str, Any]]) -> int:
        return self.estimate_context_tokens(messages)

    async def complete(self, request: ModelRequest) -> LLMResponse:
        """一次真实生成只发送一次 HTTP；重试和调用结算由 Models 拥有。"""
        try:
            body = _body(request, self.descriptor)
            headers = await _headers(self.credential)
        except (RuntimeError, TimeoutError) as error:
            if not (ModelError.matches(error)):
                raise
            error = ModelError.change(error, send_evidence="unsent")
            raise
        model = quote(self.descriptor.model.removeprefix('models/'), safe='/')
        parts: list[dict[str, Any]] = []
        usage = None
        finish = None
        tool_seen = False
        try:
            self.client.cookies.clear()
            if request.on_delta is None:
                response = await self.client.post(f'models/{model}:generateContent', json=body, headers=headers)
                _status(response)
                data = _object(response.json(), "Gemini 响应")
                current, finish, usage = _chunk(data)
                parts.extend(current)
            else:
                async with self.client.stream('POST', f'models/{model}:streamGenerateContent', params={"alt": "sse"}, json=body, headers=headers) as response:
                    if response.is_error:
                        await response.aread()
                    _status(response)
                    event: list[str] = []
                    async for line in response.aiter_lines():
                        if line.startswith('data:'):
                            event.append(line[5:].lstrip())
                        elif not line and event:
                            raw = '\n'.join(event)
                            event.clear()
                            if raw == '[DONE]':
                                continue
                            current, reason, current_usage = _chunk(_object(json.loads(raw), "Gemini SSE"))
                            parts.extend(current)
                            finish = reason or finish
                            usage = current_usage or usage
                            for part in current:
                                tool_seen = tool_seen or 'functionCall' in part
                                if part.get('text') and not tool_seen:
                                    await request.on_delta({"thinking_delta" if part.get('thought') else "content_delta": part['text']})
                    if event or finish is None:
                        raise TransportError("Gemini SSE 未完整结束").exception()
            return _answer(parts, finish, usage)
        except (RuntimeError, TimeoutError) as error:
            if not (ModelError.matches(error)):
                raise
            error = ModelError.change(error, response_delta_seen=bool(parts))
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as cause:
            error = TransportError(f"Gemini 连接未建立：{type(cause).__name__}").exception()
            error = ModelError.change(error, send_evidence="unsent")
            raise error from cause
        except httpx.TimeoutException as cause:
            raise ModelTimeoutError("Gemini 请求超时，远端效果未知").exception() from cause
        except httpx.TransportError as cause:
            raise TransportError(f"Gemini 传输中断：{type(cause).__name__}").exception() from cause
        except (json.JSONDecodeError, UnicodeDecodeError) as cause:
            raise ModelError("Gemini 返回了无效 JSON").exception() from cause


def _body(request: ModelRequest, model: BoundModelDescriptor) -> dict[str, Any]:
    """将消息投影转成原生内容；同 binding 的历史部件原样返回。"""
    if request.continuation is not None:
        raise InvalidRequestError("Gemini 不接受服务端 continuation").exception()
    contents = []
    system = [{"text": request.system_prompt}] if request.system_prompt else []
    calls = {}
    for row in request.messages:
        role = row['role']
        if role == 'system':
            system.extend(_parts(row.get('content')))
            continue
        if role == 'tool':
            identity = row['tool_call_id']
            if identity not in calls:
                raise InvalidRequestError("Gemini 工具结果没有配对调用").exception()
            name, native_id = calls.pop(identity)
            result = {"name": name, "response": {"content": _json(row.get('content'))}}
            if native_id:
                result['id'] = native_id
            contents.append({"role": "user", "parts": [{"functionResponse": result}]})
            continue
        if role not in {'assistant', 'user'}:
            raise InvalidRequestError(f"Gemini 不支持消息角色：{role}").exception()
        metadata = row.get('provider_metadata')
        wire_calls = row.get('tool_calls', ())
        if metadata is not None:
            if set(metadata) != {'gemini_content'} or role != 'assistant':
                raise InvalidRequestError("Gemini 响应协议 metadata 不匹配").exception()
            content = _json(metadata['gemini_content'])
            original = [p['functionCall'] for p in content['parts'] if 'functionCall' in p]
            if len(original) != len(wire_calls):
                raise InvalidRequestError("Gemini 原生历史与工具调用数量不匹配").exception()
            for call, native in zip(wire_calls, original):
                calls[call['id']] = (native['name'], native.get('id'))
        else:
            parts = _parts(row.get('content'))
            for call in wire_calls:
                function = call['function']
                try:
                    arguments = json.loads(function['arguments'])
                except json.JSONDecodeError as error:
                    raise InvalidRequestError("Gemini 历史工具参数不是 JSON").exception() from error
                parts.append({"functionCall": {"name": function['name'], "args": arguments, "id": call['id']}})
                calls[call['id']] = (function['name'], call['id'])
            content = {"role": "model" if role == 'assistant' else 'user', "parts": parts}
        if content['parts']:
            contents.append(content)
    if calls:
        raise InvalidRequestError("Gemini 历史包含未结算调用").exception()
    body: dict[str, Any] = {"contents": contents, "generationConfig": {"candidateCount": 1}}
    if system:
        body['systemInstruction'] = {"parts": system}
    if request.max_output_tokens:
        body['generationConfig']['maxOutputTokens'] = request.max_output_tokens
    # Gemini 没有跨模型通用的关闭思考开关；验证请求关闭摘要并省略显式思考等级。
    thinking: dict[str, Any] = {"includeThoughts": not request.disable_reasoning}
    if model.reasoning_effort and not request.disable_reasoning:
        thinking['thinkingLevel'] = model.reasoning_effort
    body['generationConfig']['thinkingConfig'] = thinking
    if request.tools:
        body['tools'] = [{"functionDeclarations": [
            {"name": tool['function']['name'], "description": tool['function'].get('description', ''),
             "parametersJsonSchema": _json(tool['function'].get('parameters', {"type": "object"}))}
            for tool in request.tools]}]
        choice = request.tool_choice
        config = {"mode": {'auto': 'AUTO', 'none': 'NONE', 'required': 'ANY'}.get(choice, '')} if isinstance(choice, str) else {"mode": "ANY", "allowedFunctionNames": [choice['function']['name']]}
        if not config['mode']:
            raise InvalidRequestError("Gemini tool_choice 不受支持").exception()
        body['toolConfig'] = {"functionCallingConfig": config}
    return body


def _parts(value: Any) -> list[dict[str, Any]]:
    """转换文本与内联图片；不能表示的输入明确拒绝。"""
    if value is None:
        return []
    if isinstance(value, str):
        return [{"text": value}] if value else []
    result = []
    for block in value:
        if block['type'] == 'text':
            result.append({"text": block['text']})
        elif block['type'] == 'image_url':
            url = block['image_url']['url']
            if not url.startswith('data:') or ';base64,' not in url:
                raise InvalidRequestError("Gemini 图片需要内联 base64 数据").exception()
            mime, data = url[5:].split(';base64,', 1)
            try:
                base64.b64decode(data, validate=True)
            except binascii.Error as error:
                raise InvalidRequestError("Gemini 图片 base64 无效").exception() from error
            result.append({"inlineData": {"mimeType": mime, "data": data}})
        else:
            raise InvalidRequestError(f"Gemini 不支持内容块：{block['type']}").exception()
    return result


def _chunk(data: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str | None, ModelUsage | None]:
    """在原生响应边界验证候选和计费字段。"""
    if data.get('promptFeedback', {}).get('blockReason'):
        raise ContentSafetyError("Gemini 拒绝了当前输入").exception()
    candidates = _list(data.get('candidates', []), "candidates")
    if len(candidates) > 1:
        raise ModelError("Gemini 返回多个候选，违反 candidateCount=1").exception()
    candidate = _object(candidates[0], "candidate") if candidates else {}
    reason = candidate.get('finishReason')
    if reason is not None:
        _string(reason, "finishReason")
    if reason in {'SAFETY', 'RECITATION', 'BLOCKLIST', 'PROHIBITED_CONTENT', 'SPII'}:
        raise ContentSafetyError(f"Gemini 输出被拒绝：{reason}").exception()
    if reason == 'MALFORMED_FUNCTION_CALL':
        # 上游生成失败，不是请求 schema 错误；丢弃整次候选，由 Models 恢复生成。
        error = ModelError(f"Gemini 生成了无法解析的工具调用：{reason}").exception()
        error = ModelError.change(error, retryable=True)
        raise error
    if reason == 'UNEXPECTED_TOOL_CALL':
        raise ModelError(f"Gemini 工具协议失败：{reason}").exception()
    content = _object(candidate.get('content', {}), "content")
    if content.get('role', 'model') != 'model':
        raise ModelError("Gemini 候选角色必须为 model").exception()
    parts = []
    for part in _list(content.get('parts', []), "parts"):
        part = _object(part, "part")
        if 'thought' in part and type(part['thought']) is not bool:
            raise ModelError("Gemini thought 必须为布尔值").exception()
        if 'thoughtSignature' in part:
            _string(part['thoughtSignature'], "thoughtSignature")
        if 'text' in part and not isinstance(part['text'], str):
            raise ModelError("Gemini text 必须为字符串").exception()
        if 'functionCall' in part:
            function = _object(part['functionCall'], "functionCall")
            _string(function.get('name'), "functionCall.name")
            _object(function.get('args', {}), "functionCall.args")
            if 'id' in function:
                _string(function['id'], "functionCall.id")
        elif 'text' not in part:
            raise ModelError("Gemini 返回了尚不支持的原生内容块").exception()
        parts.append(part)
    raw_usage = data.get('usageMetadata')
    usage = None
    if raw_usage is not None:
        raw_usage = _object(raw_usage, "usageMetadata")
        input_tokens = _count(raw_usage.get('promptTokenCount'), 'promptTokenCount')
        output_tokens = _count(raw_usage.get('candidatesTokenCount'), 'candidatesTokenCount')
        exact = input_tokens is not None and output_tokens is not None
        usage = ModelUsage(input_tokens=input_tokens, output_tokens=output_tokens,
            reasoning_output_tokens=_count(raw_usage.get('thoughtsTokenCount'), 'thoughtsTokenCount'),
            cached_input_tokens=_count(raw_usage.get('cachedContentTokenCount'), 'cachedContentTokenCount'),
            coverage='exact' if exact else 'partial',
            covered_request_count=1 if exact else 0)
    return parts, reason, usage


def _answer(parts: list[dict[str, Any]], finish: str | None, usage: ModelUsage | None) -> LLMResponse:
    """展示文本与工具请求各取所需；签名和原部件仍由账本完整保存。"""
    if finish is None or (not parts and finish != 'MAX_TOKENS'):
        raise TransportError("Gemini 返回了未结束或空的响应").exception()
    if finish not in {'STOP', 'MAX_TOKENS'}:
        raise ModelError(f"Gemini 未正常完成生成：{finish}").exception()
    calls = [ToolCall(part['functionCall'].get('id') or 'call_' + uuid.uuid4().hex,
        part['functionCall']['name'], part['functionCall'].get('args', {})) for part in parts if 'functionCall' in part]
    if len({call.id for call in calls}) != len(calls):
        raise ModelError("Gemini 返回重复工具调用 ID").exception()
    text = ''.join(p.get('text', '') for p in parts if not p.get('thought'))
    thinking = ''.join(p.get('text', '') for p in parts if p.get('thought'))
    return LLMResponse(text or None, calls, thinking or None, 'tool_calls' if calls else ('length' if finish == 'MAX_TOKENS' else 'stop'),
        usage=usage, provider_metadata={"gemini_content": {"role": "model", "parts": parts}})


def _status(response: httpx.Response) -> None:
    """按 HTTP 状态给出失败语义，不把 5xx 当成远端未处理。"""
    status = response.status_code
    if 200 <= status < 300:
        return
    if status in {401, 403}:
        error = AuthenticationError(f"Gemini 授权失败：HTTP {status}").exception()
    elif status == 429:
        error = RateLimitError("Gemini 请求限流：HTTP 429").exception()
    elif status >= 500:
        error = TransportError(f"Gemini 服务失败：HTTP {status}，远端效果未知").exception()
    elif status == 400 and any(term in response.text.lower() for term in ('input token count', 'context length', 'context_length_exceeded')):
        error = ContextLengthError("Gemini 拒绝请求：输入超过上下文限制").exception()
    else:
        # 不透传可能包含密钥或完整请求的错误正文。
        error = InvalidRequestError(f"Gemini 拒绝请求：HTTP {status}").exception()
    if status < 500:
        error = ModelError.change(error, send_evidence="rejected")
    if (value := ModelError.read(error)) is not None and value.retryable:
        error = ModelError.change(error, retry_at=retry_after_time(response.headers.get("retry-after")))
    raise error


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelError(f"{label} 必须为 JSON 对象").exception()
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ModelError(f"{label} 必须为 JSON 数组").exception()
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelError(f"{label} 必须为非空字符串").exception()
    return value


def _count(value: Any, label: str) -> int | None:
    if value is not None and (type(value) is not int or value < 0):
        raise ModelError(f"{label} 必须为非负整数").exception()
    return value


def _json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value
