"""
LLM Provider — OpenAI 兼容格式
支持所有兼容 OpenAI Chat Completions API 的服务：DeepSeek、Qwen、OpenAI 等。
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, cast

import httpx
from openai import AsyncOpenAI

from agent.model_runtime.auth.codex import CodexAuthDriver
from agent.model_runtime.auth.store import CredentialStore
from agent.model_runtime.context_policy import build_runtime_context_budget
from agent.model_runtime.errors import ContextWindowError
from agent.model_runtime.transports.responses import CodexResponsesTransport
from agent.model_runtime.types import (
    LLMResponse,
    ModelBackend,
    ModelRequest,
    ModelUsage,
    ToolCall,
    UsageCoverage,
)

if TYPE_CHECKING:
    from agent.config_models import ModelRuntimeConfig

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

logger = logging.getLogger(__name__)
_LLM_PAYLOAD_SNAPSHOT_ENABLED = False
_LAST_PAYLOAD_PATH = Path(tempfile.gettempdir()) / "akashic-last-llm-payload.json"
_PAYLOAD_SNAPSHOT_DIR = Path(tempfile.gettempdir()) / "akashic-llm-payloads"
_PAYLOAD_SNAPSHOT_SEQ = itertools.count(1)
StreamDelta = dict[str, str]

# 安全审查错误码（各厂商）
_SAFETY_ERROR_CODES = {
    "data_inspection_failed",  # Qwen / DashScope
    "content_filter",  # Azure OpenAI
    "content_policy_violation",  # OpenAI
}

_CONTEXT_LENGTH_KEYWORDS = (
    "range of input length",  # DashScope / Qwen
    "context_length_exceeded",  # OpenAI
    "maximum context length",  # OpenAI
    "context window exceeds limit",  # MiniMax
    "string too long",  # 通用
    "reduce the length",  # 通用
    "too many tokens",  # 通用
)


class ContentSafetyError(Exception):
    """LLM provider 因内容安全审查拒绝请求"""


class ContextLengthError(Exception):
    """LLM provider 因上下文超长拒绝请求"""


class LLMNetworkTimeoutError(TimeoutError):
    """LLM provider 在网络连接或读取边界超时。"""


class ProviderStrategy:
    def normalize_messages(self, messages: list[dict]) -> list[dict]:
        return _strip_reasoning_content(_normalize_chat_messages(messages))

    def prepare_request(
        self,
        kwargs: dict[str, Any],
        extra_body: dict[str, Any],
        *,
        disable_thinking: bool,
    ) -> None:
        if disable_thinking:
            _drop_thinking_keys(extra_body)
        if extra_body:
            kwargs["extra_body"] = extra_body

    def extract_message(
        self,
        msg: Any,
        raw: str | None,
    ) -> tuple[str | None, str | None, dict[str, Any]]:
        thinking: str | None = None
        if raw:
            m = _THINK_RE.search(raw)
            if m:
                thinking = m.group(1).strip()
                raw = _THINK_RE.sub("", raw).strip() or None
        return raw, thinking, {}

    def provider_fields_for_tool_call(
        self,
        fields: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        return fields

    def prepare_stream_request(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return {**kwargs, "stream": True}


class DeepSeekStrategy(ProviderStrategy):
    def normalize_messages(self, messages: list[dict]) -> list[dict]:
        return _strip_image_url_blocks(
            _normalize_chat_messages(messages, fill_tool_call_content=False)
        )

    def prepare_request(
        self,
        kwargs: dict[str, Any],
        extra_body: dict[str, Any],
        *,
        disable_thinking: bool,
    ) -> None:
        thinking_enabled = extra_body.pop("enable_thinking", None)
        reasoning_effort = extra_body.pop("reasoning_effort", None)
        thinking_requested = bool(thinking_enabled) or bool(reasoning_effort)
        if _deepseek_thinking_enabled(extra_body):
            thinking_requested = True
        named_tool_choice = isinstance(kwargs.get("tool_choice"), dict)
        if disable_thinking or named_tool_choice:
            extra_body["thinking"] = {"type": "disabled"}
            reasoning_effort = None
            thinking_requested = False
            if named_tool_choice and not disable_thinking:
                logger.info("[deepseek] 命名 tool_choice 要求本次关闭 thinking")
        elif thinking_enabled is not None and "thinking" not in extra_body:
            extra_body["thinking"] = {
                "type": "enabled" if bool(thinking_enabled) else "disabled"
            }
            thinking_requested = bool(thinking_enabled)
        if reasoning_effort and not _deepseek_thinking_disabled(extra_body):
            kwargs["reasoning_effort"] = _normalize_deepseek_effort(
                str(reasoning_effort)
            )
        if thinking_requested and not _deepseek_thinking_disabled(extra_body):
            messages = kwargs.get("messages")
            if isinstance(messages, list):
                kwargs["messages"] = _ensure_deepseek_reasoning_content(messages)
        if extra_body:
            kwargs["extra_body"] = extra_body

    def extract_message(
        self,
        msg: Any,
        raw: str | None,
    ) -> tuple[str | None, str | None, dict[str, Any]]:
        reasoning = _get_field(msg, "reasoning_content")
        if reasoning is None:
            return raw, None, {}
        text = str(reasoning)
        return raw, text, {"reasoning_content": text}

    def provider_fields_for_tool_call(
        self,
        fields: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if _deepseek_thinking_disabled(dict(kwargs.get("extra_body") or {})):
            return fields
        if "reasoning_content" in fields:
            return fields
        return {**fields, "reasoning_content": ""}

    def prepare_stream_request(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        stream_kwargs = {**kwargs, "stream": True}
        stream_options = dict(stream_kwargs.get("stream_options") or {})
        stream_options["include_usage"] = True
        stream_kwargs["stream_options"] = stream_options
        return stream_kwargs


class DashScopeStrategy(ProviderStrategy):
    def prepare_request(
        self,
        kwargs: dict[str, Any],
        extra_body: dict[str, Any],
        *,
        disable_thinking: bool,
    ) -> None:
        if disable_thinking:
            _drop_thinking_keys(extra_body)
            extra_body["enable_thinking"] = False
        if extra_body:
            kwargs["extra_body"] = extra_body


class ChatCompletionsRuntime:
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        extra_body: dict | None = None,
        read_timeout_s: float = 90.0,
        max_retries: int = 1,
        provider_name: str = "",
        force_disable_thinking: bool = False,
        payload_snapshot_enabled: bool | None = None,
    ) -> None:
        normalized_base_url = _normalize_openai_base_url(base_url)
        network_timeout = httpx.Timeout(
            connect=30.0,
            read=max(0.001, float(read_timeout_s)),
            write=30.0,
            pool=30.0,
        )
        self._client = AsyncOpenAI(
            api_key=api_key or "credential-store",
            base_url=normalized_base_url,
            timeout=network_timeout,
            max_retries=0,
        )
        self._base_url = normalized_base_url or ""
        self._provider_name = provider_name
        self._extra_body = extra_body or {}
        self._max_retries = max(0, int(max_retries))
        self._force_disable_thinking = force_disable_thinking
        self._payload_snapshot_enabled = (
            _LLM_PAYLOAD_SNAPSHOT_ENABLED
            if payload_snapshot_enabled is None
            else bool(payload_snapshot_enabled)
        )

    async def send(self, request: ModelRequest) -> LLMResponse:
        """把统一请求转换为 Chat Completions 并返回统一响应。"""
        strategy = _select_provider_strategy(
            provider_name=self._provider_name,
            base_url=self._base_url,
            model=request.model,
        )
        # 系统提示作为第一条消息（若 messages 已自带 system 消息则不再重复添加）
        messages = request.messages
        already_has_system = messages and messages[0].get("role") == "system"
        full_messages = (
            [{"role": "system", "content": request.system_prompt}, *messages]
            if request.system_prompt and not already_has_system
            else messages
        )
        full_messages = _merge_leading_system_messages(full_messages)
        full_messages = strategy.normalize_messages(full_messages)
        kwargs: dict = dict(
            model=request.model,
            max_tokens=request.max_output_tokens,
            messages=full_messages,
        )
        if request.tools:
            kwargs["tools"] = request.tools
            kwargs["tool_choice"] = request.tool_choice
        merged_extra_body = dict(self._extra_body)
        merged_extra_body.update(request.extra_body)
        strategy.prepare_request(
            kwargs,
            merged_extra_body,
            disable_thinking=self._force_disable_thinking or request.disable_thinking,
        )

        if request.on_delta is not None:
            return await self._chat_streaming(kwargs, request.on_delta, strategy)

        resp = cast(Any, await self._create_with_retry(kwargs))
        msg = resp.choices[0].message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=_parse_tool_arguments(tc.function.arguments),
                    )
                )

        raw, thinking, provider_fields = strategy.extract_message(msg, msg.content)
        cache_prompt_tokens, cache_hit_tokens = _extract_cache_usage(
            getattr(resp, "usage", None)
        )
        usage = _extract_model_usage(getattr(resp, "usage", None))
        if tool_calls:
            provider_fields = strategy.provider_fields_for_tool_call(
                provider_fields,
                kwargs,
            )
        return LLMResponse(
            content=raw,
            tool_calls=tool_calls,
            thinking=thinking,
            provider_fields=provider_fields,
            cache_prompt_tokens=cache_prompt_tokens,
            cache_hit_tokens=cache_hit_tokens,
            usage=usage,
        )

    async def _chat_streaming(
        self,
        kwargs: dict[str, Any],
        on_content_delta: Callable[[StreamDelta], Awaitable[None]],
        strategy: ProviderStrategy,
    ) -> LLMResponse:
        """消费单个 provider stream 并组装最终响应。"""

        # 1. 创建并消费流，网络 idle 超时由 SDK read timeout 统一拥有。
        stream = cast(
            Any,
            await self._create_with_retry(strategy.prepare_stream_request(kwargs)),
        )
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_call_chunks: dict[int, dict[str, str]] = {}
        tool_call_seen = False
        cache_prompt_tokens: int | None = None
        cache_hit_tokens: int | None = None
        usage: ModelUsage | None = None

        try:
            stream_iter = aiter(stream)
            while True:
                try:
                    chunk = await anext(stream_iter)
                except StopAsyncIteration:
                    break
                except Exception as exc:
                    if not self._is_network_timeout(exc):
                        raise
                    raise LLMNetworkTimeoutError("LLM 流读取网络超时") from exc
                prompt_tokens, hit_tokens = _extract_cache_usage(
                    getattr(chunk, "usage", None)
                )
                if prompt_tokens is not None:
                    cache_prompt_tokens = prompt_tokens
                    cache_hit_tokens = hit_tokens
                    usage = _extract_model_usage(getattr(chunk, "usage", None))
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                reasoning_piece = _get_field(delta, "reasoning_content")
                if isinstance(reasoning_piece, str) and reasoning_piece:
                    reasoning_parts.append(reasoning_piece)
                    if not tool_call_seen:
                        await on_content_delta({"thinking_delta": reasoning_piece})

                for tc in _iter_tool_call_deltas(delta):
                    tool_call_seen = True
                    chunk_index = int(tc["index"])
                    slot = tool_call_chunks.setdefault(chunk_index, {})
                    tc_id = str(tc["id"])
                    tc_name = str(tc["name"])
                    tc_arguments = str(tc["arguments"])
                    if tc_id:
                        slot["id"] = slot.get("id", "") + tc_id
                    if tc_name:
                        slot["name"] = slot.get("name", "") + tc_name
                    if tc_arguments:
                        slot["arguments"] = slot.get("arguments", "") + tc_arguments

                content_piece = _get_field(delta, "content")
                if isinstance(content_piece, str) and content_piece:
                    content_parts.append(content_piece)
                    if not tool_call_seen:
                        await on_content_delta({"content_delta": content_piece})
        finally:
            await stream.close()

        # 2. 将完整 tool-call 参数恢复成内部对象
        tool_calls: list[ToolCall] = []
        for idx in sorted(tool_call_chunks):
            item = tool_call_chunks[idx]
            raw_args = item.get("arguments", "") or "{}"
            tool_calls.append(
                ToolCall(
                    id=item.get("id", ""),
                    name=item.get("name", ""),
                    arguments=_parse_tool_arguments(raw_args),
                )
            )

        # 3. 组装文本、推理和 provider 字段
        raw = "".join(content_parts).strip() or None
        thinking = "".join(reasoning_parts).strip() or None
        raw, parsed_thinking, provider_fields = strategy.extract_message(
            {"reasoning_content": thinking} if thinking is not None else {},
            raw,
        )
        thinking = parsed_thinking if parsed_thinking is not None else thinking
        if tool_calls:
            provider_fields = strategy.provider_fields_for_tool_call(
                provider_fields,
                kwargs,
            )
        return LLMResponse(
            content=raw,
            tool_calls=tool_calls,
            thinking=thinking,
            provider_fields=provider_fields,
            cache_prompt_tokens=cache_prompt_tokens,
            cache_hit_tokens=cache_hit_tokens,
            usage=usage,
        )

    async def _create_with_retry(self, kwargs: dict) -> object:
        _save_llm_payload_snapshot(kwargs, enabled=self._payload_snapshot_enabled)
        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._client.chat.completions.create(**kwargs)
            except Exception as e:
                last_err = e
                logger.warning(
                    "[llm.error] model=%s stream=%s base_url=%s tools=%d extra_body_keys=%s "
                    "err=%s",
                    kwargs.get("model"),
                    bool(kwargs.get("stream")),
                    self._base_url,
                    len(kwargs.get("tools") or []),
                    sorted((kwargs.get("extra_body") or {}).keys()),
                    e,
                )
                if self._is_safety_error(e):
                    raise ContentSafetyError(str(e)) from e
                if self._is_context_length_error(e):
                    raise ContextLengthError(str(e)) from e
                retryable = self._is_retryable(e)
                exhausted = attempt >= self._max_retries
                if (not retryable) or exhausted:
                    if self._is_network_timeout(e):
                        raise LLMNetworkTimeoutError("LLM 请求网络超时") from e
                    raise
                wait_s = min(8.0, 1.0 * (2**attempt))
                logger.warning(
                    "[llm] 请求失败，将重试 attempt=%d/%d wait=%.1fs err=%s",
                    attempt + 1,
                    self._max_retries + 1,
                    wait_s,
                    type(e).__name__,
                )
                await asyncio.sleep(wait_s)
        if last_err:
            raise last_err
        raise RuntimeError("LLM request failed without exception")

    @staticmethod
    def _is_safety_error(err: Exception) -> bool:
        text = str(err)
        return any(code in text for code in _SAFETY_ERROR_CODES)

    @staticmethod
    def _is_context_length_error(err: Exception) -> bool:
        text = str(err).lower()
        return any(kw in text for kw in _CONTEXT_LENGTH_KEYWORDS)

    @staticmethod
    def _is_retryable(err: Exception) -> bool:
        if ChatCompletionsRuntime._is_network_timeout(err):
            return True
        status_code = getattr(err, "status_code", None)
        if status_code in {429, 500, 502, 503, 504}:
            return True
        text = str(err).lower()
        keywords = (
            "429",
            "timeout",
            "timed out",
            "connect",
            "connection",
            "temporarily unavailable",
            "server error",
            "502",
            "503",
            "504",
            "rate limit",
            "too many requests",
        )
        return any(k in text for k in keywords)

    @staticmethod
    def _is_network_timeout(err: Exception) -> bool:
        return isinstance(
            err,
            (TimeoutError, httpx.TimeoutException),
        ) or type(err).__name__ == "APITimeoutError"


class LLMProvider:
    """把稳定调用签名收敛为统一模型请求，并隐藏后端差异。"""

    @classmethod
    def from_runtime(
        cls,
        runtime: ModelRuntimeConfig,
        *,
        system_prompt: str,
        extra_body: dict[str, object] | None = None,
        read_timeout_s: float = 90.0,
        force_disable_thinking: bool = False,
        payload_snapshot_enabled: bool | None = None,
    ) -> LLMProvider:
        """从已校验配置构建 provider，不向调用层暴露后端参数。"""
        body = dict(extra_body or {})
        if runtime.reasoning_effort and not force_disable_thinking:
            body.setdefault("reasoning_effort", runtime.reasoning_effort)
        return cls(
            api_key=runtime.api_key,
            base_url=runtime.base_url or None,
            system_prompt=system_prompt,
            extra_body=body,
            read_timeout_s=read_timeout_s,
            provider_name=runtime.provider,
            auth_id=runtime.auth,
            runtime_id=runtime.runtime_id,
            context_window=runtime.context_window,
            effective_context_percent=runtime.effective_context_percent,
            use_responses_lite=runtime.use_responses_lite,
            supports_parallel_tool_calls=runtime.supports_parallel_tool_calls,
            reasoning_summary=runtime.reasoning_summary,
            force_disable_thinking=force_disable_thinking,
            payload_snapshot_enabled=payload_snapshot_enabled,
        )

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        system_prompt: str = "",
        extra_body: dict | None = None,
        read_timeout_s: float = 90.0,
        max_retries: int = 1,
        provider_name: str = "",
        auth_id: str = "",
        runtime_id: str = "main",
        context_window: int = 0,
        effective_context_percent: float = 0.9,
        use_responses_lite: bool = False,
        supports_parallel_tool_calls: bool = True,
        reasoning_summary: str = "none",
        force_disable_thinking: bool = False,
        payload_snapshot_enabled: bool | None = None,
    ) -> None:
        self._system = system_prompt
        self._runtime_id = runtime_id
        self._extra_body = dict(extra_body or {})
        self._context_window = int(context_window)
        self._effective_context_percent = float(effective_context_percent)
        self._force_disable_thinking = force_disable_thinking
        if self._context_window < 0:
            raise ValueError("context_window 不能小于 0")
        if not 0 < self._effective_context_percent <= 1:
            raise ValueError("effective_context_percent 必须在 (0, 1] 内")
        self._backend: ModelBackend
        if provider_name.lower() == "codex":
            auth = CodexAuthDriver(CredentialStore(), auth_id)
            self._backend = CodexResponsesTransport(
                auth,
                runtime_id=runtime_id,
                base_url=base_url or "https://chatgpt.com/backend-api/codex",
                read_timeout_s=read_timeout_s,
                use_responses_lite=use_responses_lite,
                supports_parallel_tool_calls=supports_parallel_tool_calls,
                reasoning_summary=reasoning_summary,
            )
        else:
            self._backend = ChatCompletionsRuntime(
                api_key=api_key,
                base_url=base_url,
                extra_body=extra_body,
                read_timeout_s=read_timeout_s,
                max_retries=max_retries,
                provider_name=provider_name,
                force_disable_thinking=force_disable_thinking,
                payload_snapshot_enabled=payload_snapshot_enabled,
            )

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str,
        max_tokens: int,
        tool_choice: str | dict = "auto",
        extra_body: dict | None = None,
        disable_thinking: bool = False,
        on_content_delta: Callable[[StreamDelta], Awaitable[None]] | None = None,
        cache_namespace: str = "",
    ) -> LLMResponse:
        self._enforce_context_budget(messages, tools, max_tokens)
        merged_extra = {**self._extra_body, **(extra_body or {})}
        effort = merged_extra.get("reasoning_effort")
        request = ModelRequest(
            messages=messages,
            tools=tools,
            model=model,
            max_output_tokens=max_tokens,
            system_prompt=self._system,
            tool_choice=tool_choice,
            reasoning_effort=(
                None
                if self._force_disable_thinking or disable_thinking
                else str(effort or "") or None
            ),
            prompt_cache_key=(
                _stable_prompt_cache_key(self._runtime_id, model, cache_namespace)
                if cache_namespace
                else None
            ),
            on_delta=on_content_delta,
            extra_body=dict(extra_body or {}),
            disable_thinking=self._force_disable_thinking or disable_thinking,
        )
        try:
            return await self._backend.send(request)
        except ContextWindowError as exc:
            raise ContextLengthError(str(exc)) from exc

    def _enforce_context_budget(
        self, messages: list[dict], tools: list[dict], max_tokens: int
    ) -> None:
        if not self._context_window:
            return
        budget = build_runtime_context_budget(
            self._context_window,
            self._effective_context_percent,
            max_tokens,
        )
        estimated = _estimate_context_tokens(self._system, messages, tools)
        if estimated > budget.input_budget:
            raise ContextLengthError(
                f"上下文估算超限 estimated={estimated} budget={budget.input_budget} quality=approximate"
            )


def _estimate_context_tokens(
    system_prompt: str, messages: list[dict], tools: list[dict]
) -> int:
    """估算文本与图片块预算，避免把 data URI 当作文本 token。"""
    text_chars = len(system_prompt) + len(
        json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    )
    image_tokens = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in {
                    "image_url",
                    "input_image",
                }:
                    detail = block.get("detail")
                    image = block.get("image_url")
                    if isinstance(image, dict):
                        detail = image.get("detail", detail)
                    image_tokens += 1024 if detail == "low" else 8192
                    continue
                text_chars += len(
                    json.dumps(block, ensure_ascii=False, separators=(",", ":"))
                )
        elif content is not None:
            text_chars += len(str(content))
        text_chars += len(
            json.dumps(
                {key: value for key, value in message.items() if key != "content"},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    return max(1, text_chars // 3 + image_tokens)


def _stable_prompt_cache_key(runtime_id: str, model: str, namespace: str) -> str:
    """生成不泄露 session 标识的稳定缓存路由键。"""
    raw = f"{runtime_id}\0{model}\0{namespace}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _get_field(delta: Any, name: str) -> Any:
    if isinstance(delta, dict):
        return delta.get(name)
    return getattr(delta, name, None)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    """解析 provider 工具参数并保持内部对象契约。"""

    arguments = json.loads(raw_arguments)
    if not isinstance(arguments, dict):
        raise TypeError("LLM 工具调用参数必须是 JSON 对象")
    return arguments


def _save_llm_payload_snapshot(
    kwargs: dict,
    *,
    enabled: bool | None = None,
) -> Path | None:
    if not (_LLM_PAYLOAD_SNAPSHOT_ENABLED if enabled is None else enabled):
        return None
    try:
        payload = json.dumps(kwargs, ensure_ascii=False, indent=2, default=str)
        _PAYLOAD_SNAPSHOT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        seq = next(_PAYLOAD_SNAPSHOT_SEQ)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = _PAYLOAD_SNAPSHOT_DIR / f"{ts}-{os.getpid()}-{seq:06d}.json"
        path.write_text(payload, encoding="utf-8")
        _LAST_PAYLOAD_PATH.write_text(payload, encoding="utf-8")
        logger.info("[LLM请求快照] saved=%s", path)
        return path
    except Exception as exc:
        logger.warning("[LLM请求快照] 保存失败: %s", exc)
        return None


def _extract_cache_usage(usage: Any) -> tuple[int | None, int | None]:
    hit_tokens = _coerce_int(_get_field(usage, "prompt_cache_hit_tokens"))
    miss_tokens = _coerce_int(_get_field(usage, "prompt_cache_miss_tokens"))
    if hit_tokens is not None or miss_tokens is not None:
        hit = hit_tokens or 0
        miss = miss_tokens or 0
        return hit + miss, hit

    prompt_tokens = _coerce_int(_get_field(usage, "prompt_tokens"))
    prompt_details = _get_field(usage, "prompt_tokens_details")
    cached_tokens = _coerce_int(_get_field(prompt_details, "cached_tokens"))
    if prompt_tokens is None or cached_tokens is None:
        return None, None
    return prompt_tokens, cached_tokens


def _extract_model_usage(usage: Any) -> ModelUsage | None:
    """把 Chat Completions usage 映射为规范化用量。"""
    if usage is None:
        return None
    prompt_tokens = _coerce_int(_get_field(usage, "prompt_tokens"))
    completion_tokens = _coerce_int(_get_field(usage, "completion_tokens"))
    prompt_details = _get_field(usage, "prompt_tokens_details")
    completion_details = _get_field(usage, "completion_tokens_details")
    cached_tokens = _coerce_int(_get_field(prompt_details, "cached_tokens"))
    if cached_tokens is None:
        _, cached_tokens = _extract_cache_usage(usage)
    reasoning_tokens = _coerce_int(_get_field(completion_details, "reasoning_tokens"))
    return ModelUsage(
        input_tokens=prompt_tokens,
        cached_input_tokens=cached_tokens,
        output_tokens=completion_tokens,
        reasoning_output_tokens=reasoning_tokens,
        covered_request_count=1 if prompt_tokens is not None and completion_tokens is not None else 0,
        coverage=(
            UsageCoverage.EXACT
            if prompt_tokens is not None and completion_tokens is not None
            else UsageCoverage.PARTIAL
            if prompt_tokens is not None or completion_tokens is not None
            else UsageCoverage.UNAVAILABLE
        ),
    )


def _iter_tool_call_deltas(delta: Any) -> list[dict[str, str | int]]:
    raw_items = _get_field(delta, "tool_calls") or []
    result: list[dict[str, str | int]] = []
    for idx, item in enumerate(raw_items):
        if isinstance(item, dict):
            function = item.get("function") or {}
            result.append(
                {
                    "index": int(item.get("index", idx)),
                    "id": str(item.get("id", "") or ""),
                    "name": str(function.get("name", "") or ""),
                    "arguments": str(function.get("arguments", "") or ""),
                }
            )
            continue
        function = getattr(item, "function", None)
        result.append(
            {
                "index": int(getattr(item, "index", idx)),
                "id": str(getattr(item, "id", "") or ""),
                "name": str(getattr(function, "name", "") or ""),
                "arguments": str(getattr(function, "arguments", "") or ""),
            }
        )
    return result


def _summarize_roles(messages: list[dict]) -> str:
    roles = [str(msg.get("role", "?")) for msg in messages]
    if len(roles) <= 12:
        return ",".join(roles)
    head = ",".join(roles[:6])
    tail = ",".join(roles[-3:])
    return f"{head},...,{tail}"


def _summarize_message_shapes(messages: list[dict]) -> str:
    shapes: list[str] = []
    for msg in messages[:8]:
        keys = sorted(k for k in msg.keys() if k != "content")
        content = msg.get("content")
        if isinstance(content, str):
            content_kind = "str"
        elif isinstance(content, list):
            content_kind = "list"
        elif content is None:
            content_kind = "none"
        else:
            content_kind = type(content).__name__
        role = str(msg.get("role", "?"))
        extra = ",".join(keys) if keys else "-"
        shapes.append(f"{role}[content={content_kind};keys={extra}]")
    if len(messages) > 8:
        shapes.append("...")
    return " | ".join(shapes)


def _summarize_tool_names(tools: list[dict]) -> str:
    names = [str((tool.get("function") or {}).get("name", "?")) for tool in tools[:8]]
    if len(tools) > 8:
        names.append("...")
    return ",".join(names)


def _merge_leading_system_messages(messages: list[dict]) -> list[dict]:
    merged: list[dict] = []
    system_contents: list[str] = []
    idx = 0
    while idx < len(messages) and messages[idx].get("role") == "system":
        content = messages[idx].get("content")
        if isinstance(content, str) and content:
            system_contents.append(content)
        idx += 1
    if system_contents:
        merged.append({"role": "system", "content": "\n\n".join(system_contents)})
    merged.extend(messages[idx:])
    return merged if merged else list(messages)


def _select_provider_strategy(
    *,
    provider_name: str,
    base_url: str,
    model: str,
) -> ProviderStrategy:
    provider_text = f"{provider_name} {base_url} {model}".lower()
    if "deepseek" in provider_text:
        return DeepSeekStrategy()
    if (
        "dashscope.aliyuncs.com" in provider_text
        or "dashscope" in provider_text
        or "xiaomimimo.com" in provider_text
    ):
        return DashScopeStrategy()
    return ProviderStrategy()


def _drop_thinking_keys(extra_body: dict[str, Any]) -> None:
    for key in ("enable_thinking", "thinking", "reasoning_effort"):
        extra_body.pop(key, None)


def _deepseek_thinking_disabled(extra_body: dict[str, Any]) -> bool:
    thinking = extra_body.get("thinking")
    if not isinstance(thinking, dict):
        return False
    return str(thinking.get("type", "") or "").lower() == "disabled"


def _deepseek_thinking_enabled(extra_body: dict[str, Any]) -> bool:
    thinking = extra_body.get("thinking")
    if not isinstance(thinking, dict):
        return False
    return str(thinking.get("type", "") or "").lower() == "enabled"


def _normalize_deepseek_effort(value: str) -> str:
    effort = value.strip().lower()
    if effort == "xhigh":
        return "max"
    return effort


def _ensure_deepseek_reasoning_content(messages: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for msg in messages:
        item = dict(msg)
        if item.get("role") == "assistant" and "reasoning_content" not in item:
            item["reasoning_content"] = ""
        normalized.append(item)
    return normalized


def _normalize_chat_messages(
    messages: list[dict],
    *,
    fill_tool_call_content: bool = True,
) -> list[dict]:
    normalized: list[dict] = []
    for msg in messages:
        item = dict(msg)
        role = str(item.get("role", "") or "")
        content = item.get("content")

        if fill_tool_call_content and role == "assistant" and item.get("tool_calls"):
            if content is None or (isinstance(content, str) and not content.strip()):
                tool_calls = item.get("tool_calls") or []
                first = (
                    tool_calls[0] if isinstance(tool_calls, list) and tool_calls else {}
                )
                function = first.get("function") if isinstance(first, dict) else {}
                tool_name = ""
                if isinstance(function, dict):
                    tool_name = str(function.get("name", "") or "")
                item["content"] = f"调用工具 {tool_name}" if tool_name else "调用工具"
        elif role in {"user", "assistant", "tool"}:
            if content is None:
                item["content"] = ""

        normalized.append(item)
    return normalized


def _strip_reasoning_content(messages: list[dict]) -> list[dict]:
    # 非 DeepSeek provider 不应发送 reasoning_content 字段
    return [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]


def _strip_image_url_blocks(messages: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for msg in messages:
        item = dict(msg)
        content = item.get("content")
        if isinstance(content, list):
            text_parts: list[str] = []
            image_count = 0
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
                elif block_type == "image_url":
                    image_count += 1
            if image_count:
                text_parts.append(
                    f"[已移除 {image_count} 个 image_url 图片块：DeepSeek 当前接口只接受文本消息。]"
                )
            item["content"] = "\n".join(text_parts)
        normalized.append(item)
    return normalized


def _normalize_openai_base_url(base_url: str | None) -> str | None:
    text = (base_url or "").strip()
    if not text:
        return None
    parsed = urlsplit(text)
    path = parsed.path.rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/responses"):
        if path.endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
            break
    if not path:
        path = ""
    return urlunsplit(
        (parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment)
    )
