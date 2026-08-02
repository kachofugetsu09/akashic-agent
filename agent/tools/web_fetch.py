"""
WebFetch 工具
"""

import json
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any, Callable, cast

import html2text
import httpx
from lxml import html as lxml_html
from lxml.etree import ParserError

from agent.tools.base import Tool, ToolExecutionContext
from core.net.http import (
    AddressPolicyError,
    HttpRequester,
    RequestBudget,
    get_default_http_requester,
)
from agent.tools.web_fetch_spill import (
    INLINE_MAX_BYTES,
    SPILL_MAX_FILE_BYTES,
    SpillCleanup,
    SpillLimitExceeded,
    WebFetchSpillStore,
)

_MAX_BYTES = INLINE_MAX_BYTES
_DEFAULT_TIMEOUT = 30  # 秒
_MAX_TIMEOUT = 120  # 秒，与 OpenCode 一致
_USER_AGENT = "akashic/1.0"
_MAX_TEXT_CHARS = 50_000  # 返回给 LLM 的文本字符上限（约 ~12K tokens）
_STREAM_CHUNK_BYTES = 64 * 1024


class SpillOwnerMissing(RuntimeError):
    """大响应缺少 runtime execution owner。"""


def _default_tool_context() -> ToolExecutionContext | None:
    """D1 尚未接线时不产生 owner；bootstrap 必须显式注入 runtime provider。"""

    return None


def _owner_from_context(
    context: ToolExecutionContext | None,
) -> tuple[str | None, str | None]:
    if context is None:
        return None, None
    if not isinstance(context, ToolExecutionContext):
        raise TypeError("web_fetch context_provider 必须返回 ToolExecutionContext | None")
    owner = context.execution_id.strip()
    turn = context.turn_id.strip()
    if not owner or not turn:
        return None, None
    return owner, turn

# 根据 format 设置 Accept header，引导服务端返回更合适的格式
_ACCEPT = {
    "markdown": "text/markdown;q=1.0, text/x-markdown;q=0.9, text/plain;q=0.8, text/html;q=0.7, */*;q=0.1",
    "text": "text/plain;q=1.0, text/markdown;q=0.9, text/html;q=0.8, */*;q=0.1",
    "html": "text/html;q=1.0, application/xhtml+xml;q=0.9, text/plain;q=0.8, */*;q=0.1",
}


class WebFetchTool(Tool):
    """抓取 URL 内容，支持 text / markdown / html 三种格式输出"""

    name = "web_fetch"
    description = (
        "抓取指定 URL 的内容并返回。"
        "支持 text（纯文本）、markdown（转换后的 Markdown，默认）、html（原始 HTML）三种格式。"
        "仅支持 HTTP/HTTPS；大响应会写入本次 execution 的私有临时文件。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "要抓取的完整 URL，必须以 http:// 或 https:// 开头",
            },
            "format": {
                "type": "string",
                "enum": ["text", "markdown", "html"],
                "description": "返回格式：text 纯文本 / markdown 转换后的 Markdown / html 原始 HTML。默认 markdown",
            },
            "timeout": {
                "type": "integer",
                "description": f"超时秒数，默认 {_DEFAULT_TIMEOUT}，最大 {_MAX_TIMEOUT}",
                "minimum": 1,
                "maximum": _MAX_TIMEOUT,
            },
        },
        "required": ["url"],
    }

    def __init__(
        self,
        requester: HttpRequester | None = None,
        spill_store: WebFetchSpillStore | None = None,
        context_provider: Callable[[], ToolExecutionContext | None] = _default_tool_context,
    ) -> None:
        self._requester = requester or get_default_http_requester("web_fetch")
        self._spill_store = spill_store
        self._context_provider = context_provider
        self._execution_turns: dict[str, str] = {}

    def release(self, execution_id: str):
        """释放指定 execution 的响应文件并返回可查询清理诊断。"""

        if self._spill_store is None:
            return SpillCleanup(
                execution_id=str(execution_id),
                released=True,
                status="no_spill_store",
            )
        result = self._spill_store.release(execution_id)
        if result.released:
            self._execution_turns.pop(str(execution_id), None)
        return result

    def release_turn(self, turn_id: str) -> list[SpillCleanup]:
        """释放一个 turn 产生的所有 spill，并保留失败 owner 的诊断。"""

        turn = str(turn_id or "").strip()
        execution_ids = [
            execution_id
            for execution_id, owner_turn in self._execution_turns.items()
            if owner_turn == turn
        ]
        return [self.release(execution_id) for execution_id in execution_ids]

    async def execute(self, **kwargs: Any) -> str:
        url: str = kwargs["url"]
        fmt: str = kwargs.get("format", "markdown")
        timeout: int = min(int(kwargs.get("timeout", _DEFAULT_TIMEOUT)), _MAX_TIMEOUT)

        # URL 结构校验；单人本地运行允许显式访问本机和内网 HTTP 服务。
        if not url.startswith(("http://", "https://")):
            return _err(url, "URL 必须以 http:// 或 https:// 开头")

        context = self._context_provider()
        execution_id, turn_id = _owner_from_context(context)
        validator = getattr(self._requester, "validate_external_url", None)
        if callable(validator):
            try:
                validator(url)
            except AddressPolicyError as exc:
                return _err(url, str(exc), classification="operation_rejected")

        try:
            async with _open_stream(
                self._requester,
                url,
                timeout=timeout,
                fmt=fmt,
            ) as resp:
                if resp.status_code != 200:
                    return _err(url, f"HTTP {resp.status_code}")

                content_type = resp.headers.get("content-type", "")
                is_binary = any(
                    ct in content_type
                    for ct in (
                        "application/pdf",
                        "application/octet-stream",
                        "image/",
                        "video/",
                        "audio/",
                    )
                )
                if is_binary:
                    return _err(
                        url,
                        f"不支持二进制内容（{content_type}），请使用能处理该格式的专用工具",
                    )

                declared = _declared_length(resp.headers.get("content-length"))
                if declared is not None and declared > SPILL_MAX_FILE_BYTES:
                    return _err(
                        url,
                        f"响应过大（超过 {SPILL_MAX_FILE_BYTES // (1024 * 1024)}MB 临时文件上限）",
                        classification="operation_rejected",
                    )

                if (
                    self._spill_store is not None
                    and execution_id is not None
                    and turn_id is not None
                ):
                    # Spill 可能在 _collect_response 返回前创建；先登记可恢复 owner。
                    self._execution_turns[execution_id] = turn_id
                body, spill = await self._collect_response(
                    resp,
                    execution_id=execution_id,
                )
                if spill is not None:
                    assert execution_id is not None
                    assert turn_id is not None
                    self._execution_turns[execution_id] = turn_id
                    self._spill_store.finalize(spill)
                    return json.dumps(
                        {
                            "url": url,
                            "final_url": str(resp.url),
                            "status": resp.status_code,
                            "content_type": content_type,
                            "format": "file",
                            "length": spill.size,
                            "execution_id": execution_id,
                            "turn_id": turn_id,
                            "file_path": str(spill.path),
                            "note": "响应已保存到私有临时文件，请使用 read_file 分页读取；turn 结束后由 execution owner release。",
                        },
                        ensure_ascii=False,
                    )

                if execution_id is not None:
                    self._execution_turns.pop(execution_id, None)
                assert body is not None
                encoding = resp.encoding or "utf-8"
                is_html = "text/html" in content_type
                if fmt == "html":
                    text = body.decode(encoding, errors="replace")
                elif fmt == "markdown" and is_html:
                    text = _to_markdown(body.decode(encoding, errors="replace"))
                elif fmt == "text" and is_html:
                    text = _to_text(body)
                else:
                    text = body.decode(encoding, errors="replace")

                truncated = len(text) > _MAX_TEXT_CHARS
                if truncated:
                    text = text[:_MAX_TEXT_CHARS]
                result: dict[str, Any] = {
                    "url": url,
                    "final_url": str(resp.url),
                    "status": resp.status_code,
                    "content_type": content_type,
                    "format": fmt,
                    "length": len(text),
                    "text": text,
                }
                if truncated:
                    result["truncated"] = True
                    result["note"] = (
                        f"内容已截断至 {_MAX_TEXT_CHARS} 字符，如需更多内容请缩小范围或使用其他工具"
                    )
                return json.dumps(result, ensure_ascii=False)
        except SpillLimitExceeded as exc:
            cleanup = self.release(execution_id) if execution_id is not None else None
            return _err(
                url,
                f"响应超过临时文件上限：{exc}",
                classification="operation_rejected",
                cleanup=cleanup,
            )
        except SpillOwnerMissing as exc:
            return _err(url, str(exc), classification="unit_failed")
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RequestError) as exc:
            cleanup = self.release(execution_id) if execution_id is not None else None
            return _err(url, _request_error(exc, timeout), cleanup=cleanup)
        except AddressPolicyError as exc:
            return _err(url, str(exc), classification="operation_rejected")
        except OSError as exc:
            cleanup = self.release(execution_id) if execution_id is not None else None
            return _err(
                url,
                f"响应临时文件失败：{exc}",
                classification="unit_failed",
                cleanup=cleanup,
            )
        except ValueError as exc:
            cleanup = self.release(execution_id) if execution_id is not None else None
            return _err(
                url,
                str(exc),
                classification="operation_rejected",
                cleanup=cleanup,
            )
        except BaseException:
            if execution_id is not None:
                self.release(execution_id)
            raise

    async def _collect_response(self, resp: Any, *, execution_id: str | None):
        """在内联阈值和 spill 绝对上限内收集响应。"""

        inline = bytearray()
        spill = None
        total = 0
        iterator = resp.aiter_bytes(chunk_size=_STREAM_CHUNK_BYTES)
        async for chunk in iterator:
            if not isinstance(chunk, bytes):
                raise TypeError("HTTP response chunk must be bytes")
            next_total = total + len(chunk)
            if next_total > SPILL_MAX_FILE_BYTES:
                raise SpillLimitExceeded("spill file limit exceeded")
            if spill is None and next_total <= INLINE_MAX_BYTES:
                inline.extend(chunk)
            else:
                if spill is None:
                    if execution_id is None:
                        raise SpillOwnerMissing(
                            "大响应需要 runtime execution owner 与 workspace 私有临时目录"
                        )
                    if self._spill_store is None:
                        raise SpillOwnerMissing(
                            "当前执行没有 workspace 私有临时目录，不能返回不可读 spill 引用"
                        )
                    spill = self._spill_store.open(execution_id)
                    self._spill_store.write(spill, bytes(inline))
                    inline.clear()
                self._spill_store.write(spill, chunk)
            total = next_total
        return (bytes(inline) if spill is None else None), spill


# ── 模块级工具函数 ────────────────────────────────────────────


def _err(
    url: str,
    msg: str,
    *,
    classification: str | None = None,
    cleanup: SpillCleanup | None = None,
) -> str:
    result: dict[str, Any] = {"error": msg, "url": url}
    if classification:
        result["classification"] = classification
    if cleanup is not None:
        result["cleanup"] = {
            "execution_id": cleanup.execution_id,
            "released": cleanup.released,
            "status": cleanup.status,
            "path": cleanup.path,
            "error": cleanup.error,
        }
        if not cleanup.released:
            result["cleanup_classification"] = cleanup.status
    return json.dumps(result, ensure_ascii=False)


def _declared_length(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        length = int(value)
    except ValueError as exc:
        raise ValueError("响应 Content-Length 非法") from exc
    if length < 0:
        raise ValueError("响应 Content-Length 不能为负数")
    return length


def _request_error(exc: Exception, timeout: int) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return f"请求超时（>{timeout}s）"
    if isinstance(exc, httpx.ConnectError):
        return "无法建立连接"
    return f"请求失败：{exc}"


@asynccontextmanager
async def _open_stream(
    requester: Any,
    url: str,
    *,
    timeout: int,
    fmt: str,
) -> AsyncIterator[Any]:
    """优先使用有界 HttpRequester stream，旧测试 fake 走兼容适配。"""

    stream = getattr(requester, "stream", None)
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": _ACCEPT.get(fmt, "*/*"),
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    is_requester_mock = type(requester).__module__.startswith("unittest.mock")
    stream_context = (
        stream(
            "GET",
            url,
            timeout_s=timeout,
            budget=RequestBudget(total_timeout_s=float(timeout)),
            headers=headers,
            validate_redirects=True,
        )
        if callable(stream) and not is_requester_mock
        else None
    )
    is_mock_context = type(stream_context).__module__.startswith("unittest.mock")
    is_protocol_context = not is_mock_context and (
        isinstance(stream_context, AbstractAsyncContextManager)
        or (
            stream_context is not None
            and hasattr(type(stream_context), "__aenter__")
            and hasattr(type(stream_context), "__aexit__")
        )
    )
    if isinstance(requester, HttpRequester) or is_protocol_context:
        assert stream_context is not None
        async with cast(AbstractAsyncContextManager[Any], stream_context) as response:
            yield response
        return

    response = await requester.get(
        url,
        follow_redirects=False,
        timeout_s=timeout,
        budget=RequestBudget(total_timeout_s=float(timeout)),
        headers=headers,
    )
    if not hasattr(response, "aiter_bytes"):
        content = bytes(response.content)

        async def _iter_bytes(*, chunk_size: int):
            for index in range(0, len(content), chunk_size):
                yield content[index : index + chunk_size]

        response.aiter_bytes = _iter_bytes
    yield response


def _to_markdown(raw_html: str) -> str:
    """HTML → Markdown，对应 OpenCode TS 的 TurndownService"""
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.body_width = 0  # 禁止自动折行
    h.unicode_snob = True  # 保留 Unicode 字符
    h.protect_links = True  # 防止链接被转义
    return h.handle(raw_html).strip()


def _to_text(content: bytes) -> str:
    """HTML → 纯文本，对应 OpenCode Go 的 extractTextFromHTML（goquery）"""
    try:
        doc = lxml_html.fromstring(content)
    except ParserError:
        return content.decode("utf-8", errors="replace")

    # 移除噪声标签（对应 OpenCode：script/style/noscript/iframe/object/embed）
    for tag in ("script", "style", "noscript", "iframe", "object", "embed"):
        for el in doc.xpath(f"//{tag}"):
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)

    # 合并空白（对应 OpenCode Go：strings.Fields + Join）
    return " ".join(doc.text_content().split())
