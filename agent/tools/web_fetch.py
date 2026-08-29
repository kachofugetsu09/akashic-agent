"""
WebFetch 工具
"""

import json
from typing import Any

import html2text
import httpx
from lxml import html as lxml_html
from lxml.etree import ParserError

from agent.tools.base import Tool
from core.net.http import (
    HttpRequester,
    RequestBudget,
    get_default_http_requester,
)

_MAX_BYTES = 5 * 1024 * 1024  # 5MB，与 OpenCode 一致
_DEFAULT_TIMEOUT = 30  # 秒
_MAX_TIMEOUT = 120  # 秒，与 OpenCode 一致
_USER_AGENT = "akashic/1.0"
_MAX_TEXT_CHARS = 50_000  # 返回给 LLM 的文本字符上限（约 ~12K tokens）

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
        "仅支持 HTTP/HTTPS，响应上限 5MB。"
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

    def __init__(self, requester: HttpRequester | None = None) -> None:
        self._requester = requester or get_default_http_requester("external_default")

    async def execute(self, **kwargs: Any) -> str:
        url: str = kwargs["url"]
        fmt: str = kwargs.get("format", "markdown")
        timeout: int = min(int(kwargs.get("timeout", _DEFAULT_TIMEOUT)), _MAX_TIMEOUT)

        # URL 结构校验；允许访问本机和内网 HTTP 服务。
        try:
            parsed_url = httpx.URL(url)
        except httpx.InvalidURL:
            return _err(url, "URL 格式无效")
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.host:
            return _err(url, "URL 必须以 http:// 或 https:// 开头")

        try:
            resp = await self._requester.get(
                url,
                follow_redirects=True,
                timeout_s=timeout,
                budget=RequestBudget(total_timeout_s=float(timeout)),
                headers={
                    "User-Agent": _USER_AGENT,
                    "Accept": _ACCEPT[fmt],
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                },
            )
        except httpx.TimeoutException:
            return _err(url, f"请求超时（>{timeout}s）")
        except httpx.ConnectError:
            return _err(url, "无法建立连接")
        except httpx.RequestError as exc:
            return _err(url, f"请求失败：{exc}")

        if resp.status_code != 200:
            return _err(url, f"HTTP {resp.status_code}")

        # 5MB 双重检查：先校验 Content-Length header，再检查实际 body。
        content_length = resp.headers.get("content-length")
        if content_length is not None:
            if not content_length.isascii() or not content_length.isdecimal():
                return _err(url, "响应 Content-Length 无效")
            if int(content_length) > _MAX_BYTES:
                return _err(url, "响应过大（超过 5MB 限制）")

        body = resp.content
        if len(body) > _MAX_BYTES:
            return _err(url, "响应过大（超过 5MB 限制）")

        content_type = resp.headers.get("content-type", "")
        encoding = resp.encoding or "utf-8"
        is_html = "text/html" in content_type
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
                url, f"不支持二进制内容（{content_type}），请使用能处理该格式的专用工具"
            )

        if fmt == "html":
            text = body.decode(encoding, errors="replace")
        elif fmt == "markdown" and is_html:
            text = _to_markdown(body.decode(encoding, errors="replace"))
        elif fmt == "text" and is_html:
            text = _to_text(body)
        else:
            # 非 HTML 内容（JSON、纯文本等）直接返回原文
            text = body.decode(encoding, errors="replace")

        # 截断过长文本，避免撑爆 LLM 上下文
        truncated = False
        if len(text) > _MAX_TEXT_CHARS:
            text = text[:_MAX_TEXT_CHARS]
            truncated = True

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


# ── 模块级工具函数 ────────────────────────────────────────────


def _err(url: str, msg: str) -> str:
    return json.dumps({"error": msg, "url": url}, ensure_ascii=False)


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
