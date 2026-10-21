"""HTTP 抓取提供方：直接请求 URL，按格式转换并截断正文。"""

from typing import Any

import html2text
import httpx
from lxml import html as lxml_html
from lxml.etree import ParserError
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from plugins.standard_web.contract import FetchRequest
from core.net.http import (
    HttpRequester,
    RequestBudget,
    get_default_http_requester,
    http_error_detail,
    redact_http_urls,
)



class FetchConfig(BaseModel):
    """读取上限、请求头与代理；proxy 为空时沿用系统 HTTP(S)_PROXY 环境变量。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_response_mb: float = Field(default=5, gt=0, le=100, allow_inf_nan=False)
    # 返回给模型的文本字符上限；默认约 12K tokens。
    max_text_chars: int = Field(default=50_000, ge=1_000, le=1_000_000)
    user_agent: str = Field(default="akashic/1.0", min_length=1, max_length=300)
    accept_language: str = Field(default="zh-CN,zh;q=0.9,en;q=0.8", min_length=1, max_length=200)
    proxy: HttpUrl | None = None

    @field_validator("proxy")
    @classmethod
    def check_proxy(cls, value: HttpUrl | None) -> HttpUrl | None:
        if value is not None and value.scheme not in {"http", "https", "socks5", "socks5h"}:
            raise ValueError("代理地址必须以 http://、https:// 或 socks5:// 开头")
        return value


# 根据 format 设置 Accept header，引导服务端返回更合适的格式
_ACCEPT = {
    "markdown": "text/markdown;q=1.0, text/x-markdown;q=0.9, text/plain;q=0.8, text/html;q=0.7, */*;q=0.1",
    "text": "text/plain;q=1.0, text/markdown;q=0.9, text/html;q=0.8, */*;q=0.1",
    "html": "text/html;q=1.0, application/xhtml+xml;q=0.9, text/plain;q=0.8, */*;q=0.1",
}


class HttpFetch:
    """直接 HTTP(S) 抓取，支持 text / markdown / html 三种格式输出。"""

    def __init__(self, requester: HttpRequester | None = None, config: FetchConfig | None = None) -> None:
        self._requester = requester or get_default_http_requester("external_default")
        self._config = config or FetchConfig()

    async def fetch(self, request: FetchRequest) -> dict[str, object]:
        url, fmt, timeout = request.url, request.format, request.timeout_s
        limit, max_chars = int(self._config.max_response_mb * 1024 * 1024), self._config.max_text_chars
        too_large = f"响应过大（超过 {self._config.max_response_mb:g}MB 限制）"

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
                    "User-Agent": self._config.user_agent,
                    "Accept": _ACCEPT[fmt],
                    "Accept-Language": self._config.accept_language,
                },
            )
        except httpx.RequestError as exc:
            return _err(url, f"请求失败：{http_error_detail(exc)}")

        if resp.status_code != 200:
            return _err(url, f"HTTP {resp.status_code}")

        # 大小双重检查：先校验 Content-Length header，再检查实际 body。
        content_length = resp.headers.get("content-length")
        if content_length is not None:
            if not content_length.isascii() or not content_length.isdecimal():
                return _err(url, "响应 Content-Length 无效")
            if int(content_length) > limit:
                return _err(url, too_large)

        body = resp.content
        if len(body) > limit:
            return _err(url, too_large)

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
        if len(text) > max_chars:
            text = text[:max_chars]
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
                f"内容已截断至 {max_chars} 字符，如需更多内容请缩小范围或使用其他工具"
            )

        return result


# ── 模块级工具函数 ────────────────────────────────────────────


def _err(url: str, msg: str) -> dict[str, object]:
    return {"error": msg, "url": redact_http_urls(url)}


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
