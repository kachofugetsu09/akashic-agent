"""
WebFetch 工具
"""
import ipaddress
import json
from typing import Any
from urllib.parse import urlparse

import html2text
import httpx
from lxml import html as lxml_html
from lxml.etree import ParserError

from agent.tools.base import Tool

_MAX_BYTES = 5 * 1024 * 1024  # 5MB，与 OpenCode 一致
_DEFAULT_TIMEOUT = 30          # 秒
_MAX_TIMEOUT = 120             # 秒，与 OpenCode 一致
_USER_AGENT = "akasic/1.0"

# 根据 format 设置 Accept header，引导服务端返回更合适的格式
_ACCEPT = {
    "markdown": "text/markdown;q=1.0, text/x-markdown;q=0.9, text/plain;q=0.8, text/html;q=0.7, */*;q=0.1",
    "text":     "text/plain;q=1.0, text/markdown;q=0.9, text/html;q=0.8, */*;q=0.1",
    "html":     "text/html;q=1.0, application/xhtml+xml;q=0.9, text/plain;q=0.8, */*;q=0.1",
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

    async def execute(self, **kwargs: Any) -> str:
        url: str = kwargs["url"]
        fmt: str = kwargs.get("format", "markdown")
        timeout: int = min(int(kwargs.get("timeout", _DEFAULT_TIMEOUT)), _MAX_TIMEOUT)

        # URL 安全校验
        if not url.startswith(("http://", "https://")):
            return _err(url, "URL 必须以 http:// 或 https:// 开头")
        ssrf_err = _validate_url_target(url)
        if ssrf_err:
            return _err(url, ssrf_err)

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
                resp = await client.get(
                    url,
                    headers={
                        "User-Agent": _USER_AGENT,
                        "Accept": _ACCEPT.get(fmt, "*/*"),
                        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                    },
                )
        except httpx.TimeoutException:
            return _err(url, f"请求超时（>{timeout}s）")
        except httpx.ConnectError:
            return _err(url, "无法建立连接")
        except httpx.RequestError as e:
            return _err(url, f"请求失败：{e}")

        if resp.status_code != 200:
            return _err(url, f"HTTP {resp.status_code}")

        # 5MB 双重检查：先看 Content-Length header，再看实际 body
        cl = resp.headers.get("content-length")
        if cl and int(cl) > _MAX_BYTES:
            return _err(url, "响应过大（超过 5MB 限制）")

        body = resp.content
        if len(body) > _MAX_BYTES:
            return _err(url, "响应过大（超过 5MB 限制）")

        content_type = resp.headers.get("content-type", "")
        encoding = resp.encoding or "utf-8"
        is_html = "text/html" in content_type

        if fmt == "html":
            text = body.decode(encoding, errors="replace")
        elif fmt == "markdown" and is_html:
            text = _to_markdown(body.decode(encoding, errors="replace"))
        elif fmt == "text" and is_html:
            text = _to_text(body)
        else:
            # 非 HTML 内容（JSON、纯文本等）直接返回原文
            text = body.decode(encoding, errors="replace")

        return json.dumps(
            {
                "url": url,
                "final_url": str(resp.url),
                "status": resp.status_code,
                "content_type": content_type,
                "format": fmt,
                "length": len(text),
                "text": text,
            },
            ensure_ascii=False,
        )


# ── 模块级工具函数 ────────────────────────────────────────────

def _err(url: str, msg: str) -> str:
    return json.dumps({"error": msg, "url": url}, ensure_ascii=False)


def _validate_url_target(url: str) -> str | None:
    """SSRF 防护：拒绝内网/回环/保留地址。"""
    parsed = urlparse(url)
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return "URL 缺少主机名"
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved:
            return f"禁止访问内网/本地地址：{host}"
    except ValueError:
        if host.endswith(".local") or host.endswith(".localhost"):
            return f"禁止访问本地域名：{host}"
    return None


def _to_markdown(raw_html: str) -> str:
    """HTML → Markdown，对应 OpenCode TS 的 TurndownService"""
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.body_width = 0        # 禁止自动折行
    h.unicode_snob = True   # 保留 Unicode 字符
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
