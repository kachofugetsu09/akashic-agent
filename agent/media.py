"""供 Prompt 与文件工具共用的媒体类型检查。"""


def detect_supported_image_mime(head: bytes) -> str | None:
    """根据文件签名识别 Akashic 支持的图片格式。"""

    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if head.startswith(b"BM"):
        return "image/bmp"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "image/webp"
    return None
