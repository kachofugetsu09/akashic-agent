"""供 Prompt 与文件工具共用的图片边界。"""

import base64
import io
from pathlib import Path

from PIL import Image, ImageOps


MAX_IMAGE_COUNT = 4
MAX_IMAGE_FILE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_TOTAL_BYTES = 40 * 1024 * 1024
MAX_IMAGE_DATA_URI_BYTES = 8 * 1024 * 1024
MAX_IMAGE_DATA_URI_TOTAL_BYTES = 16 * 1024 * 1024
MAX_IMAGE_EDGE = 4096
MAX_IMAGE_PIXELS = 40_000_000


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


def validate_image_attachment_budget(sizes: list[int]) -> None:
    """在取得 lease 前限制本轮图片数量和原始字节。"""

    if len(sizes) > MAX_IMAGE_COUNT:
        raise ValueError(f"每条消息最多可以添加 {MAX_IMAGE_COUNT} 张图片。")
    oversized = next((size for size in sizes if size > MAX_IMAGE_FILE_BYTES), None)
    if oversized is not None:
        raise ValueError(
            f"单张图片不能超过 {MAX_IMAGE_FILE_BYTES // 1024 // 1024}MB"
            f"（当前 {oversized / 1024 / 1024:.1f}MB）。"
        )
    total = sum(sizes)
    if total > MAX_IMAGE_TOTAL_BYTES:
        raise ValueError(
            f"每条消息的图片合计不能超过 "
            f"{MAX_IMAGE_TOTAL_BYTES // 1024 // 1024}MB"
            f"（当前 {total / 1024 / 1024:.1f}MB）。"
        )


def encode_image_data_uri(file_path: Path) -> str:
    """有界读取图片并转成经过解码验证的 data URI。"""

    file_size = file_path.stat().st_size
    validate_image_attachment_budget([file_size])
    raw = file_path.read_bytes()
    mime = detect_supported_image_mime(raw[:4096])
    if mime is None:
        raise ValueError("不支持的图片格式。仅支持 PNG、JPEG、GIF、BMP、WebP。")

    try:
        with Image.open(file_path) as image:
            _validate_image_pixels(image.width, image.height)
            image.verify()

        with Image.open(file_path) as image:
            _validate_image_pixels(image.width, image.height)
            image = ImageOps.exif_transpose(image)
            if image.mode not in ("RGB", "L"):
                canvas = Image.new("RGB", image.size, (255, 255, 255))
                alpha = image.getchannel("A") if "A" in image.getbands() else None
                canvas.paste(image.convert("RGB"), mask=alpha)
                image = canvas
            elif image.mode == "L":
                image = image.convert("RGB")

            raw_b64_len = len(base64.b64encode(raw))
            if max(image.size) > MAX_IMAGE_EDGE or raw_b64_len > MAX_IMAGE_DATA_URI_BYTES:
                image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))

            if raw_b64_len <= MAX_IMAGE_DATA_URI_BYTES and max(image.size) <= MAX_IMAGE_EDGE:
                buf = io.BytesIO()
                if mime == "image/jpeg":
                    image.save(buf, format="JPEG", quality=95, optimize=True)
                    clean_mime = "image/jpeg"
                else:
                    image.save(buf, format="PNG", optimize=True)
                    clean_mime = "image/png"
                clean_b64 = base64.b64encode(buf.getvalue())
                if len(clean_b64) <= MAX_IMAGE_DATA_URI_BYTES:
                    return f"data:{clean_mime};base64,{clean_b64.decode()}"

            compressed_b64_len = 0
            for quality in (85, 75, 65, 55, 45):
                buf = io.BytesIO()
                image.save(buf, format="JPEG", quality=quality, optimize=True)
                candidate_b64 = base64.b64encode(buf.getvalue())
                compressed_b64_len = len(candidate_b64)
                if compressed_b64_len <= MAX_IMAGE_DATA_URI_BYTES:
                    return f"data:image/jpeg;base64,{candidate_b64.decode()}"
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("图片文件无法解码或已损坏。请确认这是有效图片。") from exc

    raise ValueError(
        f"图片压缩后仍然过大（{compressed_b64_len / 1024 / 1024:.1f}MB base64），"
        f"上限为 {MAX_IMAGE_DATA_URI_BYTES / 1024 / 1024:.0f}MB。"
        "请继续压缩图片或裁剪到只包含需要分析的区域。"
    )


def _validate_image_pixels(width: int, height: int) -> None:
    """在像素解码前拒绝会显著放大内存的图片。"""

    pixels = width * height
    if pixels > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"图片像素过多（{width}×{height}），"
            f"上限为 {MAX_IMAGE_PIXELS // 1_000_000} 百万像素。"
            "请缩小图片或裁剪后重试。"
        )
