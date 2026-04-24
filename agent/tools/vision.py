"""视觉工具：使用独立的 VL 模型分析图片，返回文本描述。"""

import base64
import io
import mimetypes
import os
from pathlib import Path
from typing import Any

from agent.provider import LLMProvider
from agent.tools.base import Tool
from agent.tools.filesystem import _resolve_path

_VL_MAX_FILE_BYTES = 20 * 1024 * 1024        # 20MB 原始文件上限
_VL_MAX_DATA_URI_BYTES = 8 * 1024 * 1024     # 8MB data URI 上限（base64 编码后）
_VL_MAX_EDGE = 4096                           # 最长边像素上限，超限自动缩放


def _encode_image_data_uri(file_path: Path) -> str:
    """读取图片并编码为 data URI，大图自动缩放压缩。

    超限时抛出 ValueError 并给出可操作的错误信息。
    """
    file_size = os.path.getsize(file_path)
    if file_size > _VL_MAX_FILE_BYTES:
        raise ValueError(
            f"图片文件过大（{file_size / 1024 / 1024:.1f}MB），"
            f"上限为 {_VL_MAX_FILE_BYTES / 1024 / 1024:.0f}MB。"
            "请压缩图片后重试，或裁剪到只包含需要分析的区域。"
        )

    raw = file_path.read_bytes()
    b64 = base64.b64encode(raw).decode()
    mime, _ = mimetypes.guess_type(file_path.name)
    if not mime or not mime.startswith("image/"):
        mime = "image/png"

    if len(b64) <= _VL_MAX_DATA_URI_BYTES:
        return f"data:{mime};base64,{b64}"

    # 图片编码后仍然过大，用 Pillow 压缩
    try:
        from PIL import Image, ImageOps
    except ModuleNotFoundError:
        raise ValueError(
            f"图片编码后过大（{len(b64) / 1024 / 1024:.1f}MB base64），"
            "且当前环境未安装 Pillow，无法自动压缩。"
            "请手动压缩图片或安装 Pillow 后重试。"
        )

    with Image.open(file_path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "L"):
            canvas = Image.new("RGB", img.size, (255, 255, 255))
            alpha = img.getchannel("A") if "A" in img.getbands() else None
            canvas.paste(img.convert("RGB"), mask=alpha)
            img = canvas
        elif img.mode == "L":
            img = img.convert("RGB")

        if max(img.size) > _VL_MAX_EDGE:
            img.thumbnail((_VL_MAX_EDGE, _VL_MAX_EDGE))

        best: bytes | None = None
        for quality in (85, 75, 65, 55, 45):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            candidate = buf.getvalue()
            candidate_b64 = base64.b64encode(candidate).decode()
            best = candidate
            if len(candidate_b64) <= _VL_MAX_DATA_URI_BYTES:
                return f"data:image/jpeg;base64,{candidate_b64}"

    if best is None:
        raise ValueError("图片压缩失败")
    return f"data:image/jpeg;base64,{base64.b64encode(best).decode()}"


class ReadImageVisionTool(Tool):
    """使用 VL 模型分析图片，返回视觉理解结果。

    适用场景：主模型不支持多模态，需要单独调用视觉模型来识别图片内容。
    """

    def __init__(
        self,
        vl_provider: LLMProvider,
        vl_model: str,
        allowed_dir: Path | None = None,
    ):
        self._provider = vl_provider
        self._model = vl_model
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "read_image_vision"

    @property
    def description(self) -> str:
        return (
            "使用独立的视觉模型分析图片内容。主模型无法直接查看图片时使用此工具。"
            "你需要提供一个 prompt 来说明你想从图片中了解什么。\n\n"
            "参数说明：\n"
            "- path：图片文件的路径\n"
            "- prompt：描述你想从这张图片中了解什么内容，越具体越好。"
            "例如 '图中有什么文字？'、'描述这张图片中的物体和场景'、"
            "'这张表格中第3行的数据是什么？'\n\n"
            "限制：原始文件不超过20MB，超限图片会自动缩放至最宽/最高4096像素并压缩。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "图片文件的路径",
                },
                "prompt": {
                    "type": "string",
                    "description": "描述你想从图片中了解什么内容，越具体越好",
                },
            },
            "required": ["path", "prompt"],
        }

    async def execute(self, path: str, prompt: str, **kwargs: Any) -> str:
        try:
            file_path = _resolve_path(path, self._allowed_dir)
            if not file_path.exists():
                return f"错误：文件不存在：{path}"
            if not file_path.is_file():
                return f"错误：路径不是文件：{path}"

            data_uri = _encode_image_data_uri(file_path)
        except ValueError as e:
            return f"图片处理失败：{e}"
        except Exception as e:
            return f"读取图片文件失败：{e}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri, "detail": "high"},
                    },
                ],
            }
        ]

        try:
            response = await self._provider.chat(
                messages=messages,
                tools=[],
                model=self._model,
                max_tokens=2048,
            )
            if response.content:
                return response.content
            if response.thinking:
                return f"[VL 模型思考过程]\n{response.thinking}"
            return "视觉模型未返回任何内容，请尝试调整 prompt 后重试。"
        except Exception as e:
            return f"调用视觉模型失败：{e}"
