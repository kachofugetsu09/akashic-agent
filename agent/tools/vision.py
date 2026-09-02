"""视觉工具：使用独立的 VL 模型分析图片，返回文本描述。"""

from pathlib import Path
from typing import Any

from agent.plugin_composition import (
    CHAT_MODELS,
    ModelError,
    ModelRequest,
    ModelRole,
)
from agent.media import encode_image_data_uri
from agent.plugins.snapshot import get_current_runtime_snapshot
from agent.tools.base import Tool
from agent.tools.filesystem import (
    _resolve_path,
)


class ReadImageVisionTool(Tool):
    """使用 VL 模型分析图片，返回视觉理解结果。

    适用场景：主模型不支持多模态，需要单独调用视觉模型来识别图片内容。
    """

    def __init__(
        self,
        allowed_dir: Path | None = None,
    ):
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

            data_uri = encode_image_data_uri(file_path)
        except ValueError as e:
            return f"图片处理失败：{e}"
        except OSError as e:
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
            snapshot = get_current_runtime_snapshot()
            if snapshot is None or snapshot.composition_root is None:
                raise RuntimeError("视觉工具必须在 exact Turn snapshot 内执行")
            chat_models = snapshot.composition_root.context.require(CHAT_MODELS)
            async with chat_models.execution() as execution:
                vision = execution.chat(ModelRole.VISION)
                response = await vision.complete(
                    ModelRequest(
                        messages=messages,
                        tools=[],
                        max_output_tokens=2048,
                        disable_reasoning=True,
                    )
                )
            if response.content:
                return response.content
            if response.thinking:
                return f"[VL 模型思考过程]\n{response.thinking}"
            return "视觉模型未返回任何内容，请尝试调整 prompt 后重试。"
        except ModelError as e:
            return f"调用视觉模型失败：{e}"
