"""standard_tools 拥有的模型文件工具 schema。"""

from pathlib import Path
from typing import Any

from agent.host_bridge.filesystem import (
    EditFileOperation,
    ListDirOperation,
    ReadFileOperation,
    WriteFileOperation,
)
from agent.tool_catalog import ToolResult


class ReadFileTool(ReadFileOperation):
    """读取文件内容，支持按行分页，超大文件自动截断。"""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "读取文件内容。文本文件输出带行号格式（如 '     1→内容'），便于 edit_file 精确定位。"
            "图片文件由多模态模型直接查看；若非多模态，会提示使用 read_image_vision 工具。\n"
            "文本读取默认受 400 行和 10KB 双重上限保护；大文件须用 limit 分页，不要依赖自动截断后的续读。\n\n"
            "推荐策略：先 limit=50 预览文件结构，再按需读取目标行段（offset=N limit=M）。\n"
            "明显二进制文件不会按文本硬解码，会提示改用 shell 查看。\n"
            "并行读取：可在同一次响应中同时读取多个文件，无需逐一等待。\n"
            "参数说明：offset=跳过的行数（0-based），limit=读取行数；二者仅对文本文件生效。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要读取的文件路径",
                },
                "offset": {
                    "type": "integer",
                    "description": "起始行号（0-based），默认 0",
                    "minimum": 0,
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "最多读取行数，默认不限（受 80K 字符上限约束）",
                    "minimum": 1,
                },
            },
            "required": ["path"],
        }
    async def execute(self, path: str, **kwargs: Any) -> str | ToolResult:
        return await self.read_raw(path, **kwargs)


class WriteFileTool(WriteFileOperation):
    """将内容写入文件，自动创建所需的父目录。"""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "将内容写入文件（完整覆盖写）。不存在的父目录自动创建。\n\n"
            "使用规则：\n"
            "- 优先使用 edit_file 修改已有文件；仅在创建新文件或完整重写时使用 write_file\n"
            "- 写入已存在的文件前，必须先用 read_file 读取当前内容，禁止盲写\n"
            "- 不得主动创建文档文件（*.md、README）除非用户明确要求\n"
            "- 写入路径须为绝对路径或相对工作目录的合法路径"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要写入的文件路径"},
                "content": {"type": "string", "description": "要写入的文本内容"},
            },
            "required": ["path", "content"],
        }


class EditFileTool(EditFileOperation):
    """精确替换文件中的指定文本片段。"""

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "将文件中的 old_text 精确替换为 new_text。\n\n"
            "重要：old_text 和 new_text 是文件的原始内容，不包含 read_file 输出的行号前缀。\n"
            "从 read_file 输出复制 old_text 时，必须去掉行首的 '     N→' 前缀，只保留实际文本内容。\n"
            "old_text 必须与文件内容完全一致（含缩进和换行）。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要编辑的文件路径"},
                "old_text": {
                    "type": "string",
                    "description": "要查找并替换的原始文本（必须与文件内容完全一致，不含行号前缀）",
                },
                "new_text": {"type": "string", "description": "替换后的新文本"},
                "replace_all": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "是否替换文件中所有匹配项，默认 False（只替换第一处）。"
                        "重命名变量、批量修改相同字符串时设为 true。"
                        "不确定匹配数量时先省略，收到'出现N次'警告后再决定。"
                    ),
                },
            },
            "required": ["path", "old_text", "new_text"],
        }


class ListDirTool(ListDirOperation):
    """列举目录内容。"""

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "列举指定目录下的文件和子目录。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要列举的目录路径"}
            },
            "required": ["path"],
        }
