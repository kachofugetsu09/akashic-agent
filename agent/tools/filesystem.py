"""文件系统工具：读取、写入、编辑文件，以及列举目录。"""

import asyncio
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from agent.tools.base import Tool

if TYPE_CHECKING:
    from memory2.sop_indexer import SopIndexer

logger = logging.getLogger(__name__)


def _read_xlsx(file_path: Path) -> str:
    import openpyxl
    wb = openpyxl.load_workbook(file_path, data_only=True)
    parts = []
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        parts.append(f"## Sheet: {sheet}")
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            if any(cells):
                parts.append("\t".join(cells))
    return "\n".join(parts)


def _read_xls(file_path: Path) -> str:
    import xlrd
    wb = xlrd.open_workbook(str(file_path))
    parts = []
    for sheet in wb.sheets():
        parts.append(f"## Sheet: {sheet.name}")
        for row_idx in range(sheet.nrows):
            cells = [str(sheet.cell_value(row_idx, col)) for col in range(sheet.ncols)]
            parts.append("\t".join(cells))
    return "\n".join(parts)


def _resolve_path(path: str, allowed_dir: Path | None = None) -> Path:
    """解析路径（展开 ~ 并取绝对路径），可选限制在允许目录内。"""
    resolved = Path(path).expanduser().resolve()
    if allowed_dir and not str(resolved).startswith(str(allowed_dir.resolve())):
        raise PermissionError(f"路径 {path} 超出允许目录 {allowed_dir}")
    return resolved


_READ_MAX_CHARS = 10_000   # 默认单次最多返回 10K 字符，大文件须分页读取


class ReadFileTool(Tool):
    """读取文件内容，支持按行分页，超大文件自动截断。"""

    def __init__(self, allowed_dir: Path | None = None):
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "读取文件内容。超过 10K 字符时自动截断并提示总行数，"
            "可用 offset/limit 参数分页读取（offset=起始行，limit=读取行数）。"
            "对于大文件，先用 offset=0 limit=30 预览前几行，再决定读哪段。"
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

    async def execute(self, path: str, **kwargs: Any) -> str:
        offset: int = int(kwargs.get("offset", 0))
        limit: int | None = kwargs.get("limit")
        if limit is not None:
            limit = int(limit)
        try:
            file_path = _resolve_path(path, self._allowed_dir)
            if not file_path.exists():
                return f"错误：文件不存在：{path}"
            if not file_path.is_file():
                return f"错误：路径不是文件：{path}"

            suffix = file_path.suffix.lower()
            if suffix == ".xlsx":
                return _read_xlsx(file_path)
            if suffix == ".xls":
                return _read_xls(file_path)

            lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
            total_lines = len(lines)

            # 按行分页
            sliced = lines[offset:] if limit is None else lines[offset: offset + limit]
            text = "".join(sliced)

            # 字符上限截断
            truncated_by_chars = False
            if len(text) > _READ_MAX_CHARS:
                text = text[:_READ_MAX_CHARS]
                truncated_by_chars = True

            suffix_note = ""
            end_line = offset + len(sliced)
            if truncated_by_chars:
                suffix_note = (
                    f"\n\n[已截断：文件共 {total_lines} 行，"
                    f"本次返回前 {_READ_MAX_CHARS} 字符。"
                    f"用 offset={end_line} 继续读取。]"
                )
            elif offset > 0 or limit is not None:
                suffix_note = (
                    f"\n\n[第 {offset+1}–{end_line} 行 / 共 {total_lines} 行]"
                )
            elif total_lines > len(sliced):
                suffix_note = f"\n\n[共 {total_lines} 行]"

            return text + suffix_note
        except PermissionError as e:
            return f"错误：{e}"
        except Exception as e:
            return f"读取文件失败：{e}"


class WriteFileTool(Tool):
    """将内容写入文件，自动创建所需的父目录。"""

    def __init__(self, allowed_dir: Path | None = None, sop_indexer: "SopIndexer | None" = None):
        self._allowed_dir = allowed_dir
        self._sop_indexer = sop_indexer

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "将内容写入指定路径的文件，不存在的父目录会自动创建。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要写入的文件路径"
                },
                "content": {
                    "type": "string",
                    "description": "要写入的文本内容"
                }
            },
            "required": ["path", "content"]
        }

    async def execute(self, path: str, content: str, **kwargs: Any) -> str:
        try:
            file_path = _resolve_path(path, self._allowed_dir)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            result = f"已写入 {len(content)} 字节到 {path}"
            if self._sop_indexer and self._sop_indexer.is_sop_file(file_path):
                asyncio.create_task(self._reindex(file_path))
            return result
        except PermissionError as e:
            return f"错误：{e}"
        except Exception as e:
            return f"写入文件失败：{e}"

    async def _reindex(self, file_path: Path) -> None:
        try:
            summary = await self._sop_indexer.reindex(file_path)
            logger.info(f"write_file sop reindex: {summary}")
        except Exception as e:
            logger.warning(f"write_file sop reindex 失败: {e}")


class EditFileTool(Tool):
    """精确替换文件中的指定文本片段。"""

    def __init__(self, allowed_dir: Path | None = None, sop_indexer: "SopIndexer | None" = None):
        self._allowed_dir = allowed_dir
        self._sop_indexer = sop_indexer

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "将文件中的 old_text 替换为 new_text，old_text 必须与文件内容完全匹配。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要编辑的文件路径"
                },
                "old_text": {
                    "type": "string",
                    "description": "要查找并替换的原始文本（必须与文件内容完全一致）"
                },
                "new_text": {
                    "type": "string",
                    "description": "替换后的新文本"
                }
            },
            "required": ["path", "old_text", "new_text"]
        }

    async def execute(self, path: str, old_text: str, new_text: str, **kwargs: Any) -> str:
        try:
            file_path = _resolve_path(path, self._allowed_dir)
            if not file_path.exists():
                return f"错误：文件不存在：{path}"

            content = file_path.read_text(encoding="utf-8")

            if old_text not in content:
                return "错误：未找到 old_text，请确保与文件内容完全一致。"

            # 若匹配到多处，提示提供更多上下文以唯一定位
            count = content.count(old_text)
            if count > 1:
                return f"警告：old_text 在文件中出现了 {count} 次，请提供更多上下文以唯一匹配。"

            new_content = content.replace(old_text, new_text, 1)
            file_path.write_text(new_content, encoding="utf-8")

            if self._sop_indexer and self._sop_indexer.is_sop_file(file_path):
                asyncio.create_task(self._reindex(file_path))
            return f"已成功编辑 {path}"
        except PermissionError as e:
            return f"错误：{e}"
        except Exception as e:
            return f"编辑文件失败：{e}"

    async def _reindex(self, file_path: Path) -> None:
        try:
            summary = await self._sop_indexer.reindex(file_path)
            logger.info(f"edit_file sop reindex: {summary}")
        except Exception as e:
            logger.warning(f"edit_file sop reindex 失败: {e}")


class ListDirTool(Tool):
    """列举目录内容。"""

    def __init__(self, allowed_dir: Path | None = None):
        self._allowed_dir = allowed_dir

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
                "path": {
                    "type": "string",
                    "description": "要列举的目录路径"
                }
            },
            "required": ["path"]
        }

    async def execute(self, path: str, **kwargs: Any) -> str:
        try:
            dir_path = _resolve_path(path, self._allowed_dir)
            if not dir_path.exists():
                return f"错误：目录不存在：{path}"
            if not dir_path.is_dir():
                return f"错误：路径不是目录：{path}"

            items = []
            for item in sorted(dir_path.iterdir()):
                prefix = "📁 " if item.is_dir() else "📄 "
                items.append(f"{prefix}{item.name}")

            if not items:
                return f"目录 {path} 为空"

            return "\n".join(items)
        except PermissionError as e:
            return f"错误：{e}"
        except Exception as e:
            return f"列举目录失败：{e}"
