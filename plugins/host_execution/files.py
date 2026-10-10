"""物理文件操作由 HostExecution 创建，调用作用域持有并关闭资源。"""
from pathlib import Path
from plugins.host_execution.contract import FileOperation
from .bridge.filesystem import ReadFileOperation, WriteFileOperation, EditFileOperation, ListDirOperation
from .bridge.path_info import PathAccess


class Files:
    def open(self, name: str, allowed_dir: Path | None = None) -> FileOperation:
        return {"read_file": ReadFileOperation, "write_file": WriteFileOperation,
                "edit_file": EditFileOperation, "list_dir": ListDirOperation}[name](allowed_dir)

    def paths(self) -> PathAccess:
        return PathAccess()
