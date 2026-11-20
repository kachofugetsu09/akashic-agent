from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4



class AttachmentStore:
    """为 channel 媒体文件提供统一的持久化落盘目录。"""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _resolve_root(self) -> Path:
        if self.root.is_symlink():
            raise ValueError(f"附件目录不能是符号链接: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError(f"附件目录不能是符号链接: {self.root}")
        if not os.access(self.root, os.W_OK):
            raise PermissionError(f"附件目录不可写: {self.root}")
        return self.root

    def create_persistent_path(self, prefix: str, suffix: str) -> Path:
        """只在配置的持久目录中分配路径，目录不可用时直接失败。"""

        root = self._resolve_root()
        return root / f"{prefix}{uuid4().hex}{suffix}"
