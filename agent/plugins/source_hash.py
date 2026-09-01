from __future__ import annotations

import hashlib
import os
from pathlib import Path


def file_revision(path: Path) -> str:
    """Hash one exact file identity and its current bytes."""

    digest = hashlib.sha256()
    digest.update(str(path.resolve(strict=False)).encode())
    digest.update(file_hash(path).encode())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    """Hash one file's bytes or its exact missing state."""

    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
    else:
        digest.update(b"<missing>")
    return digest.hexdigest()


def source_revision(plugin_dir: Path) -> str:
    """Hash one plugin source tree while skipping generated local caches."""

    digest = hashlib.sha256()
    root = plugin_dir.resolve(strict=False)
    excluded = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
    }
    for current, directories, filenames in os.walk(plugin_dir, followlinks=False):
        directories[:] = sorted(name for name in directories if name not in excluded)
        current_path = Path(current)
        for name in [*directories, *sorted(filenames)]:
            path = current_path / name
            relative = path.relative_to(plugin_dir)
            if path.is_symlink():
                resolved = path.resolve(strict=False)
                _inside(root, resolved)
                digest.update(str(relative).encode())
                digest.update(os.readlink(path).encode())
                if resolved.is_file():
                    digest.update(resolved.read_bytes())
                continue
            if not path.is_file():
                continue
            resolved = path.resolve(strict=False)
            _inside(root, resolved)
            digest.update(str(relative).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _inside(root: Path, path: Path) -> None:
    if path != root and not path.is_relative_to(root):
        raise ValueError(f"插件源码路径越界: {path}")
