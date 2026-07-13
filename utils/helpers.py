from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """确保目录存在并返回原路径。"""
    path.mkdir(parents=True, exist_ok=True)
    return path
