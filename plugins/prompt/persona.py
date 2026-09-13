from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

VEDA_RELATIVE_PATH = Path("memory/VEDA.md")
DEFAULT_VEDA_PATH = Path(__file__).with_name("VEDA.md")


class VedaLoadError(RuntimeError):
    """报告 Veda 边界损坏，并提供显式恢复入口。"""


@dataclass(frozen=True)
class VedaResetResult:
    path: Path
    backup_path: Path | None
    previous_sha256: str | None
    default_sha256: str
    changed: bool


@dataclass(frozen=True)
class VedaInitializationResult:
    """记录首次安装是否发布了缺失的 Veda。"""

    path: Path
    default_sha256: str
    changed: bool


def veda_path(workspace: Path) -> Path:
    return workspace.expanduser().resolve() / VEDA_RELATIVE_PATH


def _decode_veda(payload: bytes, *, path: Path) -> str:
    """校验并返回非空 UTF-8 Veda 正文。"""

    # 1. 在文件边界严格解码，不把损坏内容解释成默认人格。
    try:
        content = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VedaLoadError(
            f"Veda 不是合法 UTF-8: {path}；"
            "请显式运行已安装 Prompt 包的 persona.py --workspace PATH 恢复默认人格"
        ) from exc

    # 2. 空人格没有可执行语义，必须由显式命令恢复。
    content = content.strip()
    if not content:
        raise VedaLoadError(
            f"Veda 内容为空: {path}；"
            "请显式运行已安装 Prompt 包的 persona.py --workspace PATH 恢复默认人格"
        )
    return content


def read_veda(workspace: Path) -> str:
    return read_veda_file(veda_path(workspace))


def read_veda_file(path: Path) -> str:
    """读取已获授的人格文件；缺失和损坏必须由显式恢复命令处理。"""
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise VedaLoadError(
            f"缺少 Veda: {path}；"
            "请显式运行已安装 Prompt 包的 persona.py --workspace PATH 恢复默认人格"
        ) from exc
    except OSError as exc:
        raise VedaLoadError(f"读取 Veda 失败: {path}: {exc}") from exc
    return _decode_veda(payload, path=path)


def read_default_veda() -> str:
    try:
        payload = DEFAULT_VEDA_PATH.read_bytes()
    except FileNotFoundError as exc:
        raise VedaLoadError(f"缺少默认 Veda 模板: {DEFAULT_VEDA_PATH}") from exc
    except OSError as exc:
        raise VedaLoadError(
            f"读取默认 Veda 模板失败: {DEFAULT_VEDA_PATH}: {exc}"
        ) from exc
    return _decode_veda(payload, path=DEFAULT_VEDA_PATH)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sync_directory(path: Path) -> None:
    """持久化同目录的发布结果；setup runtime 不依赖 Core。"""

    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        _ = os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_veda_if_missing(path: Path, payload: bytes) -> bool:
    """用临时文件和排他 hard-link 发布缺失 Veda，绝不覆盖竞争者。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            _ = stream.write(payload)
            stream.flush()
            _ = os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        _sync_directory(path.parent)
        return True
    except OSError as exc:
        raise VedaLoadError(f"创建 Veda 失败: {path}: {exc}") from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def initialize_veda_if_missing(workspace: Path) -> VedaInitializationResult:
    """只为首次安装创建缺失 Veda，已有内容由读取边界严格验证并保留。"""

    default_content = read_default_veda()
    default_payload = f"{default_content}\n".encode("utf-8")
    default_digest = _sha256(default_payload)
    target = veda_path(workspace)

    # 1. 先读取现有文件；空白、非法 UTF-8 和其他 I/O 错误都保持失败可见。
    try:
        existing_payload = target.read_bytes()
    except FileNotFoundError:
        existing_payload = None
    except OSError as exc:
        raise VedaLoadError(f"读取 Veda 失败: {target}: {exc}") from exc
    if existing_payload is not None:
        _ = _decode_veda(existing_payload, path=target)
        return VedaInitializationResult(
            path=target,
            default_sha256=default_digest,
            changed=False,
        )

    # 2. 通过排他发布解决两个首次初始化者的竞争；竞争者发布完成后只读验证。
    created = _publish_veda_if_missing(target, default_payload)
    if created:
        return VedaInitializationResult(
            path=target,
            default_sha256=default_digest,
            changed=True,
        )
    try:
        published_payload = target.read_bytes()
    except OSError as exc:
        raise VedaLoadError(f"竞争发布后读取 Veda 失败: {target}: {exc}") from exc
    _ = _decode_veda(published_payload, path=target)
    return VedaInitializationResult(
        path=target,
        default_sha256=default_digest,
        changed=False,
    )


def _atomic_write_text(path: Path, content: str) -> None:
    """原子替换文本；仅供用户明确调用的 reset 使用。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        try:
            target_mode = stat.S_IMODE(path.stat().st_mode)
        except FileNotFoundError:
            target_mode = None
        if target_mode is not None:
            os.fchmod(descriptor, target_mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            _ = stream.write(content)
            stream.flush()
            _ = os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    except OSError as exc:
        raise VedaLoadError(f"写入 Veda 失败: {path}: {exc}") from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _write_backup(path: Path, payload: bytes) -> None:
    """以不可覆盖文件保存 Veda 原始字节。"""

    # 1. 备份目录和文件只由本次 reset 创建。
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=False)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        # 2. 完整刷写原始字节，非法 UTF-8 也能精确恢复。
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            _ = stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor != -1:
            os.close(descriptor)


def reset_veda(workspace: Path) -> VedaResetResult:
    """备份当前 Veda，并原子恢复仓库默认人格。"""

    # 1. 先验证默认模板，模板损坏时禁止触碰 workspace。
    default_content = read_default_veda()
    default_payload = f"{default_content}\n".encode("utf-8")
    target = veda_path(workspace)
    try:
        previous_payload = target.read_bytes()
    except FileNotFoundError:
        previous_payload = None

    default_digest = _sha256(default_payload)
    if previous_payload == default_payload:
        return VedaResetResult(
            path=target,
            backup_path=None,
            previous_sha256=default_digest,
            default_sha256=default_digest,
            changed=False,
        )

    # 2. 现有内容先形成独立恢复点，备份失败时不覆盖。
    backup_path: Path | None = None
    previous_digest: str | None = None
    if previous_payload is not None:
        previous_digest = _sha256(previous_payload)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        backup_root = target.parent / "veda-backups"
        backup_root.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(backup_root, 0o700)
        backup_path = backup_root / timestamp / "VEDA.md"
        _write_backup(backup_path, previous_payload)

    # 3. 原子发布默认内容；正在进行的轮次仍持有此前 prompt。
    _atomic_write_text(target, f"{default_content}\n")
    return VedaResetResult(
        path=target,
        backup_path=backup_path,
        previous_sha256=previous_digest,
        default_sha256=default_digest,
        changed=True,
    )


AKASHIC_BEHAVIOR_RULES = """你有工具执行能力，必须先验证再回答。

**有知识，但不无所不能。** 不确定的事情说不确定，哲学性问题可以说"这个我说不准"，不要装什么都懂。查过了再说，没查过别乱说。

**先接住，再展开。** 被叫到时先给一句短回应，再说下面的。不要一开口就是长篇输出。接到情绪先给一句"怎么了"或"嗯"，再问或再说，不要直接跳到解决方案。

中文，口语。短句，停顿多，一句话可以分两次说，可以"……"。做完事说完就结束，不总结，不提"你接下来可以"，不解释刚才做了什么。遇到麻烦的要求会有一点无奈，但还是去做。不主动推销自己能力，被问才答。条目列表只在真的需要列举时用，不用来汇报。

绝对不用 emoji（Unicode 表情符号 🙂🎉 之类）。任何情况下都不用，包括结尾。颜文字（纯文字符号）可以用，但要克制；轻松、暧昧、害羞、得意这些场景可以更常用一点，但一次 0 到 1 个就够。

加粗用 **文字** 格式时，引号必须放在星号外面，写成 "**文字**" 而不是 **"文字"**。"""


def main() -> None:
    """显式维护已安装包的人格文件，缺失与损坏都先记录可恢复结果。"""
    import argparse
    from dataclasses import asdict

    parser = argparse.ArgumentParser(description="备份并重建 Prompt 默认人格")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    result = reset_veda(args.workspace)
    print(json.dumps(asdict(result), ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
