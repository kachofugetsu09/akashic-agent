from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Generator, cast
from uuid import uuid4

from agent.plugins.generation import PluginGeneration
from agent.plugins.reload_journal import ReloadJournal
from agent.plugins.source_hash import file_hash, file_revision, source_revision


@dataclass(frozen=True, slots=True)
class _Artifact:
    source_type: str
    path: str
    source_revision: str
    config_revision: str
    config_hash: str
    config_path: str
    data_path: str
    entrypoint: str


class _ArtifactPins:
    """Keep exact plugin code and config while durable work still names them."""

    def __init__(self, workspace: Path, journal: ReloadJournal) -> None:
        self._workspace = workspace
        self._journal = journal

    @contextmanager
    def lock(self) -> Generator[None]:
        """Serialize durable pin changes with artifact deletion checks."""

        path = self._workspace / "runtime" / "artifact-pins" / ".lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        stream: IO[str] = path.open("a+", encoding="utf-8")
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()

    def pin(self, value: str, *, owner: str, name: str) -> str:
        """Copy one mutable artifact into the shared private pin store."""

        artifact = _decode_artifact(value)
        if artifact.source_type not in {"builtin", "installed"}:
            raise RuntimeError(
                f"artifact source type 无效: {artifact.source_type}"
            )
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        source = self._pin_source(artifact, owner=owner, name=name, digest=digest)
        config = self._pin_config(artifact, owner=owner, name=name, digest=digest)
        return _encode_artifact(
            _Artifact(
                source_type=artifact.source_type,
                path=str(source),
                source_revision=artifact.source_revision,
                config_revision=artifact.config_revision,
                config_hash=artifact.config_hash,
                config_path=str(config),
                data_path=artifact.data_path,
                entrypoint=artifact.entrypoint,
            )
        )

    def clean(self) -> None:
        """Delete only pins that no durable switch or hold can still reach."""

        values = self._journal._artifact_refs()
        artifacts = (
            self._workspace / "runtime" / "artifact-pins" / "artifacts"
        ).resolve(strict=False)
        configs = (
            self._workspace / "runtime" / "artifact-pins" / "configs"
        ).resolve(strict=False)
        artifact_paths = tuple(
            Path(_decode_artifact(value).path).resolve(strict=False)
            for value in values
        )
        config_paths = tuple(
            Path(_decode_artifact(value).config_path).resolve(strict=False).parent
            for value in values
        )
        self._clean_root(artifacts, artifact_paths)
        self._clean_root(configs, config_paths)

    def has(self, path: Path) -> bool:
        """Check whether any durable switch or hold keeps this path."""

        target = path.resolve(strict=False)
        return any(
            target == pinned
            or target.is_relative_to(pinned)
            or pinned.is_relative_to(target)
            for value in self._journal._artifact_refs()
            for pinned in (Path(_decode_artifact(value).path).resolve(strict=False),)
        )

    def switching(self, path: Path) -> bool:
        """Check whether an unfinished RootSwitch covers this path."""

        target = path.resolve(strict=False)
        return any(
            target == pinned
            or target.is_relative_to(pinned)
            or pinned.is_relative_to(target)
            for value in self._journal._switch_refs()
            for pinned in (Path(_decode_artifact(value).path).resolve(strict=False),)
        )

    def _pin_source(
        self,
        artifact: _Artifact,
        *,
        owner: str,
        name: str,
        digest: str,
    ) -> Path:
        """Return exact source, copying mutable builtin code only once."""

        source = Path(artifact.path)
        root = self._workspace / "runtime" / "artifact-pins" / "artifacts"
        resolved = source.resolve(strict=False)
        if artifact.source_type == "installed" or resolved.parent == root.resolve(
            strict=False
        ):
            self._check_source(source, artifact, owner=owner, name=name)
            return source
        target = root / digest
        if target.exists():
            self._check_source(target, artifact, owner=owner, name=name)
            return target

        self._check_source(source, artifact, owner=owner, name=name)
        root.mkdir(parents=True, exist_ok=True)
        temporary = root / f".{digest}-{uuid4().hex}.tmp"
        try:
            _copy_source(source, temporary)
            if source_revision(temporary) != artifact.source_revision:
                raise RuntimeError(f"builtin pin 内容漂移: {owner}:{name}")
            _sync_tree(temporary)
            try:
                os.replace(temporary, target)
            except FileExistsError:
                shutil.rmtree(temporary)
            _sync_dir(root)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        self._check_source(target, artifact, owner=owner, name=name)
        return target

    @staticmethod
    def _check_source(
        source: Path,
        artifact: _Artifact,
        *,
        owner: str,
        name: str,
    ) -> None:
        if source_revision(source) != artifact.source_revision:
            raise RuntimeError(f"artifact source 已漂移: {owner}:{name}")
        entrypoint = source / artifact.entrypoint
        if not entrypoint.is_file() or entrypoint.is_symlink():
            raise RuntimeError(f"artifact entrypoint 无效: {owner}:{entrypoint}")

    def _pin_config(
        self,
        artifact: _Artifact,
        *,
        owner: str,
        name: str,
        digest: str,
    ) -> Path:
        """Keep one private exact config file for crash recovery."""

        source = Path(artifact.config_path)
        root = self._workspace / "runtime" / "artifact-pins" / "configs"
        resolved = source.resolve(strict=False)
        pinned = (
            resolved.parent.parent == root.resolve(strict=False)
            and resolved.name == "config.local.toml"
        )
        if pinned:
            self._check_config(source, root, artifact, owner=owner, name=name)
            return source
        target_dir = root / digest
        target = target_dir / "config.local.toml"
        if target_dir.exists():
            self._check_config(target, root, artifact, owner=owner, name=name)
            return target
        if source.is_symlink():
            raise RuntimeError(f"artifact config 不能是符号链接: {owner}")
        if file_revision(source) != artifact.config_revision:
            raise RuntimeError(f"artifact config 在 pin 前已漂移: {owner}")
        self._check_config_hash(source, artifact, owner=owner, name=name)
        root.mkdir(parents=True, exist_ok=True)
        os.chmod(root, 0o700)
        temporary = root / f".{digest}-{uuid4().hex}.tmp"
        try:
            temporary.mkdir(mode=0o700)
            if source.is_file():
                _ = shutil.copyfile(source, temporary / "config.local.toml")
                os.chmod(temporary / "config.local.toml", 0o600)
            _sync_tree(temporary)
            try:
                os.replace(temporary, target_dir)
            except FileExistsError:
                shutil.rmtree(temporary)
            _sync_dir(root)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        self._check_config(target, root, artifact, owner=owner, name=name)
        return target

    @staticmethod
    def _check_config_hash(
        path: Path,
        artifact: _Artifact,
        *,
        owner: str,
        name: str,
    ) -> None:
        if file_hash(path) != artifact.config_hash:
            raise RuntimeError(f"artifact config pin 已漂移: {owner}:{name}")

    def _check_config(
        self,
        path: Path,
        root: Path,
        artifact: _Artifact,
        *,
        owner: str,
        name: str,
    ) -> None:
        """Validate private pin location, mode, and exact content."""

        if root.stat().st_mode & 0o777 != 0o700:
            raise RuntimeError(f"artifact config root 权限无效: {root}")
        if path.parent.stat().st_mode & 0o777 != 0o700:
            raise RuntimeError(f"artifact config pin 权限无效: {path.parent}")
        if path.exists() and (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_mode & 0o777 != 0o600
        ):
            raise RuntimeError(f"artifact config file 权限无效: {path}")
        self._check_config_hash(path, artifact, owner=owner, name=name)

    @staticmethod
    def _clean_root(root: Path, paths: tuple[Path, ...]) -> None:
        """Remove unreferenced direct children from one private pin root."""

        if not root.exists():
            return
        kept = {path for path in paths if path.parent == root}
        for path in root.iterdir():
            resolved = path.resolve(strict=False)
            if resolved.parent != root:
                raise RuntimeError(f"artifact pin path 越界: {resolved}")
            if resolved not in kept:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()


def _artifact_value(generation: PluginGeneration) -> str:
    """Name one exact code and config artifact without plugin input."""

    config_path = generation.config_path or (
        generation.data_dir / "config.local.toml"
    )
    return json.dumps(
        {
            "config_hash": file_hash(config_path),
            "config_path": str(config_path.resolve(strict=False)),
            "config_revision": generation.config_revision,
            "data_path": str(generation.data_dir.resolve(strict=False)),
            "entrypoint": generation.entrypoint,
            "source_type": generation.source_type,
            "path": str(generation.plugin_dir.resolve(strict=False)),
            "source_revision": generation.source_revision,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _decode_artifact(value: str) -> _Artifact:
    """Decode one exact artifact identity from durable storage."""

    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as error:
        raise RuntimeError("artifact identity 不是 JSON") from error
    if not isinstance(loaded, dict):
        raise RuntimeError("artifact identity 结构无效")
    item = cast(dict[str, object], loaded)
    keys = {
        "source_type",
        "path",
        "source_revision",
        "config_revision",
        "config_hash",
        "config_path",
        "data_path",
        "entrypoint",
    }
    if set(item) != keys:
        raise RuntimeError("artifact identity 结构无效")
    return _Artifact(
        source_type=_stored_text(item["source_type"], "source type"),
        path=_stored_text(item["path"], "path"),
        source_revision=_stored_text(item["source_revision"], "source revision"),
        config_revision=_stored_text(item["config_revision"], "config revision"),
        config_hash=_stored_text(item["config_hash"], "config hash"),
        config_path=_stored_text(item["config_path"], "config path"),
        data_path=_stored_text(item["data_path"], "data path"),
        entrypoint=_stored_text(item["entrypoint"], "entrypoint"),
    )


def _encode_artifact(value: _Artifact) -> str:
    return json.dumps(
        {
            "config_hash": value.config_hash,
            "config_path": value.config_path,
            "config_revision": value.config_revision,
            "data_path": value.data_path,
            "entrypoint": value.entrypoint,
            "path": value.path,
            "source_revision": value.source_revision,
            "source_type": value.source_type,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _copy_source(source: Path, target: Path) -> None:
    excluded = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
    }
    _ = shutil.copytree(
        source,
        target,
        symlinks=True,
        ignore=lambda _path, names: sorted(excluded.intersection(names)),
    )


def _sync_tree(root: Path) -> None:
    for current, directories, filenames in os.walk(root, topdown=False):
        for name in filenames:
            path = Path(current) / name
            if path.is_symlink():
                continue
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        for name in directories:
            path = Path(current) / name
            if not path.is_symlink():
                _sync_dir(path)
    _sync_dir(root)


def _sync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stored_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RuntimeError(f"artifact {label} 无效")
    return value
