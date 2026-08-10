from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "docker"
        / "host-runtime"
        / "verify_release_source.py"
    )
    spec = importlib.util.spec_from_file_location("verify_release_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(module: ModuleType, root: Path) -> tuple[Path, dict[str, object]]:
    document: dict[str, object] = {
        "schemaVersion": 1,
        "sourceCommit": "a" * 40,
        "sourceTree": "b" * 40,
        "sourceArchiveSha256": "c" * 64,
        "files": module.source_entries(root),
    }
    path = root / ".akashic-source-manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, document


def test_release_source_verifies_exact_archive_tree(tmp_path: Path) -> None:
    module = _module()
    (tmp_path / "agent.py").write_text("stable\n", encoding="utf-8")
    manifest, document = _manifest(module, tmp_path)

    assert (
        module.verify_release_source(
            tmp_path,
            manifest,
            expected_commit="a" * 40,
            expected_tree="b" * 40,
            expected_archive_sha256="c" * 64,
        )
        == document
    )


@pytest.mark.parametrize("mutation", ["changed", "extra"])
def test_release_source_rejects_dirty_or_untracked_context(
    tmp_path: Path, mutation: str
) -> None:
    module = _module()
    source = tmp_path / "agent.py"
    source.write_text("stable\n", encoding="utf-8")
    manifest, _ = _manifest(module, tmp_path)
    if mutation == "changed":
        source.write_text("dirty\n", encoding="utf-8")
    else:
        (tmp_path / "config.toml").write_text("secret=true\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Docker build context"):
        module.verify_release_source(
            tmp_path,
            manifest,
            expected_commit="a" * 40,
            expected_tree="b" * 40,
            expected_archive_sha256="c" * 64,
        )


def test_runtime_image_prefers_domestic_package_cache_with_archive_fallback() -> None:
    dockerfile = (
        Path(__file__).parents[1] / "docker" / "host-runtime" / "Dockerfile"
    ).read_text(encoding="utf-8")

    tuna = dockerfile.index("CacheServer = https://mirrors.tuna.tsinghua.edu.cn")
    ustc = dockerfile.index("CacheServer = https://mirrors.ustc.edu.cn")
    archive = dockerfile.index("Server = https://archive.archlinux.org/repos/")
    assert tuna < ustc < archive
    assert dockerfile.count("pacman --disable-download-timeout") == 2
    assert "https://pypi.tuna.tsinghua.edu.cn/simple" in dockerfile
    assert '--index-url "${AKASHIC_PYPI_INDEX_URL}"' in dockerfile
    assert "https://registry.npmmirror.com" in dockerfile
    assert '--registry "${AKASHIC_NPM_REGISTRY}"' in dockerfile
