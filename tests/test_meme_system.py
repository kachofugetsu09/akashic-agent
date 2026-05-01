"""Unit tests for plugins/meme/runtime.py."""
import json
import time
from pathlib import Path

import pytest

from plugins.meme.runtime import MemeCatalog, MemeDecorator


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def memes_dir(tmp_path: Path) -> Path:
    d = tmp_path / "memes"
    d.mkdir()
    return d


def write_manifest(memes_dir: Path, categories: dict) -> None:
    (memes_dir / "manifest.json").write_text(
        json.dumps({"version": 1, "categories": categories}), encoding="utf-8"
    )


def add_image(memes_dir: Path, category: str, name: str = "001.png") -> Path:
    cat_dir = memes_dir / category
    cat_dir.mkdir(exist_ok=True)
    img = cat_dir / name
    img.write_bytes(b"\x89PNG\r\n")  # fake PNG header
    return img


# ── MemeCatalog: basic loading ─────────────────────────────────────────────────


def test_catalog_no_manifest_returns_empty(memes_dir: Path) -> None:
    catalog = MemeCatalog(memes_dir)
    assert catalog.get_enabled_categories() == []
    assert catalog.build_prompt_block() is None


def test_catalog_loads_enabled_categories(memes_dir: Path) -> None:
    write_manifest(memes_dir, {
        "happy": {"desc": "开心", "aliases": [], "enabled": True},
        "angry": {"desc": "生气", "aliases": [], "enabled": True},
    })
    catalog = MemeCatalog(memes_dir)
    names = {c.name for c in catalog.get_enabled_categories()}
    assert names == {"happy", "angry"}


def test_catalog_filters_disabled_categories(memes_dir: Path) -> None:
    write_manifest(memes_dir, {
        "happy": {"desc": "开心", "enabled": True},
        "hidden": {"desc": "隐藏", "enabled": False},
    })
    catalog = MemeCatalog(memes_dir)
    names = {c.name for c in catalog.get_enabled_categories()}
    assert names == {"happy"}
    assert "hidden" not in names


def test_catalog_prompt_block_contains_categories(memes_dir: Path) -> None:
    write_manifest(memes_dir, {
        "agree": {"desc": "收到、同意", "enabled": True},
        "shy": {"desc": "害羞", "enabled": True},
    })
    catalog = MemeCatalog(memes_dir)
    block = catalog.build_prompt_block()
    assert block is not None
    assert "agree" in block
    assert "shy" in block
    assert "收到、同意" in block
    assert "<meme:category>" in block
    assert "<meme:shy>" in block
    assert "被夸" in block


# ── MemeCatalog: hot reload ────────────────────────────────────────────────────


def test_catalog_reloads_after_manifest_change(memes_dir: Path) -> None:
    write_manifest(memes_dir, {"happy": {"desc": "开心", "enabled": True}})
    catalog = MemeCatalog(memes_dir)
    assert {c.name for c in catalog.get_enabled_categories()} == {"happy"}

    # Ensure mtime changes (some filesystems have 1s resolution)
    time.sleep(0.01)
    manifest = memes_dir / "manifest.json"
    manifest.write_text(
        json.dumps({"version": 1, "categories": {
            "happy": {"desc": "开心", "enabled": True},
            "shy": {"desc": "害羞", "enabled": True},
        }}),
        encoding="utf-8",
    )
    # Touch to guarantee mtime change on low-resolution filesystems
    manifest.touch()

    names = {c.name for c in catalog.get_enabled_categories()}
    assert "shy" in names, "新增类别应在 manifest 更新后无需重启即可生效"


def test_catalog_clears_on_manifest_deletion(memes_dir: Path) -> None:
    """manifest 被删除后，旧类别不应继续生效。"""
    write_manifest(memes_dir, {"happy": {"desc": "开心", "enabled": True}})
    catalog = MemeCatalog(memes_dir)
    assert len(catalog.get_enabled_categories()) == 1

    (memes_dir / "manifest.json").unlink()

    assert catalog.get_enabled_categories() == []
    assert catalog.build_prompt_block() is None


def test_catalog_does_not_reload_without_mtime_change(memes_dir: Path) -> None:
    write_manifest(memes_dir, {"happy": {"desc": "开心", "enabled": True}})
    catalog = MemeCatalog(memes_dir)
    _ = catalog.get_enabled_categories()  # prime cache

    # Overwrite content but keep same mtime by restoring it
    manifest = memes_dir / "manifest.json"
    original_mtime = manifest.stat().st_mtime
    manifest.write_text(
        json.dumps({"version": 1, "categories": {
            "surprise": {"desc": "惊讶", "enabled": True},
        }}),
        encoding="utf-8",
    )
    import os
    os.utime(manifest, (original_mtime, original_mtime))

    names = {c.name for c in catalog.get_enabled_categories()}
    # mtime unchanged → cache should still be used
    assert "happy" in names
    assert "surprise" not in names


# ── MemeCatalog: pick_image ────────────────────────────────────────────────────


def test_pick_image_returns_none_for_unknown_tag(memes_dir: Path) -> None:
    write_manifest(memes_dir, {"happy": {"desc": "开心", "enabled": True}})
    catalog = MemeCatalog(memes_dir)
    assert catalog.pick_image("nonexistent") is None


def test_pick_image_returns_none_for_disabled_category(memes_dir: Path) -> None:
    write_manifest(memes_dir, {"happy": {"desc": "开心", "enabled": False}})
    add_image(memes_dir, "happy")
    catalog = MemeCatalog(memes_dir)
    assert catalog.pick_image("happy") is None


def test_pick_image_returns_none_for_empty_directory(memes_dir: Path) -> None:
    write_manifest(memes_dir, {"happy": {"desc": "开心", "enabled": True}})
    (memes_dir / "happy").mkdir()  # empty dir, no images
    catalog = MemeCatalog(memes_dir)
    assert catalog.pick_image("happy") is None


def test_pick_image_returns_path_when_image_exists(memes_dir: Path) -> None:
    write_manifest(memes_dir, {"happy": {"desc": "开心", "enabled": True}})
    img = add_image(memes_dir, "happy", "001.png")
    catalog = MemeCatalog(memes_dir)
    result = catalog.pick_image("happy")
    assert result == str(img)


def test_pick_image_case_insensitive(memes_dir: Path) -> None:
    write_manifest(memes_dir, {"happy": {"desc": "开心", "enabled": True}})
    add_image(memes_dir, "happy")
    catalog = MemeCatalog(memes_dir)
    assert catalog.pick_image("HAPPY") is not None
    assert catalog.pick_image("Happy") is not None


# ── MemeDecorator ─────────────────────────────────────────────────────────────


def _make_decorator(memes_dir: Path, category: str, with_image: bool = True) -> MemeDecorator:
    write_manifest(memes_dir, {category: {"desc": "test", "enabled": True}})
    if with_image:
        add_image(memes_dir, category)
    catalog = MemeCatalog(memes_dir)
    return MemeDecorator(catalog)


def test_decorator_no_tag_passes_through(memes_dir: Path) -> None:
    dec = _make_decorator(memes_dir, "happy")
    result = dec.decorate("普通回复，没有表情")
    assert result.content == "普通回复，没有表情"
    assert result.media == []


def test_decorator_uses_explicit_tag(memes_dir: Path) -> None:
    dec = _make_decorator(memes_dir, "agree")
    result = dec.decorate("好的，我这就去做", meme_tag="agree")
    assert result.content == "好的，我这就去做"
    assert len(result.media) == 1


def test_decorator_tag_only_gives_empty_content(memes_dir: Path) -> None:
    dec = _make_decorator(memes_dir, "happy")
    result = dec.decorate("", meme_tag="happy")
    assert result.content == ""
    assert len(result.media) == 1


def test_decorator_unknown_tag_has_no_media(memes_dir: Path) -> None:
    dec = _make_decorator(memes_dir, "happy")
    result = dec.decorate("发个不存在的表情", meme_tag="nonexistent")
    assert result.content == "发个不存在的表情"
    assert result.media == []
    assert result.tag == "nonexistent"


def test_decorator_empty_dir_has_no_media(memes_dir: Path) -> None:
    dec = _make_decorator(memes_dir, "shy", with_image=False)
    result = dec.decorate("害羞一下", meme_tag="shy")
    assert result.content == "害羞一下"
    assert result.media == []


def test_decorator_picks_image_for_parsed_tag(memes_dir: Path) -> None:
    write_manifest(memes_dir, {
        "happy": {"desc": "开心", "enabled": True},
        "agree": {"desc": "同意", "enabled": True},
    })
    add_image(memes_dir, "happy", "001.png")
    add_image(memes_dir, "agree", "001.png")
    catalog = MemeCatalog(memes_dir)
    dec = MemeDecorator(catalog)
    result = dec.decorate("好的  收到", meme_tag="happy")
    assert result.content == "好的  收到"
    assert len(result.media) == 1
    assert "happy" in result.media[0]


def test_decorator_case_insensitive_tag(memes_dir: Path) -> None:
    dec = _make_decorator(memes_dir, "agree")
    result = dec.decorate("收到", meme_tag="AGREE")
    assert result.content == "收到"
    assert len(result.media) == 1


def test_decorator_reflects_hot_reload(memes_dir: Path) -> None:
    """manifest 更新后新增类别，decorator 立即可用，无需重建实例。"""
    write_manifest(memes_dir, {"happy": {"desc": "开心", "enabled": True}})
    add_image(memes_dir, "happy")
    catalog = MemeCatalog(memes_dir)
    dec = MemeDecorator(catalog)

    # 新增 shy 类别
    time.sleep(0.01)
    manifest = memes_dir / "manifest.json"
    manifest.write_text(
        json.dumps({"version": 1, "categories": {
            "happy": {"desc": "开心", "enabled": True},
            "shy": {"desc": "害羞", "enabled": True},
        }}),
        encoding="utf-8",
    )
    manifest.touch()
    add_image(memes_dir, "shy")

    result = dec.decorate("害羞一下", meme_tag="shy")
    assert len(result.media) == 1, "热更新后新类别应立即可用"
