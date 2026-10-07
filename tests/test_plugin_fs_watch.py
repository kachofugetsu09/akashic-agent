"""InotifyTreeWatcher：递归树事件、排除规则、文件目标过滤与缺失根退化。"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from agent.plugins.fs_watch import InotifyTreeWatcher

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="inotify 只在 Linux 可用"
)

_EXCLUDED = {".git", "__pycache__", "node_modules"}


async def _wait_for(flag: asyncio.Event, seconds: float = 5.0) -> bool:
    try:
        await asyncio.wait_for(flag.wait(), seconds)
    except TimeoutError:
        return False
    return True


async def _make_watcher(
    flag: asyncio.Event, *args: object, **kwargs: object
) -> InotifyTreeWatcher:
    watcher = InotifyTreeWatcher(flag.set, exclude=lambda name: name in _EXCLUDED)
    assert watcher.start()
    if args or kwargs:
        watcher.set_targets(*args, **kwargs)  # type: ignore[arg-type]
    return watcher


async def test_tree_create_modify_delete(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    root.mkdir()
    flag = asyncio.Event()
    watcher = await _make_watcher(flag, trees=[root], files=[])
    try:
        target = root / "plugin.py"
        target.write_text("x = 1")
        assert await _wait_for(flag)
        flag.clear()
        target.write_text("x = 2")
        assert await _wait_for(flag)
        flag.clear()
        target.unlink()
        assert await _wait_for(flag)
    finally:
        watcher.close()


async def test_excluded_names_do_not_fire(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    (root / "__pycache__").mkdir(parents=True)
    flag = asyncio.Event()
    watcher = await _make_watcher(flag, trees=[root], files=[])
    try:
        (root / "__pycache__" / "plugin.cpython-312.pyc").write_bytes(b"x")
        assert not await _wait_for(flag, 0.3)
        (root / "plugin.py").write_text("x = 1")
        assert await _wait_for(flag)
    finally:
        watcher.close()


async def test_new_subdirectory_is_watched(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    root.mkdir()
    flag = asyncio.Event()
    watcher = await _make_watcher(flag, trees=[root], files=[])
    try:
        child = root / "nested" / "deep"
        child.mkdir(parents=True)
        assert await _wait_for(flag)
        flag.clear()
        # 新建目录纳入监听后，其中的写入也触发回调。
        await asyncio.sleep(0.1)
        (child / "new.py").write_text("x = 1")
        assert await _wait_for(flag)
    finally:
        watcher.close()


async def test_file_target_filters_sibling_writes(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    config = data / "config.input.json"
    flag = asyncio.Event()
    watcher = await _make_watcher(flag, trees=[], files=[config])
    try:
        # 同目录的其他运行时写入不触发；只有目标文件触发。
        (data / "runtime-state.json").write_text("{}")
        assert not await _wait_for(flag, 0.3)
        config.write_text("{}")
        assert await _wait_for(flag)
    finally:
        watcher.close()


async def test_missing_root_falls_back_to_parent_watch(tmp_path: Path) -> None:
    missing = tmp_path / "later"
    flag = asyncio.Event()
    watcher = await _make_watcher(flag, trees=[missing], files=[])
    try:
        missing.mkdir()
        assert await _wait_for(flag)
        flag.clear()
        # set_targets 后根已存在，递归监听覆盖其内容。
        watcher.set_targets([missing], [])
        (missing / "plugin.py").write_text("x = 1")
        assert await _wait_for(flag)
    finally:
        watcher.close()


async def test_close_stops_events(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    root.mkdir()
    flag = asyncio.Event()
    watcher = await _make_watcher(flag, trees=[root], files=[])
    watcher.close()
    (root / "plugin.py").write_text("x = 1")
    assert not await _wait_for(flag, 0.3)
