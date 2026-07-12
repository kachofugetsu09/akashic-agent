from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import stat
from threading import Barrier

import pytest

import infra.persistence.json_store as json_store
from infra.persistence.json_store import atomic_save_json, atomic_write_text, load_json


def test_load_json_defaults_only_for_missing_file(tmp_path) -> None:
    path = tmp_path / "missing.json"
    assert load_json(path, default={"missing": True}) == {"missing": True}

    path.write_bytes(b"not json")
    with pytest.raises(RuntimeError, match=r"\[test.state\].*missing\.json"):
        load_json(path, default={"fallback": True}, domain="test.state")


def test_atomic_save_json_uses_isolated_temp_files(monkeypatch, tmp_path) -> None:
    path = tmp_path / "state.json"
    replace_barrier = Barrier(2)
    original_replace = Path.replace

    def synchronized_replace(source: Path, target: Path) -> Path:
        replace_barrier.wait(timeout=5)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", synchronized_replace)
    payloads = ({"writer": "a"}, {"writer": "b"})
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda payload: atomic_save_json(path, payload), payloads))

    assert json.loads(path.read_text(encoding="utf-8")) in payloads
    assert list(tmp_path.glob("state.json.*.tmp")) == []


def test_atomic_save_json_cleans_own_temp_after_replace_failure(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"version": "old"}', encoding="utf-8")

    def fail_replace(source: Path, target: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        atomic_save_json(path, {"version": "new"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": "old"}
    assert list(tmp_path.glob("state.json.*.tmp")) == []


def test_atomic_save_json_cleans_temp_after_serialization_failure(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"version": "old"}', encoding="utf-8")

    def fail_after_partial_write(data, stream, **kwargs) -> None:
        stream.write('{"partial":')
        raise TypeError("serialization failed")

    monkeypatch.setattr(json_store.json, "dump", fail_after_partial_write)
    with pytest.raises(TypeError, match="serialization failed"):
        atomic_save_json(path, {"version": "new"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": "old"}
    assert list(tmp_path.glob("state.json.*.tmp")) == []


def test_atomic_save_json_cleans_temp_after_file_fsync_failure(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"version": "old"}', encoding="utf-8")
    calls = 0

    def fail_file_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        raise OSError("file fsync failed")

    monkeypatch.setattr(json_store.os, "fsync", fail_file_fsync)
    with pytest.raises(OSError, match="file fsync failed"):
        atomic_save_json(path, {"version": "new"})

    assert calls == 1
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": "old"}
    assert list(tmp_path.glob("state.json.*.tmp")) == []


def test_atomic_save_json_directory_fsync_failure_keeps_new_target_visible(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"version": "old"}', encoding="utf-8")
    calls = 0
    original_fsync = json_store.os.fsync

    def fail_directory_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")
        original_fsync(fd)

    monkeypatch.setattr(json_store.os, "fsync", fail_directory_fsync)
    with pytest.raises(OSError, match="directory fsync failed"):
        atomic_save_json(path, {"version": "new"})

    assert calls == 2
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": "new"}
    assert list(tmp_path.glob("state.json.*.tmp")) == []


def test_atomic_save_json_logs_cleanup_failure_without_masking_replace_error(
    monkeypatch, tmp_path, caplog
) -> None:
    path = tmp_path / "state.json"

    def fail_replace(source: Path, target: Path) -> Path:
        raise OSError("replace failed")

    def fail_unlink(target: Path, *, missing_ok: bool = False) -> None:
        raise OSError("cleanup failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(OSError, match="replace failed"):
        atomic_save_json(path, {"version": "new"}, domain="test.state")

    assert "[test.state] 原子写清理临时文件失败" in caplog.text
    assert "cleanup failed" in caplog.text


def test_atomic_write_text_preserves_permissions_and_new_file_umask(tmp_path) -> None:
    existing = tmp_path / "existing.txt"
    existing.write_bytes(b"old")
    existing.chmod(0o751)
    atomic_write_text(existing, "\ufeffleft\r\nright\n")

    assert existing.read_bytes() == "\ufeffleft\r\nright\n".encode("utf-8")
    assert stat.S_IMODE(existing.stat().st_mode) == 0o751

    control = tmp_path / "control.txt"
    control.write_text("content", encoding="utf-8")
    new_file = tmp_path / "new.txt"
    atomic_write_text(new_file, "content")

    assert stat.S_IMODE(new_file.stat().st_mode) == stat.S_IMODE(
        control.stat().st_mode
    )


def test_atomic_write_text_encoding_failure_keeps_target_and_cleans_temp(
    tmp_path,
) -> None:
    path = tmp_path / "state.txt"
    old_content = b"old\r\ncontent\n"
    path.write_bytes(old_content)

    with pytest.raises(UnicodeEncodeError):
        atomic_write_text(path, "new\ud800")

    assert path.read_bytes() == old_content
    assert list(tmp_path.glob("state.txt.*.tmp")) == []
