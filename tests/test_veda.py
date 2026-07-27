from __future__ import annotations

import stat
from pathlib import Path

import pytest

from agent.persona import (
    VedaLoadError,
    read_default_veda,
    read_veda,
    reset_veda,
    veda_path,
)


def test_read_veda_returns_nonempty_utf8(tmp_path: Path) -> None:
    path = veda_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("\ncustom veda\n", encoding="utf-8")

    assert read_veda(tmp_path) == "custom veda"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "缺少 Veda"),
        (b" \n", "Veda 内容为空"),
        (b"\xff", "Veda 不是合法 UTF-8"),
    ],
)
def test_read_veda_fails_loud_without_fallback(
    tmp_path: Path,
    payload: bytes | None,
    message: str,
) -> None:
    if payload is not None:
        path = veda_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)

    with pytest.raises(VedaLoadError, match=message):
        read_veda(tmp_path)


def test_reset_veda_backs_up_original_bytes_and_restores_default(
    tmp_path: Path,
) -> None:
    path = veda_path(tmp_path)
    path.parent.mkdir(parents=True)
    original = b"\xffbroken"
    path.write_bytes(original)

    result = reset_veda(tmp_path)

    assert result.changed is True
    assert result.backup_path is not None
    assert result.backup_path.read_bytes() == original
    assert stat.S_IMODE(result.backup_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(result.backup_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(result.backup_path.parent.parent.stat().st_mode) == 0o700
    assert read_veda(tmp_path) == read_default_veda()
    assert result.previous_sha256 is not None


def test_reset_veda_is_idempotent_for_default_content(tmp_path: Path) -> None:
    first = reset_veda(tmp_path)
    second = reset_veda(tmp_path)

    assert first.changed is True
    assert first.backup_path is None
    assert second.changed is False
    assert second.backup_path is None
    assert not (tmp_path / "memory/veda-backups").exists()
