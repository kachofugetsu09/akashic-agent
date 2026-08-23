from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

import agent.plugins.manager as manager_module


def test_candidate_data_copy_materializes_live_wal_database(
    tmp_path: Path,
) -> None:
    source = tmp_path / "production-data"
    source.mkdir()
    database = source / "content.sqlite3"
    writer = sqlite3.connect(database)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, body TEXT NOT NULL)"
        )
        writer.executemany(
            "INSERT INTO items(body) VALUES (?)",
            [(f"item-{index}",) for index in range(32)],
        )
        writer.commit()
        assert (source / "content.sqlite3-wal").is_file()
        assert (source / "content.sqlite3-shm").is_file()
        (source / "content.sqlite3-journal").write_bytes(b"")

        inventory = (
            manager_module._copy_validation_data(  # pyright: ignore[reportPrivateUsage]
                source,
                tmp_path / "candidate-data",
                (),
            )
        )
    finally:
        writer.close()

    assert inventory == ("content.sqlite3",)
    copied = sqlite3.connect(tmp_path / "candidate-data" / "content.sqlite3")
    try:
        assert copied.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert copied.execute("SELECT COUNT(*) FROM items").fetchone() == (32,)
    finally:
        copied.close()


def test_candidate_data_copy_tracks_committed_prefix_during_wal_writes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "production-data"
    source.mkdir()
    database = source / "events.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, payload BLOB NOT NULL)"
    )
    connection.execute("CREATE INDEX events_by_payload ON events(payload)")
    connection.executemany(
        "INSERT INTO events(payload) VALUES (?)",
        [(b"x" * 2048,) for _ in range(1024)],
    )
    connection.commit()
    connection.close()

    ready = threading.Event()
    copies_finished = threading.Event()
    writer_finished = threading.Event()
    final_count: list[int] = []

    def write_committed_rows() -> None:
        """Keep committing WAL rows before and after candidate snapshots."""

        writer = sqlite3.connect(database)
        writer.execute("PRAGMA wal_autocheckpoint=0")
        committed = 0
        try:
            # 1. Prove the writer has entered its real commit loop.
            while committed < 16:
                writer.execute("INSERT INTO events(payload) VALUES (?)", (b"writer",))
                writer.commit()
                committed += 1
            ready.set()

            # 2. Continue committing while all candidate copies run.
            while not copies_finished.is_set():
                writer.execute("INSERT INTO events(payload) VALUES (?)", (b"writer",))
                writer.commit()
                committed += 1

            # 3. Prove the formal writer remains usable after materialization.
            for _ in range(16):
                writer.execute("INSERT INTO events(payload) VALUES (?)", (b"after",))
                writer.commit()
                committed += 1
            final_count.append(
                int(writer.execute("SELECT COUNT(*) FROM events").fetchone()[0])
            )
        finally:
            writer.close()
            writer_finished.set()

    thread = threading.Thread(target=write_committed_rows)
    thread.start()
    candidate_counts: list[int] = []
    try:
        assert ready.wait(timeout=5)
        for index in range(3):
            target = tmp_path / f"candidate-data-{index}"
            inventory = manager_module._copy_validation_data(  # pyright: ignore[reportPrivateUsage]
                source,
                target,
                (),
            )
            assert inventory == ("events.sqlite3",)
            candidate = sqlite3.connect(target / "events.sqlite3")
            try:
                assert candidate.execute("PRAGMA integrity_check").fetchone() == ("ok",)
                count = int(
                    candidate.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                )
                maximum = int(
                    candidate.execute("SELECT MAX(id) FROM events").fetchone()[0]
                )
                assert count == maximum
                candidate_counts.append(count)
            finally:
                candidate.close()
    finally:
        copies_finished.set()
        finished = writer_finished.wait(timeout=5)
        thread.join(timeout=5)
        assert finished

    assert not thread.is_alive()
    assert final_count
    assert candidate_counts == sorted(candidate_counts)
    assert all(1024 <= count < final_count[0] for count in candidate_counts)
    formal = sqlite3.connect(database)
    try:
        assert formal.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert formal.execute("SELECT COUNT(*) FROM events").fetchone() == (
            final_count[0],
        )
    finally:
        formal.close()


@pytest.mark.parametrize(
    ("main_bytes", "expected_inventory"),
    [
        (None, ("orphan.sqlite3-wal",)),
        (b"ordinary file", ("orphan.sqlite3", "orphan.sqlite3-wal")),
    ],
)
def test_candidate_data_copy_preserves_sidecar_without_recognized_main(
    tmp_path: Path,
    main_bytes: bytes | None,
    expected_inventory: tuple[str, ...],
) -> None:
    source = tmp_path / "production-data"
    source.mkdir()
    if main_bytes is not None:
        (source / "orphan.sqlite3").write_bytes(main_bytes)
    (source / "orphan.sqlite3-wal").write_bytes(b"not a live SQLite sidecar")

    inventory = (
        manager_module._copy_validation_data(  # pyright: ignore[reportPrivateUsage]
            source,
            tmp_path / "candidate-data",
            (),
        )
    )

    assert inventory == expected_inventory


def test_candidate_data_copy_fails_loud_and_cleans_partial_sqlite_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "production-data"
    target = tmp_path / "candidate-data"
    source.mkdir()
    (source / "before.txt").write_text("copied first", encoding="utf-8")
    (source / "broken.sqlite3").write_bytes(b"SQLite format 3\x00not-a-database")

    with pytest.raises(sqlite3.DatabaseError):
        manager_module._copy_validation_data(  # pyright: ignore[reportPrivateUsage]
            source,
            target,
            (),
        )

    assert not target.exists()
