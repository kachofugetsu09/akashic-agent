from __future__ import annotations

import json
import multiprocessing
import os
import signal
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from infra.mobile_realtime.key_protection import LoadedKeyset
from infra.mobile_realtime.protocol import (
    MobileWebUiContentPrepareCommand,
    MobileWebUiReleaseChangedControl,
    parse_frame,
)
from infra.mobile_webui.protocol import PrepareReplyWire, ReleaseViewWire
from infra.mobile_realtime.storage import DeviceRecord, MobileRealtimeStorage
from infra.mobile_webui.http import WebUiTicketIssuer, WebUiTicketError, parse_single_range
from infra.mobile_webui.manifest import (
    ManifestError,
    WebUiManifest,
    WebUiFile,
    canonical_manifest_bytes,
    derive_target_key,
    generation_id_for_manifest,
    manifest_digest,
    manifest_from_directory,
    manifest_from_json,
    validate_manifest,
)
from infra.mobile_webui.store import MobileWebUiStore, UnknownReleaseError


_SOURCE = {
    "source_repository": "https://github.com/example/repo",
    "source_commit": "a" * 40,
    "source_tree": "b" * 40,
    "input_digest": "c" * 64,
    "build_context_digest": "d" * 64,
    "dirty_provenance": None,
    "reproducible": True,
    "builder_identity": {
        "node_version": "v22.23.1",
        "npm_version": "10.9.0",
        "package_lock_digest": "e" * 64,
        "build_script_digest": "f" * 64,
    },
}


def _manifest(root: Path, text: bytes = b"<html>ok</html>\n"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "mobile.html").write_bytes(text)
    return manifest_from_directory(root, **_SOURCE)


def _kill_backup_before_rename(root: str, destination: str) -> None:
    from infra.mobile_webui import store as store_module

    original_replace = store_module.os.replace
    absolute_destination = os.path.abspath(destination)

    def kill_before_rename(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        if os.path.abspath(os.fspath(target)) == absolute_destination:
            os.kill(os.getpid(), signal.SIGKILL)
        original_replace(source, target)

    store_module.os.replace = kill_before_rename
    store = MobileWebUiStore(Path(root), server_id="server-1")
    store.backup_to(Path(destination))


def _kill_backup_after_rename(root: str, destination: str) -> None:
    from infra.mobile_webui import store as store_module

    original_replace = store_module.os.replace
    absolute_destination = os.path.abspath(destination)

    def kill_after_rename(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        original_replace(source, target)
        if os.path.abspath(os.fspath(target)) == absolute_destination:
            os.kill(os.getpid(), signal.SIGKILL)

    store_module.os.replace = kill_after_rename
    store = MobileWebUiStore(Path(root), server_id="server-1")
    store.backup_to(Path(destination))


def test_golden_manifest_and_derived_ids() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "manifest-v2.json").read_text()
    )
    manifest = manifest_from_json(fixture["manifest"])
    assert manifest_digest(manifest) == fixture["manifest_digest"]
    assert generation_id_for_manifest(manifest) == manifest.generation_id
    assert derive_target_key("server-1", manifest.generation_id, fixture["manifest_digest"]) == fixture["target_key"]
    assert canonical_manifest_bytes(manifest).decode() == json.dumps(
        fixture["manifest"], ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    mismatched = json.loads(json.dumps(fixture["manifest"]))
    mismatched["files"][0]["mime"] = "application/octet-stream"
    with pytest.raises(ManifestError):
        manifest_from_json(mismatched)


@pytest.mark.parametrize(
    "name",
    ("../escape", "/absolute", "a\\b", "a/./b", "a/../b", "a b"),
)
def test_manifest_rejects_unsafe_or_noncanonical_paths(tmp_path: Path, name: str) -> None:
    digest = "0" * 64
    files = (WebUiFile(name, digest, 0, "text/html"),)
    draft = WebUiManifest(
        generation_id="0" * 64,
        entrypoint=name,
        files=files,
        bridge_protocol_min=1,
        bridge_protocol_max=1,
        snapshot_protocol_min=7,
        snapshot_protocol_max=7,
        minimum_native_build=45,
        platforms=("android",),
        **_SOURCE,
        unpacked_size_bytes=0,
        file_count=1,
    )
    with pytest.raises(ManifestError):
        manifest_from_json({**draft.as_json(), "generation_id": draft.generation_id})


def test_manifest_rejects_ambiguous_digest_mime() -> None:
    digest = "1" * 64
    draft = WebUiManifest(
        generation_id="0" * 64,
        entrypoint="mobile.html",
        files=(
            WebUiFile("mobile.html", digest, 1, "text/html"),
            WebUiFile("script.js", digest, 1, "text/javascript"),
        ),
        bridge_protocol_min=1,
        bridge_protocol_max=1,
        snapshot_protocol_min=7,
        snapshot_protocol_max=7,
        minimum_native_build=45,
        platforms=("android",),
        **_SOURCE,
        unpacked_size_bytes=2,
        file_count=2,
    )
    with pytest.raises(ManifestError):
        validate_manifest(replace(draft, generation_id=generation_id_for_manifest(draft)))


def test_manifest_rejects_ambiguous_digest_size() -> None:
    digest = "2" * 64
    draft = WebUiManifest(
        generation_id="0" * 64,
        entrypoint="mobile.html",
        files=(
            WebUiFile("mobile.html", digest, 1, "text/html"),
            WebUiFile("screen.htm", digest, 2, "text/html"),
        ),
        bridge_protocol_min=1,
        bridge_protocol_max=1,
        snapshot_protocol_min=7,
        snapshot_protocol_max=7,
        minimum_native_build=45,
        platforms=("android",),
        **_SOURCE,
        unpacked_size_bytes=3,
        file_count=2,
    )
    with pytest.raises(ManifestError, match="size/mime"):
        validate_manifest(replace(draft, generation_id=generation_id_for_manifest(draft)))


def test_manifest_directory_uses_fixed_mime_mapping(tmp_path: Path) -> None:
    root = tmp_path / "mime"
    root.mkdir()
    (root / "mobile.html").write_bytes(b"<html/>")
    (root / "bundle.js").write_bytes(b"same")
    (root / "asset.unknown").write_bytes(b"different")
    manifest, _ = manifest_from_directory(root, **_SOURCE)
    by_path = {item.path: item.mime for item in manifest.files}
    assert by_path["bundle.js"] == "text/javascript"
    assert by_path["asset.unknown"] == "application/octet-stream"


def test_manifest_directory_covers_wire_reachable_script_and_binary_mimes(tmp_path: Path) -> None:
    root = tmp_path / "mime-golden"
    root.mkdir()
    contents = {
        "mobile.html": b"<html/>",
        "app.js": b"console.log(1);",
        "module.mjs": b"export {};",
        "worker.cjs": b"module.exports = {};",
        "style.css": b"body{}",
        "bundle.map": b"{}",
        "asset.bin": b"\x00\x01",
    }
    for path, data in contents.items():
        (root / path).write_bytes(data)
    manifest, _ = manifest_from_directory(root, **_SOURCE)
    by_path = {item.path: item.mime for item in manifest.files}
    assert by_path == {
        "app.js": "text/javascript",
        "asset.bin": "application/octet-stream",
        "bundle.map": "application/json",
        "mobile.html": "text/html",
        "module.mjs": "text/javascript",
        "style.css": "text/css",
        "worker.cjs": "text/javascript",
    }
    from infra.mobile_webui.protocol import WebUiManifestWire, WebUiFileWire

    WebUiManifestWire.model_validate(manifest.as_json(), strict=True)
    with pytest.raises(ValueError):
        WebUiFileWire.model_validate(
            {
                "path": "bad.js",
                "sha256": "0" * 64,
                "size_bytes": 0,
                "mime": "application/javascript",
            },
            strict=True,
        )


def test_wire_manifest_rejects_ambiguous_digest_size(tmp_path: Path) -> None:
    root = tmp_path / "wire-size-conflict"
    root.mkdir()
    (root / "mobile.html").write_bytes(b"<html/>")
    manifest, _ = manifest_from_directory(root, **_SOURCE)
    payload = manifest.as_json()
    payload["files"] = [
        {"path": "a.html", "sha256": "0" * 64, "size_bytes": 1, "mime": "text/html"},
        {"path": "mobile.html", "sha256": "0" * 64, "size_bytes": 2, "mime": "text/html"},
    ]
    payload["file_count"] = 2
    payload["unpacked_size_bytes"] = 3
    from infra.mobile_webui.protocol import WebUiManifestWire

    with pytest.raises(ValueError, match="size/mime"):
        WebUiManifestWire.model_validate(payload, strict=True)


def test_manifest_directory_rejects_symlink_and_special_file(tmp_path: Path) -> None:
    root = tmp_path / "special"
    root.mkdir()
    (root / "mobile.html").write_bytes(b"<html/>")
    os.symlink(root / "mobile.html", root / "link.html")
    with pytest.raises(ManifestError):
        manifest_from_directory(root, **_SOURCE)
    (root / "link.html").unlink()
    fifo = root / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(ManifestError):
        manifest_from_directory(root, **_SOURCE)


def test_store_preview_promotion_rollback_gc_and_backup(tmp_path: Path) -> None:
    first, first_contents = _manifest(tmp_path / "first")
    store = MobileWebUiStore(tmp_path / "store", server_id="server-1")
    try:
        first_release = store.publish(first, first_contents, stable=True, preview=False)
        store.pin_target(first_release.stable.target_key)  # type: ignore[union-attr]
        second, second_contents = _manifest(tmp_path / "second", b"<html>new</html>\n")
        preview = store.publish(second, second_contents, preview=True)
        assert preview.stable is not None and preview.preview is not None
        rolled_with_preview = store.rollback(first_release.stable.target_key)  # type: ignore[union-attr]
        assert rolled_with_preview.stable is not None and rolled_with_preview.stable.generation_id == first.generation_id
        assert rolled_with_preview.preview is not None and rolled_with_preview.preview.generation_id == second.generation_id
        promoted = store.promote_preview()
        assert promoted.stable is not None and promoted.stable.generation_id == second.generation_id
        rolled = store.rollback(first_release.stable.target_key)  # type: ignore[union-attr]
        assert rolled.stable is not None and rolled.stable.generation_id == first.generation_id
        cleared = store.clear_preview()
        assert cleared.stable is not None and cleared.preview is None
        backup = store.backup_to(tmp_path / "backup")
        MobileWebUiStore.verify_backup(backup, server_id="server-1")
        assert first.generation_id not in store.gc().removed_generations
    finally:
        store.close()


def test_backup_pending_kill_restart_reclaims_registration_and_gc_can_collect(tmp_path: Path) -> None:
    manifest, contents = _manifest(tmp_path / "build")
    root = tmp_path / "store"
    store = MobileWebUiStore(root, server_id="server-1")
    store.publish(manifest, contents, stable=True, preview=False)
    store.close()
    destination = tmp_path / "backup-before-rename"
    context = multiprocessing.get_context("fork")
    child = context.Process(target=_kill_backup_before_rename, args=(str(root), str(destination)))
    child.start()
    child.join(timeout=10)
    assert child.exitcode == -signal.SIGKILL
    assert list((root / "staging").glob(".backup-*.pending.json"))
    assert list(destination.parent.glob(f".{destination.name}.*.tmp"))

    reopened = MobileWebUiStore(root, server_id="server-1")
    try:
        assert reopened._db.execute("SELECT COUNT(*) FROM webui_backup_sets").fetchone()[0] == 0
        assert not list((root / "staging").glob(".backup-*.pending.json"))
        assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))
        generations = [manifest]
        for index in range(5):
            next_manifest, next_contents = _manifest(
                tmp_path / f"build-{index}",
                f"<html>{index}</html>\n".encode(),
            )
            generations.append(next_manifest)
            reopened.publish(next_manifest, next_contents, stable=True, preview=False)
        report = reopened.gc()
        assert generations[0].generation_id in report.removed_generations
    finally:
        reopened.close()


def test_backup_pending_restart_preserves_published_destination(tmp_path: Path) -> None:
    manifest, contents = _manifest(tmp_path / "build")
    root = tmp_path / "store"
    store = MobileWebUiStore(root, server_id="server-1")
    store.publish(manifest, contents, stable=True, preview=False)
    store.close()
    destination = tmp_path / "backup-after-rename"
    context = multiprocessing.get_context("fork")
    child = context.Process(target=_kill_backup_after_rename, args=(str(root), str(destination)))
    child.start()
    child.join(timeout=10)
    assert child.exitcode == -signal.SIGKILL
    assert destination.is_dir()
    assert list((root / "staging").glob(".backup-*.pending.json"))

    reopened = MobileWebUiStore(root, server_id="server-1")
    try:
        MobileWebUiStore.verify_backup(destination, server_id="server-1")
        backup_id = json.loads((destination / "backup.json").read_text(encoding="utf-8"))["backup_id"]
        assert reopened._db.execute(
            "SELECT COUNT(*) FROM webui_backup_sets WHERE backup_id = ?", (backup_id,)
        ).fetchone()[0] == 1
        assert not list((root / "staging").glob(".backup-*.pending.json"))
        reopened.release_backup(backup_id)
    finally:
        reopened.close()


def test_backup_rejects_destination_symlink_without_writing_target(tmp_path: Path) -> None:
    manifest, contents = _manifest(tmp_path / "build")
    root = tmp_path / "store"
    store = MobileWebUiStore(root, server_id="server-1")
    store.publish(manifest, contents, stable=True, preview=False)
    target = tmp_path / "symlink-target"
    destination = tmp_path / "symlink-destination"
    destination.symlink_to(target, target_is_directory=True)
    try:
        with pytest.raises(RuntimeError, match="符号链接"):
            store.backup_to(destination)
        assert not target.exists()
        assert store._db.execute("SELECT COUNT(*) FROM webui_backup_sets").fetchone()[0] == 0
    finally:
        store.close()


def test_store_rejects_corrupt_release_epoch_on_open(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = MobileWebUiStore(root, server_id="server-1")
    store.close()
    connection = sqlite3.connect(root / "publication.sqlite3")
    try:
        connection.execute(
            "UPDATE webui_meta SET value = ? WHERE key = 'release_epoch'",
            ("not-a-uuid",),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(RuntimeError, match="规范 UUID4"):
        MobileWebUiStore(root, server_id="server-1")


def test_store_rejects_duplicate_and_nonstandard_manifest_json(tmp_path: Path) -> None:
    manifest, contents = _manifest(tmp_path / "build")
    root = tmp_path / "store"
    store = MobileWebUiStore(root, server_id="server-1")
    store.publish(manifest, contents, stable=True, preview=False)
    raw = canonical_manifest_bytes(manifest)
    duplicate = raw[:-1] + b',"generation_id":"' + manifest.generation_id.encode() + b'"}'
    store._db.execute("UPDATE webui_generations SET manifest_json = ? WHERE generation_id = ?", (duplicate, manifest.generation_id))
    with pytest.raises(ManifestError, match="重复字段"):
        store.get_manifest(manifest_digest(manifest))
    store._db.execute(
        "UPDATE webui_generations SET manifest_json = ? WHERE generation_id = ?",
        (raw.replace(str(manifest.unpacked_size_bytes).encode(), b"NaN", 1), manifest.generation_id),
    )
    with pytest.raises(ManifestError, match="不允许常量"):
        store.get_manifest(manifest_digest(manifest))
    store.close()


def test_store_recovers_only_its_blob_temp_residue(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = MobileWebUiStore(root, server_id="server-1")
    digest = "a" * 64
    blob_dir = root / "blobs" / "sha256" / digest[:2]
    blob_dir.mkdir(parents=True, exist_ok=True)
    stale = blob_dir / f".{digest}.123.tmp"
    unrelated = blob_dir / ".not-a-store-temp.tmp"
    stale.write_bytes(b"partial")
    unrelated.write_bytes(b"keep")
    store.close()
    reopened = MobileWebUiStore(root, server_id="server-1")
    reopened.close()
    assert not stale.exists()
    assert unrelated.read_bytes() == b"keep"


def test_restore_appends_audit_event_and_preserves_source_backup(tmp_path: Path) -> None:
    manifest, contents = _manifest(tmp_path / "build")
    source = MobileWebUiStore(tmp_path / "source", server_id="server-1")
    source.publish(manifest, contents, stable=True, preview=False)
    backup = source.backup_to(tmp_path / "backup")
    source_descriptor = (backup / "backup.json").read_bytes()
    source.close()
    target = MobileWebUiStore(tmp_path / "target", server_id="server-1")
    target.close()
    MobileWebUiStore.restore_backup(
        backup,
        tmp_path / "target",
        server_id="server-1",
        pre_restore_backup=tmp_path / "pre-restore",
    )
    MobileWebUiStore.verify_backup(backup, server_id="server-1")
    assert (backup / "backup.json").read_bytes() == source_descriptor
    MobileWebUiStore.verify_backup(tmp_path / "target", server_id="server-1")
    restored = MobileWebUiStore(tmp_path / "target", server_id="server-1")
    try:
        assert restored.get_release().sequence == 2
        event = restored._db.execute(
            "SELECT sequence, generation_id, operation, stable, preview, actor FROM webui_publication_journal ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        assert tuple(event) == (2, manifest.generation_id, "restore", 1, 0, "restore-mobile-webui")
    finally:
        restored.close()
    MobileWebUiStore.verify_backup(tmp_path / "pre-restore", server_id="server-1")


def test_restore_empty_backup_keeps_no_state_and_records_restore(tmp_path: Path) -> None:
    source = MobileWebUiStore(tmp_path / "empty", server_id="server-1")
    backup = source.backup_to(tmp_path / "empty-backup")
    source.close()
    MobileWebUiStore.restore_backup(backup, tmp_path / "restored-empty", server_id="server-1")
    MobileWebUiStore.verify_backup(tmp_path / "restored-empty", server_id="server-1")
    restored = MobileWebUiStore(tmp_path / "restored-empty", server_id="server-1")
    try:
        assert restored.get_release().stable is None
        event = restored._db.execute(
            "SELECT sequence, generation_id, operation, stable, preview FROM webui_publication_journal"
        ).fetchone()
        assert tuple(event) == (1, None, "restore", 0, 0)
    finally:
        restored.close()


def test_restore_rejects_target_symlink_without_following_target(tmp_path: Path) -> None:
    source = MobileWebUiStore(tmp_path / "source", server_id="server-1")
    backup = source.backup_to(tmp_path / "backup")
    source.close()
    actual_target = tmp_path / "actual-target"
    target = tmp_path / "target-link"
    target.symlink_to(actual_target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="普通目录"):
        MobileWebUiStore.restore_backup(backup, target, server_id="server-1")
    assert not actual_target.exists()


def test_restore_marker_recovers_old_root_after_swap_gap(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = MobileWebUiStore(root, server_id="server-1")
    store.close()
    recovery = tmp_path / "store.recovery-test"
    os.replace(root, recovery)
    marker = MobileWebUiStore._write_restore_marker(root, recovery)
    assert marker.exists()
    recovered = MobileWebUiStore(root, server_id="server-1")
    recovered.close()
    assert root.is_dir()
    assert not recovery.exists()
    assert not marker.exists()


def test_restore_marker_keeps_new_root_when_swap_completed(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = MobileWebUiStore(root, server_id="server-1")
    store.close()
    recovery = tmp_path / "store.recovery-new"
    recovery.mkdir()
    marker = MobileWebUiStore._write_restore_marker(root, recovery)
    reopened = MobileWebUiStore(root, server_id="server-1")
    reopened.close()
    assert root.is_dir()
    assert recovery.is_dir()
    assert not marker.exists()


def test_channel_history_is_distinct_per_pointer_and_bounds_gc(tmp_path: Path) -> None:
    store = MobileWebUiStore(tmp_path / "store", server_id="server-1")
    try:
        stable_releases = []
        for index in range(6):
            manifest, contents = _manifest(tmp_path / f"stable-{index}", f"<html>stable-{index}</html>\n".encode())
            stable_releases.append(store.publish(manifest, contents, stable=True, preview=False))
        preview_releases = []
        for index in range(6):
            manifest, contents = _manifest(tmp_path / f"preview-{index}", f"<html>preview-{index}</html>\n".encode())
            preview_releases.append(store.publish(manifest, contents, preview=True))
        before = store._db.execute(
            "SELECT channel, COUNT(*) AS count FROM webui_channel_selections GROUP BY channel"
        ).fetchall()
        assert {row["channel"]: row["count"] for row in before} == {"preview": 4, "stable": 4}
        store.publish(manifest, contents, preview=True)  # no-op pointer does not add history
        store.clear_preview()
        after = store._db.execute(
            "SELECT channel, COUNT(*) AS count FROM webui_channel_selections GROUP BY channel"
        ).fetchall()
        assert {row["channel"]: row["count"] for row in after} == {"preview": 4, "stable": 4}
        assert store.rollback(preview_releases[-1].preview.target_key).preview is None  # type: ignore[union-attr]
        report = store.gc()
        assert stable_releases[0].stable.generation_id in report.removed_generations  # type: ignore[union-attr]
    finally:
        store.close()


def test_ticket_scope_and_range(tmp_path: Path) -> None:
    manifest, contents = _manifest(tmp_path / "build")
    store = MobileWebUiStore(tmp_path / "store", server_id="server-1")
    release = store.publish(manifest, contents, stable=True, preview=False)

    class FakeKeyset:
        class Manifest:
            server_id = "server-1"

        manifest = Manifest()
        identity_private_key = ec.generate_private_key(ec.SECP256R1())

    device = DeviceRecord("device-1", "pub", "test", datetime.now(timezone.utc), None, ("mobile-webui-ota-v1",))

    class FakeStorage:
        def read_device(self, device_id: str) -> DeviceRecord | None:
            return device if device_id == device.device_id else None

    issuer = WebUiTicketIssuer(
        cast(LoadedKeyset, FakeKeyset()),
        cast(MobileRealtimeStorage, FakeStorage()),
        store,
        connection_checker=lambda _id, epoch: epoch == 7,
    )
    assert release.stable is not None
    grant = issuer.issue(device_id=device.device_id, connection_epoch=7, release=release, target_key=release.stable.target_key)
    verified = issuer.verify(grant.ticket, resource_kind="manifest", resource_digest=release.stable.manifest_digest)
    assert verified.target_key == release.stable.target_key
    with pytest.raises(WebUiTicketError):
        issuer.verify(grant.ticket, resource_kind="blob", resource_digest="0" * 64)
    assert parse_single_range("bytes=0-3", 10) == (0, 3)
    assert parse_single_range("bytes=-3", 10) == (7, 9)
    with pytest.raises(WebUiTicketError):
        parse_single_range("bytes=0-1,3-4", 10)
    store.close()


def test_webui_commands_and_hint_are_strict() -> None:
    frame_id = "01J00000000000000000000000"
    command = parse_frame(json.dumps({"v": 1, "kind": "command", "type": "mobile.webui.content.prepare", "id": frame_id, "connection_epoch": 7, "payload": {"target_key": "a" * 64}}))
    assert isinstance(command, MobileWebUiContentPrepareCommand)
    control = parse_frame(json.dumps({"v": 1, "kind": "control", "type": "mobile.webui.release.changed", "connection_epoch": 7, "payload": {"server_id": "server-1", "selection_digest": "b" * 64}}))
    assert isinstance(control, MobileWebUiReleaseChangedControl)
    reply = PrepareReplyWire(
        target_key="a" * 64,
        manifest_digest="b" * 64,
        ticket="ticket",
        expires_at="2026-08-03T00:00:00Z",
    )
    assert set(reply.model_dump()) == {"target_key", "manifest_digest", "ticket", "expires_at"}
    with pytest.raises(ValueError):
        PrepareReplyWire.model_validate({**reply.model_dump(), "generation_id": "c" * 64}, strict=True)
    with pytest.raises(ValueError):
        ReleaseViewWire.model_validate(
            {
                "server_id": "server-1",
                "release_epoch": "00000000-0000-1000-8000-000000000000",
                "sequence": 0,
                "selection_digest": "b" * 64,
                "stable": None,
                "preview": None,
            },
            strict=True,
        )
