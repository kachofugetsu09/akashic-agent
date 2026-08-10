from __future__ import annotations

import json
import os
import secrets
import ssl
from pathlib import Path
from uuid import uuid4

import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from infra.mobile_realtime.key_protection import (
    EncryptedKeyBlobCodec,
    FileMasterKeyStore,
    KeyProtectionError,
    KeysetManager,
    create_server_ssl_context,
    public_key_fingerprint,
)


class _EphemeralMasterKeys:
    def __init__(self) -> None:
        self.keys: dict[str, bytes] = {}

    def create(self) -> tuple[str, bytes]:
        key_id = uuid4().hex
        key = secrets.token_bytes(32)
        self.keys[key_id] = key
        return key_id, key

    def load(self, master_key_id: str) -> bytes:
        try:
            return self.keys[master_key_id]
        except KeyError as error:
            raise KeyProtectionError("测试 master key 不存在") from error


def test_file_master_keys_persist_rotation_and_import(tmp_path: Path) -> None:
    path = tmp_path / "mobile" / "master-keys.json"
    store = FileMasterKeyStore(path)

    first_id, first = store.create()
    second_id, second = store.create()
    imported_id = uuid4().hex
    imported = secrets.token_bytes(32)
    store.import_key(imported_id, imported)
    store.import_key(imported_id, imported)

    reloaded = FileMasterKeyStore(path)
    assert reloaded.load(first_id) == first
    assert reloaded.load(second_id) == second
    assert reloaded.load(imported_id) == imported
    assert os.stat(path.parent).st_mode & 0o777 == 0o700
    assert os.stat(path).st_mode & 0o777 == 0o600


def test_file_master_keys_reject_permissions_and_conflicting_import(
    tmp_path: Path,
) -> None:
    path = tmp_path / "master-keys.json"
    store = FileMasterKeyStore(path)
    key_id, key = store.create()

    path.chmod(0o644)
    with pytest.raises(KeyProtectionError, match="0600"):
        store.load(key_id)
    path.chmod(0o600)
    with pytest.raises(KeyProtectionError, match="不同内容"):
        store.import_key(key_id, secrets.token_bytes(32))
    assert store.load(key_id) == key


def test_encrypted_blob_binds_header_aad_and_tag() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    fingerprint = public_key_fingerprint(key.public_key())
    master_key = secrets.token_bytes(32)
    plaintext = b"private-key-material"
    blob = EncryptedKeyBlobCodec.encrypt(
        plaintext,
        master_key=master_key,
        server_id="server-1",
        purpose="identity",
        keyset_version=1,
        public_fingerprint=fingerprint,
    )

    decrypted = EncryptedKeyBlobCodec.decrypt(
        blob,
        master_key=master_key,
        server_id="server-1",
        purpose="identity",
        keyset_version=1,
        public_fingerprint=fingerprint,
    )
    assert bytes(decrypted) == plaintext

    tampered = blob[:-1] + bytes([blob[-1] ^ 1])
    with pytest.raises(KeyProtectionError, match="完整性校验失败"):
        EncryptedKeyBlobCodec.decrypt(
            tampered,
            master_key=master_key,
            server_id="server-1",
            purpose="identity",
            keyset_version=1,
            public_fingerprint=fingerprint,
        )
    with pytest.raises(KeyProtectionError, match="purpose 不匹配"):
        EncryptedKeyBlobCodec.decrypt(
            blob,
            master_key=master_key,
            server_id="server-1",
            purpose="lan_tls",
            keyset_version=1,
            public_fingerprint=fingerprint,
        )


def test_keyset_never_writes_plaintext_private_keys_and_loads_via_memfd(
    tmp_path: Path,
) -> None:
    keys = _EphemeralMasterKeys()
    root = tmp_path / "keys"
    manager = KeysetManager(root, keys)

    initialized = manager.initialize(lan_hostname="akashic.local")
    loaded = manager.load(expected_server_fingerprint=initialized.server_fingerprint)
    context = create_server_ssl_context(loaded)

    assert isinstance(context, ssl.SSLContext)
    assert os.stat(root).st_mode & 0o777 == 0o700
    for path in root.rglob("*"):
        if path.is_file():
            assert b"BEGIN PRIVATE KEY" not in path.read_bytes()
            assert os.stat(path).st_mode & 0o777 == 0o600


def test_keyset_rotation_keeps_public_identity_and_old_version(tmp_path: Path) -> None:
    keys = _EphemeralMasterKeys()
    manager = KeysetManager(tmp_path / "keys", keys)
    original = manager.initialize(lan_hostname="akashic.local")

    rotated = manager.rotate_master_key()
    current = json.loads((tmp_path / "keys" / "current.json").read_text())

    assert rotated.manifest.keyset_version == 2
    assert rotated.server_fingerprint == original.server_fingerprint
    assert rotated.tls_spki_fingerprint == original.tls_spki_fingerprint
    assert current["keyset_version"] == 2
    assert (tmp_path / "keys" / "keyset-v1").is_dir()
    assert (tmp_path / "keys" / "keyset-v2").is_dir()


def test_interrupted_rotation_leaves_current_keyset_loadable(tmp_path: Path) -> None:
    keys = _EphemeralMasterKeys()
    root = tmp_path / "keys"
    stable = KeysetManager(root, keys)
    original = stable.initialize(lan_hostname="akashic.local")

    class _InterruptedManager(KeysetManager):
        def _write_current(self, keyset_version: int) -> None:
            raise OSError("injected current pointer failure")

    with pytest.raises(OSError, match="injected"):
        _InterruptedManager(root, keys).rotate_master_key()

    recovered = stable.load()
    current = json.loads((root / "current.json").read_text())
    assert current["keyset_version"] == 1
    assert recovered.server_fingerprint == original.server_fingerprint


def test_wrong_master_key_and_database_fingerprint_fail_loud(tmp_path: Path) -> None:
    keys = _EphemeralMasterKeys()
    root = tmp_path / "keys"
    manager = KeysetManager(root, keys)
    loaded = manager.initialize(lan_hostname="127.0.0.1")

    with pytest.raises(KeyProtectionError, match="数据库不一致"):
        manager.load(expected_server_fingerprint="sha256/" + "A" * 43)

    keys.keys[loaded.manifest.master_key_id] = secrets.token_bytes(32)
    with pytest.raises(KeyProtectionError, match="完整性校验失败"):
        manager.load()


def test_tampered_blob_and_manifest_path_are_rejected(tmp_path: Path) -> None:
    keys = _EphemeralMasterKeys()
    root = tmp_path / "keys"
    manager = KeysetManager(root, keys)
    _ = manager.initialize(lan_hostname="akashic.local")
    blob = root / "keyset-v1" / "server-identity.key.enc"
    tampered_blob = bytearray(blob.read_bytes())
    tampered_blob[-1] ^= 1
    blob.write_bytes(tampered_blob)

    with pytest.raises(KeyProtectionError, match="blob hash 不匹配"):
        manager.load()

    current = root / "current.json"
    current.write_text(
        json.dumps(
            {
                "format_version": 1,
                "keyset_version": 1,
                "manifest": "../manifest.json",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(KeyProtectionError, match="manifest 路径无效"):
        manager.load()
