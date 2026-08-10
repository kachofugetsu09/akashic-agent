from __future__ import annotations

import base64
import ctypes
import hashlib
import ipaddress
import json
import os
import re
import secrets
import shutil
import ssl
import stat
import struct
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

import secretstorage
from cryptography import x509
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from secretstorage.exceptions import SecretStorageException

_MAGIC = b"AKKEY"
_FORMAT_VERSION = 1
_HEADER = struct.Struct(">5sBBI12s")
_TAG_SIZE = 16
_MASTER_KEY_SIZE = 32
_PURPOSE_CODES = {"identity": 1, "lan_tls": 2}
_PURPOSE_NAMES = {code: name for name, code in _PURPOSE_CODES.items()}
_FINGERPRINT_PATTERN = re.compile(r"sha256/[A-Za-z0-9+/]{43}=")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_MAX_JSON_BYTES = 128 * 1024
_MAX_BLOB_BYTES = 128 * 1024
_MAX_MASTER_KEY_FILE_BYTES = 128 * 1024


class KeyProtectionError(RuntimeError):
    pass


class MasterKeyStore(Protocol):
    def create(self) -> tuple[str, bytes]: ...

    def load(self, master_key_id: str) -> bytes: ...


@dataclass(frozen=True)
class KeyBlobEntry:
    path: str
    purpose: str
    public_fingerprint: str
    sha256: str


@dataclass(frozen=True)
class KeysetManifest:
    keyset_version: int
    server_id: str
    master_key_id: str
    identity: KeyBlobEntry
    lan_tls: KeyBlobEntry
    tls_certificate_path: str
    tls_certificate_sha256: str


@dataclass(frozen=True)
class LoadedKeyset:
    manifest: KeysetManifest
    identity_private_key: ec.EllipticCurvePrivateKey
    tls_private_key: ec.EllipticCurvePrivateKey
    tls_certificate: x509.Certificate
    tls_certificate_path: Path

    @property
    def server_fingerprint(self) -> str:
        return self.manifest.identity.public_fingerprint

    @property
    def tls_spki_fingerprint(self) -> str:
        return self.manifest.lan_tls.public_fingerprint


class SecretServiceMasterKeyStore:
    def __init__(self, namespace: str) -> None:
        if not namespace or not _IDENTIFIER_PATTERN.fullmatch(
            namespace.replace("/", ".")
        ):
            raise ValueError("Secret Service namespace 格式无效")
        self._namespace = namespace

    def create(self) -> tuple[str, bytes]:
        """在已解锁的 Secret Service 中创建唯一 master key。"""

        # 1. 生成 item 标识和不可持久化到磁盘的随机密钥
        master_key_id = uuid4().hex
        master_key = secrets.token_bytes(_MASTER_KEY_SIZE)
        attributes = self._attributes(master_key_id)

        # 2. 明确要求可用且已解锁的默认 collection
        try:
            with closing(secretstorage.dbus_init()) as connection:
                if not secretstorage.check_service_availability(connection):
                    raise KeyProtectionError("Secret Service 当前不可用")
                collection = secretstorage.get_default_collection(connection)
                if collection.is_locked():
                    raise KeyProtectionError("Secret Service collection 已锁定")
                if list(collection.search_items(attributes)):
                    raise KeyProtectionError("Secret Service master key ID 冲突")
                _ = collection.create_item(
                    f"Akasic mobile realtime master key {master_key_id}",
                    attributes,
                    master_key,
                    replace=False,
                    content_type="application/octet-stream",
                )
        except SecretStorageException as error:
            raise KeyProtectionError(f"Secret Service 写入失败: {error}") from error
        return master_key_id, master_key

    def load(self, master_key_id: str) -> bytes:
        """从 Secret Service 精确读取 manifest 指定的 master key。"""

        if not _IDENTIFIER_PATTERN.fullmatch(master_key_id):
            raise KeyProtectionError("manifest master_key_id 格式无效")

        # 1. 在 Secret Service 信任边界查找唯一 item
        try:
            with closing(secretstorage.dbus_init()) as connection:
                if not secretstorage.check_service_availability(connection):
                    raise KeyProtectionError("Secret Service 当前不可用")
                collection = secretstorage.get_default_collection(connection)
                if collection.is_locked():
                    raise KeyProtectionError("Secret Service collection 已锁定")
                items = list(collection.search_items(self._attributes(master_key_id)))
                if len(items) != 1:
                    raise KeyProtectionError(
                        f"Secret Service master key 数量无效: {len(items)}"
                    )
                master_key = items[0].get_secret()
        except SecretStorageException as error:
            raise KeyProtectionError(f"Secret Service 读取失败: {error}") from error

        # 2. 长度属于密钥存储边界不变量
        if len(master_key) != _MASTER_KEY_SIZE:
            raise KeyProtectionError("Secret Service master key 长度无效")
        return master_key

    def _attributes(self, master_key_id: str) -> dict[str, str]:
        return {
            "application": "akasic-agent",
            "namespace": self._namespace,
            "kind": "mobile-realtime-master-key-v1",
            "master-key-id": master_key_id,
        }


class FileMasterKeyStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def create(self) -> tuple[str, bytes]:
        """创建并原子持久化一个可轮换的 master key。"""

        # 1. 加载仍需支持旧 keyset 回滚的既有密钥
        keys = self._load_all() if self._path.exists() else {}
        master_key_id = uuid4().hex
        if master_key_id in keys:
            raise KeyProtectionError("文件型 master key ID 冲突")

        # 2. 追加新密钥后原子发布完整密钥文件
        master_key = secrets.token_bytes(_MASTER_KEY_SIZE)
        keys[master_key_id] = master_key
        self._write_all(keys)
        return master_key_id, master_key

    def load(self, master_key_id: str) -> bytes:
        """从私有文件精确读取 manifest 指定的 master key。"""

        if not _IDENTIFIER_PATTERN.fullmatch(master_key_id):
            raise KeyProtectionError("manifest master_key_id 格式无效")
        keys = self._load_all()
        try:
            return keys[master_key_id]
        except KeyError as error:
            raise KeyProtectionError("文件型 master key 不存在") from error

    def import_key(self, master_key_id: str, master_key: bytes) -> None:
        """为离线迁移导入一个既有 master key，且不允许覆盖不同内容。"""

        if not _IDENTIFIER_PATTERN.fullmatch(master_key_id):
            raise KeyProtectionError("master_key_id 格式无效")
        if len(master_key) != _MASTER_KEY_SIZE:
            raise KeyProtectionError("master key 长度无效")
        keys = self._load_all() if self._path.exists() else {}
        existing = keys.get(master_key_id)
        if existing is not None:
            if existing != master_key:
                raise KeyProtectionError("同一 master_key_id 已存在不同内容")
            return
        keys[master_key_id] = master_key
        self._write_all(keys)

    def _load_all(self) -> dict[str, bytes]:
        """校验文件身份、权限和严格 JSON schema 后加载全部密钥。"""

        try:
            metadata = self._path.lstat()
        except OSError as error:
            raise KeyProtectionError(f"master key 文件不可访问: {self._path}") from error
        if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise KeyProtectionError("master key 文件必须是 0600 普通文件")
        raw = _read_json_object(self._path, max_bytes=_MAX_MASTER_KEY_FILE_BYTES)
        _require_exact_keys(raw, {"format_version", "keys"})
        if raw["format_version"] != _FORMAT_VERSION:
            raise KeyProtectionError("master key 文件 format_version 无效")
        encoded_keys = raw["keys"]
        if not isinstance(encoded_keys, dict):
            raise KeyProtectionError("master key 文件 keys 必须是 object")
        keys: dict[str, bytes] = {}
        for key_id, encoded in cast(dict[object, object], encoded_keys).items():
            if not isinstance(key_id, str) or not _IDENTIFIER_PATTERN.fullmatch(key_id):
                raise KeyProtectionError("master key 文件包含无效 ID")
            if not isinstance(encoded, str):
                raise KeyProtectionError("master key 文件包含非字符串密钥")
            try:
                key = base64.b64decode(encoded, validate=True)
            except ValueError as error:
                raise KeyProtectionError("master key 文件包含无效 base64") from error
            if len(key) != _MASTER_KEY_SIZE:
                raise KeyProtectionError("master key 文件包含错误长度密钥")
            keys[key_id] = key
        return keys

    def _write_all(self, keys: dict[str, bytes]) -> None:
        """以固定 schema 和私有权限原子发布完整密钥集合。"""

        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._path.parent.chmod(0o700)
        _atomic_write_private(
            self._path,
            _canonical_json(
                {
                    "format_version": _FORMAT_VERSION,
                    "keys": {
                        key_id: base64.b64encode(keys[key_id]).decode("ascii")
                        for key_id in sorted(keys)
                    },
                }
            ),
        )


class EncryptedKeyBlobCodec:
    @staticmethod
    def encrypt(
        private_key: bytes,
        *,
        master_key: bytes,
        server_id: str,
        purpose: str,
        keyset_version: int,
        public_fingerprint: str,
    ) -> bytes:
        """使用 AES-256-GCM 生成带严格头部和 AAD 的密钥 blob。"""

        _validate_crypto_parameters(
            master_key=master_key,
            server_id=server_id,
            purpose=purpose,
            keyset_version=keyset_version,
            public_fingerprint=public_fingerprint,
        )
        nonce = secrets.token_bytes(12)
        aad = _key_aad(server_id, purpose, keyset_version, public_fingerprint)
        encrypted = AESGCM(master_key).encrypt(nonce, private_key, aad)
        return (
            _HEADER.pack(
                _MAGIC,
                _FORMAT_VERSION,
                _PURPOSE_CODES[purpose],
                keyset_version,
                nonce,
            )
            + encrypted
        )

    @staticmethod
    def decrypt(
        blob: bytes,
        *,
        master_key: bytes,
        server_id: str,
        purpose: str,
        keyset_version: int,
        public_fingerprint: str,
    ) -> bytearray:
        """校验 blob 头部与 GCM tag 后返回可擦除的私钥缓冲区。"""

        _validate_crypto_parameters(
            master_key=master_key,
            server_id=server_id,
            purpose=purpose,
            keyset_version=keyset_version,
            public_fingerprint=public_fingerprint,
        )
        if len(blob) < _HEADER.size + _TAG_SIZE + 1:
            raise KeyProtectionError("密钥 blob 长度无效")
        magic, format_version, purpose_code, blob_version, nonce = _HEADER.unpack(
            blob[: _HEADER.size]
        )
        if magic != _MAGIC or format_version != _FORMAT_VERSION:
            raise KeyProtectionError("密钥 blob 格式无效")
        if _PURPOSE_NAMES.get(purpose_code) != purpose:
            raise KeyProtectionError("密钥 blob purpose 不匹配")
        if blob_version != keyset_version:
            raise KeyProtectionError("密钥 blob keyset_version 不匹配")

        aad = _key_aad(server_id, purpose, keyset_version, public_fingerprint)
        try:
            plaintext = AESGCM(master_key).decrypt(
                nonce,
                blob[_HEADER.size :],
                aad,
            )
        except InvalidTag as error:
            raise KeyProtectionError("密钥 blob 完整性校验失败") from error
        return bytearray(plaintext)


class KeysetManager:
    def __init__(self, keys_root: Path, master_keys: MasterKeyStore) -> None:
        self._keys_root = keys_root
        self._master_keys = master_keys

    def initialize(self, *, lan_hostname: str) -> LoadedKeyset:
        """创建首个加密 keyset，并在完整回读验证后发布 current 指针。"""

        current_path = self._keys_root / "current.json"
        if current_path.exists():
            raise KeyProtectionError("mobile keyset 已初始化")

        # 1. 在内存生成稳定应用身份、LAN TLS 密钥和公开证书
        server_id = uuid4().hex
        identity_key = ec.generate_private_key(ec.SECP256R1())
        tls_key = ec.generate_private_key(ec.SECP256R1())
        certificate = _build_self_signed_certificate(tls_key, lan_hostname)
        master_key_id, master_key = self._master_keys.create()

        # 2. 写入未发布的新版本并完成独立回读
        self._write_keyset(
            keyset_version=1,
            server_id=server_id,
            master_key_id=master_key_id,
            master_key=master_key,
            identity_key=identity_key,
            tls_key=tls_key,
            certificate=certificate,
        )
        loaded = self._load_version(1)

        # 3. 最后原子发布唯一 current 指针
        self._write_current(1)
        return loaded

    def load(self, *, expected_server_fingerprint: str | None = None) -> LoadedKeyset:
        """从 current 指针加载并交叉校验当前加密 keyset。"""

        current = _read_json_object(
            self._keys_root / "current.json",
            max_bytes=_MAX_JSON_BYTES,
        )
        _require_exact_keys(current, {"format_version", "keyset_version", "manifest"})
        if current["format_version"] != _FORMAT_VERSION:
            raise KeyProtectionError("current.json format_version 无效")
        keyset_version = _require_positive_int(
            current["keyset_version"], "keyset_version"
        )
        expected_manifest = f"keyset-v{keyset_version}/manifest.json"
        if current["manifest"] != expected_manifest:
            raise KeyProtectionError("current.json manifest 路径无效")
        loaded = self._load_version(keyset_version)
        if (
            expected_server_fingerprint is not None
            and loaded.server_fingerprint != expected_server_fingerprint
        ):
            raise KeyProtectionError("server identity fingerprint 与数据库不一致")
        return loaded

    def rotate_master_key(self) -> LoadedKeyset:
        """重新加密同一组私钥，验证成功后再切换 current 指针。"""

        # 1. 先完整加载旧版本，身份和 TLS 公钥不得变化
        current = self.load()
        new_version = current.manifest.keyset_version + 1
        master_key_id, master_key = self._master_keys.create()

        # 2. 用新 master key 写入和验证未发布版本
        self._write_keyset(
            keyset_version=new_version,
            server_id=current.manifest.server_id,
            master_key_id=master_key_id,
            master_key=master_key,
            identity_key=current.identity_private_key,
            tls_key=current.tls_private_key,
            certificate=current.tls_certificate,
        )
        rotated = self._load_version(new_version)
        if rotated.server_fingerprint != current.server_fingerprint:
            raise KeyProtectionError("master key 轮换改变了 server identity")
        if rotated.tls_spki_fingerprint != current.tls_spki_fingerprint:
            raise KeyProtectionError("master key 轮换改变了 LAN TLS 公钥")

        # 3. 只有验证成功才原子切换，旧版本继续保留供回滚
        self._write_current(new_version)
        return rotated

    def _write_keyset(
        self,
        *,
        keyset_version: int,
        server_id: str,
        master_key_id: str,
        master_key: bytes,
        identity_key: ec.EllipticCurvePrivateKey,
        tls_key: ec.EllipticCurvePrivateKey,
        certificate: x509.Certificate,
    ) -> None:
        """在 staging 目录中完整写入一个不可变 keyset。"""

        self._ensure_private_directory(self._keys_root)
        final_dir = self._keys_root / f"keyset-v{keyset_version}"
        if final_dir.exists():
            raise KeyProtectionError(f"keyset 版本已存在: {keyset_version}")
        staging_dir = self._keys_root / (
            f".keyset-v{keyset_version}.{secrets.token_hex(8)}.tmp"
        )
        staging_dir.mkdir(mode=0o700)

        # 1. 序列化只存在于内存，并分别生成绑定 AAD 的密文
        identity_bytes = bytearray(_serialize_private_key(identity_key))
        tls_bytes = bytearray(_serialize_private_key(tls_key))
        try:
            identity_fingerprint = public_key_fingerprint(identity_key.public_key())
            tls_fingerprint = public_key_fingerprint(tls_key.public_key())
            identity_blob = EncryptedKeyBlobCodec.encrypt(
                bytes(identity_bytes),
                master_key=master_key,
                server_id=server_id,
                purpose="identity",
                keyset_version=keyset_version,
                public_fingerprint=identity_fingerprint,
            )
            tls_blob = EncryptedKeyBlobCodec.encrypt(
                bytes(tls_bytes),
                master_key=master_key,
                server_id=server_id,
                purpose="lan_tls",
                keyset_version=keyset_version,
                public_fingerprint=tls_fingerprint,
            )
        finally:
            _zeroize(identity_bytes)
            _zeroize(tls_bytes)

        certificate_pem = certificate.public_bytes(serialization.Encoding.PEM)
        manifest: dict[str, object] = {
            "format_version": _FORMAT_VERSION,
            "keyset_version": keyset_version,
            "server_id": server_id,
            "master_key_id": master_key_id,
            "identity": _entry_json(
                "server-identity.key.enc",
                "identity",
                identity_fingerprint,
                identity_blob,
            ),
            "lan_tls": _entry_json(
                "lan-tls.key.enc", "lan_tls", tls_fingerprint, tls_blob
            ),
            "tls_certificate_path": "lan-tls.cert.pem",
            "tls_certificate_sha256": hashlib.sha256(certificate_pem).hexdigest(),
        }

        # 2. 所有内容落盘并 fsync 后，才把 staging 目录发布为版本目录
        try:
            _atomic_write_private(
                staging_dir / "server-identity.key.enc", identity_blob
            )
            _atomic_write_private(staging_dir / "lan-tls.key.enc", tls_blob)
            _atomic_write_private(staging_dir / "lan-tls.cert.pem", certificate_pem)
            _atomic_write_private(
                staging_dir / "manifest.json",
                _canonical_json(manifest),
            )
            _ = staging_dir.replace(final_dir)
            _fsync_directory(self._keys_root)
        except BaseException as write_error:
            try:
                shutil.rmtree(staging_dir)
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                write_error.add_note(f"staging 目录清理失败: {cleanup_error}")
            raise

    def _load_version(self, keyset_version: int) -> LoadedKeyset:
        """校验并解密指定不可变 keyset。"""

        keyset_dir = _safe_existing_path(
            self._keys_root,
            f"keyset-v{keyset_version}",
            expect_directory=True,
        )
        manifest_path = _safe_existing_path(
            keyset_dir,
            "manifest.json",
            expect_directory=False,
        )
        manifest = _parse_manifest(
            _read_json_object(manifest_path, max_bytes=_MAX_JSON_BYTES)
        )
        if manifest.keyset_version != keyset_version:
            raise KeyProtectionError("manifest keyset_version 与目录不一致")
        master_key = self._master_keys.load(manifest.master_key_id)

        # 1. 密文内容 hash 在解密前阻止文件替换和截断
        identity_blob = _read_entry_blob(keyset_dir, manifest.identity)
        tls_blob = _read_entry_blob(keyset_dir, manifest.lan_tls)
        identity_plaintext = EncryptedKeyBlobCodec.decrypt(
            identity_blob,
            master_key=master_key,
            server_id=manifest.server_id,
            purpose="identity",
            keyset_version=keyset_version,
            public_fingerprint=manifest.identity.public_fingerprint,
        )
        tls_plaintext = EncryptedKeyBlobCodec.decrypt(
            tls_blob,
            master_key=master_key,
            server_id=manifest.server_id,
            purpose="lan_tls",
            keyset_version=keyset_version,
            public_fingerprint=manifest.lan_tls.public_fingerprint,
        )

        # 2. 解密结果必须是 P-256 私钥，且公钥与 manifest 一致
        try:
            identity_key = _load_p256_private_key(identity_plaintext, "identity")
            tls_key = _load_p256_private_key(tls_plaintext, "lan_tls")
        finally:
            _zeroize(identity_plaintext)
            _zeroize(tls_plaintext)
        if (
            public_key_fingerprint(identity_key.public_key())
            != manifest.identity.public_fingerprint
        ):
            raise KeyProtectionError("server identity public fingerprint 不匹配")
        if (
            public_key_fingerprint(tls_key.public_key())
            != manifest.lan_tls.public_fingerprint
        ):
            raise KeyProtectionError("LAN TLS public fingerprint 不匹配")

        # 3. 公开证书也属于不可信文件输入，必须绑定 TLS 私钥
        certificate_path = _safe_existing_path(
            keyset_dir,
            manifest.tls_certificate_path,
            expect_directory=False,
        )
        certificate_pem = _read_limited(certificate_path, _MAX_BLOB_BYTES)
        if (
            hashlib.sha256(certificate_pem).hexdigest()
            != manifest.tls_certificate_sha256
        ):
            raise KeyProtectionError("LAN TLS certificate hash 不匹配")
        certificate = x509.load_pem_x509_certificate(certificate_pem)
        certificate_key = certificate.public_key()
        if not isinstance(certificate_key, ec.EllipticCurvePublicKey):
            raise KeyProtectionError("LAN TLS certificate 公钥类型无效")
        if (
            public_key_fingerprint(certificate_key)
            != manifest.lan_tls.public_fingerprint
        ):
            raise KeyProtectionError("LAN TLS certificate 与私钥不匹配")
        return LoadedKeyset(
            manifest=manifest,
            identity_private_key=identity_key,
            tls_private_key=tls_key,
            tls_certificate=certificate,
            tls_certificate_path=certificate_path,
        )

    def _write_current(self, keyset_version: int) -> None:
        _atomic_write_private(
            self._keys_root / "current.json",
            _canonical_json(
                {
                    "format_version": _FORMAT_VERSION,
                    "keyset_version": keyset_version,
                    "manifest": f"keyset-v{keyset_version}/manifest.json",
                }
            ),
        )

    @staticmethod
    def _ensure_private_directory(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.chmod(0o700)


def create_server_ssl_context(keyset: LoadedKeyset) -> ssl.SSLContext:
    """只通过 Linux memfd 把解密后的 TLS 私钥交给 OpenSSL。"""

    if not Path("/proc/self/fd").is_dir():
        raise KeyProtectionError("当前平台不支持 TLS 私钥 memfd 加载")
    private_bytes = bytearray(_serialize_private_key(keyset.tls_private_key))
    fd = -1
    try:
        fd = _create_memfd("akasic-mobile-tls-key")
        os.fchmod(fd, 0o600)
        with os.fdopen(os.dup(fd), "wb", closefd=True) as stream:
            _ = stream.write(private_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(
            certfile=str(keyset.tls_certificate_path),
            keyfile=f"/proc/self/fd/{fd}",
        )
        return context
    except (OSError, ssl.SSLError) as error:
        raise KeyProtectionError(f"LAN TLS SSLContext 初始化失败: {error}") from error
    finally:
        _zeroize(private_bytes)
        if fd != -1:
            os.close(fd)


def public_key_fingerprint(public_key: ec.EllipticCurvePublicKey) -> str:
    der = public_key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    digest = hashlib.sha256(der).digest()
    encoded = base64.b64encode(digest).decode("ascii")
    return f"sha256/{encoded}"


def _create_memfd(name: str) -> int:
    """调用 Python 或 libc 暴露的 Linux memfd_create。"""

    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, flags=os.MFD_CLOEXEC)

    # 部分 Python 构建未导出 os.memfd_create，仍要求内核走同一 memfd 边界。
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        memfd_create = libc.memfd_create
    except AttributeError as error:
        raise KeyProtectionError("当前 libc 不支持 memfd_create") from error
    memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    memfd_create.restype = ctypes.c_int
    fd = memfd_create(name.encode("utf-8"), 1)
    if fd == -1:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return fd


def _build_self_signed_certificate(
    private_key: ec.EllipticCurvePrivateKey,
    lan_hostname: str,
) -> x509.Certificate:
    """为固定公钥生成供 QR pin 的本地自签名服务器证书。"""

    hostname = lan_hostname.strip()
    if not hostname or len(hostname) > 253:
        raise ValueError("lan_hostname 格式无效")
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Akasic Agent"),
            x509.NameAttribute(NameOID.COMMON_NAME, hostname),
        ]
    )
    try:
        san_name: x509.GeneralName = x509.IPAddress(ipaddress.ip_address(hostname))
    except ValueError:
        san_name = x509.DNSName(hostname)
    now = datetime.now(timezone.utc)
    return (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName([san_name]), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )


def _serialize_private_key(private_key: ec.EllipticCurvePrivateKey) -> bytes:
    return private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def _load_p256_private_key(
    data: bytearray,
    purpose: str,
) -> ec.EllipticCurvePrivateKey:
    try:
        private_key = serialization.load_pem_private_key(bytes(data), password=None)
    except (TypeError, ValueError) as error:
        raise KeyProtectionError(f"{purpose} 私钥解析失败") from error
    if not isinstance(private_key, ec.EllipticCurvePrivateKey) or not isinstance(
        private_key.curve, ec.SECP256R1
    ):
        raise KeyProtectionError(f"{purpose} 私钥必须是 ECDSA P-256")
    return private_key


def _validate_crypto_parameters(
    *,
    master_key: bytes,
    server_id: str,
    purpose: str,
    keyset_version: int,
    public_fingerprint: str,
) -> None:
    if len(master_key) != _MASTER_KEY_SIZE:
        raise KeyProtectionError("AES-256-GCM master key 长度无效")
    if not _IDENTIFIER_PATTERN.fullmatch(server_id):
        raise KeyProtectionError("server_id 格式无效")
    if purpose not in _PURPOSE_CODES:
        raise KeyProtectionError("key purpose 无效")
    if type(keyset_version) is not int or keyset_version <= 0:
        raise KeyProtectionError("keyset_version 必须大于 0")
    if not _FINGERPRINT_PATTERN.fullmatch(public_fingerprint):
        raise KeyProtectionError("public fingerprint 格式无效")


def _key_aad(
    server_id: str,
    purpose: str,
    keyset_version: int,
    public_fingerprint: str,
) -> bytes:
    return _canonical_json(
        {
            "key_purpose": purpose,
            "keyset_version": keyset_version,
            "public_fingerprint": public_fingerprint,
            "server_id": server_id,
        }
    )


def _entry_json(
    path: str,
    purpose: str,
    public_fingerprint: str,
    blob: bytes,
) -> dict[str, object]:
    return {
        "path": path,
        "purpose": purpose,
        "public_fingerprint": public_fingerprint,
        "sha256": hashlib.sha256(blob).hexdigest(),
    }


def _parse_manifest(raw: dict[str, object]) -> KeysetManifest:
    """在文件边界把 manifest 转成已验证的不可变对象。"""

    _require_exact_keys(
        raw,
        {
            "format_version",
            "keyset_version",
            "server_id",
            "master_key_id",
            "identity",
            "lan_tls",
            "tls_certificate_path",
            "tls_certificate_sha256",
        },
    )
    if raw["format_version"] != _FORMAT_VERSION:
        raise KeyProtectionError("manifest format_version 无效")
    keyset_version = _require_positive_int(raw["keyset_version"], "keyset_version")
    server_id = _require_identifier(raw["server_id"], "server_id")
    master_key_id = _require_identifier(raw["master_key_id"], "master_key_id")
    certificate_path = _require_safe_relative_file(
        raw["tls_certificate_path"], "tls_certificate_path"
    )
    certificate_hash = _require_sha256(
        raw["tls_certificate_sha256"], "tls_certificate_sha256"
    )
    return KeysetManifest(
        keyset_version=keyset_version,
        server_id=server_id,
        master_key_id=master_key_id,
        identity=_parse_entry(raw["identity"], "identity"),
        lan_tls=_parse_entry(raw["lan_tls"], "lan_tls"),
        tls_certificate_path=certificate_path,
        tls_certificate_sha256=certificate_hash,
    )


def _parse_entry(raw: object, expected_purpose: str) -> KeyBlobEntry:
    if not isinstance(raw, dict):
        raise KeyProtectionError(f"manifest {expected_purpose} 必须是 object")
    entry = cast(dict[str, object], raw)
    _require_exact_keys(entry, {"path", "purpose", "public_fingerprint", "sha256"})
    if entry["purpose"] != expected_purpose:
        raise KeyProtectionError(f"manifest {expected_purpose} purpose 无效")
    fingerprint = entry["public_fingerprint"]
    if not isinstance(fingerprint, str) or not _FINGERPRINT_PATTERN.fullmatch(
        fingerprint
    ):
        raise KeyProtectionError(f"manifest {expected_purpose} fingerprint 无效")
    return KeyBlobEntry(
        path=_require_safe_relative_file(entry["path"], f"{expected_purpose}.path"),
        purpose=expected_purpose,
        public_fingerprint=fingerprint,
        sha256=_require_sha256(entry["sha256"], f"{expected_purpose}.sha256"),
    )


def _read_entry_blob(keyset_dir: Path, entry: KeyBlobEntry) -> bytes:
    path = _safe_existing_path(keyset_dir, entry.path, expect_directory=False)
    blob = _read_limited(path, _MAX_BLOB_BYTES)
    if hashlib.sha256(blob).hexdigest() != entry.sha256:
        raise KeyProtectionError(f"{entry.purpose} 密钥 blob hash 不匹配")
    return blob


def _safe_existing_path(root: Path, relative: str, *, expect_directory: bool) -> Path:
    candidate = root / _require_safe_relative_file(relative, "path")
    if candidate.is_symlink():
        raise KeyProtectionError(f"keyset 路径不允许符号链接: {relative}")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise KeyProtectionError(f"keyset 路径不可访问: {candidate}") from error
    if not resolved.is_relative_to(resolved_root):
        raise KeyProtectionError("keyset 路径逃逸")
    if expect_directory != resolved.is_dir():
        expected = "目录" if expect_directory else "文件"
        raise KeyProtectionError(f"keyset 路径必须是{expected}: {relative}")
    return resolved


def _require_safe_relative_file(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise KeyProtectionError(f"{field} 必须是非空字符串")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise KeyProtectionError(f"{field} 必须是安全相对路径")
    return value


def _read_json_object(path: Path, *, max_bytes: int) -> dict[str, object]:
    data = _read_limited(path, max_bytes)

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise KeyProtectionError(f"JSON 存在重复字段: {key}")
            result[key] = value
        return result

    try:
        parsed = json.loads(data, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise KeyProtectionError(f"JSON 解析失败: {path}") from error
    if not isinstance(parsed, dict):
        raise KeyProtectionError(f"JSON 顶层必须是 object: {path}")
    return cast(dict[str, object], parsed)


def _read_limited(path: Path, max_bytes: int) -> bytes:
    fd = -1
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        size = os.fstat(fd).st_size
        if size <= 0 or size > max_bytes:
            raise KeyProtectionError(f"文件大小无效: {path}")
        with os.fdopen(fd, "rb") as stream:
            fd = -1
            data = stream.read(max_bytes + 1)
        if len(data) != size:
            raise KeyProtectionError(f"文件读取期间发生变化: {path}")
        return data
    except OSError as error:
        raise KeyProtectionError(f"文件读取失败: {path}") from error
    finally:
        if fd != -1:
            os.close(fd)


def _require_exact_keys(raw: dict[str, object], expected: set[str]) -> None:
    if raw.keys() != expected:
        missing = sorted(expected - raw.keys())
        extra = sorted(raw.keys() - expected)
        raise KeyProtectionError(f"JSON 字段不匹配: missing={missing} extra={extra}")


def _require_positive_int(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise KeyProtectionError(f"{field} 必须是正整数")
    return value


def _require_identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise KeyProtectionError(f"{field} 格式无效")
    return value


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise KeyProtectionError(f"{field} 必须是 SHA-256 hex")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _atomic_write_private(path: Path, data: bytes) -> None:
    """以 0600 原子写文件，并同步文件与目录项。"""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            fd = -1
            _ = stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _ = temporary.replace(path)
        path.chmod(0o600)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if fd != -1:
            os.close(fd)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _zeroize(data: bytearray) -> None:
    data[:] = b"\x00" * len(data)


__all__ = [
    "EncryptedKeyBlobCodec",
    "FileMasterKeyStore",
    "KeyProtectionError",
    "KeysetManager",
    "LoadedKeyset",
    "MasterKeyStore",
    "SecretServiceMasterKeyStore",
    "create_server_ssl_context",
    "public_key_fingerprint",
]
