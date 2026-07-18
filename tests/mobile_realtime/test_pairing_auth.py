from __future__ import annotations

import base64
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from infra.mobile_realtime.auth import (
    DeviceAuthenticator,
    DeviceProofPayload,
    DeviceRevokedError,
    UnknownAuthenticationChallenge,
    device_proof_signing_bytes,
    server_challenge_signing_bytes,
)
from infra.mobile_realtime.key_protection import (
    KeyProtectionError,
    KeysetManager,
    LoadedKeyset,
)
from infra.mobile_realtime.pairing import (
    PairClaimPayload,
    PairingConfirmationError,
    PairingSecretError,
    PairingService,
    PairingSignatureError,
    pair_claim_signing_bytes,
    parse_device_public_key,
)
from infra.mobile_realtime.storage import MobileRealtimeStorage, PairingStateError


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


def _device_public_key(private_key: ec.EllipticCurvePrivateKey) -> str:
    encoded = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return base64.b64encode(encoded).decode("ascii")


def _signed_claim(
    service: PairingService,
    private_key: ec.EllipticCurvePrivateKey,
) -> tuple[PairClaimPayload, str]:
    offer = service.create_offer()
    public_key = _device_public_key(private_key)
    client_nonce = base64.urlsafe_b64encode(secrets.token_bytes(18)).decode("ascii")
    transcript = pair_claim_signing_bytes(
        server_id=offer.server_id,
        pairing_id=offer.pairing_id,
        one_time_secret=offer.one_time_secret,
        device_public_key=public_key,
        device_name="Pixel Emulator",
        capabilities=["stream-v1", "attachments-v1"],
        client_nonce=client_nonce,
    )
    signature = private_key.sign(transcript, ec.ECDSA(hashes.SHA256()))
    return (
        PairClaimPayload(
            pairing_id=offer.pairing_id,
            one_time_secret=offer.one_time_secret,
            device_public_key=public_key,
            device_name="Pixel Emulator",
            capabilities=["stream-v1", "attachments-v1"],
            client_nonce=client_nonce,
            signature=base64.b64encode(signature).decode("ascii"),
        ),
        offer.server_application_key_fingerprint,
    )


def _services(
    tmp_path: Path,
) -> tuple[MobileRealtimeStorage, PairingService, LoadedKeyset]:
    keyset = KeysetManager(
        tmp_path / "keys",
        _EphemeralMasterKeys(),
    ).initialize(lan_hostname="akashic.local")
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    service = PairingService(
        storage,
        keyset,
        lan_endpoints=("wss://akashic.local:6323/ws",),
        tunnel_endpoints=("wss://agent.example.com/ws",),
    )
    return storage, service, keyset


def test_pairing_requires_signed_claim_and_desktop_confirmation(tmp_path: Path) -> None:
    storage, service, _ = _services(tmp_path)
    device_key = ec.generate_private_key(ec.SECP256R1())
    payload, _ = _signed_claim(service, device_key)

    claim = service.claim(payload)
    with pytest.raises(PairingConfirmationError, match="确认码不一致"):
        service.approve(payload.pairing_id, "000000")

    device = service.approve(payload.pairing_id, claim.confirmation_code)
    session = storage.read_pairing_session(payload.pairing_id)

    assert storage.read_device(device.device_id) == device
    assert session is not None
    assert session.status == "consumed"
    assert session.secret_hash is None
    assert service.pending_claim(payload.pairing_id) is None
    with pytest.raises(PairingStateError, match="不能 claim"):
        service.claim(payload)
    storage.close()


def test_pairing_rejects_wrong_secret_and_signature(tmp_path: Path) -> None:
    storage, service, _ = _services(tmp_path)
    device_key = ec.generate_private_key(ec.SECP256R1())
    payload, _ = _signed_claim(service, device_key)

    wrong_secret = payload.model_copy(
        update={"one_time_secret": "A" * len(payload.one_time_secret)}
    )
    with pytest.raises(PairingSecretError, match="secret 无效"):
        service.claim(wrong_secret)

    wrong_signature = payload.model_copy(
        update={"signature": base64.b64encode(b"invalid-signature" * 5).decode()}
    )
    with pytest.raises(PairingSignatureError, match="签名无效"):
        service.claim(wrong_signature)
    storage.close()


def test_device_challenge_authentication_is_signed_one_shot_and_revocable(
    tmp_path: Path,
) -> None:
    storage, pairing, keyset = _services(tmp_path)
    device_key = ec.generate_private_key(ec.SECP256R1())
    claim_payload, _ = _signed_claim(pairing, device_key)
    claim = pairing.claim(claim_payload)
    device = pairing.approve(claim.pairing_id, claim.confirmation_code)
    authenticator = DeviceAuthenticator(storage, keyset)

    challenge = authenticator.create_challenge("connection-1")
    server_key = parse_device_public_key(challenge.server_public_key)
    server_key.verify(
        base64.b64decode(challenge.signature, validate=True),
        server_challenge_signing_bytes(challenge),
        ec.ECDSA(hashes.SHA256()),
    )
    client_nonce = base64.urlsafe_b64encode(secrets.token_bytes(18)).decode("ascii")
    proof_bytes = device_proof_signing_bytes(
        server_id=challenge.server_id,
        challenge_id=challenge.challenge_id,
        challenge_nonce=challenge.nonce,
        device_id=device.device_id,
        client_nonce=client_nonce,
    )
    signature = device_key.sign(proof_bytes, ec.ECDSA(hashes.SHA256()))
    proof = DeviceProofPayload(
        challenge_id=challenge.challenge_id,
        device_id=device.device_id,
        client_nonce=client_nonce,
        signature=base64.b64encode(signature).decode("ascii"),
    )

    authenticated = authenticator.authenticate("connection-1", proof)
    assert authenticated.device_id == device.device_id
    assert authenticated.connection_epoch == 1
    with pytest.raises(UnknownAuthenticationChallenge):
        authenticator.authenticate("connection-1", proof)

    restarted = DeviceAuthenticator(storage, keyset)
    restart_challenge = restarted.create_challenge("connection-after-restart")
    restart_bytes = device_proof_signing_bytes(
        server_id=restart_challenge.server_id,
        challenge_id=restart_challenge.challenge_id,
        challenge_nonce=restart_challenge.nonce,
        device_id=device.device_id,
        client_nonce=client_nonce,
    )
    restart_signature = device_key.sign(
        restart_bytes,
        ec.ECDSA(hashes.SHA256()),
    )
    after_restart = restarted.authenticate(
        "connection-after-restart",
        DeviceProofPayload(
            challenge_id=restart_challenge.challenge_id,
            device_id=device.device_id,
            client_nonce=client_nonce,
            signature=base64.b64encode(restart_signature).decode("ascii"),
        ),
    )
    assert after_restart.connection_epoch == 2

    _ = storage.revoke_device(device.device_id, revoked_at=datetime.now(timezone.utc))
    revoked_challenge = authenticator.create_challenge("connection-2")
    revoked_bytes = device_proof_signing_bytes(
        server_id=revoked_challenge.server_id,
        challenge_id=revoked_challenge.challenge_id,
        challenge_nonce=revoked_challenge.nonce,
        device_id=device.device_id,
        client_nonce=client_nonce,
    )
    revoked_signature = device_key.sign(revoked_bytes, ec.ECDSA(hashes.SHA256()))
    with pytest.raises(DeviceRevokedError):
        authenticator.authenticate(
            "connection-2",
            DeviceProofPayload(
                challenge_id=revoked_challenge.challenge_id,
                device_id=device.device_id,
                client_nonce=client_nonce,
                signature=base64.b64encode(revoked_signature).decode("ascii"),
            ),
        )
    storage.close()


def test_expired_pairing_secret_is_rejected(tmp_path: Path) -> None:
    now = datetime(2026, 7, 14, tzinfo=timezone.utc)
    current = [now]
    keyset = KeysetManager(
        tmp_path / "keys",
        _EphemeralMasterKeys(),
    ).initialize(lan_hostname="akashic.local")
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    service = PairingService(
        storage,
        keyset,
        lan_endpoints=("wss://akashic.local:6323/ws",),
        tunnel_endpoints=(),
        ttl=timedelta(seconds=1),
        clock=lambda: current[0],
    )
    device_key = ec.generate_private_key(ec.SECP256R1())
    payload, _ = _signed_claim(service, device_key)
    current[0] += timedelta(seconds=2)

    with pytest.raises(PairingSecretError, match="已过期"):
        service.claim(payload)
    storage.close()
