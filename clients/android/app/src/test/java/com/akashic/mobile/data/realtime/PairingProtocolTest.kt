package com.akashic.mobile.data.realtime

import java.security.KeyPairGenerator
import java.security.Signature
import java.security.spec.ECGenParameterSpec
import java.time.Instant
import java.util.Base64
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class PairingProtocolTest {
    @Test
    fun `claim transcript matches Python ensure-ascii vector`() {
        var transcript = byteArrayOf()
        val qr = PairingQrPayload(
            protocolVersion = 1,
            serverId = "server-测试",
            serverApplicationKeyFingerprint = "unused",
            serverApplicationPublicKey = "unused",
            lanEndpoints = listOf("wss://pc.local:6323/ws"),
            tunnelEndpoints = emptyList(),
            tlsSpkiPins = listOf("sha256/${"A".repeat(43)}="),
            pairingId = "pair-1",
            oneTimeSecret = "A".repeat(43),
            expiresAt = "2099-01-01T00:00:00Z",
        )
        val claim = PairingTranscripts.createClaim(
            qr = qr,
            devicePublicKeyDer = "B".repeat(124).toByteArray(),
            deviceName = "花月的 Pixel 🚀",
            capabilities = listOf("chat", "streaming", "tools", "proactive"),
            clientNonce = "C".repeat(43),
            signer = { transcript = it; byteArrayOf(1, 2, 3) },
        )

        assertEquals(PYTHON_TRANSCRIPT_BASE64, Base64.getEncoder().encodeToString(transcript))
        assertEquals("970976", claim.confirmationCode)
    }

    @Test
    fun `strict QR binds P-256 public key and rejects duplicate fields`() {
        val keys = KeyPairGenerator.getInstance("EC").run {
            initialize(ECGenParameterSpec("secp256r1"))
            generateKeyPair()
        }
        val payload = PairingQrPayload(
            protocolVersion = 1,
            serverId = "server-1",
            serverApplicationKeyFingerprint = spkiSha256Pin(keys.public),
            serverApplicationPublicKey = Base64.getEncoder().encodeToString(keys.public.encoded),
            lanEndpoints = listOf("wss://pc.local:6323/ws"),
            tunnelEndpoints = listOf("wss://mobile.example.com/ws"),
            tlsSpkiPins = listOf(spkiSha256Pin(keys.public)),
            pairingId = "pair-1",
            oneTimeSecret = "A".repeat(43),
            expiresAt = "2099-01-01T00:00:00Z",
        )
        val encoded = Json.encodeToString(payload)

        assertEquals(payload, PairingQrDecoder.decode(encoded, Instant.parse("2026-01-01T00:00:00Z")))
        val duplicate = encoded.replaceFirst("\"protocol_version\":1", "\"protocol_version\":1,\"protocol_version\":1")
        assertThrows(IllegalArgumentException::class.java) { PairingQrDecoder.decode(duplicate) }
        assertThrows(IllegalArgumentException::class.java) {
            PairingQrDecoder.decode(encoded.dropLast(1) + ",\"unknown\":true}")
        }
    }

    @Test
    fun `strict QR rejects a JSON literal with a stable boundary error`() {
        val error = assertThrows(IllegalArgumentException::class.java) {
            PairingQrDecoder.decode("\"not-a-pairing-object\"")
        }

        assertEquals("二维码内容不是配对对象", error.message)
    }

    @Test
    fun `server challenge requires cached fingerprint and valid signature`() {
        val keys = KeyPairGenerator.getInstance("EC").run {
            initialize(ECGenParameterSpec("secp256r1"))
            generateKeyPair()
        }
        val publicKey = Base64.getEncoder().encodeToString(keys.public.encoded)
        val fingerprint = spkiSha256Pin(keys.public)
        val unsigned = ServerChallengePayload(
            challengeId = "challenge-1",
            serverId = "server-1",
            serverPublicKey = publicKey,
            serverFingerprint = fingerprint,
            nonce = "N".repeat(43),
            expiresAt = "2099-01-01T00:00:00Z",
            signature = "",
        )
        val transcript = canonicalObject(
            "challenge_id" to unsigned.challengeId,
            "expires_at" to unsigned.expiresAt,
            "nonce" to unsigned.nonce,
            "protocol_version" to 1,
            "server_fingerprint" to unsigned.serverFingerprint,
            "server_id" to unsigned.serverId,
            "server_public_key" to unsigned.serverPublicKey,
        )
        val signature = Signature.getInstance("SHA256withECDSA").run {
            initSign(keys.private)
            update(transcript)
            sign()
        }
        val challenge = unsigned.copy(signature = Base64.getEncoder().encodeToString(signature))

        PairingTranscripts.verifyServerChallenge(
            challenge,
            expectedServerId = "server-1",
            expectedFingerprint = fingerprint,
            now = Instant.parse("2026-01-01T00:00:00Z"),
        )
        assertThrows(IllegalArgumentException::class.java) {
            PairingTranscripts.verifyServerChallenge(
                challenge,
                expectedServerId = "server-1",
                expectedFingerprint = "sha256/${"A".repeat(43)}=",
            )
        }
    }

    private companion object {
        const val PYTHON_TRANSCRIPT_BASE64 =
            "eyJjYXBhYmlsaXRpZXMiOlsiY2hhdCIsInN0cmVhbWluZyIsInRvb2xzIiwicHJvYWN0aXZlIl0sImNsaWVudF9ub25jZSI6IkNDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0MiLCJkZXZpY2VfbmFtZSI6Ilx1ODJiMVx1NjcwOFx1NzY4NCBQaXhlbCBcdWQ4M2RcdWRlODAiLCJkZXZpY2VfcHVibGljX2tleSI6IlFrSkNRa0pDUWtKQ1FrSkNRa0pDUWtKQ1FrSkNRa0pDUWtKQ1FrSkNRa0pDUWtKQ1FrSkNRa0pDUWtKQ1FrSkNRa0pDUWtKQ1FrSkNRa0pDUWtKQ1FrSkNRa0pDUWtKQ1FrSkNRa0pDUWtKQ1FrSkNRa0pDUWtKQ1FrSkNRa0pDUWtKQ1FrSkNRa0pDUWtKQ1FrSkNRa0pDUWtKQ1FrSkNRa0pDUWc9PSIsInBhaXJpbmdfaWQiOiJwYWlyLTEiLCJwcm90b2NvbF92ZXJzaW9uIjoxLCJzZWNyZXRfaGFzaCI6ImQ0YWVmZDlmZDVmM2RkNjY4YjMyMGNkOWM5N2U2YzZjYjJiOWNhMjJhMTI4ZTgwNTkzZTIyMzQ2YWNiODgwNTkiLCJzZXJ2ZXJfaWQiOiJzZXJ2ZXItXHU2ZDRiXHU4YmQ1In0="
    }
}
