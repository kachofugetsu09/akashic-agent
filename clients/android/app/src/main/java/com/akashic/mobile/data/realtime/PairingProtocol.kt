package com.akashic.mobile.data.realtime

import java.net.URI
import java.security.KeyFactory
import java.security.MessageDigest
import java.security.PublicKey
import java.security.Signature
import java.security.interfaces.ECPublicKey
import java.security.spec.X509EncodedKeySpec
import java.time.Instant
import java.util.Base64
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray

private val PAIRING_SECRET_DOMAIN = "akasic-mobile-pairing-secret-v1\u0000".toByteArray(Charsets.UTF_8)
private val URL_SECRET = Regex("^[A-Za-z0-9_-]{43}$")
private val SPKI_PIN = Regex("^sha256/[A-Za-z0-9+/]{43}=$")

@Serializable
data class PairingQrPayload(
    @SerialName("protocol_version") val protocolVersion: Int,
    @SerialName("server_id") val serverId: String,
    @SerialName("server_application_key_fingerprint") val serverApplicationKeyFingerprint: String,
    @SerialName("server_application_public_key") val serverApplicationPublicKey: String,
    @SerialName("lan_endpoints") val lanEndpoints: List<String>,
    @SerialName("tunnel_endpoints") val tunnelEndpoints: List<String>,
    @SerialName("tls_spki_pins") val tlsSpkiPins: List<String>,
    @SerialName("pairing_id") val pairingId: String,
    @SerialName("one_time_secret") val oneTimeSecret: String,
    @SerialName("expires_at") val expiresAt: String,
)

data class PairClaimMaterial(
    val payload: JsonObject,
    val confirmationCode: String,
)

@Serializable
data class ServerChallengePayload(
    @SerialName("challenge_id") val challengeId: String,
    @SerialName("server_id") val serverId: String,
    @SerialName("server_public_key") val serverPublicKey: String,
    @SerialName("server_fingerprint") val serverFingerprint: String,
    val nonce: String,
    @SerialName("expires_at") val expiresAt: String,
    val signature: String,
)

object PairingQrDecoder {
    private val json = Json {
        ignoreUnknownKeys = false
        explicitNulls = false
        isLenient = false
        coerceInputValues = false
    }
    private val expectedKeys = setOf(
        "protocol_version",
        "server_id",
        "server_application_key_fingerprint",
        "server_application_public_key",
        "lan_endpoints",
        "tunnel_endpoints",
        "tls_spki_pins",
        "pairing_id",
        "one_time_secret",
        "expires_at",
    )

    /** 严格解析 QR，并在落库前绑定应用公钥、TLS pin 与 endpoint。 */
    fun decode(raw: String, now: Instant = Instant.now()): PairingQrPayload {
        // 1. 在 JSON parser 分配前限制不可信 QR 字节数
        require(raw.toByteArray(Charsets.UTF_8).size <= MAX_QR_BYTES) { "Pairing QR exceeds $MAX_QR_BYTES bytes" }

        // 2. 拒绝重复、缺失和未知顶层字段
        rejectDuplicateTopLevelKeys(raw)
        val element = json.parseToJsonElement(raw).jsonObject
        require(element.keys == expectedKeys) { "QR fields do not match the v1 schema" }
        val payload = json.decodeFromString<PairingQrPayload>(raw)

        // 3. 校验一次性凭证、有效期和有限 endpoint 集合
        require(payload.protocolVersion == WIRE_PROTOCOL_VERSION) { "Unsupported QR protocol version" }
        require(payload.serverId.isNotBlank() && payload.serverId.length <= 512) { "Invalid server_id" }
        require(payload.pairingId.isNotBlank() && payload.pairingId.length <= 512) { "Invalid pairing_id" }
        require(URL_SECRET.matches(payload.oneTimeSecret)) { "Invalid one_time_secret" }
        require(Instant.parse(payload.expiresAt).isAfter(now)) { "Pairing QR has expired" }
        require(payload.lanEndpoints.size <= 16 && payload.tunnelEndpoints.size <= 16) {
            "Too many endpoints"
        }
        require(payload.lanEndpoints.isNotEmpty() || payload.tunnelEndpoints.isNotEmpty()) {
            "Pairing QR has no endpoint"
        }
        require(payload.lanEndpoints.distinct().size == payload.lanEndpoints.size) { "Duplicate LAN endpoint" }
        require(payload.tunnelEndpoints.distinct().size == payload.tunnelEndpoints.size) {
            "Duplicate tunnel endpoint"
        }
        (payload.lanEndpoints + payload.tunnelEndpoints).forEach(::requireWssEndpoint)

        // 4. 应用身份 fingerprint 必须由 QR 中的 P-256 DER 公钥导出
        val publicKey = decodeP256PublicKey(payload.serverApplicationPublicKey)
        require(spkiSha256Pin(publicKey) == payload.serverApplicationKeyFingerprint) {
            "Server application fingerprint mismatch"
        }
        require(payload.tlsSpkiPins.isNotEmpty()) { "LAN TLS pin is required" }
        require(payload.tlsSpkiPins.all(SPKI_PIN::matches)) { "Invalid TLS SPKI pin" }
        require(payload.tlsSpkiPins.distinct().size == payload.tlsSpkiPins.size) { "Duplicate TLS pin" }
        return payload
    }

    private const val MAX_QR_BYTES = 32 * 1024
}

object PairingTranscripts {
    fun createClaim(
        qr: PairingQrPayload,
        devicePublicKeyDer: ByteArray,
        deviceName: String,
        capabilities: List<String>,
        clientNonce: String,
        signer: (ByteArray) -> ByteArray,
    ): PairClaimMaterial {
        require(deviceName.isNotBlank() && deviceName.length <= 128) { "Invalid device name" }
        require(capabilities.size == capabilities.distinct().size) { "Duplicate capability" }
        require(clientNonce.length in 16..128) { "Invalid client nonce" }
        val publicKey = Base64.getEncoder().encodeToString(devicePublicKeyDer)
        val secretHash = sha256Hex(PAIRING_SECRET_DOMAIN + qr.oneTimeSecret.toByteArray(Charsets.US_ASCII))
        val transcript = canonicalObject(
            "capabilities" to capabilities,
            "client_nonce" to clientNonce,
            "device_name" to deviceName,
            "device_public_key" to publicKey,
            "pairing_id" to qr.pairingId,
            "protocol_version" to WIRE_PROTOCOL_VERSION,
            "secret_hash" to secretHash,
            "server_id" to qr.serverId,
        )
        val signature = Base64.getEncoder().encodeToString(signer(transcript))
        return PairClaimMaterial(
            payload = buildJsonObject {
                put("pairing_id", qr.pairingId)
                put("one_time_secret", qr.oneTimeSecret)
                put("device_public_key", publicKey)
                put("device_name", deviceName)
                putJsonArray("capabilities") { capabilities.forEach { add(JsonPrimitive(it)) } }
                put("client_nonce", clientNonce)
                put("signature", signature)
            },
            confirmationCode = confirmationCode(transcript),
        )
    }

    /** 验证 challenge 的应用身份签名，返回可用于 device.proof 的 nonce。 */
    fun verifyServerChallenge(
        challenge: ServerChallengePayload,
        expectedServerId: String,
        expectedFingerprint: String,
        now: Instant = Instant.now(),
    ) {
        // 1. fingerprint 绑定 QR 已确认的应用身份
        require(challenge.serverId == expectedServerId) { "Challenge server_id mismatch" }
        require(challenge.serverFingerprint == expectedFingerprint) { "Challenge fingerprint mismatch" }
        require(Instant.parse(challenge.expiresAt).isAfter(now)) { "Server challenge has expired" }
        val publicKey = decodeP256PublicKey(challenge.serverPublicKey)
        require(spkiSha256Pin(publicKey) == expectedFingerprint) { "Challenge public key mismatch" }

        // 2. 公钥必须验证完整 challenge transcript
        val transcript = canonicalObject(
            "challenge_id" to challenge.challengeId,
            "expires_at" to challenge.expiresAt,
            "nonce" to challenge.nonce,
            "protocol_version" to WIRE_PROTOCOL_VERSION,
            "server_fingerprint" to challenge.serverFingerprint,
            "server_id" to challenge.serverId,
            "server_public_key" to challenge.serverPublicKey,
        )
        val signature = Base64.getDecoder().decode(challenge.signature)
        require(Signature.getInstance("SHA256withECDSA").run {
            initVerify(publicKey)
            update(transcript)
            verify(signature)
        }) { "Server challenge signature is invalid" }
    }

    fun createDeviceProof(
        challenge: ServerChallengePayload,
        deviceId: String,
        clientNonce: String,
        signer: (ByteArray) -> ByteArray,
    ): JsonObject {
        val transcript = canonicalObject(
            "challenge_id" to challenge.challengeId,
            "challenge_nonce" to challenge.nonce,
            "client_nonce" to clientNonce,
            "device_id" to deviceId,
            "protocol_version" to WIRE_PROTOCOL_VERSION,
            "server_id" to challenge.serverId,
        )
        return buildJsonObject {
            put("challenge_id", challenge.challengeId)
            put("device_id", deviceId)
            put("client_nonce", clientNonce)
            put("signature", Base64.getEncoder().encodeToString(signer(transcript)))
        }
    }
}

fun decodeP256PublicKey(encoded: String): ECPublicKey {
    val bytes = Base64.getDecoder().decode(encoded)
    val key = KeyFactory.getInstance("EC").generatePublic(X509EncodedKeySpec(bytes))
    require(key is ECPublicKey && key.params.curve.field.fieldSize == 256) { "Public key must be P-256" }
    return key
}

fun spkiSha256Pin(publicKey: PublicKey): String =
    "sha256/" + Base64.getEncoder().encodeToString(
        MessageDigest.getInstance("SHA-256").digest(publicKey.encoded),
    )

fun randomUrlNonce(): String {
    val bytes = ByteArray(32)
    java.security.SecureRandom().nextBytes(bytes)
    return Base64.getUrlEncoder().withoutPadding().encodeToString(bytes)
}

internal fun canonicalObject(vararg entries: Pair<String, Any>): ByteArray {
    val body = entries.sortedBy { it.first }.joinToString(",") { (key, value) ->
        "${pythonJsonString(key)}:${canonicalValue(value)}"
    }
    return "{$body}".toByteArray(Charsets.UTF_8)
}

private fun canonicalValue(value: Any): String = when (value) {
    is String -> pythonJsonString(value)
    is Int -> value.toString()
    is List<*> -> value.joinToString(prefix = "[", postfix = "]", separator = ",") {
        pythonJsonString(it as String)
    }
    else -> error("Unsupported canonical JSON type: ${value::class.java.name}")
}

private fun pythonJsonString(value: String): String = buildString(value.length + 2) {
    append('"')
    value.forEach { character ->
        when (character) {
            '"' -> append("\\\"")
            '\\' -> append("\\\\")
            '\b' -> append("\\b")
            '\u000c' -> append("\\f")
            '\n' -> append("\\n")
            '\r' -> append("\\r")
            '\t' -> append("\\t")
            else -> if (character.code < 0x20 || character.code > 0x7e) {
                append("\\u")
                append(character.code.toString(16).padStart(4, '0'))
            } else {
                append(character)
            }
        }
    }
    append('"')
}

private fun sha256Hex(value: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(value).joinToString("") { "%02x".format(it) }

private fun confirmationCode(transcript: ByteArray): String {
    val digest = MessageDigest.getInstance("SHA-256").digest(transcript)
    var value = 0L
    repeat(8) { index -> value = (value shl 8) or (digest[index].toLong() and 0xff) }
    return (value.toULong() % 1_000_000u).toString().padStart(6, '0')
}

private fun requireWssEndpoint(value: String) {
    val uri = URI(value)
    require(uri.scheme?.lowercase() == "wss" && !uri.host.isNullOrBlank()) { "Endpoint must use wss://" }
    require(uri.userInfo == null && uri.fragment == null && uri.query == null) { "Endpoint contains forbidden parts" }
    require(uri.path == "/ws") { "Endpoint path must be /ws" }
}

private fun rejectDuplicateTopLevelKeys(raw: String) {
    var depth = 0
    var index = 0
    var expectsKey = false
    val keys = mutableSetOf<String>()
    while (index < raw.length) {
        when (raw[index]) {
            '{' -> {
                depth += 1
                if (depth == 1) expectsKey = true
                index += 1
            }
            '}' -> {
                depth -= 1
                index += 1
            }
            ',' -> {
                if (depth == 1) expectsKey = true
                index += 1
            }
            '"' -> {
                val start = index
                index += 1
                while (index < raw.length) {
                    if (raw[index] == '\\') index += 2 else if (raw[index++] == '"') break
                }
                require(index <= raw.length && raw[index - 1] == '"') { "Unterminated JSON string" }
                if (depth == 1 && expectsKey) {
                    val key = Json.decodeFromString<String>(raw.substring(start, index))
                    require(keys.add(key)) { "Duplicate QR field: $key" }
                    expectsKey = false
                }
            }
            else -> index += 1
        }
        require(depth >= 0) { "Invalid QR JSON nesting" }
    }
    require(depth == 0) { "Invalid QR JSON nesting" }
}
