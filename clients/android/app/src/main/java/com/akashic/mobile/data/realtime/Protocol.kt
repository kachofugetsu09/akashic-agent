package com.akashic.mobile.data.realtime

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.decodeFromJsonElement

const val WIRE_PROTOCOL_VERSION = 1
const val MAX_JSON_FRAME_BYTES = 256 * 1024

@Serializable
enum class WireKind {
    @SerialName("command")
    COMMAND,

    @SerialName("reply")
    REPLY,

    @SerialName("event")
    EVENT,

    @SerialName("ack")
    ACK,

    @SerialName("control")
    CONTROL,
}

@Serializable
data class WireEnvelope(
    val v: Int,
    val kind: WireKind,
    val type: String,
    val id: String? = null,
    @SerialName("connection_epoch")
    val connectionEpoch: Long? = null,
    @SerialName("event_seq")
    val eventSeq: Long? = null,
    @SerialName("session_id")
    val sessionId: String? = null,
    @SerialName("turn_id")
    val turnId: String? = null,
    val payload: JsonObject = JsonObject(emptyMap()),
)

@Serializable
data class MessageSendPayload(
    @SerialName("client_message_id")
    val clientMessageId: String,
    @SerialName("session_id")
    val sessionId: String,
    val text: String,
    @SerialName("media_refs")
    val mediaRefs: List<String>,
    @SerialName("client_created_at")
    val clientCreatedAt: String,
)

@Serializable
data class AttachmentBeginPayload(
    @SerialName("attachment_id") val attachmentId: String,
    val filename: String,
    @SerialName("content_type") val contentType: String,
    @SerialName("size_bytes") val sizeBytes: Long,
    val sha256: String,
)

@Serializable
data class AttachmentBeginReplyPayload(
    @SerialName("attachment_id") val attachmentId: String,
    val filename: String,
    @SerialName("content_type") val contentType: String,
    @SerialName("size_bytes") val sizeBytes: Long,
    val sha256: String,
    @SerialName("next_offset") val nextOffset: Long,
    @SerialName("chunk_size") val chunkSize: Int,
    val state: String,
)

@Serializable
data class AttachmentProgressPayload(
    @SerialName("attachment_id") val attachmentId: String,
    @SerialName("transferred_bytes") val transferredBytes: Long,
    @SerialName("size_bytes") val sizeBytes: Long,
)

@Serializable
data class AttachmentReadyPayload(
    @SerialName("attachment_id") val attachmentId: String,
    val filename: String,
    @SerialName("content_type") val contentType: String,
    @SerialName("size_bytes") val sizeBytes: Long,
    val sha256: String,
)

@Serializable
data class AttachmentFinishPayload(
    @SerialName("attachment_id") val attachmentId: String,
)

@Serializable
data class EventAckPayload(
    @SerialName("through_event_seq")
    val throughEventSeq: Long,
)

@Serializable
data class AuthAcceptedPayload(
    @SerialName("connection_epoch")
    val connectionEpoch: Long,
    @SerialName("device_id")
    val deviceId: String,
)

@Serializable
data class ResumePayload(
    @SerialName("last_ack")
    val lastAck: Long,
    @SerialName("active_turns")
    val activeTurns: List<String>,
)

@Serializable
data class PairPendingPayload(
    @SerialName("pairing_id") val pairingId: String,
    @SerialName("confirmation_code") val confirmationCode: String,
    @SerialName("device_name") val deviceName: String,
)

@Serializable
data class PairAcceptedPayload(
    @SerialName("pairing_id") val pairingId: String,
    @SerialName("device_id") val deviceId: String,
)

@Serializable
data class SessionListPayload(
    val items: List<RemoteSessionSummary>,
)

@Serializable
data class RemoteSessionSummary(
    @SerialName("session_id") val sessionId: String,
    val title: String,
    @SerialName("updated_at") val updatedAt: String,
    @SerialName("message_count") val messageCount: Int,
)

@Serializable
data class HistoryPagePayload(
    val items: List<RemoteHistoryMessage>,
    val total: Int,
    val page: Int,
    @SerialName("page_size") val pageSize: Int,
)

@Serializable
data class RemoteHistoryMessage(
    val id: String,
    @SerialName("session_key") val sessionKey: String,
    val seq: Int,
    val role: String,
    val content: String,
    @SerialName("tool_chain") val toolChain: JsonElement? = null,
    val extra: JsonObject,
    val ts: String,
)

@Serializable
data class ProtocolErrorPayload(
    val code: Int,
    val message: String,
)

object ProtocolCodec {
    @PublishedApi
    internal val json = Json {
        encodeDefaults = true
        explicitNulls = false
        ignoreUnknownKeys = false
        isLenient = false
        coerceInputValues = false
    }

    private val knownTypes = mapOf(
        WireKind.COMMAND to setOf(
            "session.list",
            "session.create",
            "session.open",
            "history.get",
            "message.send",
            "turn.stop",
            "attachment.begin",
            "attachment.finish",
            "device.update",
            "ping",
        ),
        WireKind.EVENT to setOf(
            "session.list",
            "session.created",
            "session.updated",
            "history.page",
            "turn.started",
            "react.thinking.delta",
            "react.tool.started",
            "react.tool.completed",
            "answer.delta",
            "turn.snapshot",
            "message.final",
            "turn.interrupted",
            "message.proactive",
            "attachment.progress",
            "attachment.ready",
            "connection.degraded",
            "sync.completed",
            "sync.reset_required",
            "device.revoked",
        ),
        WireKind.ACK to setOf("event.ack"),
        WireKind.CONTROL to setOf(
            "server.challenge",
            "device.proof",
            "auth.accepted",
            "resume",
            "pair.claim",
            "pair.pending",
            "pair.accepted",
            "protocol.error",
        ),
    )

    /** 在 JSON 信任边界解析并校验 wire envelope。 */
    fun decode(text: String): WireEnvelope {
        // 1. 限制外部帧大小并完成结构解析
        require(text.toByteArray(Charsets.UTF_8).size <= MAX_JSON_FRAME_BYTES) {
            "JSON frame exceeds $MAX_JSON_FRAME_BYTES bytes"
        }
        val envelope = json.decodeFromString<WireEnvelope>(text)

        // 2. 校验协议版本与 envelope 不变量
        require(envelope.v == WIRE_PROTOCOL_VERSION) { "Unsupported protocol version: ${envelope.v}" }
        validateEnvelope(envelope)
        return envelope
    }

    fun encode(envelope: WireEnvelope): String {
        require(envelope.v == WIRE_PROTOCOL_VERSION) { "Unsupported protocol version: ${envelope.v}" }
        validateEnvelope(envelope)
        val encoded = json.encodeToString(envelope)
        require(encoded.toByteArray(Charsets.UTF_8).size <= MAX_JSON_FRAME_BYTES) {
            "JSON frame exceeds $MAX_JSON_FRAME_BYTES bytes"
        }
        return encoded
    }

    fun json(): Json = json

    inline fun <reified T> decodePayload(payload: JsonObject): T = json.decodeFromJsonElement(payload)

    private fun validateEnvelope(envelope: WireEnvelope) {
        require(envelope.type.isNotBlank()) { "Envelope type is required" }
        knownTypes[envelope.kind]?.let { types ->
            require(envelope.type in types) { "Unsupported ${envelope.kind} type: ${envelope.type}" }
        }

        when (envelope.kind) {
            WireKind.COMMAND,
            WireKind.REPLY,
            WireKind.EVENT,
            -> require(!envelope.id.isNullOrBlank()) { "${envelope.kind} id is required" }

            WireKind.ACK,
            WireKind.CONTROL,
            -> Unit
        }
        if (envelope.kind == WireKind.EVENT) {
            require(envelope.eventSeq != null && envelope.eventSeq > 0) { "Event sequence must be positive" }
        } else {
            require(envelope.eventSeq == null) { "event_seq is only valid on event frames" }
        }
        when (envelope.kind) {
            WireKind.COMMAND,
            WireKind.REPLY,
            WireKind.EVENT,
            WireKind.ACK,
            -> require(envelope.connectionEpoch != null && envelope.connectionEpoch > 0) {
                "Authenticated frames require a positive connection_epoch"
            }
            WireKind.CONTROL -> when (envelope.type) {
                "auth.accepted", "resume" -> require(
                    envelope.connectionEpoch != null && envelope.connectionEpoch > 0,
                ) { "Authenticated controls require a positive connection_epoch" }
                else -> require(envelope.connectionEpoch == null) {
                    "Pre-auth controls must not carry connection_epoch"
                }
            }
        }
        if (envelope.kind == WireKind.ACK || envelope.kind == WireKind.CONTROL) {
            require(envelope.id == null) { "${envelope.kind} id is not used by protocol v1" }
        }
    }
}
