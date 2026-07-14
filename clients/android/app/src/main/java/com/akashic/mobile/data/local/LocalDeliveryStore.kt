package com.akashic.mobile.data.local

import androidx.room.withTransaction
import com.akashic.mobile.data.realtime.ProtocolCodec
import com.akashic.mobile.data.realtime.AttachmentProgressPayload
import com.akashic.mobile.data.realtime.AttachmentReadyPayload
import com.akashic.mobile.data.realtime.HistoryPagePayload
import com.akashic.mobile.data.realtime.RemoteHistoryMessage
import com.akashic.mobile.data.realtime.SessionListPayload
import com.akashic.mobile.data.realtime.WireEnvelope
import com.akashic.mobile.data.realtime.WireKind
import java.time.Instant
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull

private const val TOOL_BLOCK_V1_PREFIX = "tool.v1:"

@Serializable
internal data class StoredToolBlock(
    val name: String,
    val description: String? = null,
    val resultPreview: String? = null,
)

internal fun decodeStoredToolBlock(content: String): StoredToolBlock {
    if (!content.startsWith(TOOL_BLOCK_V1_PREFIX)) {
        return StoredToolBlock(name = "工具调用", description = content)
    }
    return ProtocolCodec.json().decodeFromString(content.removePrefix(TOOL_BLOCK_V1_PREFIX))
}

private fun encodeStoredToolBlock(content: StoredToolBlock): String =
    TOOL_BLOCK_V1_PREFIX + ProtocolCodec.json().encodeToString(content)

class LocalDeliveryStore(private val database: AppDatabase) {
    suspend fun savePairedProfile(profile: ServerProfileEntity, cursor: RealtimeCursorEntity) {
        database.withTransaction {
            database.serverProfiles().upsert(profile)
            if (database.realtimeCursors().get(cursor.deviceId) == null) {
                database.realtimeCursors().insert(cursor)
            }
        }
    }

    suspend fun enqueueMessage(
        conversation: ConversationEntity,
        message: MessageEntity,
        command: OutboxCommandEntity,
        attachmentIds: List<String>,
    ) {
        database.withTransaction {
            database.conversations().upsert(conversation)
            database.messages().upsert(message)
            database.outbox().enqueue(command)
            if (attachmentIds.isNotEmpty()) {
                check(database.attachmentTransfers().markSending(attachmentIds, message.updatedAt) == attachmentIds.size)
            }
        }
    }

    /** 在同一事务应用有序事件并推进持久化 cursor。 */
    suspend fun applyEvent(
        serverId: String,
        deviceId: String,
        envelope: WireEnvelope,
        updatedAt: Long,
    ): Long {
        require(envelope.kind == WireKind.EVENT) { "Only event envelopes can advance the cursor" }
        val eventSeq = requireNotNull(envelope.eventSeq)
        val connectionEpoch = requireNotNull(envelope.connectionEpoch)
        return database.withTransaction {
            val cursor = requireNotNull(database.realtimeCursors().get(deviceId)) {
                "Realtime cursor is missing for device $deviceId"
            }
            require(connectionEpoch >= cursor.connectionEpoch) { "Stale event connection epoch" }
            require(eventSeq == cursor.lastAcknowledgedEventSeq + 1) {
                "Event sequence gap: expected ${cursor.lastAcknowledgedEventSeq + 1}, got $eventSeq"
            }

            applyEventContent(serverId, envelope, updatedAt)
            val changed = database.realtimeCursors().advance(
                deviceId = deviceId,
                throughEventSeq = eventSeq,
                connectionEpoch = connectionEpoch,
                updatedAt = updatedAt,
            )
            check(changed == 1) { "Cursor rollback or stale connection epoch for device $deviceId" }
            eventSeq
        }
    }

    suspend fun markOutboxAttempt(commandId: String, attemptedAt: Long) {
        val changed = database.outbox().markInFlight(commandId, attemptedAt)
        check(changed == 1) { "Outbox command is not pending: $commandId" }
    }

    suspend fun retryOutbox(commandId: String) {
        val changed = database.outbox().markForRetry(commandId)
        check(changed == 1) { "Outbox command is not in flight: $commandId" }
    }

    suspend fun acknowledgeOutbox(commandId: String, updatedAt: Long): List<String> =
        database.withTransaction {
            val command = requireNotNull(database.outbox().get(commandId)) { "Unknown outbox command: $commandId" }
            val envelope = ProtocolCodec.decode(command.envelopeJson)
            val payload = ProtocolCodec.decodePayload<com.akashic.mobile.data.realtime.MessageSendPayload>(envelope.payload)
            val changed = database.messages().updateDelivery(payload.clientMessageId, "sent", updatedAt)
            check(changed == 1) { "Outbox message is missing: ${payload.clientMessageId}" }
            if (payload.mediaRefs.isNotEmpty()) {
                payload.mediaRefs.forEach { attachmentId ->
                    val transfer = requireNotNull(database.attachmentTransfers().get(attachmentId)) {
                        "Outbox attachment is missing: $attachmentId"
                    }
                    check(transfer.state == "sending") { "Outbox attachment is not sending: $attachmentId" }
                }
                check(database.attachmentTransfers().markSent(payload.mediaRefs, updatedAt) == payload.mediaRefs.size)
            }
            check(database.outbox().deleteAcknowledged(commandId) == 1) { "Outbox command disappeared: $commandId" }
            payload.mediaRefs
        }

    suspend fun failOutbox(commandId: String, updatedAt: Long) {
        database.withTransaction {
            val command = requireNotNull(database.outbox().get(commandId)) { "Unknown outbox command: $commandId" }
            val envelope = ProtocolCodec.decode(command.envelopeJson)
            val payload = ProtocolCodec.decodePayload<com.akashic.mobile.data.realtime.MessageSendPayload>(envelope.payload)
            check(database.messages().updateDelivery(payload.clientMessageId, "failed", updatedAt) == 1)
            if (payload.mediaRefs.isNotEmpty()) {
                check(database.attachmentTransfers().restoreReady(payload.mediaRefs, updatedAt) == payload.mediaRefs.size)
            }
            check(database.outbox().deleteAcknowledged(commandId) == 1)
        }
    }

    private suspend fun applyEventContent(serverId: String, envelope: WireEnvelope, updatedAt: Long) {
        when (envelope.type) {
            "session.list" -> applySessionList(serverId, envelope)
            "session.created", "session.updated" -> upsertConversation(serverId, envelope, updatedAt)
            "history.page" -> applyHistoryPage(serverId, envelope)
            "turn.started" -> ensureAssistantTurn(envelope, updatedAt)
            "react.thinking.delta" -> appendThinking(envelope, updatedAt)
            "react.tool.started" -> startTool(envelope, updatedAt)
            "react.tool.completed" -> completeTool(envelope, updatedAt)
            "answer.delta" -> appendAnswer(envelope, updatedAt)
            "message.final" -> finalizeMessage(envelope, updatedAt)
            "turn.interrupted" -> interruptTurn(envelope, updatedAt)
            "message.proactive" -> insertProactive(envelope, updatedAt)
            "attachment.progress" -> applyAttachmentProgress(envelope, updatedAt)
            "attachment.ready" -> applyAttachmentReady(envelope, updatedAt)
            else -> Unit
        }
    }

    private suspend fun applyAttachmentProgress(envelope: WireEnvelope, updatedAt: Long) {
        val payload = ProtocolCodec.decodePayload<AttachmentProgressPayload>(envelope.payload)
        val transfer = requireNotNull(database.attachmentTransfers().get(payload.attachmentId)) {
            "Attachment progress references an unknown transfer"
        }
        require(payload.sizeBytes == transfer.sizeBytes) { "Attachment progress size mismatch" }
        require(payload.transferredBytes in transfer.transferredBytes..transfer.sizeBytes) {
            "Attachment progress moved backwards or exceeded size"
        }
        val state = if (payload.transferredBytes == transfer.sizeBytes) "finishing" else "uploading"
        check(
            database.attachmentTransfers().updateState(
                attachmentId = transfer.attachmentId,
                transferredBytes = payload.transferredBytes,
                state = state,
                updatedAt = updatedAt,
            ) == 1,
        )
    }

    private suspend fun applyAttachmentReady(envelope: WireEnvelope, updatedAt: Long) {
        val payload = ProtocolCodec.decodePayload<AttachmentReadyPayload>(envelope.payload)
        val transfer = requireNotNull(database.attachmentTransfers().get(payload.attachmentId)) {
            "Attachment ready references an unknown transfer"
        }
        require(
            payload.filename == transfer.filename &&
                payload.contentType == transfer.contentType &&
                payload.sizeBytes == transfer.sizeBytes &&
                payload.sha256 == transfer.sha256
        ) { "Attachment ready metadata mismatch" }
        check(
            database.attachmentTransfers().updateState(
                attachmentId = transfer.attachmentId,
                transferredBytes = transfer.sizeBytes,
                state = "ready",
                updatedAt = updatedAt,
            ) == 1,
        )
    }

    private suspend fun applySessionList(serverId: String, envelope: WireEnvelope) {
        val payload = ProtocolCodec.decodePayload<SessionListPayload>(envelope.payload)
        payload.items.forEach { item ->
            database.conversations().upsert(
                ConversationEntity(
                    sessionId = item.sessionId,
                    serverId = serverId,
                    title = item.title,
                    updatedAt = Instant.parse(item.updatedAt).toEpochMilli(),
                ),
            )
        }
    }

    private suspend fun applyHistoryPage(
        serverId: String,
        envelope: WireEnvelope,
    ) {
        val sessionId = requireNotNull(envelope.sessionId) { "History page has no session_id" }
        val payload = ProtocolCodec.decodePayload<HistoryPagePayload>(envelope.payload)
        if (database.conversations().get(sessionId) == null) {
            database.conversations().upsert(
                ConversationEntity(sessionId, serverId, "新对话", System.currentTimeMillis()),
            )
        }
        payload.items.forEach { remote ->
            require(remote.sessionKey == sessionId) { "History item session mismatch" }
            require(remote.role in setOf("user", "assistant")) { "Unsupported history role: ${remote.role}" }
            val completedAt = Instant.parse(remote.ts).toEpochMilli()
            val duration = remote.extra["turn_duration_ms"]?.jsonPrimitive?.longOrNull ?: 0L
            val messageId = "history:${remote.id}"
            database.messages().upsert(
                MessageEntity(
                    messageId = messageId,
                    clientMessageId = null,
                    sessionId = sessionId,
                    role = remote.role,
                    text = remote.content,
                    deliveryState = "complete",
                    createdAt = (completedAt - duration).coerceAtMost(completedAt),
                    updatedAt = completedAt,
                ),
            )
            if (remote.role == "assistant") {
                database.messages().upsertBlocks(historyBlocks(messageId, remote, completedAt))
            }
        }
    }

    private fun historyBlocks(
        messageId: String,
        remote: RemoteHistoryMessage,
        updatedAt: Long,
    ): List<TurnBlockEntity> {
        val blocks = mutableListOf<TurnBlockEntity>()
        val turnId = remote.id

        fun addThinking(content: String) {
            if (content.isBlank()) return
            val ordinal = blocks.size
            blocks += TurnBlockEntity(
                blockId = "history:$turnId:$ordinal",
                messageId = messageId,
                turnId = turnId,
                ordinal = ordinal,
                kind = "thinking",
                status = "completed",
                content = content,
                updatedAt = updatedAt,
            )
        }

        (remote.toolChain as? JsonArray)?.forEach { rawGroup ->
            val group = rawGroup.jsonObject
            addThinking(jsonText(group, "reasoning_content") ?: jsonText(group, "text") ?: "")
            (group["calls"] as? JsonArray)?.forEach { rawCall ->
                val call = rawCall.jsonObject
                val name = jsonText(call, "name") ?: return@forEach
                val ordinal = blocks.size
                blocks += TurnBlockEntity(
                    blockId = "history:$turnId:$ordinal",
                    messageId = messageId,
                    turnId = turnId,
                    ordinal = ordinal,
                    kind = "tool",
                    status = if (jsonText(call, "status") == "error") "failed" else "completed",
                    content = encodeStoredToolBlock(
                        StoredToolBlock(
                            name = name,
                            description = jsonText(call, "description"),
                            resultPreview = jsonText(call, "result_preview"),
                        ),
                    ),
                    updatedAt = updatedAt,
                )
            }
        }
        addThinking(jsonText(remote.extra, "reasoning_content") ?: "")
        return blocks
    }

    private fun jsonText(payload: JsonObject, key: String): String? =
        (payload[key] as? JsonPrimitive)?.contentOrNull

    private suspend fun upsertConversation(serverId: String, envelope: WireEnvelope, updatedAt: Long) {
        val sessionId = envelope.sessionId ?: payloadText(envelope, "session_id")
        require(!sessionId.isNullOrBlank()) { "Session event has no session_id" }
        val current = database.conversations().get(sessionId)
        database.conversations().upsert(
            ConversationEntity(
                sessionId = sessionId,
                serverId = current?.serverId ?: serverId,
                title = payloadText(envelope, "title") ?: current?.title ?: "新对话",
                updatedAt = updatedAt,
            ),
        )
    }

    private suspend fun ensureAssistantTurn(envelope: WireEnvelope, updatedAt: Long): MessageEntity {
        val sessionId = requireNotNull(envelope.sessionId) { "Turn event has no session_id" }
        val turnId = requireNotNull(envelope.turnId) { "Turn event has no turn_id" }
        val messageId = "assistant:$turnId"
        val existing = database.messages().get(messageId)
        if (existing != null) return existing
        val message = MessageEntity(
            messageId = messageId,
            clientMessageId = null,
            sessionId = sessionId,
            role = "assistant",
            text = "",
            deliveryState = "streaming",
            createdAt = updatedAt,
            updatedAt = updatedAt,
        )
        database.messages().upsert(message)
        return message
    }

    private suspend fun appendThinking(envelope: WireEnvelope, updatedAt: Long) {
        val message = ensureAssistantTurn(envelope, updatedAt)
        val turnId = requireNotNull(envelope.turnId)
        val blockId = requireNotNull(payloadText(envelope, "block_id")) {
            "Thinking delta has no block_id"
        }
        val ordinal = requireNotNull(payloadLong(envelope, "ordinal")) {
            "Thinking delta has no ordinal"
        }.toInt()
        val previous = database.messages().getBlock(blockId)
        database.messages().upsertBlocks(
            listOf(
                TurnBlockEntity(
                    blockId = blockId,
                    messageId = message.messageId,
                    turnId = turnId,
                    ordinal = previous?.ordinal ?: ordinal,
                    kind = "thinking",
                    status = "running",
                    content = (previous?.content ?: "") + requireNotNull(payloadText(envelope, "delta")),
                    updatedAt = updatedAt,
                ),
            ),
        )
    }

    private suspend fun startTool(envelope: WireEnvelope, updatedAt: Long) {
        val message = ensureAssistantTurn(envelope, updatedAt)
        val turnId = requireNotNull(envelope.turnId)
        val callId = toolCallId(envelope)
        val blockId = payloadText(envelope, "block_id") ?: "tool:$callId"
        val previous = database.messages().getBlock(blockId)
        val toolName = requireNotNull(payloadText(envelope, "tool_name")) {
            "Tool start has no tool_name"
        }
        val arguments = requireNotNull(envelope.payload["arguments"] as? JsonObject) {
            "Tool start arguments must be an object"
        }
        database.messages().completeRunningThinking(message.messageId, updatedAt)
        database.messages().upsertBlocks(
            listOf(
                TurnBlockEntity(
                    blockId = blockId,
                    messageId = message.messageId,
                    turnId = turnId,
                    ordinal = previous?.ordinal ?: requireNotNull(payloadLong(envelope, "ordinal")) {
                        "Tool start has no ordinal"
                    }.toInt(),
                    kind = "tool",
                    status = "running",
                    content = encodeStoredToolBlock(
                        StoredToolBlock(
                            name = toolName,
                            description = payloadText(arguments, "description"),
                        ),
                    ),
                    updatedAt = updatedAt,
                ),
            ),
        )
    }

    private suspend fun completeTool(envelope: WireEnvelope, updatedAt: Long) {
        val message = ensureAssistantTurn(envelope, updatedAt)
        val turnId = requireNotNull(envelope.turnId)
        val callId = toolCallId(envelope)
        val blockId = payloadText(envelope, "block_id") ?: "tool:$callId"
        val previous = requireNotNull(database.messages().getBlock(blockId)) {
            "Tool completion arrived before start: $callId"
        }
        val stored = decodeStoredToolBlock(previous.content)
        val toolName = requireNotNull(payloadText(envelope, "tool_name")) {
            "Tool completion has no tool_name"
        }
        require(toolName == stored.name) { "Tool completion name mismatch: $toolName != ${stored.name}" }
        val failed = payloadText(envelope, "status") == "error"
        database.messages().upsertBlocks(
            listOf(
                TurnBlockEntity(
                    blockId = blockId,
                    messageId = message.messageId,
                    turnId = turnId,
                    ordinal = previous.ordinal,
                    kind = "tool",
                    status = if (failed) "failed" else "completed",
                    content = encodeStoredToolBlock(
                        stored.copy(resultPreview = payloadText(envelope, "result_preview")),
                    ),
                    updatedAt = updatedAt,
                ),
            ),
        )
    }

    private suspend fun appendAnswer(envelope: WireEnvelope, updatedAt: Long) {
        val current = ensureAssistantTurn(envelope, updatedAt)
        database.messages().upsert(
            current.copy(
                text = current.text + requireNotNull(payloadText(envelope, "delta")),
                deliveryState = "streaming",
                updatedAt = updatedAt,
            ),
        )
    }

    private suspend fun finalizeMessage(envelope: WireEnvelope, updatedAt: Long) {
        val current = ensureAssistantTurn(envelope, updatedAt)
        val blocks = database.messages().getBlocks(current.messageId)
        val finalThinking = payloadText(envelope, "thinking")?.trim().orEmpty()
        if (finalThinking.isNotEmpty() && blocks.none { it.kind == "thinking" }) {
            val turnId = requireNotNull(envelope.turnId)
            database.messages().upsertBlocks(
                listOf(
                    TurnBlockEntity(
                        blockId = "thinking:$turnId:final",
                        messageId = current.messageId,
                        turnId = turnId,
                        ordinal = -1,
                        kind = "thinking",
                        status = "completed",
                        content = finalThinking,
                        updatedAt = updatedAt,
                    ),
                ),
            )
        }
        database.messages().upsert(
            current.copy(
                text = payloadText(envelope, "text") ?: payloadText(envelope, "content") ?: current.text,
                deliveryState = "complete",
                updatedAt = updatedAt,
            ),
        )
        database.messages().completeRunningBlocks(current.messageId, updatedAt)
    }

    private suspend fun interruptTurn(envelope: WireEnvelope, updatedAt: Long) {
        val current = ensureAssistantTurn(envelope, updatedAt)
        database.messages().upsert(current.copy(deliveryState = "interrupted", updatedAt = updatedAt))
        database.messages().completeRunningBlocks(current.messageId, updatedAt)
    }

    private suspend fun insertProactive(envelope: WireEnvelope, updatedAt: Long) {
        val sessionId = requireNotNull(envelope.sessionId) { "Proactive event has no session_id" }
        database.messages().upsert(
            MessageEntity(
                messageId = "proactive:${requireNotNull(envelope.id)}",
                clientMessageId = null,
                sessionId = sessionId,
                role = "assistant",
                text = payloadText(envelope, "text") ?: payloadText(envelope, "content") ?: "",
                deliveryState = "complete",
                createdAt = updatedAt,
                updatedAt = updatedAt,
            ),
        )
    }

    private fun toolCallId(envelope: WireEnvelope): String =
        payloadText(envelope, "tool_call_id")
            ?: payloadText(envelope, "call_id")
            ?: payloadText(envelope, "tool_id")
            ?: requireNotNull(envelope.id)

    private fun payloadText(envelope: WireEnvelope, key: String): String? =
        (envelope.payload[key] as? JsonPrimitive)?.contentOrNull

    private fun payloadText(payload: JsonObject, key: String): String? =
        (payload[key] as? JsonPrimitive)?.contentOrNull

    private fun payloadLong(envelope: WireEnvelope, key: String): Long? =
        envelope.payload[key]?.jsonPrimitive?.longOrNull
}
