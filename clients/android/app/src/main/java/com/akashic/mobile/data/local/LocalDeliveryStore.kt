package com.akashic.mobile.data.local

import androidx.room.withTransaction
import com.akashic.mobile.data.realtime.ProtocolCodec
import com.akashic.mobile.data.realtime.WireEnvelope
import com.akashic.mobile.data.realtime.WireKind
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull

class LocalDeliveryStore(private val database: AppDatabase) {
    suspend fun savePairedProfile(profile: ServerProfileEntity, cursor: RealtimeCursorEntity) {
        database.withTransaction {
            database.serverProfiles().upsert(profile)
            database.realtimeCursors().upsert(cursor)
        }
    }

    suspend fun enqueueMessage(
        conversation: ConversationEntity,
        message: MessageEntity,
        command: OutboxCommandEntity,
    ) {
        database.withTransaction {
            database.conversations().upsert(conversation)
            database.messages().upsert(message)
            database.outbox().enqueue(command)
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

    suspend fun acknowledgeOutbox(commandId: String, updatedAt: Long) {
        database.withTransaction {
            val command = requireNotNull(database.outbox().get(commandId)) { "Unknown outbox command: $commandId" }
            val envelope = ProtocolCodec.decode(command.envelopeJson)
            val payload = ProtocolCodec.decodePayload<com.akashic.mobile.data.realtime.MessageSendPayload>(envelope.payload)
            val changed = database.messages().updateDelivery(payload.clientMessageId, "sent", updatedAt)
            check(changed == 1) { "Outbox message is missing: ${payload.clientMessageId}" }
            check(database.outbox().deleteAcknowledged(commandId) == 1) { "Outbox command disappeared: $commandId" }
        }
    }

    suspend fun failOutbox(commandId: String, updatedAt: Long) {
        database.withTransaction {
            val command = requireNotNull(database.outbox().get(commandId)) { "Unknown outbox command: $commandId" }
            val envelope = ProtocolCodec.decode(command.envelopeJson)
            val payload = ProtocolCodec.decodePayload<com.akashic.mobile.data.realtime.MessageSendPayload>(envelope.payload)
            check(database.messages().updateDelivery(payload.clientMessageId, "failed", updatedAt) == 1)
            check(database.outbox().deleteAcknowledged(commandId) == 1)
        }
    }

    private suspend fun applyEventContent(serverId: String, envelope: WireEnvelope, updatedAt: Long) {
        when (envelope.type) {
            "session.created", "session.updated" -> upsertConversation(serverId, envelope, updatedAt)
            "turn.started" -> ensureAssistantTurn(envelope, updatedAt)
            "react.thinking.delta" -> appendThinking(envelope, updatedAt)
            "react.tool.started" -> startTool(envelope, updatedAt)
            "react.tool.completed" -> completeTool(envelope, updatedAt)
            "answer.delta" -> appendAnswer(envelope, updatedAt)
            "message.final" -> finalizeMessage(envelope, updatedAt)
            "turn.interrupted" -> interruptTurn(envelope, updatedAt)
            "message.proactive" -> insertProactive(envelope, updatedAt)
            else -> Unit
        }
    }

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
                    content = payloadText(envelope, "name") ?: "工具调用",
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
        val previous = database.messages().getBlock(blockId)
        val detail = payloadText(envelope, "summary")
            ?: payloadText(envelope, "output")
            ?: previous?.content
            ?: "工具调用完成"
        database.messages().upsertBlocks(
            listOf(
                TurnBlockEntity(
                    blockId = blockId,
                    messageId = message.messageId,
                    turnId = turnId,
                    ordinal = previous?.ordinal ?: requireNotNull(payloadLong(envelope, "ordinal")) {
                        "Tool completion has no ordinal"
                    }.toInt(),
                    kind = "tool",
                    status = "completed",
                    content = detail,
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

    private fun payloadLong(envelope: WireEnvelope, key: String): Long? =
        envelope.payload[key]?.jsonPrimitive?.longOrNull
}
