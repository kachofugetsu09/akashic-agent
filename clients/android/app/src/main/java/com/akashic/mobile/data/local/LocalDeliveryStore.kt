package com.akashic.mobile.data.local

import androidx.room.withTransaction
import com.akashic.mobile.data.realtime.ProtocolCodec
import com.akashic.mobile.data.realtime.AttachmentProgressPayload
import com.akashic.mobile.data.realtime.AttachmentReadyPayload
import com.akashic.mobile.data.realtime.AttachmentDescriptor
import com.akashic.mobile.data.realtime.HistoryPagePayload
import com.akashic.mobile.data.realtime.RemoteHistoryMessage
import com.akashic.mobile.data.realtime.SessionListPayload
import com.akashic.mobile.data.realtime.WireEnvelope
import com.akashic.mobile.data.realtime.WireKind
import java.time.Instant
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.decodeFromJsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull

private const val TOOL_BLOCK_V1_PREFIX = "tool.v1:"
private const val LEGACY_IDENTITY_LOOKBACK_MS = 60 * 60 * 1_000L

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

class LocalDeliveryStore(
    private val database: AppDatabase,
    private val mediaCache: MediaCacheStore,
) {
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
        attachments: List<MediaAttachmentEntity>,
    ) {
        database.withTransaction {
            database.conversations().upsert(conversation)
            database.messages().upsert(message)
            database.outbox().enqueue(command)
            if (attachments.isNotEmpty()) {
                val attachmentIds = attachments.map { it.attachmentId }
                database.mediaAttachments().upsertAll(attachments)
                database.mediaAttachments().linkAll(
                    attachmentIds.mapIndexed { ordinal, id ->
                        MessageAttachmentEntity(message.messageId, id, ordinal)
                    },
                )
                check(database.attachmentTransfers().markSending(attachmentIds, message.updatedAt) == attachmentIds.size)
            }
        }
    }

    /** 清除可从服务端恢复的投影与附件缓存，同时保留配对和未发送工作。 */
    suspend fun clearReloadableCache(serverId: String, preservedSessionId: String?) {
        // 1. 原子移除已提交消息，保留待发送消息、草稿和连接身份
        database.withTransaction {
            database.messages().deleteReloadableServerCache(serverId)
            database.conversations().deleteEmptyProjection(serverId, preservedSessionId)
        }

        // 2. 删除失去消息引用的附件文件和描述符
        mediaCache.reconcile()
    }

    /** 在同一事务应用有序事件并推进持久化 cursor。 */
    suspend fun applyEvent(
        serverId: String,
        deviceId: String,
        envelope: WireEnvelope,
        updatedAt: Long,
        preservedSessionId: String? = null,
    ): Long {
        require(envelope.kind == WireKind.EVENT) { "Only event envelopes can advance the cursor" }
        val eventSeq = requireNotNull(envelope.eventSeq)
        val connectionEpoch = requireNotNull(envelope.connectionEpoch)
        return database.withTransaction {
            val cursor = requireNotNull(database.realtimeCursors().get(deviceId)) {
                "Realtime cursor is missing for device $deviceId"
            }
            require(cursor.serverId == serverId) { "Realtime cursor belongs to another server" }
            require(connectionEpoch >= cursor.connectionEpoch) { "Stale event connection epoch" }
            if (envelope.type == "sync.reset_required") {
                require(eventSeq >= cursor.lastAcknowledgedEventSeq) { "Stale reset event sequence" }
                database.messages().deleteServerProjection(serverId)
                database.conversations().deleteEmptyProjection(serverId, preservedSessionId)
                check(
                    database.realtimeCursors().reset(deviceId, eventSeq, connectionEpoch, updatedAt) == 1,
                ) { "Reset cursor rollback or stale connection epoch for device $deviceId" }
                return@withTransaction eventSeq
            }
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
            require(remote.id.isNotBlank() && remote.id.length <= 512) { "History message id is invalid" }
            remote.clientMessageId?.let(::requireFrameId)
            val messageId = remote.id
            val canonical = MessageEntity(
                messageId = messageId,
                clientMessageId = remote.clientMessageId,
                sessionId = sessionId,
                role = remote.role,
                text = remote.content,
                deliveryState = "complete",
                createdAt = (completedAt - duration).coerceAtMost(completedAt),
                updatedAt = completedAt,
                serverSeq = remote.seq.toLong(),
            )
            val sourceId = remote.clientMessageId?.let { "user:$it" }
                ?: legacyOptimisticSourceId(canonical)
            mergeCanonicalMessage(sourceId, canonical)
            upsertMessageAttachments(
                serverId = serverId,
                sessionId = sessionId,
                messageId = messageId,
                descriptors = remote.attachments,
                updatedAt = completedAt,
            )
            if (remote.role == "assistant") {
                database.messages().deleteBlocks(messageId)
                database.messages().upsertBlocks(historyBlocks(messageId, remote, completedAt))
            }
        }
    }

    /** 仅在唯一匹配时修复旧版缺失 client_message_id 的本地消息身份。 */
    private suspend fun legacyOptimisticSourceId(canonical: MessageEntity): String? {
        if (canonical.role != "user" || canonical.clientMessageId != null) return null
        val candidates = database.messages().findLegacyOptimisticUsers(
            sessionId = canonical.sessionId,
            text = canonical.text,
            earliestCreatedAt = canonical.createdAt - LEGACY_IDENTITY_LOOKBACK_MS,
            latestCreatedAt = canonical.updatedAt,
        )
        return candidates.singleOrNull()?.messageId
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
        val canonicalId = payloadText(envelope, "message_id")
            ?: "ephemeral:${requireNotNull(envelope.id) { "Final event has no frame id" }}"
        require(canonicalId.isNotBlank() && canonicalId.length <= 512) { "Canonical message id is invalid" }
        val canonical = current.copy(
            messageId = canonicalId,
            text = payloadText(envelope, "text") ?: payloadText(envelope, "content") ?: current.text,
            deliveryState = "complete",
            updatedAt = updatedAt,
        )
        mergeCanonicalMessage(current.messageId, canonical)
        upsertMessageAttachments(
            serverId = requireNotNull(database.conversations().get(current.sessionId)).serverId,
            sessionId = current.sessionId,
            messageId = canonicalId,
            descriptors = envelope.payload["attachments"]?.let {
                ProtocolCodec.json().decodeFromJsonElement(it)
            } ?: emptyList(),
            updatedAt = updatedAt,
        )
        database.messages().completeRunningBlocks(canonicalId, updatedAt)
    }

    /** 把流式或 optimistic 消息原子迁移到服务端 canonical identity。 */
    private suspend fun mergeCanonicalMessage(sourceId: String?, canonical: MessageEntity) {
        val messages = database.messages()
        val media = database.mediaAttachments()
        val existingCanonical = messages.get(canonical.messageId)
        if (existingCanonical != null) {
            require(
                existingCanonical.sessionId == canonical.sessionId && existingCanonical.role == canonical.role
            ) { "Canonical message identity belongs to another message" }
        }
        val source = sourceId?.let { messages.get(it) }
        if (sourceId == null || sourceId == canonical.messageId || source == null) {
            messages.upsert(canonical)
            return
        }
        require(source.sessionId == canonical.sessionId && source.role == canonical.role) {
            "Source message identity belongs to another message"
        }
        if (canonical.clientMessageId != null) {
            require(source.clientMessageId == canonical.clientMessageId) { "Optimistic client id mismatch" }
        }

        // 1. 清理已同步 canonical 子项，并释放 optimistic client id 唯一约束
        messages.deleteBlocks(canonical.messageId)
        media.deleteLinks(canonical.messageId)
        if (canonical.clientMessageId != null) {
            check(messages.clearClientMessageId(sourceId) == 1) { "Optimistic message disappeared" }
        }
        messages.upsert(canonical)

        // 2. 将流式子项迁移后删除旧身份
        messages.moveBlocks(sourceId, canonical.messageId)
        media.moveLinks(sourceId, canonical.messageId)
        check(messages.delete(sourceId) == 1) { "Source message disappeared during canonical merge" }
    }

    private suspend fun interruptTurn(envelope: WireEnvelope, updatedAt: Long) {
        val current = ensureAssistantTurn(envelope, updatedAt)
        database.messages().upsert(current.copy(deliveryState = "interrupted", updatedAt = updatedAt))
        database.messages().completeRunningBlocks(current.messageId, updatedAt)
    }

    private suspend fun insertProactive(envelope: WireEnvelope, updatedAt: Long) {
        val sessionId = requireNotNull(envelope.sessionId) { "Proactive event has no session_id" }
        val messageId = "proactive:${requireNotNull(envelope.id)}"
        database.messages().upsert(
            MessageEntity(
                messageId = messageId,
                clientMessageId = null,
                sessionId = sessionId,
                role = "assistant",
                text = payloadText(envelope, "text") ?: payloadText(envelope, "content") ?: "",
                deliveryState = "complete",
                createdAt = updatedAt,
                updatedAt = updatedAt,
            ),
        )
        val serverId = requireNotNull(database.conversations().get(sessionId)).serverId
        upsertMessageAttachments(
            serverId = serverId,
            sessionId = sessionId,
            messageId = messageId,
            descriptors = envelope.payload["attachments"]?.let {
                ProtocolCodec.json().decodeFromJsonElement(it)
            } ?: emptyList(),
            updatedAt = updatedAt,
        )
    }

    /** 持久化附件元数据并幂等关联消息。 */
    private suspend fun upsertMessageAttachments(
        serverId: String,
        sessionId: String,
        messageId: String,
        descriptors: List<AttachmentDescriptor>,
        updatedAt: Long,
    ) {
        val dao = database.mediaAttachments()
        dao.deleteLinks(messageId)
        if (descriptors.isEmpty()) return
        require(descriptors.map { it.attachmentId }.distinct().size == descriptors.size) {
            "消息附件不能重复"
        }
        val entities = descriptors.map { descriptor ->
            requireFrameId(descriptor.attachmentId)
            require(
                descriptor.filename.isNotBlank() && descriptor.filename == descriptor.filename.trim() &&
                    descriptor.filename.length <= 255 &&
                    '/' !in descriptor.filename && '\\' !in descriptor.filename &&
                    descriptor.filename.none { it.code < 32 || it.code == 127 }
            ) { "附件文件名无效" }
            require(
                descriptor.contentType.length <= 255 && MIME_TYPE.matches(descriptor.contentType)
            ) { "附件 content_type 无效" }
            require(descriptor.sizeBytes in 1..MAX_ATTACHMENT_BYTES) { "附件大小超出范围" }
            require(descriptor.sha256.matches(Regex("^[0-9a-fA-F]{64}$"))) { "附件 sha256 无效" }
            val existing = dao.get(descriptor.attachmentId)
            if (existing != null) {
                require(
                    existing.serverId == serverId &&
                        existing.sessionId == sessionId &&
                        existing.filename == descriptor.filename &&
                        existing.contentType == descriptor.contentType &&
                        existing.sizeBytes == descriptor.sizeBytes &&
                        existing.sha256.equals(descriptor.sha256, ignoreCase = true)
                ) { "附件描述与已缓存元数据不一致: ${descriptor.attachmentId}" }
                existing
            } else {
                MediaAttachmentEntity(
                    attachmentId = descriptor.attachmentId,
                    serverId = serverId,
                    sessionId = sessionId,
                    filename = descriptor.filename,
                    contentType = descriptor.contentType,
                    sizeBytes = descriptor.sizeBytes,
                    sha256 = descriptor.sha256.lowercase(),
                    transferredBytes = 0,
                    state = "pending",
                    cachePath = mediaCache.cachePath(descriptor.attachmentId),
                    lastAccessedAt = updatedAt,
                    updatedAt = updatedAt,
                )
            }
        }
        dao.upsertAll(entities)
        dao.linkAll(
            descriptors.mapIndexed { ordinal, descriptor ->
                MessageAttachmentEntity(messageId, descriptor.attachmentId, ordinal)
            },
        )
    }

    private fun requireFrameId(value: String) {
        require(FRAME_ID.matches(value)) { "Frame id must be a UUIDv7 or ULID" }
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

    private companion object {
        const val MAX_ATTACHMENT_BYTES = 50L * 1024 * 1024
        val MIME_TYPE = Regex("^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$")
        val FRAME_ID = Regex(
            "^(?:[0-9A-HJKMNP-TV-Z]{26}|[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-" +
                "7[0-9A-Fa-f]{3}-[89ABab][0-9A-Fa-f]{3}-[0-9A-Fa-f]{12})$",
        )
    }
}
