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
import com.akashic.mobile.data.realtime.deliveredFinalMessageEvent
import com.akashic.mobile.data.realtime.FinalMessageEvent
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
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

enum class RemoveUnavailableConversationResult {
    REMOVED,
    HAS_LOCAL_WORK,
    NOT_REMOTE,
}

private const val TOOL_BLOCK_V1_PREFIX = "tool.v1:"
private const val LEGACY_IDENTITY_LOOKBACK_MS = 60 * 60 * 1_000L
private const val MAX_CANONICAL_MESSAGE_ALIASES = 256

@Serializable
internal data class StoredToolBlock(
    val name: String,
    val description: String? = null,
    val resultPreview: String? = null,
    val arguments: JsonObject? = null,
    val durationMillis: Long? = null,
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
    private val projectionStateMutex = Mutex()
    private val canonicalMessageAliases = linkedMapOf<String, String>()

    /** 串行校验并保存当前电脑的一份会话草稿。 */
    suspend fun saveComposerDraft(
        sessionId: String,
        text: String,
        replyToMessageId: String?,
        expectedServerId: String?,
        updatedAt: Long,
    ) = projectionStateMutex.withLock {
        // 1. 在 WebView 边界限制消息大小并确认会话归属
        require(text.length <= 65_536) { "会话草稿超过消息长度上限" }
        val conversation = requireNotNull(database.conversations().get(sessionId)) {
            "会话草稿对应的会话不存在: $sessionId"
        }
        require(conversation.serverId == expectedServerId) { "会话草稿不属于当前电脑" }

        // 2. 引用目标消失属于可恢复状态；跨会话目标仍然是协议错误
        val resolvedReplyId = replyToMessageId?.let { messageId ->
            require(messageId.length in 1..512) { "会话草稿引用 ID 无效" }
            val resolvedMessageId = canonicalMessageAliases[messageId] ?: messageId
            val target = database.messages().get(resolvedMessageId) ?: return@let null
            require(target.sessionId == sessionId) { "会话草稿引用不属于当前会话" }
            resolvedMessageId
        }

        // 3. 空草稿删除实体，其余状态原样保存供进程重启恢复
        if (text.isEmpty() && resolvedReplyId == null) {
            database.composerDrafts().delete(conversation.serverId, sessionId)
        } else {
            database.composerDrafts().upsert(
                ComposerDraftEntity(
                    sessionId = sessionId,
                    serverId = conversation.serverId,
                    text = text,
                    replyToMessageId = resolvedReplyId,
                    updatedAt = updatedAt,
                ),
            )
        }
    }

    /** 串行校验并保存 WebView 当前可见消息锚点。 */
    suspend fun saveReadingPosition(
        sessionId: String,
        messageId: String,
        offsetPx: Int,
        expectedServerId: String?,
        updatedAt: Long,
    ): Boolean = projectionStateMutex.withLock {
        // 1. 校验当前投影边界；canonical 迁移后的旧 UI 写入已失效
        require(offsetPx in -10_000..10_000) { "阅读锚点偏移超出范围" }
        val conversation = requireNotNull(database.conversations().get(sessionId)) {
            "阅读位置会话不存在: $sessionId"
        }
        require(conversation.serverId == expectedServerId) { "阅读位置会话不属于当前电脑" }
        val resolvedMessageId = canonicalMessageAliases[messageId] ?: messageId
        val message = database.messages().get(resolvedMessageId) ?: return@withLock false
        require(message.sessionId == sessionId) { "阅读锚点不属于当前会话" }

        // 2. 在 canonical 迁移共用的锁内持久化，避免写回已删除身份
        database.conversationReadStates().savePosition(sessionId, resolvedMessageId, offsetPx, updatedAt)
        true
    }

    /** 用户主动进入会话时清除旧锚点，但不提前推进已读水位。 */
    suspend fun clearReadingPosition(
        sessionId: String,
        expectedServerId: String?,
        updatedAt: Long,
    ) = projectionStateMutex.withLock {
        val conversation = requireNotNull(database.conversations().get(sessionId)) {
            "阅读位置会话不存在: $sessionId"
        }
        require(conversation.serverId == expectedServerId) { "阅读位置会话不属于当前电脑" }
        database.conversationReadStates().clearPosition(sessionId, updatedAt)
    }

    /** 串行推进已读水位并清除旧阅读锚点。 */
    suspend fun markSessionReadThrough(
        sessionId: String,
        readAt: Long,
        expectedServerId: String?,
        updatedAt: Long,
    ) = projectionStateMutex.withLock {
        // 1. 校验 WebView 边界输入与当前电脑身份
        require(readAt >= 0) { "已读水位不能为负数" }
        val conversation = requireNotNull(database.conversations().get(sessionId)) {
            "已读水位会话不存在: $sessionId"
        }
        require(conversation.serverId == expectedServerId) { "已读水位会话不属于当前电脑" }

        // 2. 与保存及 canonical 迁移共用同一写序
        database.conversationReadStates().markReadThrough(sessionId, readAt, updatedAt)
    }

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
        projectionStateMutex.withLock {
            database.withTransaction {
                database.messages().deleteReloadableServerCache(serverId)
                database.conversations().deleteEmptyProjection(serverId, preservedSessionId)
            }
        }

        // 2. 删除失去消息引用的附件文件和描述符
        mediaCache.reconcile()
    }

    /** 删除服务端已不存在、且没有未发送工作的本机会话副本。 */
    suspend fun removeUnavailableConversation(
        serverId: String,
        sessionId: String,
    ): RemoveUnavailableConversationResult = projectionStateMutex.withLock {
        // 1. 在持久化边界重新确认会话归属和可删除状态
        val result = database.withTransaction {
            val conversation = requireNotNull(database.conversations().get(sessionId)) {
                "Unknown conversation: $sessionId"
            }
            require(conversation.serverId == serverId) { "Conversation belongs to another server" }
            if (!conversation.remoteKnown) {
                return@withTransaction RemoveUnavailableConversationResult.NOT_REMOTE
            }
            if (
                database.messages().countLocalWorkForSession(sessionId) > 0 ||
                database.attachmentTransfers().drafts(serverId, sessionId).isNotEmpty() ||
                database.composerDrafts().get(serverId, sessionId) != null
            ) {
                return@withTransaction RemoveUnavailableConversationResult.HAS_LOCAL_WORK
            }

            // 2. 删除本地投影，依靠外键清理消息、已读锚点和媒体描述符
            database.attachmentTransfers().deleteSentForSession(serverId, sessionId)
            check(database.conversations().delete(serverId, sessionId) == 1)
            RemoveUnavailableConversationResult.REMOVED
        }

        // 3. 删除失去数据库引用的缓存文件
        if (result == RemoveUnavailableConversationResult.REMOVED) mediaCache.reconcile()
        result
    }

    /** 在同一事务应用有序事件并推进持久化 cursor。 */
    suspend fun applyEvent(
        serverId: String,
        deviceId: String,
        envelope: WireEnvelope,
        updatedAt: Long,
        preservedSessionId: String? = null,
    ): Long = projectionStateMutex.withLock {
        require(envelope.kind == WireKind.EVENT) { "Only event envelopes can advance the cursor" }
        val eventSeq = requireNotNull(envelope.eventSeq)
        val connectionEpoch = requireNotNull(envelope.connectionEpoch)
        database.withTransaction {
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

            val delivered = if (envelope.type in DELIVERED_MESSAGE_EVENTS) {
                deliveredFinalMessageEvent(envelope)
            } else {
                null
            }
            envelope.sessionId?.let { sessionId ->
                if (envelope.type in REMOTE_SESSION_EVENTS) {
                    ensureRemoteConversation(serverId, sessionId, updatedAt)
                }
            }
            applyEventContent(serverId, envelope, delivered, updatedAt)
            delivered?.let { event ->
                database.pendingMessageNotifications().upsert(
                    PendingMessageNotificationEntity(
                        messageId = event.messageId,
                        serverId = serverId,
                        sessionId = event.sessionId,
                        content = event.content,
                        hasAttachments = event.hasAttachments,
                        attention = event.attention.name,
                        createdAt = updatedAt,
                    ),
                )
            }
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
            check(database.conversations().markRemoteKnown(requireNotNull(envelope.sessionId)) == 1) {
                "Outbox ACK 对应的会话投影不存在: ${envelope.sessionId}"
            }
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

    /** 保留失败命令，使消息可以安全重试或用原幂等键核对结果。 */
    suspend fun retainFailedOutbox(commandId: String, outcomeUnknown: Boolean, updatedAt: Long) {
        database.withTransaction {
            val command = requireNotNull(database.outbox().get(commandId)) { "Unknown outbox command: $commandId" }
            val envelope = ProtocolCodec.decode(command.envelopeJson)
            val payload = ProtocolCodec.decodePayload<com.akashic.mobile.data.realtime.MessageSendPayload>(envelope.payload)
            val state = if (outcomeUnknown) "outcome_unknown" else "failed_retryable"
            check(database.messages().updateDelivery(payload.clientMessageId, state, updatedAt) == 1)
            if (payload.mediaRefs.isNotEmpty()) {
                check(database.attachmentTransfers().restoreReady(payload.mediaRefs, updatedAt) == payload.mediaRefs.size)
            }
            check(database.outbox().markFailed(commandId, state) == 1)
        }
    }

    /** 在写入 WebSocket 前保留已失效会话的待发命令。 */
    suspend fun retainUnsentOutbox(commandId: String, updatedAt: Long) {
        database.withTransaction {
            val command = requireNotNull(database.outbox().get(commandId)) { "Unknown outbox command: $commandId" }
            val envelope = ProtocolCodec.decode(command.envelopeJson)
            val payload = ProtocolCodec.decodePayload<com.akashic.mobile.data.realtime.MessageSendPayload>(envelope.payload)
            check(database.messages().updateDelivery(payload.clientMessageId, "failed_retryable", updatedAt) == 1)
            if (payload.mediaRefs.isNotEmpty()) {
                check(database.attachmentTransfers().restoreReady(payload.mediaRefs, updatedAt) == payload.mediaRefs.size)
            }
            check(database.outbox().markUnsentFailed(commandId) == 1)
        }
    }

    /** 丢弃不可重试的协议坏命令，并保留原消息作为失败记录。 */
    suspend fun discardFailedOutbox(commandId: String, updatedAt: Long) {
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

    /** 原位重试失败消息，并按失败语义选择原幂等键或新命令 ID。 */
    suspend fun retryFailedMessage(messageId: String, newCommandId: String, updatedAt: Long): Boolean =
        database.withTransaction {
            // 1. 恢复失败消息、命令和附件不变量
            val message = requireNotNull(database.messages().get(messageId)) { "Unknown failed message: $messageId" }
            require(message.role == "user") { "Only user messages can be retried" }
            if (message.deliveryState in setOf("pending", "sent", "complete")) {
                return@withTransaction false
            }
            val clientMessageId = requireNotNull(message.clientMessageId) { "Failed message has no client id" }
            val command = requireNotNull(database.outbox().get(clientMessageId)) { "Failed message has no outbox command" }
            val envelope = ProtocolCodec.decode(command.envelopeJson)
            val payload = ProtocolCodec.decodePayload<com.akashic.mobile.data.realtime.MessageSendPayload>(envelope.payload)
            require(payload.clientMessageId == clientMessageId) { "Outbox client id mismatch" }
            require(command.state == message.deliveryState) { "Outbox and message failure states diverged" }

            // 2. 未知结果复用原幂等键；明确失败生成新幂等键但保留视觉消息 ID
            if (command.state == "outcome_unknown") {
                check(database.outbox().recheckUnknown(command.commandId) == 1)
                check(database.messages().updateDelivery(clientMessageId, "pending", updatedAt) == 1)
            } else {
                require(command.state == "failed_retryable") { "Message is not retryable: ${command.state}" }
                val retryPayload = payload.copy(clientMessageId = newCommandId)
                val retryEnvelope = envelope.copy(
                    id = newCommandId,
                    payload = ProtocolCodec.json().encodeToJsonElement(
                        com.akashic.mobile.data.realtime.MessageSendPayload.serializer(),
                        retryPayload,
                    ).jsonObject,
                )
                check(
                    database.messages().replaceRetryIdentity(
                        messageId,
                        clientMessageId,
                        newCommandId,
                        updatedAt,
                    ) == 1,
                )
                check(database.outbox().deleteAcknowledged(command.commandId) == 1)
                database.outbox().enqueue(
                    OutboxCommandEntity(
                        commandId = newCommandId,
                        serverId = command.serverId,
                        envelopeJson = ProtocolCodec.encode(retryEnvelope),
                        state = "pending",
                        attemptCount = 0,
                        createdAt = command.createdAt,
                        lastAttemptAt = null,
                    ),
                )
            }

            // 3. 重新占用原附件，ACK 或下一次失败负责推进最终状态
            if (payload.mediaRefs.isNotEmpty()) {
                check(database.attachmentTransfers().markSending(payload.mediaRefs, updatedAt) == payload.mediaRefs.size)
            }
            true
        }

    private suspend fun applyEventContent(
        serverId: String,
        envelope: WireEnvelope,
        delivered: FinalMessageEvent?,
        updatedAt: Long,
    ) {
        when (envelope.type) {
            "session.list" -> applySessionList(serverId, envelope)
            "session.created", "session.updated" -> upsertConversation(serverId, envelope, updatedAt)
            "history.page" -> applyHistoryPage(serverId, envelope)
            "turn.started" -> ensureAssistantTurn(envelope, updatedAt)
            "react.thinking.delta" -> appendThinking(envelope, updatedAt)
            "react.tool.started" -> startTool(envelope, updatedAt)
            "react.tool.completed" -> completeTool(envelope, updatedAt)
            "answer.delta" -> appendAnswer(envelope, updatedAt)
            "message.final" -> finalizeMessage(envelope, requireNotNull(delivered), updatedAt)
            "turn.interrupted" -> interruptTurn(envelope, updatedAt)
            "message.proactive" -> insertProactive(envelope, requireNotNull(delivered), updatedAt)
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
                    remoteKnown = true,
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
                ConversationEntity(
                    sessionId,
                    serverId,
                    "新对话",
                    System.currentTimeMillis(),
                    remoteKnown = true,
                ),
            )
        } else {
            check(database.conversations().markRemoteKnown(sessionId) == 1)
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
                replyToMessageId = remote.replyToMessageId,
                replyRole = remote.replyRole,
                replyPreview = remote.replyPreview,
            )
            val sourceId = remote.clientMessageId?.let {
                database.messages().getByClientMessageId(it)?.messageId
            }
                ?: legacyLocalSourceId(canonical)
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

    /** 仅在唯一匹配时把旧版临时消息迁移到服务端 canonical identity。 */
    private suspend fun legacyLocalSourceId(canonical: MessageEntity): String? {
        if (canonical.role == "assistant") {
            val candidates = database.messages().findEphemeralAssistants(
                canonical.sessionId,
                canonical.text,
                canonical.updatedAt - LEGACY_IDENTITY_LOOKBACK_MS,
                canonical.updatedAt + LEGACY_IDENTITY_LOOKBACK_MS,
            )
            val closestDistance = candidates.minOfOrNull {
                kotlin.math.abs(it.updatedAt - canonical.updatedAt)
            } ?: return null
            return candidates.singleOrNull {
                kotlin.math.abs(it.updatedAt - canonical.updatedAt) == closestDistance
            }?.messageId
        }
        if (canonical.role != "user" || canonical.clientMessageId != null) return null
        return database.messages().findLegacyOptimisticUsers(
            canonical.sessionId,
            canonical.text,
            canonical.createdAt - LEGACY_IDENTITY_LOOKBACK_MS,
            canonical.updatedAt,
        ).singleOrNull()?.messageId
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
                val arguments = (call["final_arguments"] ?: call["arguments"]) as? JsonObject
                val ordinal = blocks.size
                blocks += TurnBlockEntity(
                    blockId = "history:$turnId:$ordinal",
                    messageId = messageId,
                    turnId = turnId,
                    ordinal = ordinal,
                    kind = "tool",
                    status = if (jsonText(call, "status") == "success") "completed" else "failed",
                    content = encodeStoredToolBlock(
                        StoredToolBlock(
                            name = name,
                            description = arguments?.let { jsonText(it, "description") }
                                ?: jsonText(call, "description"),
                            arguments = arguments,
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
                remoteKnown = true,
            ),
        )
    }

    private suspend fun ensureRemoteConversation(serverId: String, sessionId: String, updatedAt: Long) {
        val current = database.conversations().get(sessionId)
        if (current == null) {
            database.conversations().upsert(
                ConversationEntity(sessionId, serverId, "新对话", updatedAt, remoteKnown = true),
            )
            return
        }
        require(current.serverId == serverId) { "远端事件会话不属于当前电脑" }
        if (!current.remoteKnown) {
            check(database.conversations().markRemoteKnown(sessionId) == 1)
        }
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
                            arguments = arguments,
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
        val finalArguments = when (val value = envelope.payload["arguments"]) {
            null -> stored.arguments
            is JsonObject -> value
            else -> error("Tool completion arguments must be an object")
        }
        val succeeded = payloadText(envelope, "status") == "success"
        val durationMillis = envelope.payload["duration_ms"]?.let {
            requireNotNull(payloadLong(envelope, "duration_ms")) {
                "Tool completion duration_ms must be an integer"
            }
        }
        require(durationMillis == null || durationMillis >= 0) {
            "Tool completion duration_ms must be non-negative"
        }
        database.messages().upsertBlocks(
            listOf(
                TurnBlockEntity(
                    blockId = blockId,
                    messageId = message.messageId,
                    turnId = turnId,
                    ordinal = previous.ordinal,
                    kind = "tool",
                    status = if (succeeded) "completed" else "failed",
                    content = encodeStoredToolBlock(
                        stored.copy(
                            description = finalArguments?.let { payloadText(it, "description") }
                                ?: stored.description,
                            resultPreview = payloadText(envelope, "result_preview"),
                            arguments = finalArguments,
                            durationMillis = durationMillis,
                        ),
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

    private suspend fun finalizeMessage(
        envelope: WireEnvelope,
        delivered: FinalMessageEvent,
        updatedAt: Long,
    ) {
        canonicalizeUserMessage(envelope, updatedAt)
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
        val canonicalId = delivered.messageId
        require(canonicalId.isNotBlank() && canonicalId.length <= 512) { "Canonical message id is invalid" }
        val canonical = current.copy(
            messageId = canonicalId,
            text = delivered.content.ifEmpty { current.text },
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

    /** 用服务端最终事件把 optimistic 用户消息迁移为 canonical identity。 */
    private suspend fun canonicalizeUserMessage(envelope: WireEnvelope, updatedAt: Long) {
        val clientMessageId = payloadText(envelope, "client_message_id") ?: return
        val canonicalId = requireNotNull(payloadText(envelope, "user_message_id")) {
            "Final event with client_message_id has no user_message_id"
        }
        requireFrameId(clientMessageId)
        require(canonicalId.isNotBlank() && canonicalId.length <= 512) {
            "Canonical user message id is invalid"
        }
        val source = database.messages().getByClientMessageId(clientMessageId) ?: return
        mergeCanonicalMessage(
            source.messageId,
            source.copy(
                messageId = canonicalId,
                deliveryState = "complete",
                updatedAt = updatedAt,
            ),
        )
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

        // 2. 将阅读位置与流式子项迁移后删除旧身份
        database.conversationReadStates().moveAnchor(source.sessionId, sourceId, canonical.messageId)
        database.composerDrafts().moveReplyTarget(source.sessionId, sourceId, canonical.messageId)
        messages.moveBlocks(sourceId, canonical.messageId)
        media.moveLinks(sourceId, canonical.messageId)
        check(messages.delete(sourceId) == 1) { "Source message disappeared during canonical merge" }
        canonicalMessageAliases.entries.forEach { alias ->
            if (alias.value == sourceId) alias.setValue(canonical.messageId)
        }
        canonicalMessageAliases[sourceId] = canonical.messageId
        if (canonicalMessageAliases.size > MAX_CANONICAL_MESSAGE_ALIASES) {
            canonicalMessageAliases.remove(canonicalMessageAliases.keys.first())
        }
    }

    private suspend fun interruptTurn(envelope: WireEnvelope, updatedAt: Long) {
        val current = ensureAssistantTurn(envelope, updatedAt)
        database.messages().upsert(current.copy(deliveryState = "interrupted", updatedAt = updatedAt))
        database.messages().completeRunningBlocks(current.messageId, updatedAt)
    }

    private suspend fun insertProactive(
        envelope: WireEnvelope,
        delivered: FinalMessageEvent,
        updatedAt: Long,
    ) {
        val sessionId = delivered.sessionId
        val messageId = delivered.messageId
        database.messages().upsert(
            MessageEntity(
                messageId = messageId,
                clientMessageId = null,
                sessionId = sessionId,
                role = "assistant",
                text = delivered.content,
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
                    state = if (descriptor.sizeBytes >= AUTO_DOWNLOAD_LIMIT_BYTES) "remote" else "pending",
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
        const val AUTO_DOWNLOAD_LIMIT_BYTES = 10L * 1024 * 1024
        val REMOTE_SESSION_EVENTS = setOf(
            "session.created",
            "session.updated",
            "history.page",
            "turn.started",
            "message.proactive",
        )
        val DELIVERED_MESSAGE_EVENTS = setOf("message.final", "message.proactive")
        val MIME_TYPE = Regex("^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$")
        val FRAME_ID = Regex(
            "^(?:[0-9A-HJKMNP-TV-Z]{26}|[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-" +
                "7[0-9A-Fa-f]{3}-[89ABab][0-9A-Fa-f]{3}-[0-9A-Fa-f]{12})$",
        )
    }
}
