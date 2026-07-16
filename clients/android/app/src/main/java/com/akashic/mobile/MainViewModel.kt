package com.akashic.mobile

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.akashic.mobile.data.local.MessageWithBlocks
import com.akashic.mobile.data.local.MessageAttachmentWithMedia
import com.akashic.mobile.data.local.decodeStoredToolBlock
import com.akashic.mobile.data.realtime.TransferNetworkKind
import com.akashic.mobile.domain.model.ConnectionPhase
import com.akashic.mobile.domain.model.ConnectionState
import com.akashic.mobile.ui.conversation.ConnectionStatusUi
import com.akashic.mobile.ui.conversation.CommandUi
import com.akashic.mobile.ui.conversation.ComposerAttachmentState
import com.akashic.mobile.ui.conversation.ComposerAttachmentUi
import com.akashic.mobile.ui.conversation.ConversationUiState
import com.akashic.mobile.ui.conversation.AssistantTurnStatus
import com.akashic.mobile.ui.conversation.MessageUi
import com.akashic.mobile.ui.conversation.MessageDeliveryActionUi
import com.akashic.mobile.ui.conversation.MessageReplyUi
import com.akashic.mobile.ui.conversation.MessageAttachmentState
import com.akashic.mobile.ui.conversation.MessageAttachmentUi
import com.akashic.mobile.ui.conversation.ProcessBlockKind
import com.akashic.mobile.ui.conversation.ProcessBlockState
import com.akashic.mobile.ui.conversation.ProcessBlockUi
import com.akashic.mobile.ui.conversation.ReadingPositionUi
import com.akashic.mobile.ui.conversation.NavigationTargetUi
import com.akashic.mobile.ui.conversation.PluginUiAssetUi
import com.akashic.mobile.ui.conversation.PluginUiResponseUi
import com.akashic.mobile.ui.conversation.PendingMessageUi
import com.akashic.mobile.ui.conversation.SessionUi
import com.akashic.mobile.ui.conversation.TransferStatusUi
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlin.math.ceil

private const val LARGE_TRANSFER_BYTES = 10L * 1024 * 1024

@OptIn(ExperimentalCoroutinesApi::class)
class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val container = (application as App).container
    val sessionState = container.realtimeSession.state
    private val navigationTarget = MutableStateFlow<NavigationTargetUi?>(null)

    private val messageGraph = sessionState.flatMapLatest { state ->
        state.currentSessionId?.let(container.database.messages()::observeMessageGraph) ?: flowOf(emptyList())
    }

    private val conversations = sessionState.flatMapLatest { state ->
        state.serverId?.let(container.database.conversations()::observeSummaries) ?: flowOf(emptyList())
    }

    private val attachmentDrafts = sessionState.flatMapLatest { state ->
        val serverId = state.serverId
        val sessionId = state.currentSessionId
        if (serverId == null || sessionId == null) {
            flowOf(emptyList())
        } else {
            container.database.attachmentTransfers().observeDrafts(serverId, sessionId)
        }
    }

    val conversationState = combine(
        sessionState,
        messageGraph,
        conversations,
        attachmentDrafts,
        navigationTarget,
    ) { session, graph, conversations, attachments, target ->
        val scopedGraph = graph.filter { it.message.sessionId == session.currentSessionId }
        val messages = scopedGraph.map(::toMessageUi)
        val connection = connectionPresentation(session.connection, session.errorMessage)
        val composerAttachments = attachments.map { attachment ->
            val waitingForConnection = session.connection.phase != ConnectionPhase.READY &&
                attachment.state in setOf("pending", "uploading", "finishing")
            val waitingForMeteredApproval = attachment.sizeBytes >= LARGE_TRANSFER_BYTES &&
                session.transferNetwork.kind == TransferNetworkKind.METERED &&
                !session.meteredLargeTransferApproved &&
                attachment.state in setOf("pending", "uploading", "finishing")
            ComposerAttachmentUi(
                id = attachment.attachmentId,
                filename = attachment.filename,
                contentType = attachment.contentType,
                sizeBytes = attachment.sizeBytes,
                transferredBytes = attachment.transferredBytes,
                state = when {
                    waitingForConnection -> ComposerAttachmentState.WAITING_FOR_CONNECTION
                    waitingForMeteredApproval -> ComposerAttachmentState.WAITING_FOR_METERED_APPROVAL
                    attachment.state == "ready" -> ComposerAttachmentState.READY
                    attachment.state == "failed" -> ComposerAttachmentState.FAILED
                    else -> ComposerAttachmentState.UPLOADING
                },
                canRemove = attachment.state in setOf("pending", "ready", "failed"),
            )
        }
        val largeTransfer = composerAttachments.firstOrNull {
            it.sizeBytes >= LARGE_TRANSFER_BYTES &&
                it.state !in setOf(ComposerAttachmentState.READY, ComposerAttachmentState.FAILED)
        }
        val transferStatus = largeTransfer?.let { transfer ->
            val requiresApproval = session.transferNetwork.kind == TransferNetworkKind.METERED &&
                !session.meteredLargeTransferApproved
            val progress = (transfer.transferredBytes * 100 / transfer.sizeBytes).toInt().coerceIn(0, 100)
            TransferStatusUi(
                title = if (requiresApproval) "大文件上传已暂停" else "大文件正在后台上传",
                detail = when {
                    requiresApproval && session.transferNetwork.cellular -> "当前为移动网络，确认后会从 $progress% 继续"
                    requiresApproval -> "当前网络按流量计费，确认后会从 $progress% 继续"
                    session.transferNetwork.cellular -> "移动网络 · 离开会话仍会继续"
                    else -> "切换会话或进入后台仍会继续"
                },
                progressPercent = progress,
                requiresMeteredApproval = requiresApproval,
            )
        }
        ConversationUiState(
            connectionLabel = connection.label,
            connectionStatus = connection.status,
            connectionNotice = connection.notice,
            errorNotice = session.errorMessage,
            sessions = conversations
                .filter { it.sessionId.startsWith("mobile:") }
                .map {
                    SessionUi(
                        sessionId = it.sessionId,
                        title = it.title,
                        lastMessagePreview = it.lastMessagePreview?.take(160),
                        lastMessageAtMillis = it.lastMessageAt,
                        unreadCount = it.unreadCount,
                        isRunning = it.sessionId in session.activeSessionIds,
                    )
                },
            selectedSessionId = session.currentSessionId,
            readingPosition = conversations
                .firstOrNull { it.sessionId == session.currentSessionId }
                ?.let { summary ->
                    summary.anchorMessageId?.let { ReadingPositionUi(it, summary.anchorOffsetPx) }
                },
            navigationTarget = target,
            projectionGeneration = session.projectionGeneration,
            messages = messages,
            pendingMessages = messages.filterIsInstance<MessageUi.User>()
                .filter { it.deliveryLabel == "待发送" }
                .map {
                    PendingMessageUi(
                        messageId = it.id,
                        preview = it.text.trim().replace(Regex("\\s+"), " ").take(120)
                            .ifBlank { "[附件]" },
                        createdAtMillis = it.createdAtMillis,
                    )
                },
            attachments = composerAttachments,
            transferStatus = transferStatus,
            commands = session.commands.map { CommandUi(it.command, it.description) },
            pluginUiAssets = session.pluginUiAssets.map {
                PluginUiAssetUi(it.id, it.revision, it.sha256, it.module, it.stylesheet)
            },
            pluginUiResponses = session.pluginUiResponses.map {
                PluginUiResponseUi(it.requestId, it.result?.toString(), it.error)
            },
            isStreaming = scopedGraph.any { it.message.deliveryState == "streaming" },
            isResyncing = session.connection.phase == ConnectionPhase.SYNCING,
            canResync = session.connection.phase == ConnectionPhase.READY &&
                session.activeTurnId == null &&
                !session.hasActiveAttachmentDownload,
            isStopping = session.isStopping,
            canStop = session.activeTurnId != null &&
                session.connection.phase == ConnectionPhase.READY &&
                !session.isStopping,
            canSend = session.hasProfile && composerAttachments.all { it.state == ComposerAttachmentState.READY },
        )
    }.stateIn(
        viewModelScope,
        SharingStarted.WhileSubscribed(5_000),
        ConversationUiState(
            connectionLabel = "正在连接",
            connectionStatus = ConnectionStatusUi.CONNECTING,
            connectionNotice = null,
            errorNotice = null,
            sessions = emptyList(),
            selectedSessionId = null,
            readingPosition = null,
            navigationTarget = null,
            projectionGeneration = 0,
            messages = emptyList(),
            attachments = emptyList(),
            pendingMessages = emptyList(),
            commands = emptyList(),
            isStreaming = false,
            isResyncing = false,
            canResync = false,
            isStopping = false,
            canStop = false,
            canSend = false,
        ),
    )

    fun onQrCode(value: String) = container.realtimeSession.beginPairing(value)

    fun sendMessage(
        value: String,
        replyToMessageId: String?,
        expectedAttachmentIds: List<String>,
        onPersisted: (Boolean) -> Unit,
    ) = container.realtimeSession.sendMessage(
        value,
        replyToMessageId,
        expectedAttachmentIds,
        onPersisted,
    )

    fun sendCommand(value: String) = container.realtimeSession.sendCommand(value)

    fun callPluginUi(
        requestId: String,
        sessionId: String?,
        turnId: String?,
        pluginId: String,
        method: String,
        payloadJson: String,
    ) = container.realtimeSession.callPluginUi(
        requestId,
        sessionId,
        turnId,
        pluginId,
        method,
        payloadJson,
    )

    fun acknowledgePluginUiResponses(requestIds: Set<String>) =
        container.realtimeSession.acknowledgePluginUiResponses(requestIds)

    fun stopCurrentTurn() = container.realtimeSession.stopCurrentTurn()

    fun dismissError() = container.realtimeSession.dismissError()

    fun addAttachments(uris: List<android.net.Uri>) = container.realtimeSession.addAttachments(uris)

    fun removeAttachment(attachmentId: String) = container.realtimeSession.removeAttachment(attachmentId)

    fun retryAttachment(attachmentId: String) = container.realtimeSession.retryAttachment(attachmentId)

    fun continueLargeTransfersOnMeteredNetwork() =
        container.realtimeSession.continueLargeTransfersOnMeteredNetwork()

    fun retryFailedMessage(messageId: String) = container.realtimeSession.retryFailedMessage(messageId)

    fun retryDownloadedAttachment(attachmentId: String) =
        container.realtimeSession.retryDownloadedAttachment(attachmentId)

    fun touchDownloadedAttachment(attachmentId: String) =
        container.realtimeSession.touchDownloadedAttachment(attachmentId)

    fun createSession() = container.realtimeSession.createSession()

    fun selectSession(sessionId: String) = container.realtimeSession.selectSession(sessionId)

    fun saveReadingPosition(sessionId: String, messageId: String, offsetPx: Int) {
        viewModelScope.launch {
            require(offsetPx in -10_000..10_000) { "阅读锚点偏移超出范围" }
            val conversation = requireNotNull(container.database.conversations().get(sessionId)) {
                "阅读位置会话不存在: $sessionId"
            }
            require(conversation.serverId == sessionState.value.serverId) { "阅读位置会话不属于当前电脑" }
            val message = requireNotNull(container.database.messages().get(messageId)) {
                "阅读锚点消息不存在: $messageId"
            }
            require(message.sessionId == sessionId) { "阅读锚点不属于当前会话" }
            container.database.conversationReadStates().savePosition(
                sessionId = sessionId,
                messageId = messageId,
                offsetPx = offsetPx,
                updatedAt = System.currentTimeMillis(),
            )
        }
    }

    fun markSessionReadThrough(sessionId: String, readAtMillis: Long) {
        viewModelScope.launch {
            require(readAtMillis >= 0) { "已读水位不能为负数" }
            val conversation = requireNotNull(container.database.conversations().get(sessionId)) {
                "已读水位会话不存在: $sessionId"
            }
            require(conversation.serverId == sessionState.value.serverId) { "已读水位会话不属于当前电脑" }
            container.database.conversationReadStates().markReadThrough(
                sessionId = sessionId,
                readAt = readAtMillis,
                updatedAt = System.currentTimeMillis(),
            )
        }
    }

    fun openNotificationTarget(sessionId: String, messageId: String) {
        container.realtimeSession.selectSession(sessionId)
        navigationTarget.value = NavigationTargetUi(sessionId, messageId)
    }

    fun acknowledgeNavigationTarget(messageId: String) {
        if (navigationTarget.value?.messageId == messageId) navigationTarget.value = null
    }

    fun restartPairing() = MobileConnectionService.disconnect(getApplication())

    fun reloadFromServer() = container.realtimeSession.reloadFromServer()

    private fun toMessageUi(graph: MessageWithBlocks): MessageUi {
        val message = graph.message
        if (message.role == "user") {
            return MessageUi.User(
                id = message.messageId,
                sessionId = message.sessionId,
                text = message.text,
                deliveryLabel = when (message.deliveryState) {
                    "pending" -> "待发送"
                    "sent", "complete" -> "已发送"
                    "failed" -> "发送失败"
                    "failed_retryable" -> "发送失败"
                    "outcome_unknown" -> "结果待确认"
                    else -> error("未知用户消息状态: ${message.deliveryState}")
                },
                deliveryAction = when (message.deliveryState) {
                    "failed_retryable" -> MessageDeliveryActionUi.RETRY
                    "outcome_unknown" -> MessageDeliveryActionUi.VERIFY
                    else -> null
                },
                replyable = userMessageCanReply(message.deliveryState),
                createdAtMillis = message.createdAt,
                reply = message.toReplyUi(),
                attachments = graph.attachmentLinks.toMessageAttachmentUi(),
                updatedAtMillis = message.updatedAt,
            )
        }
        return MessageUi.AssistantTurn(
            id = message.messageId,
            sessionId = message.sessionId,
            intro = null,
            blocks = graph.blocks.sortedBy { it.ordinal }.map { block ->
                val storedTool = if (block.kind == "tool") decodeStoredToolBlock(block.content) else null
                ProcessBlockUi(
                    id = block.blockId,
                    kind = if (block.kind == "thinking") ProcessBlockKind.THINKING else ProcessBlockKind.TOOL,
                    title = storedTool?.name ?: "思考",
                    detail = storedTool?.description ?: if (storedTool == null) block.content else "",
                    state = when (block.status) {
                        "running" -> ProcessBlockState.RUNNING
                        "failed" -> ProcessBlockState.FAILED
                        else -> ProcessBlockState.COMPLETED
                    },
                    arguments = storedTool?.arguments,
                    resultPreview = storedTool?.resultPreview,
                    durationMillis = storedTool?.durationMillis,
                )
            },
            answer = message.text,
            status = when (message.deliveryState) {
                "streaming" -> AssistantTurnStatus.STREAMING
                "complete" -> AssistantTurnStatus.COMPLETE
                "interrupted" -> AssistantTurnStatus.INTERRUPTED
                else -> error("未知助手消息状态: ${message.deliveryState}")
            },
            durationSeconds = turnDurationSeconds(
                startedAt = message.createdAt,
                updatedAt = message.updatedAt,
                isTerminal = message.deliveryState in setOf("complete", "interrupted"),
            ),
            createdAtMillis = message.createdAt,
            reply = message.toReplyUi(),
            attachments = graph.attachmentLinks.toMessageAttachmentUi(),
            updatedAtMillis = message.updatedAt,
        )
    }

    private fun com.akashic.mobile.data.local.MessageEntity.toReplyUi(): MessageReplyUi? {
        val target = replyToMessageId ?: return null
        return MessageReplyUi(
            messageId = target,
            role = requireNotNull(replyRole) { "引用消息缺少角色: $messageId" },
            preview = requireNotNull(replyPreview) { "引用消息缺少预览: $messageId" },
        )
    }
}

internal fun List<MessageAttachmentWithMedia>.toMessageAttachmentUi(): List<MessageAttachmentUi> =
    sortedBy { it.link.ordinal }.map { relation ->
        val attachment = relation.attachment
        MessageAttachmentUi(
            id = attachment.attachmentId,
            filename = attachment.filename,
            contentType = attachment.contentType,
            sizeBytes = attachment.sizeBytes,
            transferredBytes = attachment.transferredBytes,
            state = when (attachment.state) {
                "remote" -> MessageAttachmentState.REMOTE
                "pending" -> MessageAttachmentState.PENDING
                "downloading" -> MessageAttachmentState.DOWNLOADING
                "cached" -> MessageAttachmentState.CACHED
                "failed" -> MessageAttachmentState.FAILED
                "evicted" -> MessageAttachmentState.EVICTED
                else -> error("未知附件下载状态: ${attachment.state}")
            },
            cachePath = attachment.cachePath,
        )
    }

internal fun userMessageCanReply(deliveryState: String): Boolean = deliveryState == "complete"

internal data class ConnectionPresentation(
    val label: String,
    val status: ConnectionStatusUi,
    val notice: String?,
)

/** 把实时链路状态映射为用户可理解的连接语义。 */
internal fun connectionPresentation(
    connection: ConnectionState,
    errorMessage: String? = null,
): ConnectionPresentation {
    val reconnecting = connection.retryCount > 0 &&
        connection.phase in setOf(ConnectionPhase.CONNECTING, ConnectionPhase.DEGRADED)
    if (reconnecting) {
        return ConnectionPresentation(
            label = "正在重连",
            status = ConnectionStatusUi.RECONNECTING,
            notice = "正在重连 · 消息已缓存",
        )
    }
    return when (connection.phase) {
        ConnectionPhase.READY -> ConnectionPresentation("连接正常", ConnectionStatusUi.READY, null)
        ConnectionPhase.SYNCING -> ConnectionPresentation(
            "正在同步消息",
            ConnectionStatusUi.CONNECTING,
            "正在从电脑更新本地消息",
        )
        ConnectionPhase.DEGRADED -> ConnectionPresentation(
            "网络不稳 · 正在续传",
            ConnectionStatusUi.DEGRADED,
            "网络不稳 · 消息已缓存，正在续传",
        )
        ConnectionPhase.CLOSED -> ConnectionPresentation(
            "连接已断开",
            ConnectionStatusUi.DISCONNECTED,
            "连接已断开 · 消息已缓存",
        )
        ConnectionPhase.FAILED -> ConnectionPresentation(
            "启动失败",
            ConnectionStatusUi.DISCONNECTED,
            errorMessage ?: "启动检查失败，请重新打开应用",
        )
        else -> ConnectionPresentation("正在连接", ConnectionStatusUi.CONNECTING, null)
    }
}

internal fun turnDurationSeconds(startedAt: Long, updatedAt: Long, isTerminal: Boolean): Int? {
    if (!isTerminal) return null
    return ceil((updatedAt - startedAt).coerceAtLeast(1) / 1_000.0).toInt()
}
