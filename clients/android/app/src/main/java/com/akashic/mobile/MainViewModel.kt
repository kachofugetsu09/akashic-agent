package com.akashic.mobile

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.akashic.mobile.data.local.MessageWithBlocks
import com.akashic.mobile.data.local.MessageAttachmentWithMedia
import com.akashic.mobile.data.local.decodeStoredToolBlock
import com.akashic.mobile.domain.model.ConnectionPhase
import com.akashic.mobile.domain.model.ConnectionState
import com.akashic.mobile.ui.conversation.ConnectionStatusUi
import com.akashic.mobile.ui.conversation.CommandUi
import com.akashic.mobile.ui.conversation.ComposerAttachmentState
import com.akashic.mobile.ui.conversation.ComposerAttachmentUi
import com.akashic.mobile.ui.conversation.ConversationUiState
import com.akashic.mobile.ui.conversation.AssistantTurnStatus
import com.akashic.mobile.ui.conversation.MessageUi
import com.akashic.mobile.ui.conversation.MessageAttachmentState
import com.akashic.mobile.ui.conversation.MessageAttachmentUi
import com.akashic.mobile.ui.conversation.ProcessBlockKind
import com.akashic.mobile.ui.conversation.ProcessBlockState
import com.akashic.mobile.ui.conversation.ProcessBlockUi
import com.akashic.mobile.ui.conversation.SessionUi
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlin.math.ceil

@OptIn(ExperimentalCoroutinesApi::class)
class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val container = (application as App).container
    val sessionState = container.realtimeSession.state

    private val messageGraph = sessionState.flatMapLatest { state ->
        state.currentSessionId?.let(container.database.messages()::observeMessageGraph) ?: flowOf(emptyList())
    }

    private val conversations = sessionState.flatMapLatest { state ->
        state.serverId?.let(container.database.conversations()::observeForServer) ?: flowOf(emptyList())
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
    ) { session, graph, conversations, attachments ->
        val messages = graph.map(::toMessageUi)
        val connection = connectionPresentation(session.connection, session.errorMessage)
        ConversationUiState(
            connectionLabel = connection.label,
            connectionStatus = connection.status,
            connectionNotice = connection.notice,
            errorNotice = session.errorMessage,
            sessions = conversations
                .filter { it.sessionId.startsWith("mobile:") }
                .map { SessionUi(it.sessionId, it.title) },
            selectedSessionId = session.currentSessionId,
            messages = messages,
            attachments = attachments.map { attachment ->
                val waitingForConnection = session.connection.phase != ConnectionPhase.READY &&
                    attachment.state in setOf("pending", "uploading", "finishing")
                ComposerAttachmentUi(
                    id = attachment.attachmentId,
                    filename = attachment.filename,
                    contentType = attachment.contentType,
                    sizeBytes = attachment.sizeBytes,
                    transferredBytes = attachment.transferredBytes,
                    state = when {
                        waitingForConnection -> ComposerAttachmentState.WAITING_FOR_CONNECTION
                        attachment.state == "ready" -> ComposerAttachmentState.READY
                        attachment.state == "failed" -> ComposerAttachmentState.FAILED
                        else -> ComposerAttachmentState.UPLOADING
                    },
                    canRemove = attachment.state in setOf("pending", "ready", "failed"),
                )
            },
            commands = session.commands.map { CommandUi(it.command, it.description) },
            isStreaming = graph.any { it.message.deliveryState == "streaming" },
            isResyncing = session.connection.phase == ConnectionPhase.SYNCING,
            canResync = session.connection.phase == ConnectionPhase.READY && session.activeTurnId == null,
            isStopping = session.isStopping,
            canStop = session.activeTurnId != null &&
                session.connection.phase == ConnectionPhase.READY &&
                !session.isStopping,
            canSend = session.hasProfile,
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
            messages = emptyList(),
            attachments = emptyList(),
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

    fun sendMessage(value: String) = container.realtimeSession.sendMessage(value)

    fun stopCurrentTurn() = container.realtimeSession.stopCurrentTurn()

    fun dismissError() = container.realtimeSession.dismissError()

    fun addAttachments(uris: List<android.net.Uri>) = container.realtimeSession.addAttachments(uris)

    fun removeAttachment(attachmentId: String) = container.realtimeSession.removeAttachment(attachmentId)

    fun retryAttachment(attachmentId: String) = container.realtimeSession.retryAttachment(attachmentId)

    fun retryDownloadedAttachment(attachmentId: String) =
        container.realtimeSession.retryDownloadedAttachment(attachmentId)

    fun touchDownloadedAttachment(attachmentId: String) =
        container.realtimeSession.touchDownloadedAttachment(attachmentId)

    fun createSession() = container.realtimeSession.createSession()

    fun selectSession(sessionId: String) = container.realtimeSession.selectSession(sessionId)

    fun restartPairing() = MobileConnectionService.disconnect(getApplication())

    fun reloadFromServer() = container.realtimeSession.reloadFromServer()

    private fun toMessageUi(graph: MessageWithBlocks): MessageUi {
        val message = graph.message
        if (message.role == "user") {
            return MessageUi.User(
                id = message.messageId,
                text = message.text,
                deliveryLabel = when (message.deliveryState) {
                    "pending" -> "待发送"
                    "sent", "complete" -> "已发送"
                    "failed" -> "发送失败"
                    else -> error("未知用户消息状态: ${message.deliveryState}")
                },
                attachments = graph.attachmentLinks.toMessageAttachmentUi(),
            )
        }
        return MessageUi.AssistantTurn(
            id = message.messageId,
            intro = null,
            blocks = graph.blocks.sortedBy { it.ordinal }.map { block ->
                val storedTool = if (block.kind == "tool") decodeStoredToolBlock(block.content) else null
                ProcessBlockUi(
                    id = block.blockId,
                    kind = if (block.kind == "thinking") ProcessBlockKind.THINKING else ProcessBlockKind.TOOL,
                    title = storedTool?.name ?: "思考",
                    detail = storedTool?.description ?: storedTool?.resultPreview ?: block.content,
                    state = when (block.status) {
                        "running" -> ProcessBlockState.RUNNING
                        "failed" -> ProcessBlockState.FAILED
                        else -> ProcessBlockState.COMPLETED
                    },
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
            attachments = graph.attachmentLinks.toMessageAttachmentUi(),
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
