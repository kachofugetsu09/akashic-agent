package com.akashic.mobile.ui.web

import com.akashic.mobile.ui.conversation.AssistantTurnStatus
import com.akashic.mobile.ui.conversation.CommandUi
import com.akashic.mobile.ui.conversation.ComposerAttachmentState
import com.akashic.mobile.ui.conversation.ComposerAttachmentUi
import com.akashic.mobile.ui.conversation.ConnectionStatusUi
import com.akashic.mobile.ui.conversation.ConversationUiState
import com.akashic.mobile.ui.conversation.MessageAttachmentState
import com.akashic.mobile.ui.conversation.MessageAttachmentUi
import com.akashic.mobile.ui.conversation.MessageUi
import com.akashic.mobile.ui.conversation.MessageDeliveryActionUi
import com.akashic.mobile.ui.conversation.MessageReplyUi
import com.akashic.mobile.ui.conversation.ProcessBlockKind
import com.akashic.mobile.ui.conversation.ProcessBlockState
import com.akashic.mobile.ui.conversation.ProcessBlockUi
import com.akashic.mobile.ui.conversation.PendingMessageUi
import com.akashic.mobile.ui.conversation.ReadingPositionUi
import com.akashic.mobile.ui.conversation.NavigationTargetUi
import com.akashic.mobile.ui.conversation.SessionUi
import com.akashic.mobile.ui.conversation.TransferStatusUi
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

@Serializable
data class MobileWebSnapshot(
    val protocolVersion: Int,
    val connection: MobileWebConnection,
    val sessions: List<MobileWebSession>,
    val selectedSessionId: String?,
    val readingPosition: MobileWebReadingPosition?,
    val navigationTarget: MobileWebNavigationTarget?,
    val projectionGeneration: Long,
    val messages: List<MobileWebMessage>,
    val composer: MobileWebComposer,
)

@Serializable
data class MobileWebConnection(
    val label: String,
    val status: MobileWebConnectionStatus,
    val notice: String?,
    val error: String?,
)

@Serializable
enum class MobileWebConnectionStatus {
    @SerialName("connecting") CONNECTING,
    @SerialName("ready") READY,
    @SerialName("degraded") DEGRADED,
    @SerialName("reconnecting") RECONNECTING,
    @SerialName("disconnected") DISCONNECTED,
}

@Serializable
data class MobileWebSession(
    val id: String,
    val title: String,
    val lastMessagePreview: String?,
    val lastMessageAt: Long?,
    val unreadCount: Int,
    val isRunning: Boolean,
    val isAvailable: Boolean,
    val canRemove: Boolean,
)

@Serializable
data class MobileWebReadingPosition(
    val messageId: String,
    val offsetPx: Int,
)

@Serializable
data class MobileWebNavigationTarget(
    val sessionId: String,
    val messageId: String,
)

@Serializable
data class MobileWebPendingMessage(
    val messageId: String,
    val preview: String,
    val createdAt: Long,
)

@Serializable
data class MobileWebMessage(
    val id: String,
    val sessionId: String,
    val role: MobileWebRole,
    val content: String,
    val createdAt: Long,
    val searchRevision: Long,
    val replyable: Boolean,
    val reply: MobileWebReply? = null,
    val deliveryLabel: String? = null,
    val deliveryAction: MobileWebDeliveryAction? = null,
    val blocks: List<MobileWebProcessBlock> = emptyList(),
    val streaming: Boolean = false,
    val interrupted: Boolean = false,
    val durationSeconds: Int? = null,
    val attachments: List<MobileWebAttachment> = emptyList(),
)

@Serializable
enum class MobileWebDeliveryAction {
    @SerialName("retry") RETRY,
    @SerialName("verify") VERIFY,
}

@Serializable
data class MobileWebReply(
    val messageId: String,
    val role: String,
    val preview: String,
)

@Serializable
enum class MobileWebRole {
    @SerialName("user") USER,
    @SerialName("assistant") ASSISTANT,
}

@Serializable
data class MobileWebProcessBlock(
    val id: String,
    val kind: MobileWebProcessKind,
    val title: String,
    val detail: String,
    val state: MobileWebProcessState,
    val arguments: JsonObject? = null,
    val resultPreview: String? = null,
    val durationMillis: Long? = null,
)

@Serializable
enum class MobileWebProcessKind {
    @SerialName("thinking") THINKING,
    @SerialName("tool") TOOL,
}

@Serializable
enum class MobileWebProcessState {
    @SerialName("completed") COMPLETED,
    @SerialName("running") RUNNING,
    @SerialName("failed") FAILED,
}

@Serializable
data class MobileWebAttachment(
    val id: String,
    val filename: String,
    val contentType: String,
    val sizeBytes: Long,
    val transferredBytes: Long,
    val state: String,
    val canRemove: Boolean = false,
    val contentUrl: String? = null,
)

@Serializable
data class MobileWebCommand(
    val command: String,
    val description: String,
)

@Serializable
data class MobileWebComposer(
    val draft: MobileWebComposerDraft,
    val attachments: List<MobileWebAttachment>,
    val pendingMessages: List<MobileWebPendingMessage>,
    val transferStatus: MobileWebTransferStatus?,
    val commands: List<MobileWebCommand>,
    val isStreaming: Boolean,
    val isResyncing: Boolean,
    val canResync: Boolean,
    val isStopping: Boolean,
    val canStop: Boolean,
    val canSend: Boolean,
)

@Serializable
data class MobileWebComposerDraft(
    val text: String,
    val replyToMessageId: String?,
)

@Serializable
data class MobileWebTransferStatus(
    val title: String,
    val detail: String,
    val progressPercent: Int,
    val requiresMeteredApproval: Boolean,
)

/** 把原生持久化投影转换为版本化 WebView 快照。 */
fun ConversationUiState.toMobileWebSnapshot(): MobileWebSnapshot = MobileWebSnapshot(
    protocolVersion = 5,
    connection = MobileWebConnection(
        label = connectionLabel,
        status = connectionStatus.toMobileWebStatus(),
        notice = connectionNotice,
        error = errorNotice,
    ),
    sessions = sessions.map(SessionUi::toMobileWebSession),
    selectedSessionId = selectedSessionId,
    readingPosition = readingPosition?.toMobileWebReadingPosition(),
    navigationTarget = navigationTarget?.toMobileWebNavigationTarget(),
    projectionGeneration = projectionGeneration,
    messages = messages.map(MessageUi::toMobileWebMessage),
    composer = MobileWebComposer(
        draft = MobileWebComposerDraft(
            text = composerDraft.text,
            replyToMessageId = composerDraft.replyToMessageId,
        ),
        attachments = attachments.map(ComposerAttachmentUi::toMobileWebAttachment),
        pendingMessages = pendingMessages.map(PendingMessageUi::toMobileWebPendingMessage),
        transferStatus = transferStatus?.toMobileWebTransferStatus(),
        commands = commands.map(CommandUi::toMobileWebCommand),
        isStreaming = isStreaming,
        isResyncing = isResyncing,
        canResync = canResync,
        isStopping = isStopping,
        canStop = canStop,
        canSend = canSend,
    ),
)

private fun TransferStatusUi.toMobileWebTransferStatus() = MobileWebTransferStatus(
    title = title,
    detail = detail,
    progressPercent = progressPercent,
    requiresMeteredApproval = requiresMeteredApproval,
)

private fun SessionUi.toMobileWebSession() = MobileWebSession(
    id = sessionId,
    title = title,
    lastMessagePreview = lastMessagePreview,
    lastMessageAt = lastMessageAtMillis,
    unreadCount = unreadCount,
    isRunning = isRunning,
    isAvailable = isAvailable,
    canRemove = canRemove,
)

private fun ReadingPositionUi.toMobileWebReadingPosition() =
    MobileWebReadingPosition(messageId, offsetPx)

private fun NavigationTargetUi.toMobileWebNavigationTarget() =
    MobileWebNavigationTarget(sessionId, messageId)

private fun PendingMessageUi.toMobileWebPendingMessage() =
    MobileWebPendingMessage(messageId, preview, createdAtMillis)

private fun MessageUi.toMobileWebMessage(): MobileWebMessage = when (this) {
    is MessageUi.User -> MobileWebMessage(
        id = id,
        sessionId = sessionId,
        role = MobileWebRole.USER,
        content = text,
        createdAt = createdAtMillis,
        searchRevision = updatedAtMillis,
        replyable = replyable,
        reply = reply?.toMobileWebReply(),
        deliveryLabel = deliveryLabel,
        deliveryAction = deliveryAction?.toMobileWebDeliveryAction(),
        attachments = attachments.map(MessageAttachmentUi::toMobileWebAttachment),
    )
    is MessageUi.AssistantTurn -> MobileWebMessage(
        id = id,
        sessionId = sessionId,
        role = MobileWebRole.ASSISTANT,
        content = answer,
        createdAt = createdAtMillis,
        searchRevision = updatedAtMillis,
        replyable = status == AssistantTurnStatus.COMPLETE,
        reply = reply?.toMobileWebReply(),
        blocks = blocks.map(ProcessBlockUi::toMobileWebProcessBlock),
        streaming = status == AssistantTurnStatus.STREAMING,
        interrupted = status == AssistantTurnStatus.INTERRUPTED,
        durationSeconds = durationSeconds,
        attachments = attachments.map(MessageAttachmentUi::toMobileWebAttachment),
    )
}

private fun MessageDeliveryActionUi.toMobileWebDeliveryAction() = when (this) {
    MessageDeliveryActionUi.RETRY -> MobileWebDeliveryAction.RETRY
    MessageDeliveryActionUi.VERIFY -> MobileWebDeliveryAction.VERIFY
}

private fun ProcessBlockUi.toMobileWebProcessBlock() = MobileWebProcessBlock(
    id = id,
    kind = when (kind) {
        ProcessBlockKind.THINKING -> MobileWebProcessKind.THINKING
        ProcessBlockKind.TOOL -> MobileWebProcessKind.TOOL
    },
    title = title,
    detail = detail,
    state = when (state) {
        ProcessBlockState.COMPLETED -> MobileWebProcessState.COMPLETED
        ProcessBlockState.RUNNING -> MobileWebProcessState.RUNNING
        ProcessBlockState.FAILED -> MobileWebProcessState.FAILED
    },
    arguments = arguments,
    resultPreview = resultPreview,
    durationMillis = durationMillis,
)

private fun MessageReplyUi.toMobileWebReply() = MobileWebReply(messageId, role, preview)

private fun ComposerAttachmentUi.toMobileWebAttachment() = MobileWebAttachment(
    id = id,
    filename = filename,
    contentType = contentType,
    sizeBytes = sizeBytes,
    transferredBytes = transferredBytes,
    state = when (state) {
        ComposerAttachmentState.WAITING_FOR_CONNECTION -> "waiting"
        ComposerAttachmentState.WAITING_FOR_METERED_APPROVAL -> "metered_paused"
        ComposerAttachmentState.UPLOADING -> "uploading"
        ComposerAttachmentState.READY -> "ready"
        ComposerAttachmentState.FAILED -> "failed"
    },
    canRemove = canRemove,
)

private fun MessageAttachmentUi.toMobileWebAttachment() = MobileWebAttachment(
    id = id,
    filename = filename,
    contentType = contentType,
    sizeBytes = sizeBytes,
    transferredBytes = transferredBytes,
    state = when (state) {
        MessageAttachmentState.REMOTE -> "remote"
        MessageAttachmentState.PENDING -> "pending"
        MessageAttachmentState.DOWNLOADING -> "downloading"
        MessageAttachmentState.CACHED -> "cached"
        MessageAttachmentState.FAILED -> "failed"
        MessageAttachmentState.EVICTED -> "evicted"
    },
    contentUrl = if (state == MessageAttachmentState.CACHED) {
        "https://appassets.androidplatform.net/media/$id"
    } else {
        null
    },
)

private fun CommandUi.toMobileWebCommand() = MobileWebCommand(command, description)

private fun ConnectionStatusUi.toMobileWebStatus(): MobileWebConnectionStatus = when (this) {
    ConnectionStatusUi.CONNECTING -> MobileWebConnectionStatus.CONNECTING
    ConnectionStatusUi.READY -> MobileWebConnectionStatus.READY
    ConnectionStatusUi.DEGRADED -> MobileWebConnectionStatus.DEGRADED
    ConnectionStatusUi.RECONNECTING -> MobileWebConnectionStatus.RECONNECTING
    ConnectionStatusUi.DISCONNECTED -> MobileWebConnectionStatus.DISCONNECTED
}
