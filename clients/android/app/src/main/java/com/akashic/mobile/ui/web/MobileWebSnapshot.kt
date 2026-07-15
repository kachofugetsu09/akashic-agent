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
import com.akashic.mobile.ui.conversation.ProcessBlockKind
import com.akashic.mobile.ui.conversation.ProcessBlockState
import com.akashic.mobile.ui.conversation.ProcessBlockUi
import com.akashic.mobile.ui.conversation.PluginUiAssetUi
import com.akashic.mobile.ui.conversation.PluginUiResponseUi
import com.akashic.mobile.ui.conversation.SessionUi
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class MobileWebSnapshot(
    val protocolVersion: Int,
    val connection: MobileWebConnection,
    val sessions: List<MobileWebSession>,
    val selectedSessionId: String?,
    val messages: List<MobileWebMessage>,
    val pluginResponses: List<MobileWebPluginResponse>,
    val composer: MobileWebComposer,
)

@Serializable
data class MobileWebPluginAsset(
    val id: String,
    val revision: String,
    val sha256: String,
    val module: String,
    val stylesheet: String,
)

@Serializable
data class MobileWebPluginResponse(
    val requestId: String,
    val resultJson: String?,
    val error: String?,
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
)

@Serializable
data class MobileWebMessage(
    val id: String,
    val role: MobileWebRole,
    val content: String,
    val deliveryLabel: String? = null,
    val blocks: List<MobileWebProcessBlock> = emptyList(),
    val streaming: Boolean = false,
    val interrupted: Boolean = false,
    val durationSeconds: Int? = null,
    val attachments: List<MobileWebAttachment> = emptyList(),
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
    val attachments: List<MobileWebAttachment>,
    val commands: List<MobileWebCommand>,
    val isStreaming: Boolean,
    val isResyncing: Boolean,
    val canResync: Boolean,
    val isStopping: Boolean,
    val canStop: Boolean,
    val canSend: Boolean,
)

/** 把原生持久化投影转换为版本化 WebView 快照。 */
fun ConversationUiState.toMobileWebSnapshot(): MobileWebSnapshot = MobileWebSnapshot(
    protocolVersion = 1,
    connection = MobileWebConnection(
        label = connectionLabel,
        status = connectionStatus.toMobileWebStatus(),
        notice = connectionNotice,
        error = errorNotice,
    ),
    sessions = sessions.map(SessionUi::toMobileWebSession),
    selectedSessionId = selectedSessionId,
    messages = messages.map(MessageUi::toMobileWebMessage),
    pluginResponses = pluginUiResponses.map(PluginUiResponseUi::toMobileWebPluginResponse),
    composer = MobileWebComposer(
        attachments = attachments.map(ComposerAttachmentUi::toMobileWebAttachment),
        commands = commands.map(CommandUi::toMobileWebCommand),
        isStreaming = isStreaming,
        isResyncing = isResyncing,
        canResync = canResync,
        isStopping = isStopping,
        canStop = canStop,
        canSend = canSend,
    ),
)

private fun PluginUiAssetUi.toMobileWebPluginAsset() =
    MobileWebPluginAsset(id, revision, sha256, module, stylesheet)

fun ConversationUiState.toMobileWebPluginAssets(): List<MobileWebPluginAsset> =
    pluginUiAssets.map(PluginUiAssetUi::toMobileWebPluginAsset)

private fun PluginUiResponseUi.toMobileWebPluginResponse() =
    MobileWebPluginResponse(requestId, resultJson, error)

private fun SessionUi.toMobileWebSession() = MobileWebSession(sessionId, title)

private fun MessageUi.toMobileWebMessage(): MobileWebMessage = when (this) {
    is MessageUi.User -> MobileWebMessage(
        id = id,
        role = MobileWebRole.USER,
        content = text,
        deliveryLabel = deliveryLabel,
        attachments = attachments.map(MessageAttachmentUi::toMobileWebAttachment),
    )
    is MessageUi.AssistantTurn -> MobileWebMessage(
        id = id,
        role = MobileWebRole.ASSISTANT,
        content = answer,
        blocks = blocks.map(ProcessBlockUi::toMobileWebProcessBlock),
        streaming = status == AssistantTurnStatus.STREAMING,
        interrupted = status == AssistantTurnStatus.INTERRUPTED,
        durationSeconds = durationSeconds,
        attachments = attachments.map(MessageAttachmentUi::toMobileWebAttachment),
    )
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
)

private fun ComposerAttachmentUi.toMobileWebAttachment() = MobileWebAttachment(
    id = id,
    filename = filename,
    contentType = contentType,
    sizeBytes = sizeBytes,
    transferredBytes = transferredBytes,
    state = when (state) {
        ComposerAttachmentState.WAITING_FOR_CONNECTION -> "waiting"
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
