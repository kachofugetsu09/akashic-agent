package com.akashic.mobile.ui.conversation

import kotlinx.serialization.json.JsonObject

data class ConversationUiState(
    val connectionLabel: String,
    val connectionStatus: ConnectionStatusUi,
    val connectionNotice: String?,
    val errorNotice: String?,
    val sessions: List<SessionUi>,
    val selectedSessionId: String?,
    val readingPosition: ReadingPositionUi?,
    val navigationTarget: NavigationTargetUi?,
    val projectionGeneration: Long,
    val messages: List<MessageUi>,
    val attachments: List<ComposerAttachmentUi>,
    val pendingMessages: List<PendingMessageUi>,
    val transferStatus: TransferStatusUi? = null,
    val commands: List<CommandUi>,
    val pluginUiAssets: List<PluginUiAssetUi> = emptyList(),
    val pluginUiResponses: List<PluginUiResponseUi> = emptyList(),
    val isStreaming: Boolean,
    val isResyncing: Boolean,
    val canResync: Boolean,
    val isStopping: Boolean,
    val canStop: Boolean,
    val canSend: Boolean,
)

data class TransferStatusUi(
    val title: String,
    val detail: String,
    val progressPercent: Int,
    val requiresMeteredApproval: Boolean,
)

data class PluginUiAssetUi(
    val id: String,
    val revision: String,
    val sha256: String,
    val module: String,
    val stylesheet: String,
)

data class PluginUiResponseUi(
    val requestId: String,
    val resultJson: String?,
    val error: String?,
)

data class ComposerAttachmentUi(
    val id: String,
    val filename: String,
    val contentType: String,
    val sizeBytes: Long,
    val transferredBytes: Long,
    val state: ComposerAttachmentState,
    val canRemove: Boolean,
)

enum class ComposerAttachmentState {
    WAITING_FOR_CONNECTION,
    WAITING_FOR_METERED_APPROVAL,
    UPLOADING,
    READY,
    FAILED,
}

data class SessionUi(
    val sessionId: String,
    val title: String,
    val lastMessagePreview: String?,
    val lastMessageAtMillis: Long?,
    val unreadCount: Int,
    val isRunning: Boolean,
)

data class ReadingPositionUi(
    val messageId: String,
    val offsetPx: Int,
)

data class NavigationTargetUi(
    val sessionId: String,
    val messageId: String,
)

data class PendingMessageUi(
    val messageId: String,
    val preview: String,
    val createdAtMillis: Long,
)

data class CommandUi(
    val command: String,
    val description: String,
)

data class MessageReplyUi(
    val messageId: String,
    val role: String,
    val preview: String,
)

enum class MessageDeliveryActionUi {
    RETRY,
    VERIFY,
}

enum class ConnectionStatusUi {
    CONNECTING,
    READY,
    DEGRADED,
    RECONNECTING,
    DISCONNECTED,
}

sealed interface MessageUi {
    val id: String
    val sessionId: String
    val createdAtMillis: Long
    val updatedAtMillis: Long
    val reply: MessageReplyUi?
    val attachments: List<MessageAttachmentUi>

    data class User(
        override val id: String,
        override val sessionId: String,
        val text: String,
        val deliveryLabel: String,
        val replyable: Boolean,
        val deliveryAction: MessageDeliveryActionUi? = null,
        override val createdAtMillis: Long,
        override val reply: MessageReplyUi?,
        override val attachments: List<MessageAttachmentUi> = emptyList(),
        override val updatedAtMillis: Long = createdAtMillis,
    ) : MessageUi

    data class AssistantTurn(
        override val id: String,
        override val sessionId: String,
        val intro: String?,
        val blocks: List<ProcessBlockUi>,
        val answer: String,
        val status: AssistantTurnStatus,
        val durationSeconds: Int?,
        override val createdAtMillis: Long,
        override val reply: MessageReplyUi? = null,
        override val attachments: List<MessageAttachmentUi> = emptyList(),
        override val updatedAtMillis: Long = createdAtMillis,
    ) : MessageUi {
        val isStreaming: Boolean
            get() = status == AssistantTurnStatus.STREAMING
    }
}

enum class AssistantTurnStatus {
    STREAMING,
    COMPLETE,
    INTERRUPTED,
}

data class MessageAttachmentUi(
    val id: String,
    val filename: String,
    val contentType: String,
    val sizeBytes: Long,
    val transferredBytes: Long,
    val state: MessageAttachmentState,
    val cachePath: String,
)

enum class MessageAttachmentState {
    REMOTE,
    PENDING,
    DOWNLOADING,
    CACHED,
    FAILED,
    EVICTED,
}

data class ProcessBlockUi(
    val id: String,
    val kind: ProcessBlockKind,
    val title: String,
    val detail: String,
    val state: ProcessBlockState,
    val arguments: JsonObject? = null,
    val resultPreview: String? = null,
    val durationMillis: Long? = null,
)

enum class ProcessBlockKind {
    THINKING,
    TOOL,
}

enum class ProcessBlockState {
    COMPLETED,
    RUNNING,
    FAILED,
}

internal val EmptyConversationState = ConversationUiState(
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
)

internal val PreviewConversationState = ConversationUiState(
    connectionLabel = "网络不稳 · 正在续传",
    connectionStatus = ConnectionStatusUi.DEGRADED,
    connectionNotice = "网络不稳 · 消息已缓存，正在续传",
    errorNotice = "附件读取失败，请重新选择文件。",
    sessions = listOf(
        SessionUi("mobile:preview-1", "Android 会话设计", "正在检查实时链路", 1_752_681_601_000, 0, true),
        SessionUi("mobile:preview-2", "网络抖动恢复策略", "恢复窗口已经确认", 1_752_681_500_000, 2, false),
        SessionUi("mobile:preview-3", "Material 3 交互细节", null, null, 0, false),
    ),
    selectedSessionId = "mobile:preview-1",
    readingPosition = null,
    navigationTarget = null,
    projectionGeneration = 0,
    messages = listOf(
        MessageUi.User(
            id = "user-1",
            sessionId = "mobile:preview-1",
            text = "帮我检查移动端实时链路，尤其是网络抖动后的恢复。",
            deliveryLabel = "已发送",
            replyable = true,
            createdAtMillis = 1_752_681_600_000,
            reply = null,
        ),
        MessageUi.AssistantTurn(
            id = "assistant-1",
            sessionId = "mobile:preview-1",
            intro = "我先沿着协议和恢复链路检查。",
            blocks = listOf(
                ProcessBlockUi(
                    id = "block-1",
                    kind = ProcessBlockKind.THINKING,
                    title = "分析连接状态机",
                    detail = "核对认证、epoch 与恢复边界。",
                    state = ProcessBlockState.COMPLETED,
                ),
                ProcessBlockUi(
                    id = "block-2",
                    kind = ProcessBlockKind.TOOL,
                    title = "读取协议模型",
                    detail = "codegraph explore mobile realtime",
                    state = ProcessBlockState.COMPLETED,
                ),
                ProcessBlockUi(
                    id = "block-3",
                    kind = ProcessBlockKind.THINKING,
                    title = "检查 ACK 单调性",
                    detail = "旧 epoch 的回调必须被丢弃。",
                    state = ProcessBlockState.COMPLETED,
                ),
                ProcessBlockUi(
                    id = "block-4",
                    kind = ProcessBlockKind.TOOL,
                    title = "运行弱网恢复测试",
                    detail = "等待测试结果…",
                    state = ProcessBlockState.RUNNING,
                ),
            ),
            answer = "当前事件顺序保持一致；连接恢复后会从最后一次累计 ACK 继续。",
            status = AssistantTurnStatus.STREAMING,
            durationSeconds = null,
            createdAtMillis = 1_752_681_601_000,
            attachments = listOf(
                MessageAttachmentUi(
                    id = "preview-download",
                    filename = "弱网恢复报告.pdf",
                    contentType = "application/pdf",
                    sizeBytes = 3_200_000,
                    transferredBytes = 1_344_000,
                    state = MessageAttachmentState.DOWNLOADING,
                    cachePath = "/preview/弱网恢复报告.pdf",
                ),
            ),
        ),
    ),
    attachments = listOf(
        ComposerAttachmentUi(
            id = "preview-upload",
            filename = "network-report.png",
            contentType = "image/png",
            sizeBytes = 2_400_000,
            transferredBytes = 1_008_000,
            state = ComposerAttachmentState.UPLOADING,
            canRemove = false,
        ),
    ),
    pendingMessages = emptyList(),
    commands = listOf(
        CommandUi("undo", "撤销上一轮对话"),
        CommandUi("memorystatus", "查看记忆整理状态"),
        CommandUi("kvcache", "查看 KVCache 状态"),
    ),
    isStreaming = true,
    isResyncing = false,
    canResync = false,
    isStopping = false,
    canStop = true,
    canSend = true,
)
