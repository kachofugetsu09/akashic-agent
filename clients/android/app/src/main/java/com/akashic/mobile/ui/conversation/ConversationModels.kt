package com.akashic.mobile.ui.conversation

data class ConversationUiState(
    val connectionLabel: String,
    val connectionStatus: ConnectionStatusUi,
    val connectionNotice: String?,
    val errorNotice: String?,
    val sessions: List<SessionUi>,
    val selectedSessionId: String?,
    val messages: List<MessageUi>,
    val attachments: List<ComposerAttachmentUi>,
    val isStreaming: Boolean,
    val isStopping: Boolean,
    val canStop: Boolean,
    val canSend: Boolean,
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
    UPLOADING,
    READY,
    FAILED,
}

data class SessionUi(
    val sessionId: String,
    val title: String,
)

enum class ConnectionStatusUi {
    CONNECTING,
    READY,
    DEGRADED,
    RECONNECTING,
    DISCONNECTED,
}

sealed interface MessageUi {
    val id: String
    val attachments: List<MessageAttachmentUi>

    data class User(
        override val id: String,
        val text: String,
        val deliveryLabel: String,
        override val attachments: List<MessageAttachmentUi> = emptyList(),
    ) : MessageUi

    data class AssistantTurn(
        override val id: String,
        val intro: String?,
        val blocks: List<ProcessBlockUi>,
        val answer: String,
        val isStreaming: Boolean,
        val durationSeconds: Int?,
        override val attachments: List<MessageAttachmentUi> = emptyList(),
    ) : MessageUi
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
    messages = emptyList(),
    attachments = emptyList(),
    isStreaming = false,
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
        SessionUi("mobile:preview-1", "Android 会话设计"),
        SessionUi("mobile:preview-2", "网络抖动恢复策略"),
        SessionUi("mobile:preview-3", "Material 3 交互细节"),
    ),
    selectedSessionId = "mobile:preview-1",
    messages = listOf(
        MessageUi.User(
            id = "user-1",
            text = "帮我检查移动端实时链路，尤其是网络抖动后的恢复。",
            deliveryLabel = "已发送",
        ),
        MessageUi.AssistantTurn(
            id = "assistant-1",
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
            isStreaming = true,
            durationSeconds = null,
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
    isStreaming = true,
    isStopping = false,
    canStop = true,
    canSend = true,
)
