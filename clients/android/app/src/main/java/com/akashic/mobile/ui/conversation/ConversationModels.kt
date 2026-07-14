package com.akashic.mobile.ui.conversation

data class ConversationUiState(
    val title: String,
    val connectionLabel: String,
    val messages: List<MessageUi>,
    val isConnectionDegraded: Boolean,
    val isStreaming: Boolean,
    val canSend: Boolean,
)

sealed interface MessageUi {
    val id: String

    data class User(
        override val id: String,
        val text: String,
        val deliveryLabel: String,
    ) : MessageUi

    data class AssistantTurn(
        override val id: String,
        val intro: String?,
        val blocks: List<ProcessBlockUi>,
        val answer: String,
    ) : MessageUi
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
    title = "新对话",
    connectionLabel = "等待安全连接",
    messages = emptyList(),
    isConnectionDegraded = false,
    isStreaming = false,
    canSend = false,
)

internal val PreviewConversationState = ConversationUiState(
    title = "Akashic",
    connectionLabel = "设备已认证 · 局域网",
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
        ),
    ),
    isConnectionDegraded = true,
    isStreaming = true,
    canSend = true,
)
