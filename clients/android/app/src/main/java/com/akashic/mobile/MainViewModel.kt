package com.akashic.mobile

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.akashic.mobile.data.local.MessageWithBlocks
import com.akashic.mobile.domain.model.ConnectionPhase
import com.akashic.mobile.ui.conversation.ConversationUiState
import com.akashic.mobile.ui.conversation.MessageUi
import com.akashic.mobile.ui.conversation.ProcessBlockKind
import com.akashic.mobile.ui.conversation.ProcessBlockState
import com.akashic.mobile.ui.conversation.ProcessBlockUi
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn

@OptIn(ExperimentalCoroutinesApi::class)
class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val container = (application as App).container
    val sessionState = container.realtimeSession.state

    private val messageGraph = sessionState.flatMapLatest { state ->
        state.currentSessionId?.let(container.database.messages()::observeMessageGraph) ?: flowOf(emptyList())
    }

    val conversationState = combine(sessionState, messageGraph) { session, graph ->
        val messages = graph.map(::toMessageUi)
        val phase = session.connection.phase
        ConversationUiState(
            title = "Akashic",
            connectionLabel = connectionLabel(phase),
            messages = messages,
            isConnectionDegraded = phase == ConnectionPhase.DEGRADED,
            isStreaming = graph.any { it.message.deliveryState == "streaming" },
            canSend = session.hasProfile,
        )
    }.stateIn(
        viewModelScope,
        SharingStarted.WhileSubscribed(5_000),
        ConversationUiState("Akashic", "正在载入", emptyList(), false, false, false),
    )

    init {
        container.realtimeSession.start()
    }

    fun onQrCode(value: String) = container.realtimeSession.beginPairing(value)

    fun sendMessage(value: String) = container.realtimeSession.sendMessage(value)

    private fun toMessageUi(graph: MessageWithBlocks): MessageUi {
        val message = graph.message
        if (message.role == "user") {
            return MessageUi.User(
                id = message.messageId,
                text = message.text,
                deliveryLabel = when (message.deliveryState) {
                    "pending" -> "待发送"
                    "sent" -> "已发送"
                    "failed" -> "发送失败"
                    else -> message.deliveryState
                },
            )
        }
        return MessageUi.AssistantTurn(
            id = message.messageId,
            intro = null,
            blocks = graph.blocks.sortedBy { it.ordinal }.map { block ->
                ProcessBlockUi(
                    id = block.blockId,
                    kind = if (block.kind == "thinking") ProcessBlockKind.THINKING else ProcessBlockKind.TOOL,
                    title = if (block.kind == "thinking") "思考" else block.content.lineSequence().firstOrNull() ?: "工具",
                    detail = block.content,
                    state = when (block.status) {
                        "running" -> ProcessBlockState.RUNNING
                        "failed" -> ProcessBlockState.FAILED
                        else -> ProcessBlockState.COMPLETED
                    },
                )
            },
            answer = message.text,
        )
    }

    private fun connectionLabel(phase: ConnectionPhase): String = when (phase) {
        ConnectionPhase.IDLE -> "等待安全连接"
        ConnectionPhase.CONNECTING -> "正在连接"
        ConnectionPhase.SERVER_CHALLENGE -> "正在验证电脑"
        ConnectionPhase.DEVICE_PROOF -> "正在认证设备"
        ConnectionPhase.AUTHENTICATED, ConnectionPhase.SYNCING -> "正在同步"
        ConnectionPhase.READY -> "设备已认证"
        ConnectionPhase.DEGRADED -> "网络不稳 · 正在续传"
        ConnectionPhase.CLOSED -> "连接已关闭"
    }
}
