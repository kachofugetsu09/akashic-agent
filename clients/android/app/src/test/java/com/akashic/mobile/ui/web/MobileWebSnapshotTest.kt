package com.akashic.mobile.ui.web

import com.akashic.mobile.ui.conversation.ConnectionStatusUi
import com.akashic.mobile.ui.conversation.ConversationUiState
import com.akashic.mobile.ui.conversation.AssistantTurnStatus
import com.akashic.mobile.ui.conversation.CommandUi
import com.akashic.mobile.ui.conversation.MessageUi
import com.akashic.mobile.ui.conversation.MessageReplyUi
import com.akashic.mobile.ui.conversation.ProcessBlockKind
import com.akashic.mobile.ui.conversation.ProcessBlockState
import com.akashic.mobile.ui.conversation.ProcessBlockUi
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MobileWebSnapshotTest {
    @Test
    fun serializesVersionedConversationSnapshot() {
        val snapshot = ConversationUiState(
            connectionLabel = "正在重连",
            connectionStatus = ConnectionStatusUi.RECONNECTING,
            connectionNotice = "消息已缓存",
            errorNotice = null,
            sessions = emptyList(),
            selectedSessionId = "mobile:test",
            messages = listOf(
                MessageUi.User(
                    id = "message-1",
                    sessionId = "mobile:test",
                    text = "你好",
                    deliveryLabel = "已发送",
                    replyable = true,
                    createdAtMillis = 1_752_681_600_000,
                    reply = MessageReplyUi("message-0", "assistant", "之前的回答"),
                ),
                MessageUi.AssistantTurn(
                    id = "message-2",
                    sessionId = "mobile:test",
                    intro = null,
                    blocks = listOf(
                        ProcessBlockUi(
                            id = "thinking-1",
                            kind = ProcessBlockKind.THINKING,
                            title = "思考",
                            detail = "正在检查上下文",
                            state = ProcessBlockState.RUNNING,
                        ),
                    ),
                    answer = "正在处理",
                    status = AssistantTurnStatus.STREAMING,
                    durationSeconds = null,
                    createdAtMillis = 1_752_681_601_000,
                ),
            ),
            attachments = emptyList(),
            commands = listOf(CommandUi("memorystatus", "查看记忆整理状态")),
            isStreaming = true,
            isResyncing = false,
            canResync = false,
            isStopping = false,
            canStop = false,
            canSend = true,
        ).toMobileWebSnapshot()

        val encoded = Json.encodeToString(snapshot)

        assertEquals(2, snapshot.protocolVersion)
        assertEquals(MobileWebConnectionStatus.RECONNECTING, snapshot.connection.status)
        assertEquals(listOf("message-1", "message-2"), snapshot.messages.map { it.id })
        assertEquals(listOf("mobile:test", "mobile:test"), snapshot.messages.map { it.sessionId })
        assertTrue(snapshot.messages.first().replyable)
        assertTrue(!snapshot.messages.last().replyable)
        assertTrue(snapshot.messages.last().streaming)
        assertEquals(MobileWebProcessState.RUNNING, snapshot.messages.last().blocks.single().state)
        assertEquals("memorystatus", snapshot.composer.commands.single().command)
        assertEquals("之前的回答", snapshot.messages.first().reply?.preview)
        assertEquals(2, Json.parseToJsonElement(encoded).jsonObject
            .getValue("protocolVersion").jsonPrimitive.content.toInt())
        assertTrue(encoded.contains("\"status\":\"reconnecting\""))
    }
}
