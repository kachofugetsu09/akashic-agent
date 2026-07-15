package com.akashic.mobile.ui.web

import com.akashic.mobile.ui.conversation.ConnectionStatusUi
import com.akashic.mobile.ui.conversation.ConversationUiState
import com.akashic.mobile.ui.conversation.MessageUi
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
            messages = listOf(MessageUi.User("message-1", "你好", "已发送")),
            attachments = emptyList(),
            commands = emptyList(),
            isStreaming = false,
            isResyncing = false,
            canResync = false,
            isStopping = false,
            canStop = false,
            canSend = true,
        ).toMobileWebSnapshot()

        val encoded = Json.encodeToString(snapshot)

        assertEquals(1, snapshot.protocolVersion)
        assertEquals(MobileWebConnectionStatus.RECONNECTING, snapshot.connection.status)
        assertEquals(listOf("message-1"), snapshot.messages.map { it.id })
        assertEquals(1, Json.parseToJsonElement(encoded).jsonObject
            .getValue("protocolVersion").jsonPrimitive.content.toInt())
        assertTrue(encoded.contains("\"status\":\"reconnecting\""))
    }
}
