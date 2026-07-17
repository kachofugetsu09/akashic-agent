package com.akashic.mobile

import com.akashic.mobile.data.local.ConversationSummary
import com.akashic.mobile.data.local.canRemoveFrom
import com.akashic.mobile.data.local.isRemoteMissingIn
import com.akashic.mobile.domain.model.ConnectionPhase
import com.akashic.mobile.domain.model.ConnectionState
import com.akashic.mobile.ui.conversation.ConnectionStatusUi
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class MainViewModelTest {
    @Test
    fun connectionPresentationUsesLinkHealthInsteadOfAuthenticationState() {
        val ready = connectionPresentation(ConnectionState(phase = ConnectionPhase.READY))
        val syncing = connectionPresentation(ConnectionState(phase = ConnectionPhase.SYNCING))
        val degraded = connectionPresentation(ConnectionState(phase = ConnectionPhase.DEGRADED))
        val disconnected = connectionPresentation(ConnectionState(phase = ConnectionPhase.CLOSED))
        val failed = connectionPresentation(
            ConnectionState(phase = ConnectionPhase.FAILED),
            "启动缓存检查失败：permission denied",
        )

        assertEquals("连接正常", ready.label)
        assertEquals("正在同步消息", syncing.label)
        assertEquals("正在从电脑更新本地消息", syncing.notice)
        assertEquals(ConnectionStatusUi.READY, ready.status)
        assertNull(ready.notice)
        assertEquals("网络不稳 · 正在续传", degraded.label)
        assertEquals(ConnectionStatusUi.DEGRADED, degraded.status)
        assertEquals("连接已断开", disconnected.label)
        assertEquals(ConnectionStatusUi.DISCONNECTED, disconnected.status)
        assertEquals("启动失败", failed.label)
        assertEquals(ConnectionStatusUi.DISCONNECTED, failed.status)
        assertEquals("启动缓存检查失败：permission denied", failed.notice)
    }

    @Test
    fun connectionPresentationSeparatesReconnectFromInitialConnect() {
        val initial = connectionPresentation(ConnectionState(phase = ConnectionPhase.CONNECTING))
        val reconnecting = connectionPresentation(
            ConnectionState(phase = ConnectionPhase.DEGRADED, retryCount = 2),
        )

        assertEquals("正在连接", initial.label)
        assertEquals(ConnectionStatusUi.CONNECTING, initial.status)
        assertEquals("正在重连", reconnecting.label)
        assertEquals(ConnectionStatusUi.RECONNECTING, reconnecting.status)
        assertEquals("正在重连 · 消息已缓存", reconnecting.notice)
    }

    @Test
    fun turnDurationRoundsUpOnlyAfterTerminalMessage() {
        assertNull(turnDurationSeconds(startedAt = 1_000, updatedAt = 2_001, isTerminal = false))
        assertEquals(2, turnDurationSeconds(startedAt = 1_000, updatedAt = 2_001, isTerminal = true))
    }

    @Test
    fun userMessageBecomesReplyableOnlyAfterCanonicalCommit() {
        assertEquals(false, userMessageCanReply("pending"))
        assertEquals(false, userMessageCanReply("sent"))
        assertEquals(false, userMessageCanReply("failed"))
        assertEquals(true, userMessageCanReply("complete"))
    }

    @Test
    fun remoteMissingBlocksSendingWhileLocalWorkOnlyBlocksRemoval() {
        val remote = ConversationSummary(
            sessionId = "mobile:remote",
            title = "旧会话",
            lastMessagePreview = "历史",
            lastMessageAt = 1,
            unreadCount = 0,
            isRunning = false,
            anchorMessageId = null,
            anchorOffsetPx = 0,
            remoteKnown = true,
            hasLocalWork = false,
        )
        val local = remote.copy(sessionId = "mobile:local", remoteKnown = false)
        val pending = remote.copy(sessionId = "mobile:pending", hasLocalWork = true)

        assertEquals(false, remote.isRemoteMissingIn(null))
        assertEquals(false, remote.isRemoteMissingIn(setOf("mobile:remote")))
        assertEquals(true, remote.isRemoteMissingIn(emptySet()))
        assertEquals(false, local.isRemoteMissingIn(emptySet()))
        assertEquals(true, pending.isRemoteMissingIn(emptySet()))
        assertEquals(true, remote.canRemoveFrom(emptySet()))
        assertEquals(false, pending.canRemoveFrom(emptySet()))
    }
}
