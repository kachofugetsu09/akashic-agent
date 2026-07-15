package com.akashic.mobile

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
        val degraded = connectionPresentation(ConnectionState(phase = ConnectionPhase.DEGRADED))
        val disconnected = connectionPresentation(ConnectionState(phase = ConnectionPhase.CLOSED))
        val failed = connectionPresentation(
            ConnectionState(phase = ConnectionPhase.FAILED),
            "启动缓存检查失败：permission denied",
        )

        assertEquals("连接正常", ready.label)
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
}
