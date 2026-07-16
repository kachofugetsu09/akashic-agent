package com.akashic.mobile.data.realtime

import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class TurnStopCoordinatorTest {
    @Test
    fun duplicateStopAndReconnectReuseOneCommandIdentity() {
        val sent = mutableListOf<TurnStopRequest>()
        var changes = 0
        val coordinator = coordinator(sent = sent, onStateChanged = { changes += 1 })
        coordinator.onTurnStarted("mobile:one", "turn-1")

        val first = coordinator.requestStop("mobile:one")
        val duplicate = coordinator.requestStop("mobile:one")
        coordinator.onConnectionReady()

        assertEquals(first, duplicate)
        assertEquals(listOf(first, first), sent)
        assertTrue(coordinator.isStopping("mobile:one"))
        assertTrue(changes >= 2)
    }

    @Test
    fun activeSessionsRemainVisibleWhenCurrentConversationChanges() {
        val coordinator = coordinator()
        coordinator.onTurnStarted("mobile:one", "turn-1")
        coordinator.onTurnStarted("mobile:two", "turn-2")

        assertEquals(setOf("mobile:one", "mobile:two"), coordinator.activeSessionIds())

        coordinator.onTurnTerminal("mobile:one", "turn-1")
        assertEquals(setOf("mobile:two"), coordinator.activeSessionIds())
    }

    @Test
    fun stopReplyClearsPendingAndProtocolErrorIsVisible() {
        val sent = mutableListOf<TurnStopRequest>()
        val errors = mutableListOf<String>()
        val coordinator = coordinator(sent = sent, onError = errors::add)
        coordinator.onTurnStarted("mobile:one", "turn-1")
        val request = coordinator.requestStop("mobile:one")

        assertTrue(
            coordinator.onReply(
                WireEnvelope(
                    v = 1,
                    kind = WireKind.REPLY,
                    type = "turn.stop.error",
                    id = request.commandId,
                    sessionId = request.sessionId,
                    turnId = request.turnId,
                    payload = buildJsonObject { put("message", "目标 turn 已结束") },
                ),
            ),
        )
        assertFalse(coordinator.isStopping("mobile:one"))
        assertEquals(listOf("目标 turn 已结束"), errors)
    }

    @Test
    fun terminalThenReplyAllowsStoppingNextTurn() {
        val sent = mutableListOf<TurnStopRequest>()
        val coordinator = coordinator(sent = sent)
        coordinator.onTurnStarted("mobile:one", "turn-1")
        val request = coordinator.requestStop("mobile:one")

        coordinator.onTurnTerminal("mobile:one", "turn-1")

        assertEquals(null, coordinator.activeTurnId("mobile:one"))
        assertFalse(coordinator.isStopping("mobile:one"))
        assertTrue(
            coordinator.onReply(
                WireEnvelope(
                    v = 1,
                    kind = WireKind.REPLY,
                    type = "turn.stop.ok",
                    id = request.commandId,
                    sessionId = request.sessionId,
                    turnId = request.turnId,
                    payload = buildJsonObject { put("status", "interrupted") },
                ),
            ),
        )
        coordinator.onTurnStarted("mobile:one", "turn-2")

        val next = coordinator.requestStop("mobile:one")

        assertEquals("turn-2", next.turnId)
    }

    @Test
    fun replyThenTerminalAllowsStoppingNextTurn() {
        val coordinator = coordinator()
        coordinator.onTurnStarted("mobile:one", "turn-1")
        val request = coordinator.requestStop("mobile:one")
        assertTrue(coordinator.onReply(okReply(request)))

        coordinator.onTurnTerminal("mobile:one", "turn-1")
        coordinator.onTurnStarted("mobile:one", "turn-2")

        assertEquals("turn-2", coordinator.requestStop("mobile:one").turnId)
    }

    @Test
    fun terminalThenErrorAllowsStoppingNextTurn() {
        val errors = mutableListOf<String>()
        val coordinator = coordinator(onError = errors::add)
        coordinator.onTurnStarted("mobile:one", "turn-1")
        val request = coordinator.requestStop("mobile:one")
        coordinator.onTurnTerminal("mobile:one", "turn-1")

        assertTrue(coordinator.onReply(errorReply(request)))
        coordinator.onTurnStarted("mobile:one", "turn-2")

        assertEquals("turn-2", coordinator.requestStop("mobile:one").turnId)
        assertTrue(errors.isEmpty())
    }

    private fun okReply(request: TurnStopRequest) = WireEnvelope(
        v = 1,
        kind = WireKind.REPLY,
        type = "turn.stop.ok",
        id = request.commandId,
        sessionId = request.sessionId,
        turnId = request.turnId,
        payload = buildJsonObject { put("status", "interrupted") },
    )

    private fun errorReply(request: TurnStopRequest) = WireEnvelope(
        v = 1,
        kind = WireKind.REPLY,
        type = "turn.stop.error",
        id = request.commandId,
        sessionId = request.sessionId,
        turnId = request.turnId,
        payload = buildJsonObject { put("message", "当前会话没有正在生成的内容") },
    )

    private fun coordinator(
        sent: MutableList<TurnStopRequest> = mutableListOf(),
        onError: (String) -> Unit = {},
        onStateChanged: () -> Unit = {},
    ) = TurnStopCoordinator(
        send = { request -> sent.add(request) },
        onTransportUnavailable = {},
        onError = onError,
        onStateChanged = onStateChanged,
    )
}
