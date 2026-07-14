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
    fun terminalEventClearsOnlyMatchingActiveTurn() {
        val coordinator = coordinator()
        coordinator.onTurnStarted("mobile:one", "turn-1")

        coordinator.onTurnTerminal("mobile:one", "turn-1")

        assertEquals(null, coordinator.activeTurnId("mobile:one"))
    }

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
