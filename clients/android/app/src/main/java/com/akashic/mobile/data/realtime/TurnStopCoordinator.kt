package com.akashic.mobile.data.realtime

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonPrimitive

internal data class TurnStopRequest(
    val commandId: String,
    val sessionId: String,
    val turnId: String,
)

internal class TurnStopCoordinator(
    private val send: (TurnStopRequest) -> Boolean,
    private val onTransportUnavailable: (String) -> Unit,
    private val onError: (String) -> Unit,
    private val onStateChanged: () -> Unit,
) {
    private data class PendingStop(
        val request: TurnStopRequest,
        var replyReceived: Boolean = false,
    )

    private val activeTurns = mutableMapOf<String, String>()
    private val pendingStops = mutableMapOf<String, PendingStop>()
    private val terminalAwaitingReplies = mutableMapOf<String, TurnStopRequest>()

    fun onTurnStarted(sessionId: String, turnId: String) {
        val existing = activeTurns[sessionId]
        require(existing == null || existing == turnId) {
            "同一会话出现重叠 turn: $existing -> $turnId"
        }
        activeTurns[sessionId] = turnId
        onStateChanged()
    }

    fun onTurnTerminal(sessionId: String, turnId: String) {
        val existing = activeTurns[sessionId] ?: return
        require(existing == turnId) { "终态 turn 与当前活动 turn 不匹配" }
        activeTurns.remove(sessionId)
        pendingStops[sessionId]?.let { pending ->
            require(pending.request.turnId == turnId) { "终态 turn 与停止请求不匹配" }
            pendingStops.remove(sessionId)
            if (!pending.replyReceived) {
                terminalAwaitingReplies[pending.request.commandId] = pending.request
            }
        }
        onStateChanged()
    }

    fun requestStop(sessionId: String): TurnStopRequest {
        val turnId = requireNotNull(activeTurns[sessionId]) { "当前会话没有正在生成的内容" }
        pendingStops[sessionId]?.let { pending ->
            require(pending.request.turnId == turnId) { "停止请求属于旧 turn" }
            return pending.request
        }
        val request = TurnStopRequest(Ulid.next(), sessionId, turnId)
        pendingStops[sessionId] = PendingStop(request)
        onStateChanged()
        if (!send(request)) {
            onTransportUnavailable("停止生成命令未进入 WebSocket 队列")
        }
        return request
    }

    fun onConnectionReady() {
        terminalAwaitingReplies.clear()
        pendingStops.values.forEach { pending ->
            if (!send(pending.request)) {
                onTransportUnavailable("停止生成命令未进入 WebSocket 队列")
                return
            }
        }
    }

    fun onReply(envelope: WireEnvelope): Boolean {
        val commandId = envelope.id ?: return false
        terminalAwaitingReplies.remove(commandId)?.let { request ->
            validateReply(envelope, request)
            return true
        }
        val pending = pendingStops.values.firstOrNull { it.request.commandId == commandId } ?: return false
        validateReply(envelope, pending.request)
        if (envelope.type == "turn.stop.error") {
            pendingStops.remove(pending.request.sessionId)
            onStateChanged()
            onError(replyMessage(envelope.payload))
        } else {
            pending.replyReceived = true
        }
        return true
    }

    fun activeTurnId(sessionId: String?): String? = sessionId?.let(activeTurns::get)

    fun activeSessionIds(): Set<String> = activeTurns.keys.toSet()

    fun isStopping(sessionId: String?): Boolean {
        if (sessionId == null) return false
        val pending = pendingStops[sessionId] ?: return false
        return activeTurns[sessionId] == pending.request.turnId
    }

    fun reset() {
        activeTurns.clear()
        pendingStops.clear()
        terminalAwaitingReplies.clear()
        onStateChanged()
    }

    private fun replyMessage(payload: JsonObject): String =
        payload["message"]?.jsonPrimitive?.content ?: "停止生成失败"

    private fun validateReply(envelope: WireEnvelope, request: TurnStopRequest) {
        require(envelope.sessionId == request.sessionId) { "停止生成 reply session 不匹配" }
        require(envelope.turnId == request.turnId) { "停止生成 reply turn 不匹配" }
        require(envelope.type in setOf("turn.stop.ok", "turn.stop.error")) {
            "停止生成 reply 类型无效: ${envelope.type}"
        }
    }
}
