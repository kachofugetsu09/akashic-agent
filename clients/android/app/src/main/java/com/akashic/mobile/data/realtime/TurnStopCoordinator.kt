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
    private val activeTurns = mutableMapOf<String, String>()
    private val pendingStops = mutableMapOf<String, TurnStopRequest>()

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
        onStateChanged()
    }

    fun requestStop(sessionId: String): TurnStopRequest {
        val turnId = requireNotNull(activeTurns[sessionId]) { "当前会话没有正在生成的内容" }
        pendingStops[sessionId]?.let { pending ->
            require(pending.turnId == turnId) { "停止请求属于旧 turn" }
            return pending
        }
        val request = TurnStopRequest(Ulid.next(), sessionId, turnId)
        pendingStops[sessionId] = request
        onStateChanged()
        if (!send(request)) {
            onTransportUnavailable("停止生成命令未进入 WebSocket 队列")
        }
        return request
    }

    fun onConnectionReady() {
        pendingStops.values.forEach { request ->
            if (!send(request)) {
                onTransportUnavailable("停止生成命令未进入 WebSocket 队列")
                return
            }
        }
    }

    fun onReply(envelope: WireEnvelope): Boolean {
        val commandId = envelope.id ?: return false
        val pending = pendingStops.values.firstOrNull { it.commandId == commandId } ?: return false
        require(envelope.sessionId == pending.sessionId) { "停止生成 reply session 不匹配" }
        require(envelope.turnId == pending.turnId) { "停止生成 reply turn 不匹配" }
        require(envelope.type in setOf("turn.stop.ok", "turn.stop.error")) {
            "停止生成 reply 类型无效: ${envelope.type}"
        }
        pendingStops.remove(pending.sessionId)
        onStateChanged()
        if (envelope.type == "turn.stop.error") {
            onError(replyMessage(envelope.payload))
        }
        return true
    }

    fun activeTurnId(sessionId: String?): String? = sessionId?.let(activeTurns::get)

    fun activeSessionIds(): Set<String> = activeTurns.keys.toSet()

    fun isStopping(sessionId: String?): Boolean = sessionId != null && sessionId in pendingStops

    fun reset() {
        activeTurns.clear()
        pendingStops.clear()
        onStateChanged()
    }

    private fun replyMessage(payload: JsonObject): String =
        payload["message"]?.jsonPrimitive?.content ?: "停止生成失败"
}
