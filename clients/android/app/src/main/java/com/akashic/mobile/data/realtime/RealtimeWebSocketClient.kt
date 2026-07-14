package com.akashic.mobile.data.realtime

import com.akashic.mobile.domain.model.ServerEndpoint
import java.net.URI
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import kotlinx.serialization.SerializationException
import okhttp3.CertificatePinner
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString

interface RealtimeSocketListener {
    fun onOpen(attemptId: Long, endpoint: ServerEndpoint)

    fun onEnvelope(attemptId: Long, envelope: WireEnvelope)

    fun onClosed(attemptId: Long, code: Int, reason: String)

    fun onFailure(attemptId: Long, error: Throwable)

    fun onProtocolFailure(attemptId: Long, error: IllegalArgumentException)
}

class RealtimeWebSocketClient(
    private val listener: RealtimeSocketListener,
    private val allowInsecureTransport: Boolean,
) {
    private val attemptSequence = AtomicLong(0)
    private val baseClient = OkHttpClient.Builder()
        .pingInterval(PING_INTERVAL_SECONDS, TimeUnit.SECONDS)
        .build()

    @Volatile
    private var activeSocket: WebSocket? = null

    /** 启动新的连接尝试，并使旧 socket 回调失效。 */
    fun connect(endpoint: ServerEndpoint): Long {
        // 1. 在网络边界验证 transport 与 pin 配置
        val uri = URI(endpoint.url)
        val scheme = uri.scheme?.lowercase()
        require(scheme == "wss" || (scheme == "ws" && allowInsecureTransport)) {
            "Only wss:// endpoints are allowed"
        }
        val host = requireNotNull(uri.host) { "WebSocket endpoint host is required" }

        // 2. 切换 generation，旧异步回调不再影响当前状态
        val attemptId = attemptSequence.incrementAndGet()
        activeSocket?.close(CLOSE_REPLACED, "replaced by a newer connection attempt")

        val client = if (endpoint.tlsSpkiPins.isEmpty()) {
            baseClient
        } else {
            val pinner = CertificatePinner.Builder()
                .add(host, *endpoint.tlsSpkiPins.toTypedArray())
                .build()
            baseClient.newBuilder().certificatePinner(pinner).build()
        }
        val request = Request.Builder().url(endpoint.url).build()
        activeSocket = client.newWebSocket(request, socketListener(attemptId, endpoint))
        return attemptId
    }

    fun send(envelope: WireEnvelope): Boolean = activeSocket?.send(ProtocolCodec.encode(envelope)) ?: false

    fun send(bytes: ByteString): Boolean = activeSocket?.send(bytes) ?: false

    fun close(code: Int = 1000, reason: String = "client closed") {
        attemptSequence.incrementAndGet()
        activeSocket?.close(code, reason)
        activeSocket = null
    }

    private fun socketListener(attemptId: Long, endpoint: ServerEndpoint) = object : WebSocketListener() {
        override fun onOpen(webSocket: WebSocket, response: Response) {
            if (!isCurrent(attemptId)) {
                webSocket.close(CLOSE_REPLACED, "stale connection attempt")
                return
            }
            listener.onOpen(attemptId, endpoint)
        }

        override fun onMessage(webSocket: WebSocket, text: String) {
            if (!isCurrent(attemptId)) return
            try {
                listener.onEnvelope(attemptId, ProtocolCodec.decode(text))
            } catch (error: SerializationException) {
                rejectProtocol(webSocket, attemptId, error)
            } catch (error: IllegalArgumentException) {
                rejectProtocol(webSocket, attemptId, error)
            }
        }

        override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
            if (isCurrent(attemptId)) listener.onClosed(attemptId, code, reason)
        }

        override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
            if (isCurrent(attemptId)) listener.onFailure(attemptId, t)
        }
    }

    private fun rejectProtocol(webSocket: WebSocket, attemptId: Long, error: IllegalArgumentException) {
        listener.onProtocolFailure(attemptId, error)
        webSocket.close(CLOSE_PROTOCOL_ERROR, "invalid protocol frame")
    }

    private fun isCurrent(attemptId: Long): Boolean = attemptSequence.get() == attemptId

    private companion object {
        const val PING_INTERVAL_SECONDS = 25L
        const val CLOSE_REPLACED = 4000
        const val CLOSE_PROTOCOL_ERROR = 4406
    }
}
