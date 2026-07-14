package com.akashic.mobile.data.realtime

import com.akashic.mobile.domain.model.EndpointRoute
import com.akashic.mobile.domain.model.ServerEndpoint
import java.net.URI
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import javax.net.ssl.SSLContext
import kotlinx.serialization.SerializationException
import okhttp3.CertificatePinner
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener

data class SocketCandidateId(val generation: Long, val ordinal: Int)

interface RealtimeSocketListener {
    fun onOpen(candidateId: SocketCandidateId, endpoint: ServerEndpoint)

    fun onEnvelope(candidateId: SocketCandidateId, envelope: WireEnvelope)

    fun onClosed(candidateId: SocketCandidateId, code: Int, reason: String)

    fun onFailure(candidateId: SocketCandidateId, error: Throwable)

    fun onProtocolFailure(candidateId: SocketCandidateId, error: IllegalArgumentException)

    fun onRaceExhausted(generation: Long, error: Throwable)
}

class RealtimeWebSocketClient(
    private val listener: RealtimeSocketListener,
    private val allowInsecureTransport: Boolean,
) {
    private data class RaceState(
        val generation: Long,
        val sockets: MutableMap<SocketCandidateId, WebSocket> = mutableMapOf(),
        var nextOrdinal: Int = 0,
        var winner: SocketCandidateId? = null,
        var tunnelStarted: Boolean = false,
        var delayedTunnel: ScheduledFuture<*>? = null,
        var lastError: Throwable? = null,
    )

    private val generationSequence = AtomicLong(0)
    private val lock = Any()
    private val scheduler = Executors.newSingleThreadScheduledExecutor { task ->
        Thread(task, "mobile-endpoint-race").apply { isDaemon = true }
    }
    private val baseClient = OkHttpClient.Builder()
        .pingInterval(PING_INTERVAL_SECONDS, TimeUnit.SECONDS)
        .build()

    @Volatile
    private var state: RaceState? = null

    /** LAN 立即连接，750ms 后并发启动 tunnel；业务层认证成功后才选 winner。 */
    fun connectRace(lan: List<ServerEndpoint>, tunnel: List<ServerEndpoint>): Long {
        require(lan.all { it.route == EndpointRoute.LAN }) { "LAN race contains a tunnel endpoint" }
        require(tunnel.all { it.route == EndpointRoute.TUNNEL }) { "Tunnel race contains a LAN endpoint" }
        require(lan.isNotEmpty() || tunnel.isNotEmpty()) { "No realtime endpoint configured" }
        (lan + tunnel).forEach(::validateEndpoint)

        val generation = generationSequence.incrementAndGet()
        val previous = synchronized(lock) {
            val old = state
            state = RaceState(generation = generation)
            old
        }
        closeState(previous, "replaced by a newer connection generation")

        if (lan.isEmpty()) {
            synchronized(lock) { state?.takeIf { it.generation == generation }?.tunnelStarted = true }
            tunnel.forEach { startCandidate(generation, it) }
        } else {
            lan.forEach { startCandidate(generation, it) }
            val delayed = scheduler.schedule(
                {
                    synchronized(lock) { state?.takeIf { it.generation == generation }?.tunnelStarted = true }
                    tunnel.forEach { startCandidate(generation, it) }
                    reportExhaustedIfNeeded(generation)
                },
                TUNNEL_RACE_DELAY_MILLIS,
                TimeUnit.MILLISECONDS,
            )
            synchronized(lock) { state?.takeIf { it.generation == generation }?.delayedTunnel = delayed }
        }
        return generation
    }

    /** 只有通过应用身份与设备认证的候选才能成为活动连接。 */
    fun promote(candidateId: SocketCandidateId): Boolean {
        val losers = synchronized(lock) {
            val current = state ?: return false
            if (current.generation != candidateId.generation || candidateId !in current.sockets) return false
            if (current.winner != null) return current.winner == candidateId
            current.winner = candidateId
            current.delayedTunnel?.cancel(false)
            val stale = current.sockets.filterKeys { it != candidateId }.values.toList()
            current.sockets.keys.removeAll { it != candidateId }
            stale
        }
        losers.forEach { it.close(CLOSE_REPLACED, "authenticated candidate won") }
        return true
    }

    fun send(candidateId: SocketCandidateId, envelope: WireEnvelope): Boolean {
        val socket = synchronized(lock) { state?.sockets?.get(candidateId) } ?: return false
        return socket.send(ProtocolCodec.encode(envelope))
    }

    fun send(envelope: WireEnvelope): Boolean {
        val socket = synchronized(lock) {
            val current = state ?: return false
            current.winner?.let(current.sockets::get)
        } ?: return false
        return socket.send(ProtocolCodec.encode(envelope))
    }

    fun reject(candidateId: SocketCandidateId, code: Int, reason: String) {
        val socket = synchronized(lock) {
            val current = state ?: return
            if (current.generation != candidateId.generation || current.winner == candidateId) return
            current.sockets.remove(candidateId)
        } ?: return
        socket.close(code, reason)
        reportExhaustedIfNeeded(candidateId.generation)
    }

    fun close(code: Int = 1000, reason: String = "client closed") {
        generationSequence.incrementAndGet()
        val previous = synchronized(lock) { state.also { state = null } }
        previous?.delayedTunnel?.cancel(false)
        previous?.sockets?.values?.forEach { it.close(code, reason) }
    }

    private fun startCandidate(generation: Long, endpoint: ServerEndpoint) {
        val candidateId = synchronized(lock) {
            val current = state ?: return
            if (current.generation != generation || current.winner != null) return
            SocketCandidateId(generation, current.nextOrdinal++)
        }
        val client = clientFor(endpoint)
        val request = Request.Builder().url(endpoint.url).build()
        val socket = client.newWebSocket(request, socketListener(candidateId, endpoint))
        synchronized(lock) {
            val current = state
            if (current == null || current.generation != generation || current.winner != null) {
                socket.close(CLOSE_REPLACED, "stale connection candidate")
            } else {
                current.sockets[candidateId] = socket
            }
        }
    }

    private fun clientFor(endpoint: ServerEndpoint): OkHttpClient {
        if (endpoint.route == EndpointRoute.TUNNEL) return baseClient
        val uri = URI(endpoint.url)
        val host = requireNotNull(uri.host) { "WebSocket endpoint host is required" }
        val trustManager = LanPinnedTrustManager(endpoint.tlsSpkiPins.toSet())
        val sslContext = SSLContext.getInstance("TLS").apply { init(null, arrayOf(trustManager), null) }
        val pinner = CertificatePinner.Builder()
            .add(host, *endpoint.tlsSpkiPins.toTypedArray())
            .build()
        return baseClient.newBuilder()
            .sslSocketFactory(sslContext.socketFactory, trustManager)
            .certificatePinner(pinner)
            .build()
    }

    private fun socketListener(candidateId: SocketCandidateId, endpoint: ServerEndpoint) =
        object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                if (isCurrent(candidateId)) listener.onOpen(candidateId, endpoint)
                else webSocket.close(CLOSE_REPLACED, "stale connection candidate")
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                if (!isCurrent(candidateId)) return
                try {
                    listener.onEnvelope(candidateId, ProtocolCodec.decode(text))
                } catch (error: SerializationException) {
                    rejectProtocol(webSocket, candidateId, IllegalArgumentException(error.message, error))
                } catch (error: IllegalArgumentException) {
                    rejectProtocol(webSocket, candidateId, error)
                }
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                val wasWinner = removeCandidate(candidateId, null)
                if (wasWinner) listener.onClosed(candidateId, code, reason)
                reportExhaustedIfNeeded(candidateId.generation)
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                val wasWinner = removeCandidate(candidateId, t)
                if (wasWinner) listener.onFailure(candidateId, t)
                reportExhaustedIfNeeded(candidateId.generation)
            }
        }

    private fun rejectProtocol(
        webSocket: WebSocket,
        candidateId: SocketCandidateId,
        error: IllegalArgumentException,
    ) {
        listener.onProtocolFailure(candidateId, error)
        webSocket.close(CLOSE_PROTOCOL_ERROR, "invalid protocol frame")
    }

    private fun removeCandidate(candidateId: SocketCandidateId, error: Throwable?): Boolean = synchronized(lock) {
        val current = state ?: return false
        if (current.generation != candidateId.generation) return false
        current.sockets.remove(candidateId)
        if (error != null) current.lastError = error
        current.winner == candidateId
    }

    private fun reportExhaustedIfNeeded(generation: Long) {
        val error = synchronized(lock) {
            val current = state ?: return
            if (current.generation != generation || current.winner != null) return
            if (!current.tunnelStarted || current.sockets.isNotEmpty()) return
            current.lastError ?: IllegalStateException("All realtime endpoints closed before authentication")
        }
        listener.onRaceExhausted(generation, error)
    }

    private fun isCurrent(candidateId: SocketCandidateId): Boolean = synchronized(lock) {
        val current = state
        current?.generation == candidateId.generation && candidateId in current.sockets
    }

    private fun validateEndpoint(endpoint: ServerEndpoint) {
        val uri = URI(endpoint.url)
        val scheme = uri.scheme?.lowercase()
        require(scheme == "wss" || (scheme == "ws" && allowInsecureTransport)) {
            "Only wss:// endpoints are allowed"
        }
        require(!uri.host.isNullOrBlank()) { "WebSocket endpoint host is required" }
        validateEndpointSecurity(endpoint)
    }

    private fun closeState(previous: RaceState?, reason: String) {
        previous?.delayedTunnel?.cancel(false)
        previous?.sockets?.values?.forEach { it.close(CLOSE_REPLACED, reason) }
    }

    private companion object {
        const val PING_INTERVAL_SECONDS = 25L
        const val TUNNEL_RACE_DELAY_MILLIS = 750L
        const val CLOSE_REPLACED = 4000
        const val CLOSE_PROTOCOL_ERROR = 4406
    }
}
