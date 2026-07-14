package com.akashic.mobile.domain.model

data class ServerEndpoint(
    val url: String,
    val tlsSpkiPins: List<String>,
    val route: EndpointRoute,
)

enum class EndpointRoute {
    LAN,
    TUNNEL,
}

enum class ConnectionPhase {
    IDLE,
    CONNECTING,
    SERVER_CHALLENGE,
    DEVICE_PROOF,
    AUTHENTICATED,
    SYNCING,
    READY,
    DEGRADED,
    FAILED,
    CLOSED,
}

data class ConnectionState(
    val phase: ConnectionPhase = ConnectionPhase.IDLE,
    val endpoint: ServerEndpoint? = null,
    val connectionEpoch: Long? = null,
    val retryCount: Int = 0,
    val lastErrorCode: String? = null,
)
