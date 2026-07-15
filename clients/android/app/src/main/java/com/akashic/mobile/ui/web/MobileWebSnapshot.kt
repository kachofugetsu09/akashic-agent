package com.akashic.mobile.ui.web

import com.akashic.mobile.ui.conversation.ConnectionStatusUi
import com.akashic.mobile.ui.conversation.ConversationUiState
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class MobileWebSnapshot(
    val protocolVersion: Int,
    val connection: MobileWebConnection,
    val selectedSessionId: String?,
    val messages: List<MobileWebMessageRef>,
)

@Serializable
data class MobileWebConnection(
    val label: String,
    val status: MobileWebConnectionStatus,
    val notice: String?,
)

@Serializable
enum class MobileWebConnectionStatus {
    @SerialName("connecting")
    CONNECTING,

    @SerialName("ready")
    READY,

    @SerialName("degraded")
    DEGRADED,

    @SerialName("reconnecting")
    RECONNECTING,

    @SerialName("disconnected")
    DISCONNECTED,
}

@Serializable
data class MobileWebMessageRef(val id: String)

fun ConversationUiState.toMobileWebSnapshot(): MobileWebSnapshot = MobileWebSnapshot(
    protocolVersion = 1,
    connection = MobileWebConnection(
        label = connectionLabel,
        status = connectionStatus.toMobileWebStatus(),
        notice = connectionNotice,
    ),
    selectedSessionId = selectedSessionId,
    messages = messages.map { MobileWebMessageRef(it.id) },
)

private fun ConnectionStatusUi.toMobileWebStatus(): MobileWebConnectionStatus = when (this) {
    ConnectionStatusUi.CONNECTING -> MobileWebConnectionStatus.CONNECTING
    ConnectionStatusUi.READY -> MobileWebConnectionStatus.READY
    ConnectionStatusUi.DEGRADED -> MobileWebConnectionStatus.DEGRADED
    ConnectionStatusUi.RECONNECTING -> MobileWebConnectionStatus.RECONNECTING
    ConnectionStatusUi.DISCONNECTED -> MobileWebConnectionStatus.DISCONNECTED
}
