package com.akashic.mobile.data.realtime

import android.os.Build
import com.akashic.mobile.data.local.AppDatabase
import com.akashic.mobile.data.local.AppPreferences
import com.akashic.mobile.data.local.ConversationEntity
import com.akashic.mobile.data.local.LocalDeliveryStore
import com.akashic.mobile.data.local.MessageEntity
import com.akashic.mobile.data.local.OutboxCommandEntity
import com.akashic.mobile.data.local.RealtimeCursorEntity
import com.akashic.mobile.data.local.ServerProfileEntity
import com.akashic.mobile.domain.model.ConnectionPhase
import com.akashic.mobile.domain.model.ConnectionState
import com.akashic.mobile.domain.model.EndpointRoute
import com.akashic.mobile.domain.model.ServerEndpoint
import java.time.Instant
import java.util.UUID
import kotlin.math.min
import kotlin.random.Random
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.put

data class MobileSessionState(
    val initialized: Boolean = false,
    val scanGeneration: Long = 0,
    val connection: ConnectionState = ConnectionState(),
    val hasProfile: Boolean = false,
    val pairingConfirmationCode: String? = null,
    val currentSessionId: String? = null,
    val errorMessage: String? = null,
)

internal object FullJitterBackoff {
    fun maximumDelayMillis(retry: Int): Long {
        require(retry >= 0) { "retry must not be negative" }
        val exponent = min(retry, 8)
        return min(30_000L, 500L * (1L shl exponent))
    }

    fun nextDelayMillis(retry: Int, random: Random = Random.Default): Long =
        random.nextLong(maximumDelayMillis(retry) + 1)
}

class RealtimeSession(
    private val database: AppDatabase,
    private val deliveryStore: LocalDeliveryStore,
    private val preferences: AppPreferences,
    private val deviceKeys: DeviceKeyStore,
    private val scope: CoroutineScope,
    allowInsecureTransport: Boolean,
) : RealtimeSocketListener {
    private data class PendingPairing(
        val qr: PairingQrPayload,
        val keyAlias: String,
        val claim: PairClaimMaterial,
    )

    private val json = Json { encodeDefaults = true; explicitNulls = false }
    private val mutex = Mutex()
    private val socket = RealtimeWebSocketClient(this, allowInsecureTransport)
    private val mutableState = MutableStateFlow(MobileSessionState())
    val state: StateFlow<MobileSessionState> = mutableState.asStateFlow()

    private var profile: ServerProfileEntity? = null
    private var pendingPairing: PendingPairing? = null
    private val challengedCandidates = mutableSetOf<SocketCandidateId>()
    private val candidateEndpoints = mutableMapOf<SocketCandidateId, ServerEndpoint>()
    private var activeCandidate: SocketCandidateId? = null
    private var activeEpoch: Long? = null
    private var retryCount = 0
    private var reconnectJob: Job? = null
    private var ackJob: Job? = null
    private var pendingAckCount = 0
    private var pendingAckSeq = 0L

    fun start() {
        scope.launch {
            mutex.withLock {
                val settings = preferences.settings.first()
                profile = settings.currentServerId?.let { database.serverProfiles().get(it) }
                    ?: database.serverProfiles().first()
                val selected = settings.currentSessionId
                mutableState.value = mutableState.value.copy(
                    initialized = true,
                    hasProfile = profile != null,
                    currentSessionId = selected,
                )
                profile?.let {
                    preferences.selectServer(it.serverId)
                    database.outbox().resetInFlight(it.serverId)
                    connectProfile(it)
                }
            }
        }
    }

    fun beginPairing(rawQr: String) {
        scope.launch {
            try {
                mutex.withLock {
                    val qr = PairingQrDecoder.decode(rawQr)
                    check(profile == null) { "Remove the current server before pairing another one" }
                    val alias = deviceKeys.aliasForServer(qr.serverId)
                    if (!deviceKeys.contains(alias)) deviceKeys.create(alias)
                    val claim = PairingTranscripts.createClaim(
                        qr = qr,
                        devicePublicKeyDer = deviceKeys.publicKeyDer(alias),
                        deviceName = Build.MODEL.take(128).ifBlank { "Android" },
                        capabilities = CAPABILITIES,
                        clientNonce = randomUrlNonce(),
                        signer = { deviceKeys.sign(alias, it) },
                    )
                    pendingPairing = PendingPairing(qr, alias, claim)
                    mutableState.value = MobileSessionState(
                        initialized = true,
                        connection = ConnectionState(phase = ConnectionPhase.CONNECTING),
                    )
                    connectQr(qr)
                }
            } catch (error: IllegalArgumentException) {
                mutex.withLock {
                    mutableState.value = mutableState.value.copy(
                        initialized = true,
                        scanGeneration = mutableState.value.scanGeneration + 1,
                        errorMessage = "二维码无效：${error.message}",
                    )
                }
            }
        }
    }

    fun sendMessage(text: String) {
        scope.launch {
            mutex.withLock {
                val currentProfile = requireNotNull(profile) { "Pair a server before sending" }
                val body = text.trim()
                require(body.isNotEmpty()) { "Message text is empty" }
                val sessionId = mutableState.value.currentSessionId
                    ?: "mobile:${UUID.randomUUID()}".also {
                        preferences.selectSession(it)
                        mutableState.value = mutableState.value.copy(currentSessionId = it)
                    }
                require(MOBILE_SESSION.matches(sessionId)) { "Invalid mobile session_id" }
                val now = System.currentTimeMillis()
                val commandId = Ulid.next(now)
                val clientMessageId = Ulid.next(now)
                val payload = MessageSendPayload(
                    clientMessageId = clientMessageId,
                    sessionId = sessionId,
                    text = body,
                    mediaRefs = emptyList(),
                    clientCreatedAt = Instant.ofEpochMilli(now).toString(),
                )
                val envelope = WireEnvelope(
                    v = WIRE_PROTOCOL_VERSION,
                    kind = WireKind.COMMAND,
                    type = "message.send",
                    id = commandId,
                    connectionEpoch = 1,
                    sessionId = sessionId,
                    payload = ProtocolCodec.json().encodeToJsonElement(MessageSendPayload.serializer(), payload).jsonObject,
                )
                deliveryStore.enqueueMessage(
                    conversation = ConversationEntity(sessionId, currentProfile.serverId, "新对话", now),
                    message = MessageEntity(
                        messageId = "user:$clientMessageId",
                        clientMessageId = clientMessageId,
                        sessionId = sessionId,
                        role = "user",
                        text = body,
                        deliveryState = "pending",
                        createdAt = now,
                        updatedAt = now,
                    ),
                    command = OutboxCommandEntity(
                        commandId = commandId,
                        serverId = currentProfile.serverId,
                        envelopeJson = ProtocolCodec.encode(envelope),
                        state = "pending",
                        attemptCount = 0,
                        createdAt = now,
                        lastAttemptAt = null,
                    ),
                )
                if (mutableState.value.connection.phase == ConnectionPhase.READY) flushOutbox()
            }
        }
    }

    override fun onOpen(candidateId: SocketCandidateId, endpoint: ServerEndpoint) {
        scope.launch {
            mutex.withLock {
                if (candidateId.generation != currentGeneration()) return@withLock
                candidateEndpoints[candidateId] = endpoint
                mutableState.value = mutableState.value.copy(
                    connection = mutableState.value.connection.copy(
                        phase = ConnectionPhase.SERVER_CHALLENGE,
                        endpoint = endpoint,
                    ),
                    errorMessage = null,
                )
            }
        }
    }

    override fun onEnvelope(candidateId: SocketCandidateId, envelope: WireEnvelope) {
        scope.launch {
            mutex.withLock {
                try {
                    handleEnvelope(candidateId, envelope)
                } catch (error: IllegalArgumentException) {
                    mutableState.value = mutableState.value.copy(
                        errorMessage = "连接协议校验失败：${error.message}",
                    )
                    socket.reject(candidateId, 4406, "invalid authenticated server frame")
                }
            }
        }
    }

    override fun onClosed(candidateId: SocketCandidateId, code: Int, reason: String) {
        scope.launch {
            mutex.withLock {
                if (candidateId == activeCandidate) scheduleReconnect("连接关闭：$code $reason")
            }
        }
    }

    override fun onFailure(candidateId: SocketCandidateId, error: Throwable) {
        scope.launch {
            mutex.withLock {
                if (candidateId == activeCandidate) scheduleReconnect(error.message ?: "连接失败")
            }
        }
    }

    override fun onProtocolFailure(candidateId: SocketCandidateId, error: IllegalArgumentException) {
        scope.launch {
            mutex.withLock {
                mutableState.value = mutableState.value.copy(errorMessage = error.message)
            }
        }
    }

    override fun onRaceExhausted(generation: Long, error: Throwable) {
        scope.launch {
            mutex.withLock {
                if (generation == currentGeneration()) scheduleReconnect(error.message ?: "所有 endpoint 均不可用")
            }
        }
    }

    private suspend fun handleEnvelope(candidateId: SocketCandidateId, envelope: WireEnvelope) {
        when (envelope.type) {
            "server.challenge" -> handleChallenge(candidateId, envelope)
            "pair.pending" -> handlePairPending(candidateId, envelope)
            "pair.accepted" -> handlePairAccepted(candidateId, envelope)
            "auth.accepted" -> handleAuthAccepted(candidateId, envelope)
            "protocol.error" -> {
                val error = ProtocolCodec.decodePayload<ProtocolErrorPayload>(envelope.payload)
                mutableState.value = mutableState.value.copy(errorMessage = error.message)
            }
            else -> handleAuthenticatedFrame(candidateId, envelope)
        }
    }

    private fun handleChallenge(candidateId: SocketCandidateId, envelope: WireEnvelope) {
        require(envelope.kind == WireKind.CONTROL) { "server.challenge must be a control frame" }
        val challenge = ProtocolCodec.decodePayload<ServerChallengePayload>(envelope.payload)
        val pairing = pendingPairing
        val expectedServerId = pairing?.qr?.serverId ?: requireNotNull(profile).serverId
        val expectedFingerprint = pairing?.qr?.serverApplicationKeyFingerprint
            ?: requireNotNull(profile).applicationKeyFingerprint
        PairingTranscripts.verifyServerChallenge(challenge, expectedServerId, expectedFingerprint)
        challengedCandidates += candidateId
        if (pairing != null) {
            check(socket.send(candidateId, control("pair.claim", pairing.claim.payload)))
            return
        }
        val currentProfile = requireNotNull(profile)
        val proof = PairingTranscripts.createDeviceProof(
            challenge = challenge,
            deviceId = currentProfile.deviceId,
            clientNonce = randomUrlNonce(),
            signer = { deviceKeys.sign(currentProfile.keyAlias, it) },
        )
        check(socket.send(candidateId, control("device.proof", proof)))
        mutableState.value = mutableState.value.copy(
            connection = mutableState.value.connection.copy(phase = ConnectionPhase.DEVICE_PROOF),
        )
    }

    private fun handlePairPending(candidateId: SocketCandidateId, envelope: WireEnvelope) {
        require(candidateId in challengedCandidates) { "pair.pending arrived before a verified challenge" }
        val pairing = requireNotNull(pendingPairing)
        val pending = ProtocolCodec.decodePayload<PairPendingPayload>(envelope.payload)
        require(pending.pairingId == pairing.qr.pairingId) { "pair.pending pairing_id mismatch" }
        require(pending.confirmationCode == pairing.claim.confirmationCode) { "Pairing confirmation code mismatch" }
        mutableState.value = mutableState.value.copy(pairingConfirmationCode = pending.confirmationCode)
    }

    private suspend fun handlePairAccepted(candidateId: SocketCandidateId, envelope: WireEnvelope) {
        require(candidateId in challengedCandidates) { "pair.accepted arrived before a verified challenge" }
        val pairing = requireNotNull(pendingPairing)
        val accepted = ProtocolCodec.decodePayload<PairAcceptedPayload>(envelope.payload)
        require(accepted.pairingId == pairing.qr.pairingId) { "pair.accepted pairing_id mismatch" }
        val now = System.currentTimeMillis()
        val saved = ServerProfileEntity(
            serverId = pairing.qr.serverId,
            displayName = pairing.qr.serverId,
            deviceId = accepted.deviceId,
            keyAlias = pairing.keyAlias,
            applicationKeyFingerprint = pairing.qr.serverApplicationKeyFingerprint,
            lanEndpointsJson = json.encodeToString(pairing.qr.lanEndpoints),
            tunnelEndpointsJson = json.encodeToString(pairing.qr.tunnelEndpoints),
            tlsSpkiPinsJson = json.encodeToString(pairing.qr.tlsSpkiPins),
            createdAt = now,
        )
        deliveryStore.savePairedProfile(
            saved,
            RealtimeCursorEntity(accepted.deviceId, saved.serverId, 0, 0, now),
        )
        preferences.selectServer(saved.serverId)
        profile = saved
        pendingPairing = null
        mutableState.value = MobileSessionState(
            initialized = true,
            connection = ConnectionState(phase = ConnectionPhase.CONNECTING),
            hasProfile = true,
        )
        socket.close(reason = "pairing accepted; reconnecting with device proof")
        connectProfile(saved)
    }

    private suspend fun handleAuthAccepted(candidateId: SocketCandidateId, envelope: WireEnvelope) {
        require(candidateId in challengedCandidates) { "auth.accepted arrived before a verified challenge" }
        require(envelope.kind == WireKind.CONTROL) { "auth.accepted must be a control frame" }
        val accepted = ProtocolCodec.decodePayload<AuthAcceptedPayload>(envelope.payload)
        require(accepted.connectionEpoch == envelope.connectionEpoch) { "auth.accepted epoch mismatch" }
        val currentProfile = requireNotNull(profile)
        require(accepted.deviceId == currentProfile.deviceId) { "auth.accepted device mismatch" }
        if (!socket.promote(candidateId)) return
        activeCandidate = candidateId
        activeEpoch = accepted.connectionEpoch
        retryCount = 0
        val cursor = requireNotNull(database.realtimeCursors().get(currentProfile.deviceId))
        check(
            socket.send(
                candidateId,
                WireEnvelope(
                    v = WIRE_PROTOCOL_VERSION,
                    kind = WireKind.CONTROL,
                    type = "resume",
                    connectionEpoch = accepted.connectionEpoch,
                    payload = buildJsonObject {
                        put("last_ack", cursor.lastAcknowledgedEventSeq)
                        put("active_turns", kotlinx.serialization.json.JsonArray(emptyList()))
                    },
                ),
            ),
        )
        mutableState.value = mutableState.value.copy(
            connection = ConnectionState(
                phase = ConnectionPhase.SYNCING,
                endpoint = candidateEndpoints[candidateId],
                connectionEpoch = accepted.connectionEpoch,
            ),
            errorMessage = null,
        )
        database.outbox().resetInFlight(currentProfile.serverId)
        flushOutbox()
    }

    private suspend fun handleAuthenticatedFrame(candidateId: SocketCandidateId, envelope: WireEnvelope) {
        if (candidateId != activeCandidate) return
        require(envelope.connectionEpoch == activeEpoch) { "Authenticated frame epoch mismatch" }
        when (envelope.kind) {
            WireKind.EVENT -> {
                val currentProfile = requireNotNull(profile)
                val eventSeq = deliveryStore.applyEvent(
                    serverId = currentProfile.serverId,
                    deviceId = currentProfile.deviceId,
                    envelope = envelope,
                    updatedAt = System.currentTimeMillis(),
                )
                recordAck(eventSeq)
                when (envelope.type) {
                    "sync.completed" -> mutableState.value = mutableState.value.copy(
                        connection = mutableState.value.connection.copy(phase = ConnectionPhase.READY),
                    )
                    "connection.degraded" -> mutableState.value = mutableState.value.copy(
                        connection = mutableState.value.connection.copy(phase = ConnectionPhase.DEGRADED),
                    )
                    "sync.reset_required" -> mutableState.value = mutableState.value.copy(
                        errorMessage = "服务端要求重新同步；当前版本尚不支持历史全量重建",
                    )
                }
            }
            WireKind.REPLY -> {
                val id = requireNotNull(envelope.id)
                if (envelope.type.endsWith(".ok")) {
                    deliveryStore.acknowledgeOutbox(id, System.currentTimeMillis())
                } else if (envelope.type.endsWith(".error")) {
                    deliveryStore.failOutbox(id, System.currentTimeMillis())
                    mutableState.value = mutableState.value.copy(
                        errorMessage = envelope.payload["message"]?.toString()?.trim('"') ?: "消息发送失败",
                    )
                }
            }
            else -> error("Unexpected authenticated server frame: ${envelope.kind}")
        }
    }

    private suspend fun flushOutbox() {
        val currentProfile = requireNotNull(profile)
        val epoch = activeEpoch ?: return
        val candidate = activeCandidate ?: return
        database.outbox().pending(currentProfile.serverId).forEach { command ->
            val stored = ProtocolCodec.decode(command.envelopeJson)
            val wire = stored.copy(connectionEpoch = epoch)
            deliveryStore.markOutboxAttempt(command.commandId, System.currentTimeMillis())
            if (!socket.send(candidate, wire)) {
                deliveryStore.retryOutbox(command.commandId)
                return
            }
        }
    }

    private fun recordAck(eventSeq: Long) {
        pendingAckSeq = eventSeq
        pendingAckCount += 1
        if (pendingAckCount >= ACK_EVENT_LIMIT) {
            sendPendingAck()
            return
        }
        if (ackJob == null) {
            ackJob = scope.launch {
                delay(ACK_DELAY_MILLIS)
                mutex.withLock { sendPendingAck() }
            }
        }
    }

    private fun sendPendingAck() {
        ackJob?.cancel()
        ackJob = null
        if (pendingAckCount == 0) return
        val epoch = activeEpoch ?: return
        val candidate = activeCandidate ?: return
        val sent = socket.send(
            candidate,
            WireEnvelope(
                v = WIRE_PROTOCOL_VERSION,
                kind = WireKind.ACK,
                type = "event.ack",
                connectionEpoch = epoch,
                payload = buildJsonObject { put("through_event_seq", pendingAckSeq) },
            ),
        )
        if (sent) pendingAckCount = 0
    }

    private fun connectQr(qr: PairingQrPayload) {
        reconnectJob?.cancel()
        resetGenerationState()
        currentGenerationValue = socket.connectRace(
            qr.lanEndpoints.lanEndpoints(qr.tlsSpkiPins),
            qr.tunnelEndpoints.tunnelEndpoints(),
        )
    }

    private fun connectProfile(value: ServerProfileEntity) {
        reconnectJob?.cancel()
        resetGenerationState()
        val pins = json.decodeFromString<List<String>>(value.tlsSpkiPinsJson)
        val lan = json.decodeFromString<List<String>>(value.lanEndpointsJson).lanEndpoints(pins)
        val tunnel = json.decodeFromString<List<String>>(value.tunnelEndpointsJson).tunnelEndpoints()
        currentGenerationValue = socket.connectRace(lan, tunnel)
        mutableState.value = mutableState.value.copy(
            connection = ConnectionState(phase = ConnectionPhase.CONNECTING, retryCount = retryCount),
            hasProfile = true,
        )
    }

    private fun scheduleReconnect(message: String) {
        if (reconnectJob?.isActive == true) return
        activeCandidate = null
        activeEpoch = null
        ackJob?.cancel()
        ackJob = null
        pendingAckCount = 0
        retryCount += 1
        mutableState.value = mutableState.value.copy(
            connection = mutableState.value.connection.copy(
                phase = ConnectionPhase.DEGRADED,
                retryCount = retryCount,
            ),
            errorMessage = message,
        )
        reconnectJob = scope.launch {
            delay(FullJitterBackoff.nextDelayMillis(retryCount))
            mutex.withLock {
                pendingPairing?.let { connectQr(it.qr) } ?: profile?.let(::connectProfile)
            }
        }
    }

    private fun resetGenerationState() {
        challengedCandidates.clear()
        candidateEndpoints.clear()
        activeCandidate = null
        activeEpoch = null
    }

    private fun control(type: String, payload: kotlinx.serialization.json.JsonObject) = WireEnvelope(
        v = WIRE_PROTOCOL_VERSION,
        kind = WireKind.CONTROL,
        type = type,
        payload = payload,
    )

    private var currentGenerationValue = 0L

    private fun currentGeneration(): Long = currentGenerationValue

    private fun List<String>.lanEndpoints(pins: List<String>): List<ServerEndpoint> =
        map { ServerEndpoint(it, pins, EndpointRoute.LAN) }

    private fun List<String>.tunnelEndpoints(): List<ServerEndpoint> =
        map { ServerEndpoint(it, emptyList(), EndpointRoute.TUNNEL) }

    private companion object {
        val CAPABILITIES = listOf("chat", "streaming", "tools", "proactive")
        val MOBILE_SESSION = Regex("^mobile:(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$")
        const val ACK_DELAY_MILLIS = 100L
        const val ACK_EVENT_LIMIT = 32
    }
}
