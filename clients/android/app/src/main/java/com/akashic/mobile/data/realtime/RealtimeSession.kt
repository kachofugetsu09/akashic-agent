package com.akashic.mobile.data.realtime

import android.os.Build
import android.net.Uri
import com.akashic.mobile.data.local.AttachmentDraftStore
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
    val serverId: String? = null,
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
    private val attachmentDrafts: AttachmentDraftStore,
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
    private val uploads = AttachmentUploadCoordinator(
        dao = database.attachmentTransfers(),
        sourceFile = attachmentDrafts::fileFor,
        sendCommand = ::sendAttachmentCommand,
        sendBinary = socket::sendBinary,
        onTransportUnavailable = ::scheduleReconnect,
        onUploadFailed = { message ->
            mutableState.value = mutableState.value.copy(errorMessage = message)
        },
    )
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
            attachmentDrafts.reconcile()
            mutex.withLock {
                val settings = preferences.settings.first()
                profile = settings.currentServerId?.let { database.serverProfiles().get(it) }
                    ?: database.serverProfiles().first()
                val selected = settings.currentSessionId
                mutableState.value = mutableState.value.copy(
                    initialized = true,
                    hasProfile = profile != null,
                    serverId = profile?.serverId,
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

    /** 暂停当前连接并进入扫码页，保留设备密钥、会话和本地消息。 */
    fun restartPairing() {
        scope.launch {
            mutex.withLock {
                reconnectJob?.cancel()
                reconnectJob = null
                ackJob?.cancel()
                ackJob = null
                socket.close(reason = "user requested re-pairing")
                uploads.onDisconnected()
                pendingPairing = null
                profile = null
                resetGenerationState()
                preferences.selectServer(null)
                mutableState.value = MobileSessionState(
                    initialized = true,
                    scanGeneration = mutableState.value.scanGeneration + 1,
                    currentSessionId = mutableState.value.currentSessionId,
                )
            }
        }
    }

    /** 把系统文档复制为可跨重启续传的附件草稿。 */
    fun addAttachments(uris: List<Uri>) {
        scope.launch {
            try {
                val target = mutex.withLock {
                    val currentProfile = requireNotNull(profile) { "Pair a server before attaching" }
                    currentProfile.serverId to ensureCurrentSession(currentProfile)
                }
                attachmentDrafts.import(
                    serverId = target.first,
                    sessionId = target.second,
                    uris = uris,
                    now = System.currentTimeMillis(),
                )
                mutex.withLock {
                    if (mutableState.value.connection.phase == ConnectionPhase.READY) {
                        uploads.resumeIfIdle(target.first)
                    }
                }
            } catch (error: IllegalArgumentException) {
                mutex.withLock {
                    mutableState.value = mutableState.value.copy(errorMessage = error.message)
                }
            } catch (error: java.io.IOException) {
                mutex.withLock {
                    mutableState.value = mutableState.value.copy(errorMessage = "读取附件失败：${error.message}")
                }
            } catch (error: SecurityException) {
                mutex.withLock {
                    mutableState.value = mutableState.value.copy(errorMessage = "没有读取附件的权限")
                }
            }
        }
    }

    fun removeAttachment(attachmentId: String) {
        scope.launch {
            try {
                attachmentDrafts.remove(attachmentId)
            } catch (error: IllegalStateException) {
                mutex.withLock {
                    mutableState.value = mutableState.value.copy(errorMessage = error.message)
                }
            }
        }
    }

    fun retryAttachment(attachmentId: String) {
        scope.launch {
            attachmentDrafts.retry(attachmentId, System.currentTimeMillis())
            mutex.withLock {
                profile?.serverId?.let { serverId ->
                    if (mutableState.value.connection.phase == ConnectionPhase.READY) {
                        uploads.resumeIfIdle(serverId)
                    }
                }
            }
        }
    }

    /** 创建本地消息和 outbox 命令，并在链路可用时立即发送。 */
    fun sendMessage(text: String) {
        scope.launch {
            mutex.withLock {
                // 1. 确定当前手机会话
                val currentProfile = requireNotNull(profile) { "Pair a server before sending" }
                val body = text.trim()
                val sessionId = ensureCurrentSession(currentProfile)
                require(MOBILE_SESSION.matches(sessionId)) { "Invalid mobile session_id" }
                val attachments = database.attachmentTransfers().drafts(currentProfile.serverId, sessionId)
                require(body.isNotEmpty() || attachments.isNotEmpty()) { "消息和附件不能同时为空" }
                require(attachments.all { it.state == "ready" }) { "请等待附件上传完成" }

                // 2. 持久化消息和可重放命令
                val now = System.currentTimeMillis()
                val commandId = Ulid.next(now)
                val clientMessageId = Ulid.next(now)
                val payload = MessageSendPayload(
                    clientMessageId = clientMessageId,
                    sessionId = sessionId,
                    text = body,
                    mediaRefs = attachments.map { it.attachmentId },
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
                    conversation = conversationForMessage(
                        currentProfile,
                        sessionId,
                        body.ifBlank { attachments.joinToString("、") { it.filename } },
                        now,
                    ),
                    message = MessageEntity(
                        messageId = "user:$clientMessageId",
                        clientMessageId = clientMessageId,
                        sessionId = sessionId,
                        role = "user",
                        text = body.ifBlank {
                            attachments.joinToString("、", prefix = "附件：") { it.filename }
                        },
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
                    attachmentIds = attachments.map { it.attachmentId },
                )
                if (mutableState.value.connection.phase == ConnectionPhase.READY) flushOutbox()
            }
        }
    }

    /** 创建并选中一个独立的手机会话。 */
    fun createSession() {
        scope.launch {
            mutex.withLock {
                // 1. 创建本地会话
                val currentProfile = requireNotNull(profile) { "Pair a server before creating a session" }
                val now = System.currentTimeMillis()
                val sessionId = "mobile:${UUID.randomUUID()}"
                database.conversations().upsert(
                    ConversationEntity(sessionId, currentProfile.serverId, "新对话", now),
                )

                // 2. 切换当前会话
                preferences.selectSession(sessionId)
                mutableState.value = mutableState.value.copy(currentSessionId = sessionId)
            }
        }
    }

    /** 选择属于当前电脑的现有手机会话。 */
    fun selectSession(sessionId: String) {
        scope.launch {
            mutex.withLock {
                // 1. 校验会话归属
                require(MOBILE_SESSION.matches(sessionId)) { "Invalid mobile session_id" }
                val currentProfile = requireNotNull(profile) { "Pair a server before selecting a session" }
                val conversation = requireNotNull(database.conversations().get(sessionId)) {
                    "Unknown mobile session: $sessionId"
                }
                require(conversation.serverId == currentProfile.serverId) { "Mobile session belongs to another server" }

                // 2. 持久化选择
                preferences.selectSession(sessionId)
                mutableState.value = mutableState.value.copy(currentSessionId = sessionId)
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
            serverId = saved.serverId,
            currentSessionId = mutableState.value.currentSessionId,
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
                    "sync.completed" -> {
                        mutableState.value = mutableState.value.copy(
                            connection = mutableState.value.connection.copy(phase = ConnectionPhase.READY),
                        )
                        requestSessionList()
                        uploads.onConnectionReady(currentProfile.serverId)
                        flushOutbox()
                    }
                    "session.list" -> requestMissingHistory(envelope)
                    "history.page" -> requestNextHistoryPage(envelope)
                    "attachment.progress" -> uploads.onProgress(
                        ProtocolCodec.decodePayload(envelope.payload),
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
                if (uploads.onReply(envelope)) return
                val id = requireNotNull(envelope.id)
                when (envelope.type) {
                    "message.send.ok" -> attachmentDrafts.deleteSentFiles(
                        deliveryStore.acknowledgeOutbox(id, System.currentTimeMillis()),
                    )
                    "message.send.error" -> {
                        deliveryStore.failOutbox(id, System.currentTimeMillis())
                        mutableState.value = mutableState.value.copy(
                            errorMessage = envelope.payload["message"]?.toString()?.trim('"') ?: "消息发送失败",
                        )
                    }
                    "session.list.ok", "history.get.ok" -> Unit
                    else -> {
                        require(envelope.type.endsWith(".error")) { "Unexpected reply type: ${envelope.type}" }
                        mutableState.value = mutableState.value.copy(
                            errorMessage = envelope.payload["message"]?.toString()?.trim('"') ?: "历史同步失败",
                        )
                    }
                }
            }
            else -> error("Unexpected authenticated server frame: ${envelope.kind}")
        }
    }

    private fun requestSessionList() {
        sendSyncCommand(type = "session.list", sessionId = null, payload = buildJsonObject {})
    }

    private suspend fun requestMissingHistory(envelope: WireEnvelope) {
        val payload = ProtocolCodec.decodePayload<SessionListPayload>(envelope.payload)
        payload.items.forEach { session ->
            if (session.messageCount > 0 && database.messages().countForSession(session.sessionId) == 0) {
                requestHistoryPage(session.sessionId, page = 1)
            }
        }
    }

    private fun requestNextHistoryPage(envelope: WireEnvelope) {
        val sessionId = requireNotNull(envelope.sessionId) { "History page has no session_id" }
        val payload = ProtocolCodec.decodePayload<HistoryPagePayload>(envelope.payload)
        if (payload.page * payload.pageSize < payload.total) {
            requestHistoryPage(sessionId, payload.page + 1)
        }
    }

    private fun requestHistoryPage(sessionId: String, page: Int) {
        sendSyncCommand(
            type = "history.get",
            sessionId = sessionId,
            payload = buildJsonObject {
                put("page", page)
                put("page_size", HISTORY_PAGE_SIZE)
            },
        )
    }

    private fun sendSyncCommand(
        type: String,
        sessionId: String?,
        payload: kotlinx.serialization.json.JsonObject,
    ) {
        val epoch = requireNotNull(activeEpoch) { "History sync requires an authenticated connection" }
        val candidate = requireNotNull(activeCandidate) { "History sync requires an active endpoint" }
        check(
            socket.send(
                candidate,
                WireEnvelope(
                    v = WIRE_PROTOCOL_VERSION,
                    kind = WireKind.COMMAND,
                    type = type,
                    id = Ulid.next(),
                    connectionEpoch = epoch,
                    sessionId = sessionId,
                    payload = payload,
                ),
            ),
        )
    }

    private fun sendAttachmentCommand(
        type: String,
        id: String,
        sessionId: String,
        payload: kotlinx.serialization.json.JsonObject,
    ): Boolean {
        val epoch = activeEpoch ?: return false
        val candidate = activeCandidate ?: return false
        return socket.send(
            candidate,
            WireEnvelope(
                v = WIRE_PROTOCOL_VERSION,
                kind = WireKind.COMMAND,
                type = type,
                id = id,
                connectionEpoch = epoch,
                sessionId = sessionId,
                payload = payload,
            ),
        )
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
            serverId = value.serverId,
        )
    }

    private suspend fun conversationForMessage(
        currentProfile: ServerProfileEntity,
        sessionId: String,
        body: String,
        now: Long,
    ): ConversationEntity {
        val current = database.conversations().get(sessionId)
        val title = if (current == null || current.title == "新对话") {
            body.lineSequence().first().take(32)
        } else {
            current.title
        }
        return ConversationEntity(sessionId, currentProfile.serverId, title, now)
    }

    private suspend fun ensureCurrentSession(currentProfile: ServerProfileEntity): String {
        val existing = mutableState.value.currentSessionId
        if (existing != null) return existing
        val sessionId = "mobile:${UUID.randomUUID()}"
        val now = System.currentTimeMillis()
        database.conversations().upsert(
            ConversationEntity(sessionId, currentProfile.serverId, "新对话", now),
        )
        preferences.selectSession(sessionId)
        mutableState.value = mutableState.value.copy(currentSessionId = sessionId)
        return sessionId
    }

    private fun scheduleReconnect(message: String) {
        if (reconnectJob?.isActive == true) return
        activeCandidate = null
        activeEpoch = null
        ackJob?.cancel()
        ackJob = null
        pendingAckCount = 0
        uploads.onDisconnected()
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
        val CAPABILITIES = listOf("chat", "streaming", "tools", "proactive", "attachments-v1")
        val MOBILE_SESSION = Regex("^mobile:(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$")
        const val HISTORY_PAGE_SIZE = 10
        const val ACK_DELAY_MILLIS = 100L
        const val ACK_EVENT_LIMIT = 32
    }
}
