package com.akashic.mobile.data.realtime

import com.akashic.mobile.data.local.AttachmentTransferDao
import com.akashic.mobile.data.local.AttachmentTransferEntity
import java.io.RandomAccessFile
import java.io.File
import kotlin.math.min
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.put

class AttachmentUploadCoordinator(
    private val dao: AttachmentTransferDao,
    private val sourceFile: (String) -> File,
    private val sendCommand: (String, String, String, JsonObject) -> Boolean,
    private val sendBinary: (okio.ByteString) -> Boolean,
    private val onTransportUnavailable: (String) -> Unit,
    private val onUploadFailed: (String) -> Unit,
    private val canTransfer: suspend (AttachmentTransferEntity) -> Boolean = { true },
) {
    private data class PendingCommand(val type: String, val attachmentId: String)

    private data class ActiveUpload(
        val transfer: AttachmentTransferEntity,
        val expectedProgress: Long,
    )

    private val pendingCommands = mutableMapOf<String, PendingCommand>()
    private var serverId: String? = null
    private var active: ActiveUpload? = null

    suspend fun onConnectionReady(currentServerId: String) {
        serverId = currentServerId
        active = null
        pendingCommands.clear()
        startNext()
    }

    fun onDisconnected() {
        serverId = null
        active = null
        pendingCommands.clear()
    }

    suspend fun resumeIfIdle(currentServerId: String) {
        if (serverId == currentServerId && active == null && pendingCommands.isEmpty()) {
            startNext()
        }
    }

    /** 消费 begin/finish reply；普通消息 reply 返回 false 交回会话处理。 */
    suspend fun onReply(envelope: WireEnvelope): Boolean {
        val commandId = requireNotNull(envelope.id) { "Attachment reply has no id" }
        val pending = pendingCommands.remove(commandId) ?: return false
        if (envelope.type.endsWith(".error")) {
            fail(pending.attachmentId, replyMessage(envelope))
            return true
        }
        when (pending.type) {
            "attachment.begin" -> handleBeginReply(pending.attachmentId, envelope)
            "attachment.finish" -> handleFinishReply(pending.attachmentId, envelope)
            else -> error("Unknown attachment command: ${pending.type}")
        }
        return true
    }

    suspend fun onProgress(payload: AttachmentProgressPayload) {
        val current = active ?: return
        if (payload.attachmentId != current.transfer.attachmentId) return
        require(payload.sizeBytes == current.transfer.sizeBytes) { "Attachment progress size mismatch" }
        require(payload.transferredBytes >= current.expectedProgress) {
            "Attachment progress did not reach the sent window"
        }
        if (payload.transferredBytes == payload.sizeBytes) {
            sendFinish(current.transfer)
        } else if (!canTransfer(current.transfer)) {
            update(current.transfer, payload.transferredBytes, "pending")
            active = null
            startNext()
        } else {
            sendWindow(current.transfer.copy(transferredBytes = payload.transferredBytes))
        }
    }

    private suspend fun handleBeginReply(attachmentId: String, envelope: WireEnvelope) {
        require(envelope.type == "attachment.begin.ok") { "Unexpected attachment begin reply" }
        val payload = ProtocolCodec.decodePayload<AttachmentBeginReplyPayload>(envelope.payload)
        require(payload.attachmentId == attachmentId) { "Attachment begin id mismatch" }
        val transfer = requireNotNull(dao.get(attachmentId)) { "Attachment transfer disappeared" }
        require(
            payload.filename == transfer.filename &&
                payload.contentType == transfer.contentType &&
                payload.sizeBytes == transfer.sizeBytes &&
                payload.sha256 == transfer.sha256
        ) { "Attachment begin metadata mismatch" }
        require(payload.nextOffset in 0..transfer.sizeBytes) { "Attachment begin offset out of range" }
        if (payload.state == "ready") {
            update(transfer, transfer.sizeBytes, "ready")
            active = null
            startNext()
            return
        }
        require(payload.state == "transferring") { "Unsupported attachment state: ${payload.state}" }
        require(payload.chunkSize in 1..AttachmentChunkCodec.MAX_CHUNK_BYTES) {
            "Attachment chunk size out of range"
        }
        update(transfer, payload.nextOffset, "uploading")
        val resumed = transfer.copy(transferredBytes = payload.nextOffset)
        if (payload.nextOffset == transfer.sizeBytes) {
            sendFinish(resumed)
        } else {
            sendWindow(resumed)
        }
    }

    private suspend fun handleFinishReply(attachmentId: String, envelope: WireEnvelope) {
        require(envelope.type == "attachment.finish.ok") { "Unexpected attachment finish reply" }
        val transfer = requireNotNull(dao.get(attachmentId)) { "Attachment transfer disappeared" }
        update(transfer, transfer.sizeBytes, "ready")
        active = null
        startNext()
    }

    private suspend fun startNext() {
        val currentServer = serverId ?: return
        if (active != null || pendingCommands.isNotEmpty()) return
        var transfer: AttachmentTransferEntity
        while (true) {
            transfer = dao.pendingUploads(currentServer).firstOrNull { canTransfer(it) } ?: return
            if (dao.claimUploading(transfer.attachmentId, System.currentTimeMillis()) == 1) break
        }
        val commandId = Ulid.next()
        pendingCommands[commandId] = PendingCommand("attachment.begin", transfer.attachmentId)
        val sent = sendCommand(
            "attachment.begin",
            commandId,
            transfer.sessionId,
            ProtocolCodec.json().encodeToJsonElement(
                AttachmentBeginPayload.serializer(),
                AttachmentBeginPayload(
                    attachmentId = transfer.attachmentId,
                    filename = transfer.filename,
                    contentType = transfer.contentType,
                    sizeBytes = transfer.sizeBytes,
                    sha256 = transfer.sha256,
                ),
            ).jsonObject,
        )
        if (!sent) transportUnavailable("附件 begin 未进入 WebSocket 队列")
    }

    private suspend fun sendWindow(transfer: AttachmentTransferEntity) {
        val windowEnd = min(
            transfer.sizeBytes,
            (transfer.transferredBytes / CONFIRMATION_WINDOW_BYTES + 1) * CONFIRMATION_WINDOW_BYTES,
        )
        active = ActiveUpload(transfer, windowEnd)
        RandomAccessFile(sourceFile(transfer.attachmentId), "r").use { file ->
            require(file.length() == transfer.sizeBytes) { "Attachment source size changed" }
            file.seek(transfer.transferredBytes)
            var offset = transfer.transferredBytes
            while (offset < windowEnd) {
                val size = min(AttachmentChunkCodec.MAX_CHUNK_BYTES.toLong(), windowEnd - offset).toInt()
                val payload = ByteArray(size)
                file.readFully(payload)
                if (!sendBinary(AttachmentChunkCodec.encode(transfer.attachmentId, offset, payload))) {
                    transportUnavailable("附件分片未进入 WebSocket 队列")
                    return
                }
                offset += size
            }
        }
    }

    private suspend fun sendFinish(transfer: AttachmentTransferEntity) {
        update(transfer, transfer.sizeBytes, "finishing")
        val commandId = Ulid.next()
        pendingCommands[commandId] = PendingCommand("attachment.finish", transfer.attachmentId)
        val sent = sendCommand(
            "attachment.finish",
            commandId,
            transfer.sessionId,
            buildJsonObject { put("attachment_id", transfer.attachmentId) },
        )
        if (!sent) transportUnavailable("附件 finish 未进入 WebSocket 队列")
    }

    private suspend fun fail(attachmentId: String, message: String) {
        val transfer = requireNotNull(dao.get(attachmentId)) { "Attachment transfer disappeared" }
        update(transfer, transfer.transferredBytes, "failed")
        active = null
        onUploadFailed(message)
        startNext()
    }

    private suspend fun update(transfer: AttachmentTransferEntity, offset: Long, state: String) {
        check(
            dao.updateState(
                attachmentId = transfer.attachmentId,
                transferredBytes = offset,
                state = state,
                updatedAt = System.currentTimeMillis(),
            ) == 1,
        ) { "Attachment transfer disappeared: ${transfer.attachmentId}" }
    }

    private fun transportUnavailable(message: String) {
        onDisconnected()
        onTransportUnavailable(message)
    }

    private fun replyMessage(envelope: WireEnvelope): String =
        envelope.payload["message"]?.toString()?.trim('"') ?: "附件上传失败"

    private companion object {
        const val CONFIRMATION_WINDOW_BYTES = 1024L * 1024
    }
}
