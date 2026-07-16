package com.akashic.mobile.data.realtime

import com.akashic.mobile.data.local.MediaAttachmentDao
import com.akashic.mobile.data.local.MediaAttachmentEntity
import com.akashic.mobile.data.local.MediaCacheStore
import java.io.RandomAccessFile
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject

class AttachmentDownloadCoordinator(
    private val dao: MediaAttachmentDao,
    private val cache: MediaCacheStore,
    private val sendCommand: (String, String, String, JsonObject) -> Boolean,
    private val onTransportUnavailable: (String) -> Unit,
    private val onDownloadFailed: (String) -> Unit,
    private val onStateChanged: (Boolean) -> Unit = {},
) {
    private data class ActiveDownload(
        val commandId: String,
        val transfer: MediaAttachmentEntity,
        val received: AttachmentChunkCodec.DecodedChunk? = null,
    )

    private var serverId: String? = null
    private var active: ActiveDownload? = null

    suspend fun onConnectionReady(currentServerId: String) {
        if (serverId != currentServerId) {
            serverId = currentServerId
            updateActive(null)
        }
        startNext()
    }

    fun onDisconnected() {
        serverId = null
        updateActive(null)
    }

    suspend fun resumeIfIdle(currentServerId: String) {
        if (serverId == currentServerId && active == null) startNext()
    }

    suspend fun retry(attachmentId: String) {
        val changed = dao.requestDownload(attachmentId, System.currentTimeMillis())
        if (changed == 0) {
            val current = requireNotNull(dao.get(attachmentId)) { "附件不存在: $attachmentId" }
            check(current.state in setOf("pending", "downloading", "cached")) {
                "附件当前不可开始下载: $attachmentId"
            }
        }
        startNext()
    }

    /** 接收服务端先于 reply 发来的单个附件分片并持久化 offset。 */
    suspend fun onBinary(chunk: AttachmentChunkCodec.DecodedChunk) {
        val current = requireNotNull(active) { "收到未请求的附件二进制帧" }
        require(current.received == null) { "同一下载命令收到多个二进制帧" }
        require(chunk.attachmentId == current.transfer.attachmentId) { "附件二进制 id 不匹配" }
        require(chunk.offset == current.transfer.transferredBytes) { "附件二进制 offset 不连续" }
        val nextOffset = Math.addExact(chunk.offset, chunk.payload.size.toLong())
        require(nextOffset <= current.transfer.sizeBytes) { "附件二进制超过声明大小" }

        // 1. 分片 fsync 成功后才推进持久化 offset
        val partial = cache.partialFile(current.transfer)
        check(partial.parentFile?.isDirectory == true || partial.parentFile?.mkdirs() == true) {
            "无法创建附件缓存目录"
        }
        RandomAccessFile(partial, "rw").use { file ->
            require(file.length() >= chunk.offset) { "附件临时文件短于已提交 offset" }
            file.setLength(chunk.offset)
            file.seek(chunk.offset)
            file.write(chunk.payload)
            file.fd.sync()
        }
        check(
            dao.updateDownload(
                attachmentId = current.transfer.attachmentId,
                transferredBytes = nextOffset,
                state = "downloading",
                updatedAt = System.currentTimeMillis(),
            ) == 1,
        ) { "下载记录已消失: ${current.transfer.attachmentId}" }
        updateActive(current.copy(received = chunk))
    }

    /** 消费 attachment.download reply；其他 reply 返回 false。 */
    suspend fun onReply(envelope: WireEnvelope): Boolean {
        val current = active ?: return false
        if (envelope.id != current.commandId) return false
        if (envelope.type == "attachment.download.error") {
            fail(current.transfer, replyMessage(envelope))
            return true
        }
        require(envelope.type == "attachment.download.ok") { "Unexpected attachment download reply" }
        val chunk = requireNotNull(current.received) { "附件下载 reply 先于二进制分片" }
        val reply = ProtocolCodec.decodePayload<AttachmentDownloadReplyPayload>(envelope.payload)
        validateReply(current.transfer, chunk, reply)
        val nextOffset = Math.addExact(chunk.offset, chunk.payload.size.toLong())
        updateActive(null)
        if (reply.complete) finish(current.transfer, nextOffset) else startNext()
        return true
    }

    private suspend fun startNext() {
        val currentServerId = serverId ?: return
        if (active != null) return
        var reconciled: MediaAttachmentEntity
        while (true) {
            val transfer = dao.pendingDownloads(currentServerId).firstOrNull() ?: return
            try {
                reconciled = reconcilePartial(transfer)
                if (reconciled.transferredBytes == reconciled.sizeBytes) {
                    finish(reconciled, reconciled.transferredBytes)
                    return
                }
                cache.reserve(reconciled)
                break
            } catch (error: IllegalArgumentException) {
                failPreflight(transfer, error.message ?: "附件下载参数无效")
            } catch (error: IllegalStateException) {
                failPreflight(transfer, error.message ?: "附件下载状态无效")
            } catch (error: ArithmeticException) {
                failPreflight(transfer, "附件下载长度溢出")
            } catch (error: SecurityException) {
                failPreflight(transfer, "附件缓存路径不安全")
            }
        }
        val commandId = Ulid.next()
        updateActive(ActiveDownload(commandId, reconciled))
        val sent = sendCommand(
            "attachment.download",
            commandId,
            reconciled.sessionId,
            ProtocolCodec.json().encodeToJsonElement(
                AttachmentDownloadPayload.serializer(),
                AttachmentDownloadPayload(reconciled.attachmentId, reconciled.transferredBytes),
            ).jsonObject,
        )
        if (!sent) transportUnavailable("附件下载命令未进入 WebSocket 队列")
    }

    private suspend fun failPreflight(transfer: MediaAttachmentEntity, message: String) {
        check(
            dao.updateDownload(
                transfer.attachmentId,
                transfer.transferredBytes,
                "failed",
                System.currentTimeMillis(),
            ) == 1,
        ) { "下载记录已消失: ${transfer.attachmentId}" }
        onDownloadFailed(message)
    }

    private suspend fun reconcilePartial(transfer: MediaAttachmentEntity): MediaAttachmentEntity {
        val partial = cache.partialFile(transfer)
        val actual = if (partial.exists()) partial.length() else 0L
        require(actual <= transfer.sizeBytes) { "附件临时文件超过声明大小" }
        if (actual == transfer.transferredBytes) return transfer
        check(
            dao.updateDownload(transfer.attachmentId, actual, "downloading", System.currentTimeMillis()) == 1,
        ) { "下载记录已消失: ${transfer.attachmentId}" }
        return transfer.copy(transferredBytes = actual, state = "downloading")
    }

    private fun validateReply(
        transfer: MediaAttachmentEntity,
        chunk: AttachmentChunkCodec.DecodedChunk,
        reply: AttachmentDownloadReplyPayload,
    ) {
        require(
            reply.attachmentId == transfer.attachmentId &&
                reply.filename == transfer.filename &&
                reply.contentType == transfer.contentType &&
                reply.sizeBytes == transfer.sizeBytes &&
                reply.sha256.equals(transfer.sha256, ignoreCase = true)
        ) { "附件下载元数据不匹配" }
        require(reply.offset == chunk.offset) { "附件下载 reply offset 不匹配" }
        require(reply.nextOffset == Math.addExact(chunk.offset, chunk.payload.size.toLong())) {
            "附件下载 reply next_offset 不匹配"
        }
        require(reply.complete == (reply.nextOffset == transfer.sizeBytes)) { "附件下载 complete 状态不匹配" }
    }

    private suspend fun finish(transfer: MediaAttachmentEntity, nextOffset: Long) {
        val partial = cache.partialFile(transfer)
        require(nextOffset == transfer.sizeBytes && partial.length() == transfer.sizeBytes) {
            "附件尚未下载完整"
        }
        if (cache.sha256(partial) != transfer.sha256.lowercase()) {
            check(partial.delete()) { "无法删除摘要错误的附件临时文件" }
            check(
                dao.updateDownload(transfer.attachmentId, 0, "failed", System.currentTimeMillis()) == 1,
            ) { "下载记录已消失: ${transfer.attachmentId}" }
            onDownloadFailed("附件 SHA-256 校验失败")
            startNext()
            return
        }
        val target = cache.finalFile(transfer)
        Files.move(
            partial.toPath(),
            target.toPath(),
            StandardCopyOption.ATOMIC_MOVE,
            StandardCopyOption.REPLACE_EXISTING,
        )
        check(dao.markCached(transfer.attachmentId, System.currentTimeMillis()) == 1) {
            "下载记录状态不允许完成: ${transfer.attachmentId}"
        }
        cache.enforceQuota()
        startNext()
    }

    private suspend fun fail(transfer: MediaAttachmentEntity, message: String) {
        check(
            dao.updateDownload(
                transfer.attachmentId,
                transfer.transferredBytes,
                "failed",
                System.currentTimeMillis(),
            ) == 1,
        ) { "下载记录已消失: ${transfer.attachmentId}" }
        updateActive(null)
        onDownloadFailed(message)
        startNext()
    }

    private fun transportUnavailable(message: String) {
        onDisconnected()
        onTransportUnavailable(message)
    }

    private fun updateActive(next: ActiveDownload?) {
        val changed = (active == null) != (next == null)
        active = next
        if (changed) onStateChanged(next != null)
    }

    private fun replyMessage(envelope: WireEnvelope): String =
        envelope.payload["message"]?.toString()?.trim('"') ?: "附件下载失败"
}
