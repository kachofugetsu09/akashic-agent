package com.akashic.mobile.data.realtime

import com.akashic.mobile.data.local.MediaAttachmentDao
import com.akashic.mobile.data.local.MediaAttachmentEntity
import com.akashic.mobile.data.local.MessageAttachmentEntity
import com.akashic.mobile.data.local.MediaCacheStore
import java.io.File
import java.security.MessageDigest
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class AttachmentDownloadCoordinatorTest {
    @get:Rule
    val temporary = TemporaryFolder()

    @Test
    fun `downloads serial chunks and atomically publishes verified file`() = runBlocking {
        val content = ByteArray(AttachmentChunkCodec.MAX_CHUNK_BYTES + 3) { (it % 251).toByte() }
        val transfer = transfer(content)
        val dao = FakeMediaAttachmentDao(transfer)
        val commands = mutableListOf<SentCommand>()
        val failures = mutableListOf<String>()
        val coordinator = coordinator(dao, commands, failures)

        coordinator.onConnectionReady("server")
        val first = commands.single()
        coordinator.onBinary(chunk(transfer, 0, content.copyOfRange(0, AttachmentChunkCodec.MAX_CHUNK_BYTES)))
        coordinator.onReply(reply(first, transfer, AttachmentChunkCodec.MAX_CHUNK_BYTES.toLong(), complete = false))

        assertEquals(2, commands.size)
        assertEquals(AttachmentChunkCodec.MAX_CHUNK_BYTES.toLong(), commands.last().offset)
        val second = commands.last()
        coordinator.onBinary(
            chunk(
                transfer,
                AttachmentChunkCodec.MAX_CHUNK_BYTES.toLong(),
                content.copyOfRange(AttachmentChunkCodec.MAX_CHUNK_BYTES, content.size),
            ),
        )
        coordinator.onReply(reply(second, transfer, content.size.toLong(), complete = true))

        assertEquals("cached", dao.get(transfer.attachmentId)!!.state)
        assertEquals(content.toList(), File(transfer.cachePath).readBytes().toList())
        assertFalse(File("${transfer.cachePath}.part").exists())
        assertTrue(failures.isEmpty())
    }

    @Test
    fun `reconnect resumes from fsynced offset before missing reply`() = runBlocking {
        val content = ByteArray(AttachmentChunkCodec.MAX_CHUNK_BYTES + 1) { 7 }
        val transfer = transfer(content)
        val dao = FakeMediaAttachmentDao(transfer)
        val firstCommands = mutableListOf<SentCommand>()
        val first = coordinator(dao, firstCommands, mutableListOf())

        first.onConnectionReady("server")
        first.onBinary(chunk(transfer, 0, content.copyOf(AttachmentChunkCodec.MAX_CHUNK_BYTES)))
        first.onDisconnected()

        val resumedCommands = mutableListOf<SentCommand>()
        coordinator(dao, resumedCommands, mutableListOf()).onConnectionReady("server")

        assertEquals(AttachmentChunkCodec.MAX_CHUNK_BYTES.toLong(), dao.get(transfer.attachmentId)!!.transferredBytes)
        assertEquals(AttachmentChunkCodec.MAX_CHUNK_BYTES.toLong(), resumedCommands.single().offset)
    }

    @Test
    fun `same connection ready transition keeps the in-flight command`() = runBlocking {
        val content = byteArrayOf(1, 2, 3)
        val transfer = transfer(content)
        val dao = FakeMediaAttachmentDao(transfer)
        val commands = mutableListOf<SentCommand>()
        val activeStates = mutableListOf<Boolean>()
        val coordinator = coordinator(dao, commands, mutableListOf(), activeStates)

        coordinator.onConnectionReady("server")
        coordinator.onConnectionReady("server")

        assertEquals(1, commands.size)
        assertEquals(listOf(true), activeStates)
        coordinator.onBinary(chunk(transfer, 0, content))
        coordinator.onReply(reply(commands.single(), transfer, content.size.toLong(), complete = true))
        assertEquals("cached", dao.get(transfer.attachmentId)!!.state)
        assertEquals(listOf(true, false), activeStates)
    }

    @Test
    fun `reconnect publishes a fully fsynced chunk before its missing reply`() = runBlocking {
        val content = byteArrayOf(1, 2, 3)
        val transfer = transfer(content)
        val dao = FakeMediaAttachmentDao(transfer)
        val first = coordinator(dao, mutableListOf(), mutableListOf())

        first.onConnectionReady("server")
        first.onBinary(chunk(transfer, 0, content))
        first.onDisconnected()

        val resumedCommands = mutableListOf<SentCommand>()
        coordinator(dao, resumedCommands, mutableListOf()).onConnectionReady("server")

        assertTrue(resumedCommands.isEmpty())
        assertEquals("cached", dao.get(transfer.attachmentId)!!.state)
        assertEquals(content.toList(), File(transfer.cachePath).readBytes().toList())
    }

    @Test
    fun `checksum failure remains retryable and never publishes final file`() = runBlocking {
        val content = byteArrayOf(1, 2, 3)
        val transfer = transfer(content).copy(sha256 = "0".repeat(64))
        val dao = FakeMediaAttachmentDao(transfer)
        val commands = mutableListOf<SentCommand>()
        val failures = mutableListOf<String>()
        val coordinator = coordinator(dao, commands, failures)

        coordinator.onConnectionReady("server")
        coordinator.onBinary(chunk(transfer, 0, content))
        coordinator.onReply(reply(commands.single(), transfer, content.size.toLong(), complete = true))

        assertEquals("failed", dao.get(transfer.attachmentId)!!.state)
        assertFalse(File(transfer.cachePath).exists())
        assertEquals(listOf("附件 SHA-256 校验失败"), failures)
        coordinator.retry(transfer.attachmentId)
        assertEquals(2, commands.size)
        assertEquals(0L, commands.last().offset)
    }

    @Test
    fun `reconnect resumes queue without rescanning the whole cache`() = runBlocking {
        val transfer = transfer(byteArrayOf(1)).copy(state = "cached", transferredBytes = 1)
        val orphan = temporary.root.resolve("orphan.bin").apply { writeBytes(byteArrayOf(9)) }
        val coordinator = coordinator(FakeMediaAttachmentDao(transfer), mutableListOf(), mutableListOf())

        coordinator.onConnectionReady("server")

        assertTrue(orphan.exists())
    }

    private fun coordinator(
        dao: FakeMediaAttachmentDao,
        commands: MutableList<SentCommand>,
        failures: MutableList<String>,
        activeStates: MutableList<Boolean>? = null,
    ) = AttachmentDownloadCoordinator(
        dao = dao,
        cache = MediaCacheStore(temporary.root, dao),
        sendCommand = { type, id, sessionId, payload ->
            commands += SentCommand(type, id, sessionId, payload["offset"]!!.toString().toLong())
            true
        },
        onTransportUnavailable = { error(it) },
        onDownloadFailed = failures::add,
        onStateChanged = { active -> activeStates?.add(active) },
    )

    private fun transfer(content: ByteArray): MediaAttachmentEntity {
        val id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        return MediaAttachmentEntity(
            attachmentId = id,
            serverId = "server",
            sessionId = "mobile:test",
            filename = "result.bin",
            contentType = "application/octet-stream",
            sizeBytes = content.size.toLong(),
            sha256 = MessageDigest.getInstance("SHA-256").digest(content).joinToString("") { "%02x".format(it) },
            transferredBytes = 0,
            state = "pending",
            cachePath = temporary.root.resolve("${cacheKey(id)}.bin").absolutePath,
            lastAccessedAt = 1,
            updatedAt = 1,
        )
    }

    private fun cacheKey(id: String): String =
        MessageDigest.getInstance("SHA-256").digest(id.toByteArray()).joinToString("") { "%02x".format(it) }

    private fun chunk(transfer: MediaAttachmentEntity, offset: Long, bytes: ByteArray) =
        AttachmentChunkCodec.DecodedChunk(transfer.attachmentId, offset, bytes)

    private fun reply(
        command: SentCommand,
        transfer: MediaAttachmentEntity,
        nextOffset: Long,
        complete: Boolean,
    ) = WireEnvelope(
        v = 1,
        kind = WireKind.REPLY,
        type = "attachment.download.ok",
        id = command.id,
        connectionEpoch = 1,
        sessionId = transfer.sessionId,
        payload = buildJsonObject {
            put("attachment_id", transfer.attachmentId)
            put("filename", transfer.filename)
            put("content_type", transfer.contentType)
            put("size_bytes", transfer.sizeBytes)
            put("sha256", transfer.sha256)
            put("offset", command.offset)
            put("next_offset", nextOffset)
            put("complete", complete)
        },
    )

    private data class SentCommand(
        val type: String,
        val id: String,
        val sessionId: String,
        val offset: Long,
    )

    private class FakeMediaAttachmentDao(initial: MediaAttachmentEntity) : MediaAttachmentDao {
        private var value = initial

        override suspend fun upsert(attachment: MediaAttachmentEntity) {
            value = attachment
        }

        override suspend fun upsertAll(attachments: List<MediaAttachmentEntity>) {
            value = attachments.single()
        }

        override suspend fun linkAll(links: List<MessageAttachmentEntity>): List<Long> =
            links.map { 1L }

        override suspend fun deleteLinks(messageId: String): Int = 0

        override suspend fun moveLinks(sourceId: String, targetId: String): Int = 0

        override suspend fun get(attachmentId: String): MediaAttachmentEntity? =
            value.takeIf { it.attachmentId == attachmentId }

        override suspend fun pendingDownloads(serverId: String): List<MediaAttachmentEntity> =
            listOf(value).filter { it.serverId == serverId && it.state in setOf("pending", "downloading") }

        override suspend fun updateDownload(
            attachmentId: String,
            transferredBytes: Long,
            state: String,
            updatedAt: Long,
        ): Int {
            if (value.attachmentId != attachmentId) return 0
            value = value.copy(transferredBytes = transferredBytes, state = state, updatedAt = updatedAt)
            return 1
        }

        override suspend fun requestDownload(attachmentId: String, updatedAt: Long): Int {
            if (value.attachmentId != attachmentId || value.state !in setOf("failed", "evicted")) return 0
            value = value.copy(state = "pending", updatedAt = updatedAt)
            return 1
        }

        override suspend fun touch(attachmentId: String, accessedAt: Long): Int {
            if (value.attachmentId != attachmentId || value.state != "cached") return 0
            value = value.copy(lastAccessedAt = accessedAt)
            return 1
        }

        override suspend fun markCached(attachmentId: String, updatedAt: Long): Int {
            if (value.attachmentId != attachmentId) return 0
            value = value.copy(
                transferredBytes = value.sizeBytes,
                state = "cached",
                lastAccessedAt = updatedAt,
                updatedAt = updatedAt,
            )
            return 1
        }

        override suspend fun markEvicted(attachmentId: String, updatedAt: Long): Int {
            if (value.attachmentId != attachmentId) return 0
            value = value.copy(transferredBytes = 0, state = "evicted", updatedAt = updatedAt)
            return 1
        }

        override suspend fun getAll(ids: List<String>): List<MediaAttachmentEntity> =
            listOf(value).filter { it.attachmentId in ids }

        override suspend fun all(): List<MediaAttachmentEntity> = listOf(value)

        override suspend fun unreferenced(): List<MediaAttachmentEntity> = emptyList()

        override suspend fun delete(attachmentId: String): Int = 0

        override suspend fun forMessage(messageId: String): List<MediaAttachmentEntity> = emptyList()
    }
}
