package com.akashic.mobile.data.realtime

import com.akashic.mobile.data.local.AttachmentTransferDao
import com.akashic.mobile.data.local.AttachmentTransferEntity
import java.io.File
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import okio.ByteString
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class AttachmentUploadCoordinatorTest {
    @get:Rule
    val temporary = TemporaryFolder()

    @Test
    fun `uploads one confirmed window then resumes with a new begin id`() = runBlocking {
        val attachmentId = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        val dao = FakeAttachmentDao(transfer(attachmentId, 0, "pending"))
        val source = temporary.newFile("$attachmentId.upload").apply {
            writeBytes(ByteArray(1024 * 1024 + 3) { (it % 251).toByte() })
        }
        val commands = mutableListOf<SentCommand>()
        val chunks = mutableListOf<ByteString>()
        val coordinator = coordinator(dao, source, commands, chunks)

        coordinator.onConnectionReady("server")
        val firstBegin = commands.single()
        assertEquals("uploading", dao.get(attachmentId)!!.state)
        coordinator.onReply(beginReply(firstBegin.id, attachmentId, nextOffset = 0))
        assertEquals(8, chunks.size)

        coordinator.onDisconnected()
        coordinator.onConnectionReady("server")
        val secondBegin = commands.last()
        assertNotEquals(firstBegin.id, secondBegin.id)
        assertEquals("attachment.begin", secondBegin.type)
    }

    @Test
    fun `finish is sent only after server confirms the complete offset`() = runBlocking {
        val attachmentId = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        val dao = FakeAttachmentDao(transfer(attachmentId, 0, "pending"))
        val source = temporary.newFile("$attachmentId.upload").apply {
            writeBytes(ByteArray(1024 * 1024 + 3))
        }
        val commands = mutableListOf<SentCommand>()
        val chunks = mutableListOf<ByteString>()
        val coordinator = coordinator(dao, source, commands, chunks)

        coordinator.onConnectionReady("server")
        coordinator.onReply(beginReply(commands.single().id, attachmentId, nextOffset = 0))
        assertEquals(listOf("attachment.begin"), commands.map { it.type })

        dao.updateState(attachmentId, 1024 * 1024L, "uploading", 2)
        coordinator.onProgress(
            AttachmentProgressPayload(attachmentId, 1024 * 1024L, 1024 * 1024L + 3),
        )
        assertEquals(9, chunks.size)
        assertEquals(listOf("attachment.begin"), commands.map { it.type })

        dao.updateState(attachmentId, 1024 * 1024L + 3, "finishing", 3)
        coordinator.onProgress(
            AttachmentProgressPayload(attachmentId, 1024 * 1024L + 3, 1024 * 1024L + 3),
        )
        assertEquals("attachment.finish", commands.last().type)

        coordinator.onReply(finishReply(commands.last().id, attachmentId))
        assertEquals("ready", dao.get(attachmentId)!!.state)
    }

    @Test
    fun `begin at complete offset immediately resumes finish`() = runBlocking {
        val attachmentId = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        val complete = 1024 * 1024L + 3
        val dao = FakeAttachmentDao(transfer(attachmentId, complete, "finishing"))
        val source = temporary.newFile("$attachmentId.upload").apply {
            writeBytes(ByteArray(complete.toInt()))
        }
        val commands = mutableListOf<SentCommand>()
        val chunks = mutableListOf<ByteString>()
        val coordinator = coordinator(dao, source, commands, chunks)

        coordinator.onConnectionReady("server")
        coordinator.onReply(beginReply(commands.single().id, attachmentId, complete))

        assertTrue(chunks.isEmpty())
        assertEquals(listOf("attachment.begin", "attachment.finish"), commands.map { it.type })
    }

    @Test
    fun `metered policy pauses the persisted queue until approval`() = runBlocking {
        val attachmentId = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        val dao = FakeAttachmentDao(transfer(attachmentId, 0, "pending"))
        val source = temporary.newFile("$attachmentId.upload").apply {
            writeBytes(ByteArray(1024 * 1024 + 3))
        }
        val commands = mutableListOf<SentCommand>()
        var approved = false
        val coordinator = coordinator(dao, source, commands, mutableListOf()) { approved }

        coordinator.onConnectionReady("server")
        assertTrue(commands.isEmpty())
        assertEquals("pending", dao.get(attachmentId)!!.state)

        approved = true
        coordinator.resumeIfIdle("server")
        assertEquals("attachment.begin", commands.single().type)
        assertEquals("uploading", dao.get(attachmentId)!!.state)
    }

    @Test
    fun `network policy change pauses at the confirmed window offset`() = runBlocking {
        val attachmentId = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        val dao = FakeAttachmentDao(transfer(attachmentId, 0, "pending"))
        val source = temporary.newFile("$attachmentId.upload").apply {
            writeBytes(ByteArray(1024 * 1024 + 3))
        }
        val commands = mutableListOf<SentCommand>()
        val chunks = mutableListOf<ByteString>()
        var allowed = true
        val coordinator = coordinator(dao, source, commands, chunks) { allowed }

        coordinator.onConnectionReady("server")
        coordinator.onReply(beginReply(commands.single().id, attachmentId, nextOffset = 0))
        allowed = false
        coordinator.onProgress(
            AttachmentProgressPayload(attachmentId, 1024 * 1024L, 1024 * 1024L + 3),
        )

        assertEquals("pending", dao.get(attachmentId)!!.state)
        assertEquals(1024 * 1024L, dao.get(attachmentId)!!.transferredBytes)
        assertEquals(listOf("attachment.begin"), commands.map { it.type })
    }

    @Test
    fun `paused large upload does not block a later eligible upload`() = runBlocking {
        val blockedId = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        val eligibleId = "01ARZ3NDEKTSV4RRFFQ69G5FAW"
        val dao = FakeAttachmentDao(
            transfer(blockedId, 0, "pending"),
            transfer(eligibleId, 0, "pending").copy(updatedAt = 2),
        )
        val source = temporary.newFile("eligible.upload").apply {
            writeBytes(ByteArray(1024 * 1024 + 3))
        }
        val commands = mutableListOf<SentCommand>()
        val coordinator = coordinator(dao, source, commands, mutableListOf()) {
            it.attachmentId == eligibleId
        }

        coordinator.onConnectionReady("server")

        assertEquals(eligibleId, commands.single().payload["attachment_id"]?.toString()?.trim('"'))
        assertEquals("pending", dao.get(blockedId)!!.state)
        assertEquals("uploading", dao.get(eligibleId)!!.state)
    }

    @Test
    fun `pausing active upload immediately schedules the next eligible upload`() = runBlocking {
        val pausedId = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        val eligibleId = "01ARZ3NDEKTSV4RRFFQ69G5FAW"
        val dao = FakeAttachmentDao(
            transfer(pausedId, 0, "pending"),
            transfer(eligibleId, 0, "pending").copy(updatedAt = 2),
        )
        val source = temporary.newFile("queue.upload").apply {
            writeBytes(ByteArray(1024 * 1024 + 3))
        }
        val commands = mutableListOf<SentCommand>()
        var pauseFirst = false
        val coordinator = coordinator(dao, source, commands, mutableListOf()) {
            it.attachmentId != pausedId || !pauseFirst
        }

        coordinator.onConnectionReady("server")
        coordinator.onReply(beginReply(commands.single().id, pausedId, nextOffset = 0))
        pauseFirst = true
        coordinator.onProgress(
            AttachmentProgressPayload(pausedId, 1024 * 1024L, 1024 * 1024L + 3),
        )

        assertEquals("pending", dao.get(pausedId)!!.state)
        assertEquals(1024 * 1024L, dao.get(pausedId)!!.transferredBytes)
        assertEquals(eligibleId, commands.last().payload["attachment_id"]?.toString()?.trim('"'))
        assertEquals("uploading", dao.get(eligibleId)!!.state)
    }

    private fun coordinator(
        dao: FakeAttachmentDao,
        source: File,
        commands: MutableList<SentCommand>,
        chunks: MutableList<ByteString>,
        canTransfer: (AttachmentTransferEntity) -> Boolean = { true },
    ) = AttachmentUploadCoordinator(
        dao = dao,
        sourceFile = { source },
        sendCommand = { type, id, sessionId, payload ->
            commands += SentCommand(type, id, sessionId, payload)
            true
        },
        sendBinary = { chunks += it; true },
        onTransportUnavailable = { error(it) },
        onUploadFailed = { error(it) },
        canTransfer = canTransfer,
    )

    private fun beginReply(id: String, attachmentId: String, nextOffset: Long) = WireEnvelope(
        v = 1,
        kind = WireKind.REPLY,
        type = "attachment.begin.ok",
        id = id,
        connectionEpoch = 1,
        sessionId = "mobile:test",
        payload = buildJsonObject {
            put("attachment_id", attachmentId)
            put("filename", "image.png")
            put("content_type", "image/png")
            put("size_bytes", 1024 * 1024 + 3)
            put("sha256", "a".repeat(64))
            put("next_offset", nextOffset)
            put("chunk_size", 128 * 1024)
            put("state", "transferring")
        },
    )

    private fun finishReply(id: String, attachmentId: String) = WireEnvelope(
        v = 1,
        kind = WireKind.REPLY,
        type = "attachment.finish.ok",
        id = id,
        connectionEpoch = 1,
        sessionId = "mobile:test",
        payload = buildJsonObject { put("attachment_id", attachmentId); put("state", "ready") },
    )

    private fun transfer(id: String, offset: Long, state: String) = AttachmentTransferEntity(
        attachmentId = id,
        serverId = "server",
        sessionId = "mobile:test",
        filename = "image.png",
        contentType = "image/png",
        sizeBytes = 1024 * 1024L + 3,
        sha256 = "a".repeat(64),
        transferredBytes = offset,
        state = state,
        updatedAt = 1,
    )

    private data class SentCommand(
        val type: String,
        val id: String,
        val sessionId: String,
        val payload: JsonObject,
    )

    private class FakeAttachmentDao(vararg initial: AttachmentTransferEntity) : AttachmentTransferDao {
        private val values = initial.associateByTo(linkedMapOf()) { it.attachmentId }

        override suspend fun upsert(transfer: AttachmentTransferEntity) {
            values[transfer.attachmentId] = transfer
        }

        override suspend fun upsertAll(transfers: List<AttachmentTransferEntity>) {
            transfers.forEach { values[it.attachmentId] = it }
        }

        override suspend fun get(attachmentId: String): AttachmentTransferEntity? =
            values[attachmentId]

        override suspend fun all(): List<AttachmentTransferEntity> = values.values.toList()

        override fun observeDrafts(serverId: String, sessionId: String): Flow<List<AttachmentTransferEntity>> =
            flowOf(values.values.filter { it.serverId == serverId && it.sessionId == sessionId })

        override suspend fun drafts(serverId: String, sessionId: String): List<AttachmentTransferEntity> =
            values.values.filter { it.serverId == serverId && it.sessionId == sessionId }

        override suspend fun pendingUploads(serverId: String): List<AttachmentTransferEntity> =
            values.values.filter {
                it.serverId == serverId && it.state in setOf("pending", "uploading", "finishing")
            }.sortedBy { it.updatedAt }

        override fun observeActiveUploads(serverId: String): Flow<List<AttachmentTransferEntity>> =
            flowOf(values.values.filter {
                it.serverId == serverId && it.state in setOf("pending", "uploading", "finishing")
            })

        override suspend fun claimUploading(attachmentId: String, updatedAt: Long): Int {
            val value = values[attachmentId] ?: return 0
            if (value.state !in setOf("pending", "uploading", "finishing")) return 0
            values[attachmentId] = value.copy(state = "uploading", updatedAt = updatedAt)
            return 1
        }

        override suspend fun updateState(
            attachmentId: String,
            transferredBytes: Long,
            state: String,
            updatedAt: Long,
        ): Int {
            val value = values[attachmentId] ?: return 0
            values[attachmentId] = value.copy(
                transferredBytes = transferredBytes,
                state = state,
                updatedAt = updatedAt,
            )
            return 1
        }

        override suspend fun retryFailed(attachmentId: String, updatedAt: Long): Int {
            val value = values[attachmentId] ?: return 0
            if (value.state != "failed") return 0
            values[attachmentId] = value.copy(state = "pending", updatedAt = updatedAt)
            return 1
        }

        override suspend fun markSent(ids: List<String>, updatedAt: Long): Int = updateMany(ids, "sent", updatedAt)

        override suspend fun markSending(ids: List<String>, updatedAt: Long): Int =
            updateMany(ids, "sending", updatedAt)

        override suspend fun restoreReady(ids: List<String>, updatedAt: Long): Int =
            updateMany(ids, "ready", updatedAt)

        override suspend fun deleteDraft(attachmentId: String): Int {
            val value = values[attachmentId] ?: return 0
            if (value.state !in setOf("pending", "ready", "failed")) return 0
            values.remove(attachmentId)
            return 1
        }

        override suspend fun deleteSent(): Int = 0

        override suspend fun deleteSentForSession(serverId: String, sessionId: String): Int = 0

        private fun updateMany(ids: List<String>, state: String, updatedAt: Long): Int {
            var changed = 0
            ids.forEach { id ->
                val value = values[id] ?: return@forEach
                values[id] = value.copy(state = state, updatedAt = updatedAt)
                changed += 1
            }
            return changed
        }
    }
}
