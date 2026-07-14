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

    private fun coordinator(
        dao: FakeAttachmentDao,
        source: File,
        commands: MutableList<SentCommand>,
        chunks: MutableList<ByteString>,
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

    private class FakeAttachmentDao(initial: AttachmentTransferEntity) : AttachmentTransferDao {
        private var value = initial

        override suspend fun upsert(transfer: AttachmentTransferEntity) {
            value = transfer
        }

        override suspend fun upsertAll(transfers: List<AttachmentTransferEntity>) {
            require(transfers.size == 1)
            value = transfers.single()
        }

        override suspend fun get(attachmentId: String): AttachmentTransferEntity? =
            value.takeIf { it.attachmentId == attachmentId }

        override suspend fun all(): List<AttachmentTransferEntity> = listOf(value)

        override fun observeDrafts(serverId: String, sessionId: String): Flow<List<AttachmentTransferEntity>> =
            flowOf(listOf(value))

        override suspend fun drafts(serverId: String, sessionId: String): List<AttachmentTransferEntity> =
            listOf(value)

        override suspend fun pendingUploads(serverId: String): List<AttachmentTransferEntity> =
            listOf(value).filter { it.state in setOf("pending", "uploading", "finishing") }

        override suspend fun claimUploading(attachmentId: String, updatedAt: Long): Int {
            if (value.attachmentId != attachmentId || value.state !in setOf("pending", "uploading", "finishing")) return 0
            value = value.copy(state = "uploading", updatedAt = updatedAt)
            return 1
        }

        override suspend fun updateState(
            attachmentId: String,
            transferredBytes: Long,
            state: String,
            updatedAt: Long,
        ): Int {
            if (value.attachmentId != attachmentId) return 0
            value = value.copy(
                transferredBytes = transferredBytes,
                state = state,
                updatedAt = updatedAt,
            )
            return 1
        }

        override suspend fun markSent(ids: List<String>, updatedAt: Long): Int = updateMany(ids, "sent", updatedAt)

        override suspend fun markSending(ids: List<String>, updatedAt: Long): Int =
            updateMany(ids, "sending", updatedAt)

        override suspend fun restoreReady(ids: List<String>, updatedAt: Long): Int =
            updateMany(ids, "ready", updatedAt)

        override suspend fun deleteDraft(attachmentId: String): Int = 0

        override suspend fun deleteSent(): Int = 0

        private fun updateMany(ids: List<String>, state: String, updatedAt: Long): Int {
            if (value.attachmentId !in ids) return 0
            value = value.copy(state = state, updatedAt = updatedAt)
            return 1
        }
    }
}
