package com.akashic.mobile.data.local

import java.io.File
import java.security.MessageDigest
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class MediaCacheStoreTest {
    @get:Rule
    val temporary = TemporaryFolder()

    @Test
    fun `reconcile commits valid final left before database update`() = runBlocking {
        val dao = FakeMediaDao()
        val cache = MediaCacheStore(temporary.root, dao)
        val content = byteArrayOf(1, 2, 3)
        val transfer = transfer(cache, "01ARZ3NDEKTSV4RRFFQ69G5FAV", content, "downloading")
        dao.values += transfer
        File(transfer.cachePath).writeBytes(content)

        cache.reconcile()

        assertEquals("cached", dao.get(transfer.attachmentId)!!.state)
        assertEquals(content.size.toLong(), dao.get(transfer.attachmentId)!!.transferredBytes)
    }

    @Test
    fun `reconcile publishes fully fsynced part left before reply`() = runBlocking {
        val dao = FakeMediaDao()
        val cache = MediaCacheStore(temporary.root, dao)
        val content = byteArrayOf(4, 5, 6)
        val transfer = transfer(cache, "01ARZ3NDEKTSV4RRFFQ69G5FAV", content, "downloading")
        dao.values += transfer.copy(transferredBytes = content.size.toLong())
        File("${transfer.cachePath}.part").writeBytes(content)

        cache.reconcile()

        assertEquals("cached", dao.get(transfer.attachmentId)!!.state)
        assertTrue(File(transfer.cachePath).exists())
        assertFalse(File("${transfer.cachePath}.part").exists())
    }

    @Test
    fun `reconcile removes orphan and preserves evicted state`() = runBlocking {
        val dao = FakeMediaDao()
        val cache = MediaCacheStore(temporary.root, dao)
        val transfer = transfer(cache, "01ARZ3NDEKTSV4RRFFQ69G5FAV", byteArrayOf(1), "evicted")
        dao.values += transfer
        val stalePart = File("${transfer.cachePath}.part").apply { writeBytes(byteArrayOf(1)) }
        val orphan = temporary.root.resolve("orphan.bin").apply { writeBytes(byteArrayOf(2)) }

        cache.reconcile()

        assertEquals("evicted", dao.get(transfer.attachmentId)!!.state)
        assertFalse(stalePart.exists())
        assertFalse(orphan.exists())
    }

    @Test
    fun `quota evicts least recently used without scheduling automatic redownload`() = runBlocking {
        val dao = FakeMediaDao()
        val cache = MediaCacheStore(temporary.root, dao, quotaBytes = 5)
        val old = transfer(cache, "01ARZ3NDEKTSV4RRFFQ69G5FAV", byteArrayOf(1, 1, 1, 1), "cached", 1)
        val recent = transfer(cache, "01ARZ3NDEKTSV4RRFFQ69G5FAW", byteArrayOf(2, 2, 2, 2), "cached", 2)
        dao.values += listOf(old, recent)
        File(old.cachePath).writeBytes(byteArrayOf(1, 1, 1, 1))
        File(recent.cachePath).writeBytes(byteArrayOf(2, 2, 2, 2))

        cache.enforceQuota()

        assertEquals("evicted", dao.get(old.attachmentId)!!.state)
        assertEquals("cached", dao.get(recent.attachmentId)!!.state)
        assertTrue(dao.pendingDownloads("server").isEmpty())
    }

    @Test
    fun `outbound draft is atomically copied into received cache`() = runBlocking {
        val dao = FakeMediaDao()
        val cache = MediaCacheStore(temporary.root.resolve("received"), dao)
        val content = byteArrayOf(7, 8, 9)
        val source = temporary.root.resolve("draft.upload").apply { writeBytes(content) }
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(content)
            .joinToString("") { "%02x".format(it) }
        val draft = AttachmentTransferEntity(
            attachmentId = "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            serverId = "server",
            sessionId = "mobile:test",
            filename = "photo.png",
            contentType = "image/png",
            sizeBytes = content.size.toLong(),
            sha256 = digest,
            transferredBytes = content.size.toLong(),
            state = "ready",
            updatedAt = 1,
        )

        val cached = cache.importOutbound(draft, source, updatedAt = 2)

        assertEquals("cached", cached.state)
        assertEquals(content.toList(), File(cached.cachePath).readBytes().toList())
        assertTrue(source.exists())
        assertFalse(temporary.root.resolve("received").listFiles()!!.any { it.extension == "import" })
    }

    private fun transfer(
        cache: MediaCacheStore,
        id: String,
        content: ByteArray,
        state: String,
        accessedAt: Long = 1,
    ) = MediaAttachmentEntity(
        attachmentId = id,
        serverId = "server",
        sessionId = "mobile:test",
        filename = "file.bin",
        contentType = "application/octet-stream",
        sizeBytes = content.size.toLong(),
        sha256 = MessageDigest.getInstance("SHA-256").digest(content).joinToString("") { "%02x".format(it) },
        transferredBytes = if (state == "cached") content.size.toLong() else 0,
        state = state,
        cachePath = cache.cachePath(id),
        lastAccessedAt = accessedAt,
        updatedAt = accessedAt,
    )

    private class FakeMediaDao : MediaAttachmentDao {
        val values = mutableListOf<MediaAttachmentEntity>()

        override suspend fun upsert(attachment: MediaAttachmentEntity) = replace(attachment)
        override suspend fun upsertAll(attachments: List<MediaAttachmentEntity>) = attachments.forEach(::replace)
        override suspend fun linkAll(links: List<MessageAttachmentEntity>): List<Long> = links.map { 1L }
        override suspend fun deleteLinks(messageId: String): Int = 0
        override suspend fun moveLinks(sourceId: String, targetId: String): Int = 0
        override suspend fun get(attachmentId: String) = values.firstOrNull { it.attachmentId == attachmentId }
        override suspend fun pendingDownloads(serverId: String) =
            values.filter { it.serverId == serverId && it.state in setOf("pending", "downloading") }

        override suspend fun updateDownload(
            attachmentId: String,
            transferredBytes: Long,
            state: String,
            updatedAt: Long,
        ) = update(attachmentId) { it.copy(transferredBytes = transferredBytes, state = state, updatedAt = updatedAt) }

        override suspend fun requestDownload(attachmentId: String, updatedAt: Long) =
            updateIf(attachmentId, { it.state in setOf("failed", "evicted") }) {
                it.copy(state = "pending", updatedAt = updatedAt)
            }

        override suspend fun touch(attachmentId: String, accessedAt: Long) =
            updateIf(attachmentId, { it.state == "cached" }) { it.copy(lastAccessedAt = accessedAt) }

        override suspend fun markCached(attachmentId: String, updatedAt: Long) = update(attachmentId) {
            it.copy(
                transferredBytes = it.sizeBytes,
                state = "cached",
                lastAccessedAt = updatedAt,
                updatedAt = updatedAt,
            )
        }

        override suspend fun markEvicted(attachmentId: String, updatedAt: Long) = update(attachmentId) {
            it.copy(transferredBytes = 0, state = "evicted", updatedAt = updatedAt)
        }

        override suspend fun getAll(ids: List<String>) = values.filter { it.attachmentId in ids }
        override suspend fun all() = values.toList()
        override suspend fun unreferenced() = emptyList<MediaAttachmentEntity>()
        override suspend fun delete(attachmentId: String): Int = if (values.removeIf { it.attachmentId == attachmentId }) 1 else 0
        override suspend fun forMessage(messageId: String) = emptyList<MediaAttachmentEntity>()

        private fun replace(value: MediaAttachmentEntity) {
            values.removeIf { it.attachmentId == value.attachmentId }
            values += value
        }

        private fun update(id: String, transform: (MediaAttachmentEntity) -> MediaAttachmentEntity): Int =
            updateIf(id, { true }, transform)

        private fun updateIf(
            id: String,
            predicate: (MediaAttachmentEntity) -> Boolean,
            transform: (MediaAttachmentEntity) -> MediaAttachmentEntity,
        ): Int {
            val index = values.indexOfFirst { it.attachmentId == id && predicate(it) }
            if (index < 0) return 0
            values[index] = transform(values[index])
            return 1
        }
    }
}
