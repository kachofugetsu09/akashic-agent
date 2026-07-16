package com.akashic.mobile.data.local

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class AttachmentDraftStoreTest {
    private lateinit var database: AppDatabase
    private lateinit var root: File
    private lateinit var store: AttachmentDraftStore

    @Before
    fun setUp() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<Context>()
        database = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java).build()
        root = context.cacheDir.resolve("attachment-drafts-${System.nanoTime()}")
        store = AttachmentDraftStore(context.contentResolver, root, database.attachmentTransfers())
        database.serverProfiles().upsert(
            ServerProfileEntity("server", "server", "device", "alias", "pin", "[]", "[]", "[]", 1),
        )
        database.conversations().upsert(ConversationEntity("mobile:test", "server", "test", 1))
    }

    @After
    fun tearDown() {
        database.close()
        root.deleteRecursively()
    }

    @Test
    fun removeDeletesRecordAndFileExactlyOnce() = runBlocking {
        val transfer = transfer("remove", "pending")
        database.attachmentTransfers().upsert(transfer)
        val file = store.fileFor(transfer.attachmentId).apply {
            parentFile!!.mkdirs()
            writeText("payload")
        }

        assertTrue(store.remove(transfer.attachmentId))
        assertFalse(store.remove(transfer.attachmentId))
        assertNull(database.attachmentTransfers().get(transfer.attachmentId))
        assertFalse(file.exists())
    }

    @Test
    fun retryOwnsFailedToPendingTransitionExactlyOnce() = runBlocking {
        val transfer = transfer("retry", "failed")
        database.attachmentTransfers().upsert(transfer)

        assertTrue(store.retry(transfer.attachmentId, 2))
        assertFalse(store.retry(transfer.attachmentId, 3))
        val persisted = database.attachmentTransfers().get(transfer.attachmentId)!!
        assertEquals("pending", persisted.state)
        assertEquals(2, persisted.updatedAt)
    }

    private fun transfer(id: String, state: String) = AttachmentTransferEntity(
        attachmentId = id,
        serverId = "server",
        sessionId = "mobile:test",
        filename = "$id.png",
        contentType = "image/png",
        sizeBytes = 7,
        sha256 = "a".repeat(64),
        transferredBytes = 0,
        state = state,
        updatedAt = 1,
    )
}
