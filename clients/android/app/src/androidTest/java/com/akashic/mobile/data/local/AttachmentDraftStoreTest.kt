package com.akashic.mobile.data.local

import android.content.Context
import androidx.core.content.FileProvider
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.akashic.mobile.BuildConfig
import com.akashic.mobile.data.realtime.AttachmentOperationOwner
import com.akashic.mobile.data.realtime.attachmentDraftMatchesExpected
import java.io.File
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.async
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
    private lateinit var context: Context
    private lateinit var database: AppDatabase
    private lateinit var root: File
    private lateinit var store: AttachmentDraftStore

    @Before
    fun setUp() = runBlocking {
        context = ApplicationProvider.getApplicationContext()
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

    @Test
    fun sendWinningOwnershipPreventsReadyDraftRemoval() = runBlocking {
        val transfer = transfer("send-wins", "ready")
        database.attachmentTransfers().upsert(transfer)
        store.fileFor(transfer.attachmentId).apply {
            parentFile!!.mkdirs()
            writeText("payload")
        }
        val owner = AttachmentOperationOwner()
        val sendEntered = CompletableDeferred<Unit>()
        val releaseSend = CompletableDeferred<Unit>()
        val send = async {
            owner.perform {
                val drafts = database.attachmentTransfers().drafts("server", "mobile:test")
                assertTrue(attachmentDraftMatchesExpected(drafts.map { it.attachmentId }, listOf(transfer.attachmentId)))
                sendEntered.complete(Unit)
                releaseSend.await()
                assertEquals(1, database.attachmentTransfers().markSending(listOf(transfer.attachmentId), 2))
            }
        }
        sendEntered.await()
        val remove = async { owner.perform { store.remove(transfer.attachmentId) } }

        releaseSend.complete(Unit)
        send.await()

        assertFalse(remove.await())
        assertEquals("sending", database.attachmentTransfers().get(transfer.attachmentId)?.state)
        assertTrue(store.fileFor(transfer.attachmentId).exists())
    }

    @Test
    fun removeWinningOwnershipMakesSendRejectTheStaleDraftSet() = runBlocking {
        val transfer = transfer("remove-wins", "ready")
        database.attachmentTransfers().upsert(transfer)
        store.fileFor(transfer.attachmentId).apply {
            parentFile!!.mkdirs()
            writeText("payload")
        }
        val owner = AttachmentOperationOwner()
        val removeEntered = CompletableDeferred<Unit>()
        val releaseRemove = CompletableDeferred<Unit>()
        val remove = async {
            owner.perform {
                removeEntered.complete(Unit)
                releaseRemove.await()
                store.remove(transfer.attachmentId)
            }
        }
        removeEntered.await()
        val sendAccepted = async {
            owner.perform {
                val actual = database.attachmentTransfers().drafts("server", "mobile:test")
                attachmentDraftMatchesExpected(actual.map { it.attachmentId }, listOf(transfer.attachmentId))
            }
        }

        releaseRemove.complete(Unit)

        assertTrue(remove.await())
        assertFalse(sendAccepted.await())
        assertNull(database.attachmentTransfers().get(transfer.attachmentId))
        assertFalse(store.fileFor(transfer.attachmentId).exists())
    }

    @Test
    fun stableShareIdsMakeImportIdempotentAcrossRetries() = runBlocking {
        val source = context.cacheDir.resolve("shared-text/idempotent-share.txt").apply {
            parentFile!!.mkdirs()
            writeText("stable payload")
        }
        val uri = FileProvider.getUriForFile(
            context,
            "${BuildConfig.APPLICATION_ID}.files",
            source,
        )

        val first = store.import(
            serverId = "server",
            sessionId = "mobile:test",
            uris = listOf(uri),
            now = 2,
            attachmentIds = listOf("01STABLESHAREATTACHMENT00"),
        )
        val second = store.import(
            serverId = "server",
            sessionId = "mobile:test",
            uris = listOf(uri),
            now = 3,
            attachmentIds = listOf("01STABLESHAREATTACHMENT00"),
        )

        assertEquals(first, second)
        assertEquals(1, database.attachmentTransfers().all().size)
        source.delete()
        Unit
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
