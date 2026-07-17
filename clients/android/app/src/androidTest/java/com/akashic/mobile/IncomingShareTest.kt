package com.akashic.mobile

import android.content.ClipData
import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.akashic.mobile.data.local.IncomingShareStore
import com.akashic.mobile.data.local.fallbackAttachmentFilename
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class IncomingShareTest {
    @Test
    fun combinesSubjectWithSharedUrl() {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_SUBJECT, "页面标题")
            putExtra(Intent.EXTRA_TEXT, "https://example.com/page")
        }

        val share = requireNotNull(parseIncomingShare(intent))

        assertEquals("页面标题\nhttps://example.com/page", share.text)
        assertEquals(emptyList<Uri>(), share.uris)
    }

    @Test
    fun acceptsSubjectOnlyAndRejectsEmptyOrOversizedText() {
        assertEquals("标题", normalizeSharedText(text = "  ", subject = " 标题 "))
        assertThrows(IllegalArgumentException::class.java) {
            normalizeSharedText("x".repeat(65_537), null)
        }
        assertThrows(IllegalArgumentException::class.java) {
            parseIncomingShare(Intent(Intent.ACTION_SEND).apply { type = "text/plain" })
        }
    }

    @Test
    fun mergesAndDeduplicatesGrantedContentUris() {
        val first = Uri.parse("content://fixture/first")
        val second = Uri.parse("content://fixture/second")
        val intent = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
            type = "image/*"
            clipData = ClipData.newRawUri("shared files", first).apply {
                addItem(ClipData.Item(second))
            }
            putParcelableArrayListExtra(Intent.EXTRA_STREAM, arrayListOf(first, second))
        }

        val share = requireNotNull(parseIncomingShare(intent))

        assertEquals(listOf(first, second), share.uris)
    }

    @Test
    fun rejectsUntrustedFileUrisAtTheIntentBoundary() {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "application/octet-stream"
            putExtra(Intent.EXTRA_STREAM, Uri.parse("file:///sdcard/private.txt"))
        }

        assertThrows(IllegalArgumentException::class.java) { parseIncomingShare(intent) }
    }

    @Test
    fun ignoresShareIntentRestoredFromActivityHistory() {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            flags = Intent.FLAG_ACTIVITY_LAUNCHED_FROM_HISTORY
            putExtra(Intent.EXTRA_TEXT, "不应再次导入")
        }

        assertNull(parseIncomingShare(intent))
    }

    @Test
    fun reparsingTheSameIntentKeepsItsDurableReceiveIdentity() {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_TEXT, "同一份分享")
        }

        assertEquals(parseIncomingShare(intent)?.id, parseIncomingShare(intent)?.id)
    }

    @Test
    fun rejectsForgedReceiveIdentityAtTheIntentBoundary() {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_TEXT, "恶意分享")
            putExtra("com.akashic.mobile.extra.INCOMING_SHARE_ID", "../../escape")
        }

        assertThrows(IllegalArgumentException::class.java) { parseIncomingShare(intent) }
    }

    @Test
    fun fallbackFilenameUsesMimeExtensionWithoutTrustingTheUriPath() {
        assertEquals("shared-01TEST.png", fallbackAttachmentFilename("01TEST", "image/png"))
        assertEquals(
            "shared-01UNKNOWN.bin",
            fallbackAttachmentFilename("01UNKNOWN", "application/octet-stream"),
        )
    }

    @Test
    fun incomingShareSurvivesStoreRecreationUntilEachPartIsConsumed() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val root = File(context.filesDir, "incoming-shares")
        root.deleteRecursively()
        context.getSharedPreferences("incoming_shares", android.content.Context.MODE_PRIVATE)
            .edit().clear().commit()
        val source = File(context.filesDir, "received-attachments/share-store-source.txt")
        source.parentFile?.mkdirs()
        source.writeText("durable share fixture")
        val uri = FileProvider.getUriForFile(
            context,
            "${BuildConfig.APPLICATION_ID}.files",
            source,
        )
        val store = IncomingShareStore(context, root)

        val shareId = "8a525db5-bca7-481b-a8c8-69f585c1568a"
        val accepted = store.enqueue(IncomingShare(shareId, "共享文字", listOf(uri)))
        val repeated = store.enqueue(IncomingShare(shareId, "共享文字", listOf(uri)))
        val replayed = store.enqueue(
            IncomingShare("e7338075-bc1e-4134-a62e-378d57442d36", "共享文字", listOf(uri)),
        )
        assertEquals(accepted.content.attachmentIds, repeated.content.attachmentIds)
        assertEquals(shareId, replayed.content.id)
        assertEquals(1, store.load().size)
        store.claimTarget(shareId, "mobile:durable-target")

        val restored = IncomingShareStore(context, root).load().single()
        assertEquals("共享文字", restored.content.text)
        assertEquals("mobile:durable-target", restored.targetSessionId)
        val attachmentIds = requireNotNull(restored.content.attachmentIds)
        assertEquals(1, attachmentIds.size)
        assertEquals("durable share fixture", context.contentResolver.openInputStream(
            restored.content.uris.single(),
        )?.bufferedReader()?.use { it.readText() })

        store.prepareText(
            shareId,
            "已有草稿\n共享文字",
            "reply-message",
            "已有草稿",
            null,
            41L,
        )
        val prepared = IncomingShareStore(context, root).load().single()
        assertNull(prepared.content.text)
        assertEquals("已有草稿\n共享文字", prepared.preparedText)
        assertEquals("reply-message", prepared.preparedReplyToMessageId)
        assertEquals("已有草稿", prepared.preparedBaseText)
        assertEquals(41L, prepared.preparedBaseUpdatedAt)
        assertEquals(attachmentIds, prepared.content.attachmentIds)

        store.consumeText(shareId)
        assertNull(store.load().single().content.text)
        store.consumeFiles(shareId)
        assertEquals(emptyList<Any>(), store.load())
        source.delete()
        Unit
    }
}
