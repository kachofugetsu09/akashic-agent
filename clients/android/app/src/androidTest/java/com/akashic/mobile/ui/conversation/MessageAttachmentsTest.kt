package com.akashic.mobile.ui.conversation

import androidx.compose.ui.test.assertRangeInfoEquals
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onAllNodesWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.semantics.ProgressBarRangeInfo
import com.akashic.mobile.ui.design.AkashicTheme
import android.content.Context
import androidx.test.core.app.ApplicationProvider
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test

class MessageAttachmentsTest {
    @get:Rule
    val compose = createComposeRule()

    @Test
    fun downloadProgressAndFailureRetryStayOnOneListPlane() {
        var retried: String? = null
        compose.setContent {
            AkashicTheme {
                MessageAttachments(
                    attachments = listOf(
                        attachment("active", MessageAttachmentState.DOWNLOADING, 42),
                        attachment("failed", MessageAttachmentState.FAILED, 0),
                    ),
                    onRetry = { retried = it },
                    onOpen = {},
                )
            }
        }

        compose.onNodeWithTag("message-attachment-active").assertRangeInfoEquals(
            ProgressBarRangeInfo(0.42f, 0f..1f),
        )
        compose.onNodeWithContentDescription("重试下载 failed.pdf").performClick()
        assertEquals("failed", retried)
    }

    @Test
    fun cachedImageUsesInlinePreviewInsteadOfFileRow() {
        compose.setContent {
            AkashicTheme {
                MessageAttachments(
                    attachments = listOf(
                        attachment(
                            id = "image",
                            state = MessageAttachmentState.CACHED,
                            transferredBytes = 100,
                            contentType = "image/gif",
                        ),
                    ),
                    onRetry = {},
                    onOpen = {},
                )
            }
        }

        compose.onNodeWithTag("message-image-image").assertExists()
        compose.onNodeWithTag("message-attachment-image").assertDoesNotExist()
    }

    @Test
    fun evictedAttachmentOffersRealDownloadRetry() {
        var retried: String? = null
        compose.setContent {
            AkashicTheme {
                MessageAttachments(
                    attachments = listOf(
                        attachment("evicted", MessageAttachmentState.EVICTED, 100),
                    ),
                    onRetry = { retried = it },
                    onOpen = {},
                )
            }
        }

        compose.onNodeWithContentDescription("重试下载 evicted.pdf").performClick()
        assertEquals("evicted", retried)
    }

    @Test
    fun invalidCachedImageFallsBackToOpenableFileRow() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val root = File(context.filesDir, "received-attachments").apply { mkdirs() }
        val invalidGif = File(root, "invalid.gif").apply { writeBytes("not-a-gif".encodeToByteArray()) }
        compose.setContent {
            AkashicTheme {
                MessageAttachments(
                    attachments = listOf(
                        attachment(
                            id = "invalid",
                            state = MessageAttachmentState.CACHED,
                            transferredBytes = 100,
                            contentType = "image/gif",
                            cachePath = invalidGif.path,
                        ),
                    ),
                    onRetry = {},
                    onOpen = {},
                )
            }
        }

        compose.waitUntil(timeoutMillis = 5_000) {
            compose.onAllNodesWithTag("message-attachment-invalid")
                .fetchSemanticsNodes().isNotEmpty()
        }
        compose.onNodeWithContentDescription("打开文件 invalid.gif").assertExists()
        invalidGif.delete()
    }

    private fun attachment(
        id: String,
        state: MessageAttachmentState,
        transferredBytes: Long,
        contentType: String = "application/pdf",
        cachePath: String = "/cache/$id",
    ) = MessageAttachmentUi(
        id = id,
        filename = "$id.${if (contentType == "image/gif") "gif" else "pdf"}",
        contentType = contentType,
        sizeBytes = 100,
        transferredBytes = transferredBytes,
        state = state,
        cachePath = cachePath,
    )
}
