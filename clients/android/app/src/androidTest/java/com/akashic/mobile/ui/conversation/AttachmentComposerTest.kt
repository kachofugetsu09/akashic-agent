package com.akashic.mobile.ui.conversation

import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.assertRangeInfoEquals
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.performClick
import androidx.compose.ui.semantics.ProgressBarRangeInfo
import com.akashic.mobile.ui.design.AkashicTheme
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test

class AttachmentComposerTest {
    @get:Rule
    val compose = createComposeRule()

    @Test
    fun uploadingAttachmentExposesProgressAndBlocksPureAttachmentSend() {
        compose.setContent {
            AkashicTheme {
                ConversationScreen(
                    state = state(ComposerAttachmentState.UPLOADING, 42),
                    onAttach = {},
                    onRemoveAttachment = {},
                    onRetryAttachment = {},
                    onRetryDownloadedAttachment = {},
                    onOpenDownloadedAttachment = {},
                    onDismissError = {},
                    onSend = {},
                    onStop = {},
                )
            }
        }

        compose.onNodeWithTag("attachment-draft-file").assertRangeInfoEquals(
            ProgressBarRangeInfo(0.42f, 0f..1f),
        )
        compose.onNodeWithTag("composer-send-stop").assertIsNotEnabled()
    }

    @Test
    fun readyAttachmentAllowsPureAttachmentSend() {
        var sends = 0
        compose.setContent {
            AkashicTheme {
                ConversationScreen(
                    state = state(ComposerAttachmentState.READY, 100),
                    onAttach = {},
                    onRemoveAttachment = {},
                    onRetryAttachment = {},
                    onRetryDownloadedAttachment = {},
                    onOpenDownloadedAttachment = {},
                    onDismissError = {},
                    onSend = { sends += 1 },
                    onStop = {},
                )
            }
        }

        compose.onNodeWithTag("composer-send-stop").assertIsEnabled().performClick()
        assertEquals(1, sends)
    }

    @Test
    fun pendingAttachmentCanBeRemoved() {
        compose.setContent {
            AkashicTheme {
                ConversationScreen(
                    state = state(ComposerAttachmentState.WAITING_FOR_CONNECTION, 0),
                    onAttach = {},
                    onRemoveAttachment = {},
                    onRetryAttachment = {},
                    onRetryDownloadedAttachment = {},
                    onOpenDownloadedAttachment = {},
                    onDismissError = {},
                    onSend = {},
                    onStop = {},
                )
            }
        }
        compose.onNodeWithContentDescription("移除附件 报告.pdf").assertExists()
    }

    @Test
    fun activeUploadCannotBeRemoved() {
        compose.setContent {
            AkashicTheme {
                ConversationScreen(
                    state = state(ComposerAttachmentState.UPLOADING, 42),
                    onAttach = {},
                    onRemoveAttachment = {},
                    onRetryAttachment = {},
                    onRetryDownloadedAttachment = {},
                    onOpenDownloadedAttachment = {},
                    onDismissError = {},
                    onSend = {},
                    onStop = {},
                )
            }
        }
        compose.onNodeWithContentDescription("移除附件 报告.pdf").assertDoesNotExist()
    }

    private fun state(attachmentState: ComposerAttachmentState, progress: Int) =
        EmptyConversationState.copy(
            canSend = true,
            attachments = listOf(
                ComposerAttachmentUi(
                    id = "file",
                    filename = "报告.pdf",
                    contentType = "application/pdf",
                    sizeBytes = 100,
                    transferredBytes = progress.toLong(),
                    state = attachmentState,
                    canRemove = attachmentState in setOf(
                        ComposerAttachmentState.WAITING_FOR_CONNECTION,
                        ComposerAttachmentState.READY,
                        ComposerAttachmentState.FAILED,
                    ),
                ),
            ),
        )
}
