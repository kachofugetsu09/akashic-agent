package com.akashic.mobile

import com.akashic.mobile.data.local.MediaAttachmentEntity
import com.akashic.mobile.data.local.MessageAttachmentEntity
import com.akashic.mobile.data.local.MessageAttachmentWithMedia
import com.akashic.mobile.ui.conversation.MessageAttachmentState
import com.akashic.mobile.ui.conversation.MessageAttachmentUi
import com.akashic.mobile.ui.conversation.progress
import com.akashic.mobile.ui.conversation.stateLabel
import org.junit.Assert.assertEquals
import org.junit.Test

class MessageAttachmentPresentationTest {
    @Test
    fun messageAttachmentsKeepServerOrderAndMapDownloadState() {
        val mapped = listOf(
            relation(id = "second", ordinal = 1, state = "cached"),
            relation(id = "first", ordinal = 0, state = "downloading", transferredBytes = 25),
            relation(id = "evicted", ordinal = 2, state = "evicted"),
        ).toMessageAttachmentUi()

        assertEquals(listOf("first", "second", "evicted"), mapped.map { it.id })
        assertEquals(MessageAttachmentState.DOWNLOADING, mapped.first().state)
        assertEquals(MessageAttachmentState.CACHED, mapped[1].state)
        assertEquals(MessageAttachmentState.EVICTED, mapped.last().state)
        assertEquals("/cache/first", mapped.first().cachePath)
    }

    @Test
    fun progressAndLabelUseStableTabularPercentage() {
        val attachment = MessageAttachmentUi(
            id = "report",
            filename = "report.pdf",
            contentType = "application/pdf",
            sizeBytes = 200,
            transferredBytes = 85,
            state = MessageAttachmentState.DOWNLOADING,
            cachePath = "/cache/report",
        )

        assertEquals(0.425f, attachment.progress)
        assertEquals("下载中 43%", attachment.stateLabel())
    }

    private fun relation(
        id: String,
        ordinal: Int,
        state: String,
        transferredBytes: Long = 100,
    ) = MessageAttachmentWithMedia(
        link = MessageAttachmentEntity(
            messageId = "message",
            attachmentId = id,
            ordinal = ordinal,
        ),
        attachment = MediaAttachmentEntity(
            attachmentId = id,
            serverId = "server",
            sessionId = "mobile:test",
            filename = "$id.pdf",
            contentType = "application/pdf",
            sizeBytes = 100,
            sha256 = "a".repeat(64),
            transferredBytes = transferredBytes,
            state = state,
            cachePath = "/cache/$id",
            updatedAt = 1,
            lastAccessedAt = 1,
        ),
    )
}
