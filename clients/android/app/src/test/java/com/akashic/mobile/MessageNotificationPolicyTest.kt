package com.akashic.mobile

import com.akashic.mobile.data.realtime.FinalMessageEvent
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class MessageNotificationPolicyTest {
    private val event = FinalMessageEvent(
        sessionId = "mobile:session-a",
        messageId = "message-1",
        content = "最终回答第一行\n不会进入通知预览",
        hasAttachments = false,
    )

    @Test
    fun foregroundCurrentSessionDoesNotDuplicateFinalMessage() {
        assertFalse(
            MessageNotificationPolicy.shouldNotify(
                appVisible = true,
                currentSessionId = event.sessionId,
                event = event,
            ),
        )
    }

    @Test
    fun backgroundOrDifferentSessionReceivesFinalMessage() {
        assertTrue(
            MessageNotificationPolicy.shouldNotify(
                appVisible = false,
                currentSessionId = event.sessionId,
                event = event,
            ),
        )
        assertTrue(
            MessageNotificationPolicy.shouldNotify(
                appVisible = true,
                currentSessionId = "mobile:session-b",
                event = event,
            ),
        )
    }

    @Test
    fun previewUsesOnlyFinalAnswerAndHasPrivateFallback() {
        assertEquals("最终回答第一行", MessageNotificationPolicy.preview(event))
        assertEquals(
            "收到一个附件",
            MessageNotificationPolicy.preview(event.copy(content = "", hasAttachments = true)),
        )
        assertEquals(
            "收到一条新回复",
            MessageNotificationPolicy.preview(event.copy(content = "", hasAttachments = false)),
        )
    }
}
