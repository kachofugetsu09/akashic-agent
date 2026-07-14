package com.akashic.mobile

import com.akashic.mobile.data.realtime.FinalMessageEvent

internal object MessageNotificationPolicy {
    fun shouldNotify(
        appVisible: Boolean,
        currentSessionId: String?,
        event: FinalMessageEvent,
    ): Boolean = !appVisible || currentSessionId != event.sessionId

    fun preview(event: FinalMessageEvent): String {
        val text = event.content.trim().lineSequence().firstOrNull().orEmpty()
        if (text.isNotEmpty()) return text.take(MAX_PREVIEW_CHARS)
        return if (event.hasAttachments) "收到一个附件" else "收到一条新回复"
    }

    private const val MAX_PREVIEW_CHARS = 120
}
