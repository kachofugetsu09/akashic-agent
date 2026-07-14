package com.akashic.mobile.data.realtime

data class FinalMessageEvent(
    val sessionId: String,
    val messageId: String,
    val content: String,
    val hasAttachments: Boolean,
)
