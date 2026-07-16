package com.akashic.mobile.data.realtime

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonPrimitive

data class FinalMessageEvent(
    val sessionId: String,
    val messageId: String,
    val content: String,
    val hasAttachments: Boolean,
    val attention: FinalMessageAttention = FinalMessageAttention.COMPLETE,
)

enum class FinalMessageAttention {
    COMPLETE,
    CONFIRMATION,
}

internal fun finalMessageAttention(payload: JsonObject): FinalMessageAttention {
    """读取服务端显式声明的移动端通知语义。"""

    // 1. 缺省终态就是普通完成
    val rawMetadata = payload["metadata"] ?: return FinalMessageAttention.COMPLETE

    // 2. 在协议边界校验结构与枚举值
    val metadata = rawMetadata as? JsonObject ?: error("message.final metadata 必须是对象")
    return when (metadata["mobile_attention"]?.jsonPrimitive?.content) {
        null -> FinalMessageAttention.COMPLETE
        "confirmation" -> FinalMessageAttention.CONFIRMATION
        else -> error("message.final mobile_attention 无效")
    }
}
