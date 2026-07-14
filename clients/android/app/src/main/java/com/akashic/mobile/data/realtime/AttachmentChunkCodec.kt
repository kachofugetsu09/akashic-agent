package com.akashic.mobile.data.realtime

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import okio.ByteString
import okio.ByteString.Companion.toByteString

object AttachmentChunkCodec {
    /** 编码服务端约定的绝对 offset WebSocket 二进制分片。 */
    fun encode(attachmentId: String, offset: Long, payload: ByteArray): ByteString {
        require(offset >= 0) { "附件 offset 不能为负数" }
        require(payload.isNotEmpty() && payload.size <= MAX_CHUNK_BYTES) {
            "附件分片必须在 1..$MAX_CHUNK_BYTES 字节"
        }
        val header = ProtocolCodec.json().encodeToString(
            buildJsonObject {
                put("attachment_id", attachmentId)
                put("offset", offset)
            },
        ).toByteArray(Charsets.UTF_8)
        require(header.size <= MAX_HEADER_BYTES) { "附件分片 header 过大" }
        return ByteBuffer.allocate(Int.SIZE_BYTES + header.size + payload.size)
            .order(ByteOrder.BIG_ENDIAN)
            .putInt(header.size)
            .put(header)
            .put(payload)
            .array()
            .toByteString()
    }

    const val MAX_CHUNK_BYTES = 128 * 1024
    const val MAX_HEADER_BYTES = 1024
}
