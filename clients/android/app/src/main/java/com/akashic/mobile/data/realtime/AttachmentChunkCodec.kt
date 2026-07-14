package com.akashic.mobile.data.realtime

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import okio.ByteString
import okio.ByteString.Companion.toByteString

object AttachmentChunkCodec {
    data class DecodedChunk(
        val attachmentId: String,
        val offset: Long,
        val payload: ByteArray,
    )

    @Serializable
    private data class ChunkHeader(
        @SerialName("attachment_id") val attachmentId: String,
        val offset: Long,
    )

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

    /** 严格解析服务端二进制附件分片。 */
    fun decode(frame: ByteString): DecodedChunk {
        // 1. 校验固定长度和 JSON header 边界
        require(frame.size <= MAX_BINARY_FRAME_BYTES) { "附件二进制帧超过协议上限" }
        val raw = frame.toByteArray()
        require(raw.size > Int.SIZE_BYTES) { "附件二进制帧缺少 header 长度" }
        val buffer = ByteBuffer.wrap(raw).order(ByteOrder.BIG_ENDIAN)
        val headerSize = buffer.int
        require(headerSize in 1..MAX_HEADER_BYTES) { "附件二进制帧 header 长度无效" }
        require(raw.size > Int.SIZE_BYTES + headerSize) { "附件二进制帧缺少分片数据" }

        // 2. 解析 header 并限制分片大小
        val headerBytes = ByteArray(headerSize)
        buffer.get(headerBytes)
        val header = ProtocolCodec.json().decodeFromString<ChunkHeader>(headerBytes.toString(Charsets.UTF_8))
        val payload = ByteArray(buffer.remaining())
        buffer.get(payload)
        require(header.offset >= 0) { "附件 offset 不能为负数" }
        require(payload.size <= MAX_CHUNK_BYTES) { "附件二进制分片超过 $MAX_CHUNK_BYTES 字节" }
        return DecodedChunk(header.attachmentId, header.offset, payload)
    }

    const val MAX_CHUNK_BYTES = 128 * 1024
    const val MAX_HEADER_BYTES = 1024
    const val MAX_BINARY_FRAME_BYTES = Int.SIZE_BYTES + MAX_HEADER_BYTES + MAX_CHUNK_BYTES
}
