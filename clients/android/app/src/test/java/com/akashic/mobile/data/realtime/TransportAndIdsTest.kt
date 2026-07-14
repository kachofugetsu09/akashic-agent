package com.akashic.mobile.data.realtime

import com.akashic.mobile.domain.model.EndpointRoute
import com.akashic.mobile.domain.model.ServerEndpoint
import kotlin.random.Random
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class TransportAndIdsTest {
    @Test
    fun `LAN requires QR pin and tunnel forbids it`() {
        val pin = "sha256/${"A".repeat(43)}="
        validateEndpointSecurity(ServerEndpoint("wss://pc.local:6323/ws", listOf(pin), EndpointRoute.LAN))
        validateEndpointSecurity(ServerEndpoint("wss://example.com/ws", emptyList(), EndpointRoute.TUNNEL))
        assertThrows(IllegalArgumentException::class.java) {
            validateEndpointSecurity(ServerEndpoint("wss://pc.local:6323/ws", emptyList(), EndpointRoute.LAN))
        }
        assertThrows(IllegalArgumentException::class.java) {
            validateEndpointSecurity(ServerEndpoint("wss://example.com/ws", listOf(pin), EndpointRoute.TUNNEL))
        }
    }

    @Test
    fun `full jitter remains inside exponential cap`() {
        repeat(100) {
            val delay = FullJitterBackoff.nextDelayMillis(5, Random(it))
            assertTrue(delay in 0..FullJitterBackoff.maximumDelayMillis(5))
        }
        assertEquals(30_000L, FullJitterBackoff.maximumDelayMillis(20))
    }

    @Test
    fun `generated ULID satisfies wire frame grammar`() {
        val value = Ulid.next(1_752_460_800_000)
        assertTrue(Regex("^[0-9A-HJKMNP-TV-Z]{26}$").matches(value))
    }

    @Test
    fun `attachment chunk uses big endian json header and absolute offset`() {
        val attachmentId = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        val encoded = AttachmentChunkCodec.encode(attachmentId, 1_048_576, byteArrayOf(1, 2, 3)).toByteArray()
        val buffer = ByteBuffer.wrap(encoded).order(ByteOrder.BIG_ENDIAN)
        val headerSize = buffer.int
        val headerBytes = ByteArray(headerSize)
        buffer.get(headerBytes)
        val header = ProtocolCodec.json().parseToJsonElement(headerBytes.toString(Charsets.UTF_8)).jsonObject
        val payload = ByteArray(buffer.remaining())
        buffer.get(payload)

        assertEquals(attachmentId, header.getValue("attachment_id").jsonPrimitive.content)
        assertEquals("1048576", header.getValue("offset").jsonPrimitive.content)
        assertTrue(payload.contentEquals(byteArrayOf(1, 2, 3)))
    }

    @Test
    fun `attachment chunk rejects payload beyond server limit`() {
        assertThrows(IllegalArgumentException::class.java) {
            AttachmentChunkCodec.encode(
                "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                0,
                ByteArray(AttachmentChunkCodec.MAX_CHUNK_BYTES + 1),
            )
        }
    }
}
