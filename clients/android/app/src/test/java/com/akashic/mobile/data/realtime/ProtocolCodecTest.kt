package com.akashic.mobile.data.realtime

import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.decodeFromJsonElement
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.put
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import okio.ByteString.Companion.toByteString

class ProtocolCodecTest {
    @Test
    fun `decodes shared v1 golden frames`() {
        val resource = requireNotNull(javaClass.getResource("/frames-v1.json"))
        val frames = ProtocolCodec.json().parseToJsonElement(resource.readText()).jsonArray

        val envelopes = frames.map { frame -> ProtocolCodec.decode(frame.toString()) }
        val messagePayload = ProtocolCodec.json().decodeFromJsonElement<MessageSendPayload>(
            envelopes.first().payload,
        )
        val resumePayload = ProtocolCodec.json().decodeFromJsonElement<ResumePayload>(
            envelopes[5].payload,
        )

        assertEquals(7, envelopes.size)
        assertEquals("2026-07-14T12:00:00+08:00", messagePayload.clientCreatedAt)
        assertEquals(listOf("turn-1"), resumePayload.activeTurns)
    }

    @Test
    fun `decodes ordered answer event`() {
        val frame = """
            {
              "v": 1,
              "kind": "event",
              "type": "answer.delta",
              "id": "01J00000000000000000000002",
              "connection_epoch": 7,
              "event_seq": 1842,
              "session_id": "mobile:session",
              "turn_id": "turn-1",
              "payload": {"delta": "你好"}
            }
        """.trimIndent()

        val envelope = ProtocolCodec.decode(frame)

        assertEquals(WireKind.EVENT, envelope.kind)
        assertEquals(1842L, envelope.eventSeq)
        assertEquals("你好", envelope.payload["delta"]?.toString()?.trim('"'))
    }

    @Test
    fun `round trips cumulative ack`() {
        val envelope = WireEnvelope(
            v = WIRE_PROTOCOL_VERSION,
            kind = WireKind.ACK,
            type = "event.ack",
            connectionEpoch = 7,
            payload = buildJsonObject { put("through_event_seq", 1842) },
        )

        assertEquals(envelope, ProtocolCodec.decode(ProtocolCodec.encode(envelope)))
    }

    @Test
    fun `round trips turn stop with current identity`() {
        val envelope = WireEnvelope(
            v = WIRE_PROTOCOL_VERSION,
            kind = WireKind.COMMAND,
            type = "turn.stop",
            id = "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            connectionEpoch = 7,
            sessionId = "mobile:one",
            turnId = "turn-1",
            payload = buildJsonObject {},
        )

        assertEquals(envelope, ProtocolCodec.decode(ProtocolCodec.encode(envelope)))
    }

    @Test
    fun `rejects unsupported version at boundary`() {
        val frame = """{"v":2,"kind":"control","type":"server.challenge","payload":{}}"""

        assertThrows(IllegalArgumentException::class.java) { ProtocolCodec.decode(frame) }
    }

    @Test
    fun `rejects event without sequence`() {
        val frame = """{"v":1,"kind":"event","type":"answer.delta","id":"event-1","payload":{}}"""

        assertThrows(IllegalArgumentException::class.java) { ProtocolCodec.decode(frame) }
    }

    @Test
    fun `decodes attachment descriptor download reply and binary chunk`() {
        val frame = AttachmentChunkCodec.encode(
            "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            128,
            byteArrayOf(1, 2, 3),
        )

        val decoded = AttachmentChunkCodec.decode(frame)

        assertEquals("01ARZ3NDEKTSV4RRFFQ69G5FAV", decoded.attachmentId)
        assertEquals(128L, decoded.offset)
        assertEquals(listOf<Byte>(1, 2, 3), decoded.payload.toList())
    }

    @Test
    fun `rejects message command whose id differs from client message id`() {
        val frame = requireNotNull(javaClass.getResource("/frames-v1.json")).readText()
            .replaceFirst("01J00000000000000000000000", "01J00000000000000000000009")
        val command = ProtocolCodec.json().parseToJsonElement(frame).jsonArray.first().toString()

        assertThrows(IllegalArgumentException::class.java) { ProtocolCodec.decode(command) }
    }

    @Test
    fun `rejects oversized binary frame before header parsing`() {
        val frame = ByteArray(AttachmentChunkCodec.MAX_BINARY_FRAME_BYTES + 1).toByteString()

        assertThrows(IllegalArgumentException::class.java) { AttachmentChunkCodec.decode(frame) }
    }
}
