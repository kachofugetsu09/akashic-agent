package com.akashic.mobile.data.realtime

import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.decodeFromJsonElement
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.put
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

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
              "id": "019event",
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
    fun `rejects unsupported version at boundary`() {
        val frame = """{"v":2,"kind":"control","type":"server.challenge","payload":{}}"""

        assertThrows(IllegalArgumentException::class.java) { ProtocolCodec.decode(frame) }
    }

    @Test
    fun `rejects event without sequence`() {
        val frame = """{"v":1,"kind":"event","type":"answer.delta","id":"event-1","payload":{}}"""

        assertThrows(IllegalArgumentException::class.java) { ProtocolCodec.decode(frame) }
    }
}
