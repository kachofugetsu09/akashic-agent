package com.akashic.mobile.data.realtime

import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.put
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class RealtimeEventsTest {
    @Test
    fun finalMessageUsesExplicitConfirmationSemantic() {
        val confirmation = buildJsonObject {
            put("metadata", buildJsonObject { put("mobile_attention", "confirmation") })
        }

        assertEquals(FinalMessageAttention.CONFIRMATION, finalMessageAttention(confirmation))
        assertEquals(FinalMessageAttention.COMPLETE, finalMessageAttention(buildJsonObject {}))
    }

    @Test
    fun finalMessageRejectsUnknownAttentionSemantic() {
        val payload = buildJsonObject {
            put("metadata", buildJsonObject { put("mobile_attention", "guess-from-copy") })
        }

        assertThrows(IllegalStateException::class.java) { finalMessageAttention(payload) }
    }

    @Test
    fun finalMessageRejectsNonObjectMetadataAtProtocolBoundary() {
        val payload = buildJsonObject { put("metadata", JsonPrimitive("broken")) }

        assertThrows(IllegalStateException::class.java) { finalMessageAttention(payload) }
    }
}
