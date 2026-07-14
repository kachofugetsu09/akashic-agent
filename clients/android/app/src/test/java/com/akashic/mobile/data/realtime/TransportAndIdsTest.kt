package com.akashic.mobile.data.realtime

import com.akashic.mobile.domain.model.EndpointRoute
import com.akashic.mobile.domain.model.ServerEndpoint
import kotlin.random.Random
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
}
