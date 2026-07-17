package com.akashic.mobile.data.local

import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.akashic.mobile.data.realtime.WIRE_PROTOCOL_VERSION
import com.akashic.mobile.data.realtime.WireEnvelope
import com.akashic.mobile.data.realtime.WireKind
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class LocalDeliveryStoreCursorTest {
    private lateinit var database: AppDatabase
    private lateinit var store: LocalDeliveryStore

    @Before
    fun setUp() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        database = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java).build()
        store = LocalDeliveryStore(
            database,
            MediaCacheStore(context.cacheDir.resolve("cursor-test-${System.nanoTime()}"), database.mediaAttachments()),
        )
        runBlocking {
            store.savePairedProfile(
                ServerProfileEntity(
                    serverId = "server-1",
                    displayName = "Test",
                    deviceId = "device-1",
                    keyAlias = "key",
                    applicationKeyFingerprint = "fingerprint",
                    lanEndpointsJson = "[]",
                    tunnelEndpointsJson = "[]",
                    tlsSpkiPinsJson = "[]",
                    createdAt = 1,
                ),
                RealtimeCursorEntity("device-1", "server-1", 4, 1, 1),
            )
        }
    }

    @After
    fun tearDown() {
        database.close()
    }

    @Test
    fun invalidFinalAttentionDoesNotAdvanceCursor() {
        val envelope = WireEnvelope(
            v = WIRE_PROTOCOL_VERSION,
            kind = WireKind.EVENT,
            type = "message.final",
            id = "final-frame",
            connectionEpoch = 1,
            eventSeq = 5,
            sessionId = "mobile:test",
            turnId = "turn-1",
            payload = buildJsonObject { put("metadata", JsonPrimitive("broken")) },
        )

        assertThrows(IllegalStateException::class.java) {
            runBlocking { store.applyEvent("server-1", "device-1", envelope, updatedAt = 2) }
        }
        assertEquals(
            4L,
            runBlocking { database.realtimeCursors().get("device-1")?.lastAcknowledgedEventSeq },
        )
    }

    @Test
    fun invalidProactiveAttentionDoesNotAdvanceCursor() {
        val envelope = WireEnvelope(
            v = WIRE_PROTOCOL_VERSION,
            kind = WireKind.EVENT,
            type = "message.proactive",
            id = "proactive-frame",
            connectionEpoch = 1,
            eventSeq = 5,
            sessionId = "mobile:test",
            payload = buildJsonObject { put("metadata", JsonPrimitive("broken")) },
        )

        assertThrows(IllegalStateException::class.java) {
            runBlocking { store.applyEvent("server-1", "device-1", envelope, updatedAt = 2) }
        }
        assertEquals(
            4L,
            runBlocking { database.realtimeCursors().get("device-1")?.lastAcknowledgedEventSeq },
        )
    }
}
