package com.akashic.mobile.data.local

import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.akashic.mobile.data.realtime.WireEnvelope
import com.akashic.mobile.data.realtime.WireKind
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class LocalDeliveryStoreTest {
    private lateinit var database: AppDatabase
    private lateinit var store: LocalDeliveryStore

    @Before
    fun setUp() = runBlocking {
        database = Room.inMemoryDatabaseBuilder(
            ApplicationProvider.getApplicationContext(),
            AppDatabase::class.java,
        ).build()
        store = LocalDeliveryStore(database)
        store.savePairedProfile(
            ServerProfileEntity("server", "server", "device", "alias", "pin", "[]", "[]", "[]", 1),
            RealtimeCursorEntity("device", "server", 0, 0, 1),
        )
        database.conversations().upsert(ConversationEntity("mobile:test", "server", "test", 1))
    }

    @After
    fun tearDown() = database.close()

    @Test
    fun reducerPreservesThinkingToolThinkingOrder() = runBlocking {
        val events = listOf(
            event(1, "turn.started", buildJsonObject {}),
            event(2, "react.thinking.delta", buildJsonObject {
                put("block_id", "think-1"); put("ordinal", 0); put("delta", "先分析")
            }),
            event(3, "react.tool.started", buildJsonObject {
                put("block_id", "tool-1"); put("tool_call_id", "call-1"); put("ordinal", 1); put("name", "shell")
            }),
            event(4, "react.tool.completed", buildJsonObject {
                put("block_id", "tool-1"); put("tool_call_id", "call-1"); put("ordinal", 1); put("summary", "完成")
            }),
            event(5, "react.thinking.delta", buildJsonObject {
                put("block_id", "think-2"); put("ordinal", 2); put("delta", "再判断")
            }),
            event(6, "answer.delta", buildJsonObject { put("delta", "答案") }),
        )
        events.forEach { store.applyEvent("server", "device", it, it.eventSeq!!) }

        val blocks = database.messages().getBlocks("assistant:turn")
        assertEquals(listOf("think-1", "tool-1", "think-2"), blocks.map { it.blockId })
        assertEquals("答案", database.messages().get("assistant:turn")!!.text)
    }

    private fun event(sequence: Long, type: String, payload: kotlinx.serialization.json.JsonObject) = WireEnvelope(
        v = 1,
        kind = WireKind.EVENT,
        type = type,
        id = "01J00000000000000000000000",
        connectionEpoch = 1,
        eventSeq = sequence,
        sessionId = "mobile:test",
        turnId = "turn",
        payload = payload,
    )
}
