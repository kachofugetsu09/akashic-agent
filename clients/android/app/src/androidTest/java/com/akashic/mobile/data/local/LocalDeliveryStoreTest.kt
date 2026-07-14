package com.akashic.mobile.data.local

import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.akashic.mobile.data.realtime.WireEnvelope
import com.akashic.mobile.data.realtime.WireKind
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.add
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
                put("block_id", "tool-1"); put("call_id", "call-1"); put("ordinal", 1); put("tool_name", "shell")
                put("arguments", buildJsonObject { put("description", "读取运行日志") })
            }),
            event(4, "react.tool.completed", buildJsonObject {
                put("block_id", "tool-1"); put("call_id", "call-1"); put("ordinal", 1); put("tool_name", "shell")
                put("status", "success"); put("result_preview", "完成")
            }),
            event(5, "react.thinking.delta", buildJsonObject {
                put("block_id", "think-2"); put("ordinal", 2); put("delta", "再判断")
            }),
            event(6, "answer.delta", buildJsonObject { put("delta", "答案") }),
        )
        events.forEach { store.applyEvent("server", "device", it, it.eventSeq!!) }

        val blocks = database.messages().getBlocks("assistant:turn")
        assertEquals(listOf("think-1", "tool-1", "think-2"), blocks.map { it.blockId })
        assertEquals(listOf("completed", "completed", "running"), blocks.map { it.status })
        assertEquals(
            StoredToolBlock("shell", "读取运行日志", "完成"),
            decodeStoredToolBlock(blocks[1].content),
        )
        assertEquals("答案", database.messages().get("assistant:turn")!!.text)
    }

    @Test
    fun finalMessageAddsReasoningWhenProviderDidNotStreamThinking() = runBlocking {
        val events = listOf(
            event(1, "turn.started", buildJsonObject {}),
            event(2, "react.tool.started", buildJsonObject {
                put("block_id", "tool-1"); put("call_id", "call-1"); put("ordinal", 0); put("tool_name", "shell")
                put("arguments", buildJsonObject { put("description", "检查进程") })
            }),
            event(3, "react.tool.completed", buildJsonObject {
                put("block_id", "tool-1"); put("call_id", "call-1"); put("ordinal", 0); put("tool_name", "shell")
                put("status", "success"); put("result_preview", "完成")
            }),
            event(4, "message.final", buildJsonObject {
                put("content", "答案"); put("thinking", "先确认运行状态，再给出结论。")
            }),
        )
        events.forEach { store.applyEvent("server", "device", it, it.eventSeq!!) }

        val blocks = database.messages().getBlocks("assistant:turn")
        assertEquals(listOf("thinking", "tool"), blocks.map { it.kind })
        assertEquals(listOf(-1, 0), blocks.map { it.ordinal })
        assertEquals("先确认运行状态，再给出结论。", blocks.first().content)
    }

    @Test
    fun historyPageRestoresMessagesThinkingAndTools() = runBlocking {
        val history = event(
            1,
            "history.page",
            buildJsonObject {
                put("total", 2)
                put("page", 1)
                put("page_size", 10)
                put("items", buildJsonArray {
                    add(buildJsonObject {
                        put("id", "mobile:test:0")
                        put("session_key", "mobile:test")
                        put("seq", 0)
                        put("role", "user")
                        put("content", "恢复问题")
                        put("extra", buildJsonObject {})
                        put("ts", "2026-07-14T16:00:00Z")
                    })
                    add(buildJsonObject {
                        put("id", "mobile:test:1")
                        put("session_key", "mobile:test")
                        put("seq", 1)
                        put("role", "assistant")
                        put("content", "恢复回答")
                        put("tool_chain", buildJsonArray {
                            add(buildJsonObject {
                                put("text", "先检查")
                                put("calls", buildJsonArray {
                                    add(buildJsonObject {
                                        put("call_id", "call-1")
                                        put("name", "shell")
                                        put("status", "success")
                                        put("description", "读取状态")
                                        put("result_preview", "完成")
                                    })
                                })
                            })
                        })
                        put("extra", buildJsonObject { put("reasoning_content", "最后判断") })
                        put("ts", "2026-07-14T16:00:05Z")
                    })
                })
            },
        )

        store.applyEvent("server", "device", history, 1)

        assertEquals("恢复问题", database.messages().get("history:mobile:test:0")!!.text)
        assertEquals("恢复回答", database.messages().get("history:mobile:test:1")!!.text)
        val blocks = database.messages().getBlocks("history:mobile:test:1")
        assertEquals(listOf("thinking", "tool", "thinking"), blocks.map { it.kind })
        assertEquals(StoredToolBlock("shell", "读取状态", "完成"), decodeStoredToolBlock(blocks[1].content))
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
