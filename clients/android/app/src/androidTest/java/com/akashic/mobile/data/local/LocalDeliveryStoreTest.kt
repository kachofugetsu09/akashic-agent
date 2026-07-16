package com.akashic.mobile.data.local

import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import android.content.Context
import com.akashic.mobile.data.realtime.WireEnvelope
import com.akashic.mobile.data.realtime.WireKind
import com.akashic.mobile.data.realtime.AttachmentDownloadCoordinator
import com.akashic.mobile.data.realtime.MessageSendPayload
import com.akashic.mobile.data.realtime.ProtocolCodec
import java.time.Instant
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.flow.first
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.add
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.put
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue

@RunWith(AndroidJUnit4::class)
class LocalDeliveryStoreTest {
    private lateinit var database: AppDatabase
    private lateinit var mediaCache: MediaCacheStore
    private lateinit var store: LocalDeliveryStore

    @Before
    fun setUp() = runBlocking {
        database = Room.inMemoryDatabaseBuilder(
            ApplicationProvider.getApplicationContext(),
            AppDatabase::class.java,
        ).build()
        mediaCache = MediaCacheStore(
            ApplicationProvider.getApplicationContext<Context>().cacheDir.resolve("media-${System.nanoTime()}"),
            database.mediaAttachments(),
        )
        store = LocalDeliveryStore(database, mediaCache)
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
                put("status", "success"); put("result_preview", "完成"); put("duration_ms", 615)
                put("arguments", buildJsonObject { put("description", "读取实际运行日志") })
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
            StoredToolBlock(
                name = "shell",
                description = "读取实际运行日志",
                resultPreview = "完成",
                arguments = buildJsonObject { put("description", "读取实际运行日志") },
                durationMillis = 615,
            ),
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
                put("status", "denied"); put("result_preview", "策略拒绝"); put("duration_ms", 20)
                put("arguments", buildJsonObject { put("description", "检查实际进程") })
            }),
            event(4, "message.final", buildJsonObject {
                put("message_id", "mobile:test:assistant:final")
                put("content", "答案"); put("thinking", "先确认运行状态，再给出结论。")
            }),
        )
        events.forEach { store.applyEvent("server", "device", it, it.eventSeq!!) }

        val blocks = database.messages().getBlocks("mobile:test:assistant:final")
        assertEquals(listOf("thinking", "tool"), blocks.map { it.kind })
        assertEquals(listOf(-1, 0), blocks.map { it.ordinal })
        assertEquals("failed", blocks.last().status)
        assertEquals("先确认运行状态，再给出结论。", blocks.first().content)
    }

    @Test
    fun finalWithoutMessageIdUsesFrameScopedEphemeralIdentity() = runBlocking {
        store.applyEvent(
            "server",
            "device",
            event(1, "message.final", buildJsonObject { put("content", "控制指令结果") }),
            2,
        )

        val ephemeralId = "ephemeral:01J00000000000000000000000"
        assertEquals("控制指令结果", database.messages().get(ephemeralId)!!.text)
        assertEquals(null, database.messages().get("assistant:turn"))

        store.applyEvent(
            "server",
            "device",
            event(2, "history.page", buildJsonObject {
                put("total", 1)
                put("page", 1)
                put("page_size", 10)
                put("items", buildJsonArray {
                    add(buildJsonObject {
                        put("id", "mobile:test:history:canonical")
                        put("session_key", "mobile:test")
                        put("seq", 1)
                        put("role", "assistant")
                        put("content", "持久化回答")
                        put("extra", buildJsonObject { put("reasoning_content", "历史思考") })
                        put("ts", "2026-07-14T16:00:05Z")
                    })
                })
            }),
            3,
        )

        assertEquals(2, database.messages().countForSession("mobile:test"))
        assertEquals("控制指令结果", database.messages().get(ephemeralId)!!.text)
        assertEquals("持久化回答", database.messages().get("mobile:test:history:canonical")!!.text)
        assertEquals(
            listOf("历史思考"),
            database.messages().getBlocks("mobile:test:history:canonical").map { it.content },
        )
    }

    @Test
    fun historyRepairsRepeatedEphemeralAssistantsByCompletionTime() = runBlocking {
        val firstCompletedAt = Instant.parse("2026-07-14T16:00:05Z").toEpochMilli()
        val secondCompletedAt = Instant.parse("2026-07-14T16:10:05Z").toEpochMilli()
        listOf(firstCompletedAt, secondCompletedAt).forEachIndexed { index, completedAt ->
            database.messages().upsert(
                MessageEntity(
                    messageId = "ephemeral:$index",
                    clientMessageId = null,
                    sessionId = "mobile:test",
                    role = "assistant",
                    text = "相同回答",
                    deliveryState = "complete",
                    createdAt = completedAt - 5_000,
                    updatedAt = completedAt,
                ),
            )
        }

        store.applyEvent(
            "server",
            "device",
            event(1, "history.page", buildJsonObject {
                put("total", 2)
                put("page", 1)
                put("page_size", 10)
                put("items", buildJsonArray {
                    listOf(firstCompletedAt, secondCompletedAt).forEachIndexed { index, _ ->
                        add(buildJsonObject {
                            put("id", "mobile:test:canonical:$index")
                            put("session_key", "mobile:test")
                            put("seq", index)
                            put("role", "assistant")
                            put("content", "相同回答")
                            put("extra", buildJsonObject {})
                            put("ts", if (index == 0) "2026-07-14T16:00:05Z" else "2026-07-14T16:10:05Z")
                        })
                    }
                })
            }),
            secondCompletedAt,
        )

        assertEquals(2, database.messages().countForSession("mobile:test"))
        assertEquals(null, database.messages().get("ephemeral:0"))
        assertEquals(null, database.messages().get("ephemeral:1"))
        assertEquals("相同回答", database.messages().get("mobile:test:canonical:0")!!.text)
        assertEquals("相同回答", database.messages().get("mobile:test:canonical:1")!!.text)
    }

    @Test
    fun historyReplacesLiveBlocksForTheSameCanonicalMessage() = runBlocking {
        store.applyEvent(
            "server",
            "device",
            event(1, "react.thinking.delta", buildJsonObject {
                put("block_id", "live-thinking")
                put("ordinal", 0)
                put("delta", "流式思考")
            }),
            2,
        )
        store.applyEvent(
            "server",
            "device",
            event(2, "message.final", buildJsonObject {
                put("message_id", "mobile:test:assistant:same")
                put("content", "实时回答")
            }),
            3,
        )
        store.applyEvent(
            "server",
            "device",
            event(3, "history.page", buildJsonObject {
                put("total", 1)
                put("page", 1)
                put("page_size", 10)
                put("items", buildJsonArray {
                    add(buildJsonObject {
                        put("id", "mobile:test:assistant:same")
                        put("session_key", "mobile:test")
                        put("seq", 1)
                        put("role", "assistant")
                        put("content", "历史回答")
                        put("extra", buildJsonObject { put("reasoning_content", "历史思考") })
                        put("ts", "2026-07-14T16:00:05Z")
                    })
                })
            }),
            4,
        )

        assertEquals(1, database.messages().countForSession("mobile:test"))
        assertEquals("历史回答", database.messages().get("mobile:test:assistant:same")!!.text)
        val blocks = database.messages().getBlocks("mobile:test:assistant:same")
        assertEquals(listOf("history:mobile:test:assistant:same:0"), blocks.map { it.blockId })
        assertEquals(listOf("历史思考"), blocks.map { it.content })
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
                                        put("status", "blocked")
                                        put("description", "读取状态")
                                        put("result_preview", "未执行")
                                    })
                                })
                            })
                        })
                        put("extra", buildJsonObject {
                            put("reasoning_content", "最后判断")
                            put("turn_duration_ms", 30_000)
                        })
                        put("ts", "2026-07-14T16:00:05Z")
                    })
                })
            },
        )

        store.applyEvent("server", "device", history, 1)

        assertEquals("恢复问题", database.messages().get("mobile:test:0")!!.text)
        assertEquals("恢复回答", database.messages().get("mobile:test:1")!!.text)
        assertEquals(
            listOf("恢复问题", "恢复回答"),
            database.messages().observeMessages("mobile:test").first().map { it.text },
        )
        val blocks = database.messages().getBlocks("mobile:test:1")
        assertEquals(listOf("thinking", "tool", "thinking"), blocks.map { it.kind })
        assertEquals("failed", blocks[1].status)
        assertEquals(StoredToolBlock("shell", "读取状态", "未执行"), decodeStoredToolBlock(blocks[1].content))
    }

    @Test
    fun historyCanonicalIdReplacesOptimisticUserMessage() = runBlocking {
        val clientId = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        val attachmentId = "01ARZ3NDEKTSV4RRFFQ69G5FAW"
        database.messages().upsert(
            MessageEntity(
                messageId = "user:$clientId",
                clientMessageId = clientId,
                sessionId = "mobile:test",
                role = "user",
                text = "本地问题",
                deliveryState = "sent",
                createdAt = 1,
                updatedAt = 1,
            ),
        )
        database.mediaAttachments().upsert(mediaAttachment(attachmentId))
        database.mediaAttachments().linkAll(
            listOf(MessageAttachmentEntity("user:$clientId", attachmentId, 0)),
        )
        store.applyEvent(
            "server",
            "device",
            event(1, "history.page", buildJsonObject {
                put("total", 1)
                put("page", 1)
                put("page_size", 10)
                put("items", buildJsonArray {
                    add(buildJsonObject {
                        put("id", "mobile:test:user:canonical")
                        put("client_message_id", clientId)
                        put("session_key", "mobile:test")
                        put("seq", 0)
                        put("role", "user")
                        put("content", "本地问题")
                        put("extra", buildJsonObject {})
                        put("attachments", buildJsonArray {
                            add(buildJsonObject {
                                put("attachment_id", attachmentId)
                                put("filename", "image.png")
                                put("content_type", "image/png")
                                put("size_bytes", 1_048_579)
                                put("sha256", "a".repeat(64))
                            })
                        })
                        put("ts", "2026-07-14T16:00:00Z")
                    })
                })
            }),
            2,
        )

        assertEquals(null, database.messages().get("user:$clientId"))
        val canonical = database.messages().get("mobile:test:user:canonical")!!
        assertEquals(clientId, canonical.clientMessageId)
        assertEquals("complete", canonical.deliveryState)
        assertEquals(
            listOf(attachmentId),
            database.mediaAttachments().forMessage("mobile:test:user:canonical").map { it.attachmentId },
        )
    }

    @Test
    fun finalCanonicalIdReplacesReplyingOptimisticUserMessage() = runBlocking {
        val clientId = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        val attachmentId = "01ARZ3NDEKTSV4RRFFQ69G5FAW"
        database.messages().upsert(
            MessageEntity(
                messageId = "user:$clientId",
                clientMessageId = clientId,
                sessionId = "mobile:test",
                role = "user",
                text = "继续",
                deliveryState = "sent",
                createdAt = 1,
                updatedAt = 1,
                replyToMessageId = "mobile:test:0",
                replyRole = "assistant",
                replyPreview = "旧回答",
            ),
        )
        database.mediaAttachments().upsert(mediaAttachment(attachmentId))
        database.mediaAttachments().linkAll(
            listOf(MessageAttachmentEntity("user:$clientId", attachmentId, 0)),
        )

        store.applyEvent(
            "server",
            "device",
            event(1, "message.final", buildJsonObject {
                put("user_message_id", "mobile:test:user:canonical")
                put("client_message_id", clientId)
                put("message_id", "mobile:test:assistant:canonical")
                put("content", "回答")
            }),
            2,
        )

        assertEquals(null, database.messages().get("user:$clientId"))
        val canonical = database.messages().get("mobile:test:user:canonical")!!
        assertEquals("complete", canonical.deliveryState)
        assertEquals("mobile:test:0", canonical.replyToMessageId)
        assertEquals("assistant", canonical.replyRole)
        assertEquals("旧回答", canonical.replyPreview)
        assertEquals(
            listOf(attachmentId),
            database.mediaAttachments().forMessage(canonical.messageId).map { it.attachmentId },
        )
        assertEquals("回答", database.messages().get("mobile:test:assistant:canonical")!!.text)
    }

    @Test
    fun legacyHistoryWithoutClientIdRepairsUniqueOptimisticUserMessage() = runBlocking {
        val clientId = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        val sentAt = Instant.parse("2026-07-14T16:00:00Z").toEpochMilli()
        database.messages().upsert(
            MessageEntity(
                messageId = "user:$clientId",
                clientMessageId = clientId,
                sessionId = "mobile:test",
                role = "user",
                text = "旧版问题",
                deliveryState = "sent",
                createdAt = sentAt,
                updatedAt = sentAt,
            ),
        )

        store.applyEvent(
            "server",
            "device",
            event(1, "history.page", buildJsonObject {
                put("total", 1)
                put("page", 1)
                put("page_size", 10)
                put("items", buildJsonArray {
                    add(buildJsonObject {
                        put("id", "mobile:test:user:canonical")
                        put("session_key", "mobile:test")
                        put("seq", 0)
                        put("role", "user")
                        put("content", "旧版问题")
                        put("extra", buildJsonObject {})
                        put("ts", "2026-07-14T16:00:09Z")
                    })
                })
            }),
            sentAt + 10_000,
        )

        assertEquals(1, database.messages().countForSession("mobile:test"))
        assertEquals(null, database.messages().get("user:$clientId"))
        assertEquals("旧版问题", database.messages().get("mobile:test:user:canonical")!!.text)
    }

    @Test
    fun legacyHistoryDoesNotGuessBetweenRepeatedIdenticalMessages() = runBlocking {
        val sentAt = Instant.parse("2026-07-14T16:00:00Z").toEpochMilli()
        listOf("01ARZ3NDEKTSV4RRFFQ69G5FAV", "01ARZ3NDEKTSV4RRFFQ69G5FAW").forEachIndexed { index, clientId ->
            database.messages().upsert(
                MessageEntity(
                    messageId = "user:$clientId",
                    clientMessageId = clientId,
                    sessionId = "mobile:test",
                    role = "user",
                    text = "重复问题",
                    deliveryState = "sent",
                    createdAt = sentAt + index,
                    updatedAt = sentAt + index,
                ),
            )
        }

        store.applyEvent(
            "server",
            "device",
            event(1, "history.page", buildJsonObject {
                put("total", 1)
                put("page", 1)
                put("page_size", 10)
                put("items", buildJsonArray {
                    add(buildJsonObject {
                        put("id", "mobile:test:user:canonical")
                        put("session_key", "mobile:test")
                        put("seq", 0)
                        put("role", "user")
                        put("content", "重复问题")
                        put("extra", buildJsonObject {})
                        put("ts", "2026-07-14T16:00:09Z")
                    })
                })
            }),
            sentAt + 10_000,
        )

        assertEquals(3, database.messages().countForSession("mobile:test"))
    }

    @Test
    fun legacyHistoryDoesNotConsumeANewerIdenticalUserMessage() = runBlocking {
        val clientId = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        val historyAt = Instant.parse("2026-07-14T16:00:00Z").toEpochMilli()
        database.messages().upsert(
            MessageEntity(
                messageId = "user:$clientId",
                clientMessageId = clientId,
                sessionId = "mobile:test",
                role = "user",
                text = "稍后又问的相同问题",
                deliveryState = "sent",
                createdAt = historyAt + 30 * 60 * 1_000,
                updatedAt = historyAt + 30 * 60 * 1_000,
            ),
        )

        store.applyEvent(
            "server",
            "device",
            event(1, "history.page", buildJsonObject {
                put("total", 1)
                put("page", 1)
                put("page_size", 10)
                put("items", buildJsonArray {
                    add(buildJsonObject {
                        put("id", "mobile:test:user:old-canonical")
                        put("session_key", "mobile:test")
                        put("seq", 0)
                        put("role", "user")
                        put("content", "稍后又问的相同问题")
                        put("extra", buildJsonObject {})
                        put("ts", "2026-07-14T16:00:00Z")
                    })
                })
            }),
            historyAt + 30 * 60 * 1_000,
        )

        assertEquals(2, database.messages().countForSession("mobile:test"))
        assertNotNull(database.messages().get("user:$clientId"))
        assertNotNull(database.messages().get("mobile:test:user:old-canonical"))
    }

    @Test
    fun gapResetClearsOnlyServerProjectionAndPreservesLocalWork() = runBlocking {
        val clientId = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        database.messages().upsert(
            MessageEntity("remote", null, "mobile:test", "assistant", "旧投影", "complete", 1, 1),
        )
        database.messages().upsert(
            MessageEntity("user:$clientId", clientId, "mobile:test", "user", "待发送", "pending", 2, 2),
        )
        database.outbox().enqueue(
            OutboxCommandEntity(clientId, "server", "{}", "pending", 0, 2, null),
        )
        database.attachmentTransfers().upsert(transfer("ready", 1_048_579, "draft"))
        store.savePairedProfile(
            ServerProfileEntity("other", "other", "other-device", "alias", "pin", "[]", "[]", "[]", 1),
            RealtimeCursorEntity("other-device", "other", 0, 0, 1),
        )
        database.conversations().upsert(ConversationEntity("mobile:other", "other", "other", 1))
        database.messages().upsert(
            MessageEntity("other-message", null, "mobile:other", "assistant", "其他电脑", "complete", 1, 1),
        )

        store.applyEvent(
            "server",
            "device",
            event(50, "sync.reset_required", buildJsonObject { put("reason", "inbox_retention_exceeded") }),
            3,
            preservedSessionId = "mobile:test",
        )

        assertEquals(50, database.realtimeCursors().get("device")!!.lastAcknowledgedEventSeq)
        assertEquals(null, database.messages().get("remote"))
        assertNotNull(database.messages().get("user:$clientId"))
        assertNotNull(database.outbox().get(clientId))
        assertNotNull(database.attachmentTransfers().get("draft"))
        assertEquals("其他电脑", database.messages().get("other-message")!!.text)
    }

    @Test
    fun manualReloadClearsCommittedCacheButPreservesPairingAndUnsentWork() = runBlocking {
        val pendingId = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        database.messages().upsert(
            MessageEntity("remote", null, "mobile:test", "assistant", "旧投影", "complete", 1, 1, 1),
        )
        database.messages().upsert(
            MessageEntity("sent", pendingId, "mobile:test", "user", "已提交", "sent", 2, 2),
        )
        database.messages().upsert(
            MessageEntity("pending", "pending-id", "mobile:test", "user", "待发送", "pending", 3, 3),
        )
        database.messages().upsert(
            MessageEntity("failed", "failed-id", "mobile:test", "user", "发送失败", "failed", 4, 4),
        )
        database.outbox().enqueue(
            OutboxCommandEntity("pending-id", "server", "{}", "pending", 0, 3, null),
        )
        database.attachmentTransfers().upsert(transfer("ready", 1_048_579, "draft"))

        store.clearReloadableCache("server", "mobile:test")

        assertEquals(null, database.messages().get("remote"))
        assertEquals(null, database.messages().get("sent"))
        assertNotNull(database.messages().get("pending"))
        assertNotNull(database.messages().get("failed"))
        assertNotNull(database.outbox().get("pending-id"))
        assertNotNull(database.attachmentTransfers().get("draft"))
        assertNotNull(database.serverProfiles().get("server"))
        assertNotNull(database.realtimeCursors().get("device"))
        assertNotNull(database.conversations().get("mobile:test"))
    }

    @Test
    fun repeatedResetRebuildsHistoryAttachmentsAndThenAcceptsLiveEvents() = runBlocking {
        store.applyEvent(
            "server",
            "device",
            event(100, "sync.reset_required", buildJsonObject { put("reason", "inbox_retention_exceeded") }),
            2,
        )
        store.applyEvent(
            "server",
            "device",
            event(200, "sync.reset_required", buildJsonObject { put("reason", "inbox_retention_exceeded") }),
            3,
        )
        val descriptor = buildJsonObject {
            put("attachment_id", "01ARZ3NDEKTSV4RRFFQ69G5FAW")
            put("filename", "history.png")
            put("content_type", "image/png")
            put("size_bytes", 3)
            put("sha256", "a".repeat(64))
        }
        store.applyEvent(
            "server",
            "device",
            event(201, "history.page", buildJsonObject {
                put("total", 1)
                put("page", 1)
                put("page_size", 10)
                put("items", buildJsonArray {
                    add(buildJsonObject {
                        put("id", "mobile:test:history")
                        put("session_key", "mobile:test")
                        put("seq", 1)
                        put("role", "assistant")
                        put("content", "历史恢复")
                        put("extra", buildJsonObject {})
                        put("attachments", buildJsonArray { add(descriptor) })
                        put("ts", "2026-07-14T16:00:05Z")
                    })
                })
            }),
            4,
        )
        store.applyEvent(
            "server",
            "device",
            event(202, "message.final", buildJsonObject {
                put("message_id", "mobile:test:live")
                put("content", "重建后的实时消息")
            }),
            5,
        )

        assertEquals(202, database.realtimeCursors().get("device")!!.lastAcknowledgedEventSeq)
        assertEquals("历史恢复", database.messages().get("mobile:test:history")!!.text)
        assertEquals("重建后的实时消息", database.messages().get("mobile:test:live")!!.text)
        assertEquals(
            listOf("01ARZ3NDEKTSV4RRFFQ69G5FAW"),
            database.mediaAttachments().forMessage("mobile:test:history").map { it.attachmentId },
        )
    }

    @Test
    fun attachmentProgressAndCursorCommitTogether() = runBlocking {
        database.attachmentTransfers().upsert(transfer(state = "uploading", offset = 0))
        store.applyEvent(
            "server",
            "device",
            event(1, "attachment.progress", buildJsonObject {
                put("attachment_id", "attachment")
                put("transferred_bytes", 1_048_576)
                put("size_bytes", 1_048_579)
            }),
            2,
        )

        assertEquals(1_048_576, database.attachmentTransfers().get("attachment")!!.transferredBytes)
        assertEquals(1, database.realtimeCursors().get("device")!!.lastAcknowledgedEventSeq)

        store.applyEvent(
            "server",
            "device",
            event(2, "attachment.ready", buildJsonObject {
                put("attachment_id", "attachment")
                put("filename", "image.png")
                put("content_type", "image/png")
                put("size_bytes", 1_048_579)
                put("sha256", "a".repeat(64))
            }),
            3,
        )
        assertEquals("ready", database.attachmentTransfers().get("attachment")!!.state)
    }

    @Test
    fun attachmentMessageMovesReadyToSendingThenSent() = runBlocking {
        database.attachmentTransfers().upsert(transfer(state = "ready", offset = 1_048_579))
        val payload = MessageSendPayload(
            clientMessageId = "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            sessionId = "mobile:test",
            text = "",
            mediaRefs = listOf("attachment"),
            clientCreatedAt = Instant.EPOCH.toString(),
        )
        val command = WireEnvelope(
            v = 1,
            kind = WireKind.COMMAND,
            type = "message.send",
            id = payload.clientMessageId,
            connectionEpoch = 1,
            sessionId = "mobile:test",
            payload = ProtocolCodec.json().encodeToJsonElement(
                MessageSendPayload.serializer(),
                payload,
            ).jsonObject,
        )
        store.enqueueMessage(
            conversation = ConversationEntity("mobile:test", "server", "image.png", 2),
            message = MessageEntity(
                messageId = "user:${payload.clientMessageId}",
                clientMessageId = payload.clientMessageId,
                sessionId = "mobile:test",
                role = "user",
                text = "",
                deliveryState = "pending",
                createdAt = 2,
                updatedAt = 2,
            ),
            command = OutboxCommandEntity(
                commandId = command.id!!,
                serverId = "server",
                envelopeJson = ProtocolCodec.encode(command),
                state = "pending",
                attemptCount = 0,
                createdAt = 2,
                lastAttemptAt = null,
            ),
            attachments = listOf(mediaAttachment("attachment")),
        )

        assertEquals("sending", database.attachmentTransfers().get("attachment")!!.state)
        assertEquals(
            listOf("attachment"),
            database.mediaAttachments().forMessage("user:${payload.clientMessageId}").map { it.attachmentId },
        )
        assertEquals(
            listOf("attachment"),
            store.acknowledgeOutbox(command.id, 3),
        )
        assertEquals("sent", database.attachmentTransfers().get("attachment")!!.state)
        assertEquals(
            listOf("attachment"),
            database.mediaAttachments().forMessage("user:${payload.clientMessageId}").map { it.attachmentId },
        )
    }

    @Test
    fun retryDefiniteFailureKeepsVisualMessageAndCreatesNewIdempotencyKey() = runBlocking {
        val oldCommandId = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        val newCommandId = "01ARZ3NDEKTSV4RRFFQ69G5FAW"
        enqueueRetryableMessage(oldCommandId, createdAt = 2_000)
        store.markOutboxAttempt(oldCommandId, attemptedAt = 2_100)
        store.retainFailedOutbox(oldCommandId, outcomeUnknown = false, updatedAt = 2_200)

        assertTrue(store.retryFailedMessage("visual-message", newCommandId, updatedAt = 2_300))
        assertTrue(!store.retryFailedMessage("visual-message", newCommandId, updatedAt = 2_301))

        val message = requireNotNull(database.messages().get("visual-message"))
        val command = requireNotNull(database.outbox().get(newCommandId))
        val payload = ProtocolCodec.decodePayload<MessageSendPayload>(ProtocolCodec.decode(command.envelopeJson).payload)
        assertEquals(newCommandId, message.clientMessageId)
        assertEquals("pending", message.deliveryState)
        assertEquals(newCommandId, payload.clientMessageId)
        assertEquals(2_000, command.createdAt)
        assertEquals(null, database.outbox().get(oldCommandId))
        assertEquals(1, database.messages().countForSession("mobile:test"))
    }

    @Test
    fun verifyUnknownOutcomeReusesOriginalIdempotencyKey() = runBlocking {
        val oldCommandId = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        val unusedCommandId = "01ARZ3NDEKTSV4RRFFQ69G5FAW"
        enqueueRetryableMessage(oldCommandId, createdAt = 2_000)
        store.markOutboxAttempt(oldCommandId, attemptedAt = 2_100)
        store.retainFailedOutbox(oldCommandId, outcomeUnknown = true, updatedAt = 2_200)

        assertTrue(store.retryFailedMessage("visual-message", unusedCommandId, updatedAt = 2_300))
        assertTrue(!store.retryFailedMessage("visual-message", unusedCommandId, updatedAt = 2_301))

        val message = requireNotNull(database.messages().get("visual-message"))
        val command = requireNotNull(database.outbox().get(oldCommandId))
        assertEquals(oldCommandId, message.clientMessageId)
        assertEquals("pending", message.deliveryState)
        assertEquals("retry", command.state)
        assertEquals(null, database.outbox().get(unusedCommandId))
    }

    @Test
    fun failedLocalMessageKeepsChronologicalPositionAmongServerMessages() = runBlocking {
        database.messages().upsert(retryMessage("server-before", null, 1_000, 1))
        database.messages().upsert(retryMessage("failed-local", "failed-command", 2_000, null, "failed_retryable"))
        database.messages().upsert(retryMessage("server-after", null, 3_000, 2))

        val ordered = database.messages().observeMessages("mobile:test").first()

        assertEquals(listOf("server-before", "failed-local", "server-after"), ordered.map { it.messageId })
    }

    @Test
    fun sentAttachmentLinkAndCachePathSurviveDatabaseRestart() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val name = "sent-media-${System.nanoTime()}.db"
        val cachedFile = context.cacheDir.resolve("sent-media-${System.nanoTime()}.bin")
            .apply { writeBytes(byteArrayOf(1, 2, 3)) }
        val messageId = "user:01ARZ3NDEKTSV4RRFFQ69G5FAV"
        val attachment = mediaAttachment("01ARZ3NDEKTSV4RRFFQ69G5FAW").copy(
            sizeBytes = 3,
            transferredBytes = 3,
            cachePath = cachedFile.absolutePath,
        )
        var persisted = Room.databaseBuilder(context, AppDatabase::class.java, name).build()
        try {
            persisted.serverProfiles().upsert(
                ServerProfileEntity("restart", "restart", "restart-device", "alias", "pin", "[]", "[]", "[]", 1),
            )
            persisted.conversations().upsert(ConversationEntity("mobile:restart", "restart", "restart", 1))
            persisted.messages().upsert(
                MessageEntity(messageId, messageId.removePrefix("user:"), "mobile:restart", "user", "附件：image.png", "sent", 1, 1),
            )
            persisted.mediaAttachments().upsert(attachment.copy(serverId = "restart", sessionId = "mobile:restart"))
            persisted.mediaAttachments().linkAll(
                listOf(MessageAttachmentEntity(messageId, attachment.attachmentId, 0)),
            )
            persisted.close()

            persisted = Room.databaseBuilder(context, AppDatabase::class.java, name).build()
            val restored = persisted.mediaAttachments().forMessage(messageId).single()
            assertEquals(cachedFile.absolutePath, restored.cachePath)
            assertEquals(byteArrayOf(1, 2, 3).toList(), cachedFile.readBytes().toList())
        } finally {
            persisted.close()
            context.deleteDatabase(name)
            cachedFile.delete()
        }
    }

    @Test
    fun liveAndHistoryMessagesIdempotentlyShareOutboundAttachment() = runBlocking {
        val descriptor = buildJsonObject {
            put("attachment_id", "01ARZ3NDEKTSV4RRFFQ69G5FAV")
            put("filename", "answer.png")
            put("content_type", "image/png")
            put("size_bytes", 3)
            put("sha256", "a".repeat(64))
        }
        store.applyEvent(
            "server",
            "device",
            event(1, "message.final", buildJsonObject {
                put("message_id", "mobile:test:assistant:attachment")
                put("content", "完成")
                put("attachments", buildJsonArray { add(descriptor) })
            }),
            2,
        )
        store.applyEvent(
            "server",
            "device",
            event(2, "history.page", buildJsonObject {
                put("total", 1)
                put("page", 1)
                put("page_size", 10)
                put("items", buildJsonArray {
                    add(buildJsonObject {
                        put("id", "mobile:test:1")
                        put("session_key", "mobile:test")
                        put("seq", 1)
                        put("role", "assistant")
                        put("content", "完成")
                        put("extra", buildJsonObject {})
                        put("ts", "2026-07-14T16:00:05Z")
                        put("attachments", buildJsonArray { add(descriptor) })
                    })
                })
            }),
            3,
        )

        val attachment = database.mediaAttachments().get("01ARZ3NDEKTSV4RRFFQ69G5FAV")
        assertNotNull(attachment)
        assertEquals("pending", attachment!!.state)
        assertEquals(
            listOf(attachment.attachmentId),
            database.mediaAttachments().forMessage("mobile:test:assistant:attachment").map { it.attachmentId },
        )
        assertEquals(
            listOf(attachment.attachmentId),
            database.mediaAttachments().forMessage("mobile:test:1").map { it.attachmentId },
        )
    }

    @Test
    fun largeMessageAttachmentWaitsForExplicitDownload() = runBlocking {
        val smallId = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        val largeId = "01ARZ3NDEKTSV4RRFFQ69G5FAW"
        fun descriptor(id: String, sizeBytes: Long) = buildJsonObject {
            put("attachment_id", id)
            put("filename", "$id.pdf")
            put("content_type", "application/pdf")
            put("size_bytes", sizeBytes)
            put("sha256", "a".repeat(64))
        }
        store.applyEvent(
            "server",
            "device",
            event(1, "message.final", buildJsonObject {
                put("message_id", "mobile:test:assistant:large-attachment")
                put("content", "附件")
                put("attachments", buildJsonArray {
                    add(descriptor(smallId, 10L * 1024 * 1024 - 1))
                    add(descriptor(largeId, 10L * 1024 * 1024))
                })
            }),
            2,
        )

        assertEquals("pending", database.mediaAttachments().get(smallId)!!.state)
        assertEquals("remote", database.mediaAttachments().get(largeId)!!.state)
        assertEquals(listOf(smallId), database.mediaAttachments().pendingDownloads("server").map { it.attachmentId })

        mediaCache.reconcile()
        assertEquals("remote", database.mediaAttachments().get(largeId)!!.state)

        assertEquals(1, database.mediaAttachments().updateDownload(smallId, 0, "failed", 3))
        val sentTypes = mutableListOf<String>()
        val downloads = AttachmentDownloadCoordinator(
            database.mediaAttachments(),
            mediaCache,
            sendCommand = { type, _, _, _ -> sentTypes += type; true },
            onTransportUnavailable = {},
            onDownloadFailed = {},
        )
        downloads.retry(largeId)
        downloads.retry(largeId)
        downloads.onConnectionReady("server")

        assertEquals(listOf("attachment.download"), sentTypes)
        assertEquals(
            listOf(largeId),
            database.mediaAttachments().pendingDownloads("server").map { it.attachmentId },
        )
    }

    @Test
    fun attachmentReconcileDeletesSentAndOrphanFiles() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val root = context.cacheDir.resolve("attachment-reconcile-${System.nanoTime()}")
        root.mkdirs()
        val dao = database.attachmentTransfers()
        dao.upsert(transfer(state = "sent", offset = 1_048_579))
        dao.upsert(transfer(id = "pending", state = "pending", offset = 0))
        root.resolve("attachment.upload").writeText("sent")
        root.resolve("pending.upload").writeText("pending")
        root.resolve("orphan.upload").writeText("orphan")
        val drafts = AttachmentDraftStore(context.contentResolver, root, dao)

        drafts.reconcile()

        assertEquals(null, dao.get("attachment"))
        assertEquals(false, root.resolve("attachment.upload").exists())
        assertEquals(false, root.resolve("orphan.upload").exists())
        assertEquals(true, root.resolve("pending.upload").exists())
        root.deleteRecursively()
        Unit
    }

    @Test
    fun reliabilityGateOrdersLocalFailureAndTracksDurableReadingState() = runBlocking {
        // 1. 模拟断网消息先于稍后到达的服务端消息创建
        database.messages().upsert(retryMessage("remote-before", null, 1_000, 10))
        database.messages().upsert(retryMessage("local-failed", "client-failed", 2_000, null, "failed_retryable"))
        database.messages().upsert(retryMessage("remote-after", null, 3_000, 11))
        val ordered = database.messages().observeMessages("mobile:test").first()
        assertEquals(listOf("remote-before", "local-failed", "remote-after"), ordered.map { it.messageId })

        // 2. 阅读水位只统计之后完成的助手消息，锚点独立保存
        database.messages().upsert(
            MessageEntity("assistant-old", null, "mobile:test", "assistant", "旧回答", "complete", 4_000, 4_000, 12),
        )
        database.conversationReadStates().markReadThrough("mobile:test", 4_000, 4_100)
        database.conversationReadStates().savePosition("mobile:test", "local-failed", -14, 4_200)
        database.messages().upsert(
            MessageEntity("assistant-new", null, "mobile:test", "assistant", "新回答", "complete", 5_000, 5_000, 13),
        )
        val summary = database.conversations().observeSummaries("server").first().single()
        assertEquals(1, summary.unreadCount)
        assertEquals("新回答", summary.lastMessagePreview)
        assertEquals("local-failed", summary.anchorMessageId)
        assertEquals(-14, summary.anchorOffsetPx)
    }

    @Test
    fun canonicalMergeMigratesOptimisticAndStreamingReadingAnchors() = runBlocking {
        // 1. 历史同步将 optimistic 用户消息与阅读锚点一起迁移
        val clientId = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        val optimisticId = "user:$clientId"
        database.messages().upsert(
            MessageEntity(
                optimisticId,
                clientId,
                "mobile:test",
                "user",
                "本地问题",
                "sent",
                1,
                1,
            ),
        )
        database.conversationReadStates().savePosition("mobile:test", optimisticId, -12, 2)
        store.applyEvent(
            "server",
            "device",
            event(1, "history.page", buildJsonObject {
                put("total", 1)
                put("page", 1)
                put("page_size", 10)
                put("items", buildJsonArray {
                    add(buildJsonObject {
                        put("id", "mobile:test:user:canonical")
                        put("client_message_id", clientId)
                        put("session_key", "mobile:test")
                        put("seq", 0)
                        put("role", "user")
                        put("content", "本地问题")
                        put("extra", buildJsonObject {})
                        put("ts", "2026-07-14T16:00:00Z")
                    })
                })
            }),
            3,
        )
        var summary = database.conversations().observeSummaries("server").first().single()
        assertEquals("mobile:test:user:canonical", summary.anchorMessageId)
        assertEquals(-12, summary.anchorOffsetPx)

        // 2. 最终事件将 streaming 助手消息与阅读锚点一起迁移
        store.applyEvent("server", "device", event(2, "turn.started", buildJsonObject {}), 4)
        database.conversationReadStates().savePosition("mobile:test", "assistant:turn", -20, 5)
        store.applyEvent(
            "server",
            "device",
            event(3, "message.final", buildJsonObject {
                put("message_id", "mobile:test:assistant:canonical")
                put("content", "最终回答")
            }),
            6,
        )
        summary = database.conversations().observeSummaries("server").first().single()
        assertEquals("mobile:test:assistant:canonical", summary.anchorMessageId)
        assertEquals(-20, summary.anchorOffsetPx)
    }

    private fun transfer(state: String, offset: Long, id: String = "attachment") = AttachmentTransferEntity(
        attachmentId = id,
        serverId = "server",
        sessionId = "mobile:test",
        filename = "image.png",
        contentType = "image/png",
        sizeBytes = 1_048_579,
        sha256 = "a".repeat(64),
        transferredBytes = offset,
        state = state,
        updatedAt = 1,
    )

    private fun mediaAttachment(id: String) = MediaAttachmentEntity(
        attachmentId = id,
        serverId = "server",
        sessionId = "mobile:test",
        filename = "image.png",
        contentType = "image/png",
        sizeBytes = 1_048_579,
        sha256 = "a".repeat(64),
        transferredBytes = 1_048_579,
        state = "cached",
        cachePath = "cache/$id.bin",
        lastAccessedAt = 1,
        updatedAt = 1,
    )

    private suspend fun enqueueRetryableMessage(commandId: String, createdAt: Long) {
        val payload = MessageSendPayload(
            clientMessageId = commandId,
            sessionId = "mobile:test",
            text = "需要恢复的消息",
            mediaRefs = emptyList(),
            clientCreatedAt = "2026-07-16T08:00:00Z",
        )
        val envelope = WireEnvelope(
            v = 1,
            kind = WireKind.COMMAND,
            type = "message.send",
            id = commandId,
            connectionEpoch = 1,
            sessionId = "mobile:test",
            payload = ProtocolCodec.json().encodeToJsonElement(MessageSendPayload.serializer(), payload).jsonObject,
        )
        store.enqueueMessage(
            ConversationEntity("mobile:test", "server", "test", 1),
            retryMessage("visual-message", commandId, createdAt, null, "pending"),
            OutboxCommandEntity(commandId, "server", ProtocolCodec.encode(envelope), "pending", 0, createdAt, null),
            emptyList(),
        )
    }

    private fun retryMessage(
        messageId: String,
        clientMessageId: String?,
        createdAt: Long,
        serverSeq: Long?,
        state: String = "complete",
    ) = MessageEntity(
        messageId = messageId,
        clientMessageId = clientMessageId,
        sessionId = "mobile:test",
        role = "user",
        text = messageId,
        deliveryState = state,
        createdAt = createdAt,
        updatedAt = createdAt,
        serverSeq = serverSeq,
    )

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
