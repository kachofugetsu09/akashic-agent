package com.akashic.mobile.data.local

import androidx.room.testing.MigrationTestHelper
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class AppDatabaseMigrationTest {
    @get:Rule
    val helper = MigrationTestHelper(
        InstrumentationRegistry.getInstrumentation(),
        AppDatabase::class.java,
    )

    @After
    fun removeTestDatabases() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        listOf(
            DATABASE_1_2,
            DATABASE_2_3,
            DATABASE_3_4,
            DATABASE_4_5,
            DATABASE_5_6,
            DATABASE_6_7,
            DATABASE_7_8,
        )
            .forEach(context::deleteDatabase)
    }

    @Test
    fun migrate1To2CreatesMediaCacheTables() {
        helper.createDatabase(DATABASE_1_2, 1).apply {
            execSQL(
                """
                INSERT INTO server_profiles VALUES(
                    'server', '电脑', 'device', 'alias', 'pin', '[]', '[]', '[]', 1
                )
                """.trimIndent(),
            )
            execSQL("INSERT INTO conversations VALUES('mobile:test', 'server', '旧会话', 2)")
            execSQL(
                """
                INSERT INTO messages VALUES(
                    'old-message', NULL, 'mobile:test', 'assistant', '旧消息', 'complete', 3, 3
                )
                """.trimIndent(),
            )
            close()
        }

        helper.runMigrationsAndValidate(
            DATABASE_1_2,
            2,
            true,
            AppDatabase.MIGRATION_1_2,
        ).use { database ->
            database.query("SELECT text FROM messages WHERE messageId = 'old-message'").use { cursor ->
                check(cursor.moveToFirst())
                assertEquals("旧消息", cursor.getString(0))
            }
            database.query("SELECT COUNT(*) FROM media_attachments").use { cursor ->
                check(cursor.moveToFirst())
                assertEquals(0, cursor.getInt(0))
            }
        }
    }

    @Test
    fun migrate2To3AddsServerSequenceWithoutLosingMessages() {
        helper.createDatabase(DATABASE_2_3, 2).apply {
            execSQL(
                "INSERT INTO server_profiles VALUES('server', '电脑', 'device', 'alias', 'pin', '[]', '[]', '[]', 1)",
            )
            execSQL("INSERT INTO conversations VALUES('mobile:test', 'server', '旧会话', 2)")
            execSQL(
                "INSERT INTO messages VALUES('old-message', NULL, 'mobile:test', 'assistant', '旧消息', 'complete', 3, 4)",
            )
            close()
        }

        helper.runMigrationsAndValidate(
            DATABASE_2_3,
            3,
            true,
            AppDatabase.MIGRATION_2_3,
        ).use { database ->
            database.query("SELECT text, serverSeq FROM messages WHERE messageId = 'old-message'").use { cursor ->
                check(cursor.moveToFirst())
                assertEquals("旧消息", cursor.getString(0))
                assertEquals(true, cursor.isNull(1))
            }
        }
    }

    @Test
    fun migrate3To4AddsReplyProjectionWithoutDroppingMessages() {
        helper.createDatabase(DATABASE_3_4, 3).apply {
            execSQL(
                """
                INSERT INTO messages(
                    messageId, clientMessageId, sessionId, role, text, deliveryState,
                    createdAt, updatedAt, serverSeq
                ) VALUES ('message-1', NULL, 'mobile:test', 'assistant', '保留我', 'complete', 1, 1, 1)
                """.trimIndent(),
            )
            close()
        }

        helper.runMigrationsAndValidate(
            DATABASE_3_4,
            4,
            true,
            AppDatabase.MIGRATION_3_4,
        ).use { database ->
            database.query(
                "SELECT text, replyToMessageId, replyRole, replyPreview FROM messages WHERE messageId = 'message-1'",
            ).use { cursor ->
                check(cursor.moveToFirst()) { "迁移后消息丢失" }
                assertEquals("保留我", cursor.getString(0))
                assertEquals(null, cursor.getString(1))
                assertEquals(null, cursor.getString(2))
                assertEquals(null, cursor.getString(3))
            }
        }
    }

    @Test
    fun migrate4To5AddsConversationReadingState() {
        helper.createDatabase(DATABASE_4_5, 4).apply {
            execSQL(
                "INSERT INTO server_profiles VALUES('server', '电脑', 'device', 'alias', 'pin', '[]', '[]', '[]', 1)",
            )
            execSQL("INSERT INTO conversations VALUES('mobile:test', 'server', '旧会话', 2)")
            close()
        }

        helper.runMigrationsAndValidate(
            DATABASE_4_5,
            5,
            true,
            AppDatabase.MIGRATION_4_5,
        ).use { database ->
            database.execSQL(
                "INSERT INTO conversation_read_states VALUES('mobile:test', 3, 'message-1', -12, 4)",
            )
            database.query(
                "SELECT lastReadAt, anchorMessageId, anchorOffsetPx FROM conversation_read_states",
            ).use { cursor ->
                check(cursor.moveToFirst())
                assertEquals(3L, cursor.getLong(0))
                assertEquals("message-1", cursor.getString(1))
                assertEquals(-12, cursor.getInt(2))
            }
        }
    }

    @Test
    fun migrate5To6PersistsKnownRemoteConversationIdentity() {
        helper.createDatabase(DATABASE_5_6, 5).apply {
            execSQL(
                "INSERT INTO server_profiles VALUES('server', '电脑', 'device', 'alias', 'pin', '[]', '[]', '[]', 1)",
            )
            execSQL("INSERT INTO conversations VALUES('mobile:remote', 'server', '远端会话', 2)")
            execSQL("INSERT INTO conversations VALUES('mobile:live', 'server', '实时会话', 3)")
            execSQL("INSERT INTO conversations VALUES('mobile:local', 'server', '本机会话', 3)")
            execSQL("INSERT INTO conversations VALUES('mobile:failed', 'server', '失败草稿', 3)")
            execSQL(
                """
                INSERT INTO messages(
                    messageId, clientMessageId, sessionId, role, text, deliveryState,
                    createdAt, updatedAt, serverSeq, replyToMessageId, replyRole, replyPreview
                ) VALUES(
                    'remote-message', NULL, 'mobile:remote', 'assistant', '旧消息', 'complete',
                    4, 4, 1, NULL, NULL, NULL
                )
                """.trimIndent(),
            )
            execSQL(
                """
                INSERT INTO messages(
                    messageId, clientMessageId, sessionId, role, text, deliveryState,
                    createdAt, updatedAt, serverSeq, replyToMessageId, replyRole, replyPreview
                ) VALUES(
                    'live-final', NULL, 'mobile:live', 'assistant', '刚完成的回答', 'complete',
                    5, 5, NULL, NULL, NULL, NULL
                )
                """.trimIndent(),
            )
            execSQL(
                """
                INSERT INTO messages(
                    messageId, clientMessageId, sessionId, role, text, deliveryState,
                    createdAt, updatedAt, serverSeq, replyToMessageId, replyRole, replyPreview
                ) VALUES(
                    'local-pending', 'local-client', 'mobile:local', 'user', '还没发送', 'pending',
                    6, 6, NULL, NULL, NULL, NULL
                )
                """.trimIndent(),
            )
            execSQL(
                """
                INSERT INTO messages(
                    messageId, clientMessageId, sessionId, role, text, deliveryState,
                    createdAt, updatedAt, serverSeq, replyToMessageId, replyRole, replyPreview
                ) VALUES(
                    'failed-assistant', NULL, 'mobile:failed', 'assistant', '', 'failed',
                    7, 7, NULL, NULL, NULL, NULL
                )
                """.trimIndent(),
            )
            close()
        }

        helper.runMigrationsAndValidate(
            DATABASE_5_6,
            6,
            true,
            AppDatabase.MIGRATION_5_6,
        ).use { database ->
            database.query("SELECT sessionId, remoteKnown FROM conversations ORDER BY sessionId").use { cursor ->
                check(cursor.moveToFirst())
                assertEquals("mobile:failed", cursor.getString(0))
                assertEquals(0, cursor.getInt(1))
                check(cursor.moveToNext())
                assertEquals("mobile:live", cursor.getString(0))
                assertEquals(1, cursor.getInt(1))
                check(cursor.moveToNext())
                assertEquals("mobile:local", cursor.getString(0))
                assertEquals(0, cursor.getInt(1))
                check(cursor.moveToNext())
                assertEquals("mobile:remote", cursor.getString(0))
                assertEquals(1, cursor.getInt(1))
            }
        }
    }

    @Test
    fun migrate6To7CreatesConversationOwnedComposerDrafts() {
        helper.createDatabase(DATABASE_6_7, 6).apply {
            execSQL(
                "INSERT INTO server_profiles VALUES('server', '电脑', 'device', 'alias', 'pin', '[]', '[]', '[]', 1)",
            )
            execSQL("INSERT INTO conversations VALUES('mobile:test', 'server', '旧会话', 2, 1)")
            close()
        }

        helper.runMigrationsAndValidate(
            DATABASE_6_7,
            7,
            true,
            AppDatabase.MIGRATION_6_7,
        ).use { database ->
            database.execSQL(
                """
                INSERT INTO composer_drafts(
                    sessionId, serverId, text, replyToMessageId, updatedAt
                ) VALUES('mobile:test', 'server', '保留草稿', 'missing-message', 3)
                """.trimIndent(),
            )
            database.query(
                "SELECT text, replyToMessageId, updatedAt FROM composer_drafts WHERE sessionId = 'mobile:test'",
            ).use { cursor ->
                check(cursor.moveToFirst())
                assertEquals("保留草稿", cursor.getString(0))
                assertEquals("missing-message", cursor.getString(1))
                assertEquals(3L, cursor.getLong(2))
            }

        }
    }

    @Test
    fun migrate7To8CreatesDurableMessageNotificationQueue() {
        helper.createDatabase(DATABASE_7_8, 7).apply {
            execSQL(
                "INSERT INTO server_profiles VALUES('server', '电脑', 'device', 'alias', 'pin', '[]', '[]', '[]', 1)",
            )
            execSQL("INSERT INTO conversations VALUES('mobile:test', 'server', '旧会话', 2, 1)")
            execSQL(
                "INSERT INTO messages VALUES('message-1', NULL, 'mobile:test', 'assistant', '完成', 'complete', 3, 3, NULL, NULL, NULL, NULL)",
            )
            close()
        }

        helper.runMigrationsAndValidate(
            DATABASE_7_8,
            8,
            true,
            AppDatabase.MIGRATION_7_8,
        ).use { database ->
            database.execSQL(
                "INSERT INTO pending_message_notifications VALUES('message-1', 'server', 'mobile:test', '完成', 0, 'COMPLETE', 4)",
            )
            database.query(
                "SELECT content, attention FROM pending_message_notifications WHERE messageId = 'message-1'",
            ).use { cursor ->
                check(cursor.moveToFirst())
                assertEquals("完成", cursor.getString(0))
                assertEquals("COMPLETE", cursor.getString(1))
            }
        }
    }

    private companion object {
        const val DATABASE_1_2 = "migration-1-2"
        const val DATABASE_2_3 = "migration-2-3"
        const val DATABASE_3_4 = "migration-3-4"
        const val DATABASE_4_5 = "migration-4-5"
        const val DATABASE_5_6 = "migration-5-6"
        const val DATABASE_6_7 = "migration-6-7"
        const val DATABASE_7_8 = "migration-7-8"
    }
}
