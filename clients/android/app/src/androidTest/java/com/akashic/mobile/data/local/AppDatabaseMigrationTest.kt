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
        listOf(DATABASE_1_2, DATABASE_2_3, DATABASE_3_4).forEach(context::deleteDatabase)
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

    private companion object {
        const val DATABASE_1_2 = "migration-1-2"
        const val DATABASE_2_3 = "migration-2-3"
        const val DATABASE_3_4 = "migration-3-4"
    }
}
