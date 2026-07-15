package com.akashic.mobile.data.local

import androidx.room.testing.MigrationTestHelper
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.Assert.assertEquals

@RunWith(AndroidJUnit4::class)
class AppDatabaseMigrationTest {
    @get:Rule
    val helper = MigrationTestHelper(
        InstrumentationRegistry.getInstrumentation(),
        AppDatabase::class.java,
    )

    @Test
    fun migrate1To2CreatesMediaCacheTables() {
        helper.createDatabase(DATABASE_NAME, 1).apply {
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

        helper.runMigrationsAndValidate(DATABASE_NAME, 2, true, AppDatabase.MIGRATION_1_2).apply {
            query("SELECT text FROM messages WHERE messageId = 'old-message'").use { cursor ->
                check(cursor.moveToFirst())
                assertEquals("旧消息", cursor.getString(0))
            }
            query("SELECT COUNT(*) FROM media_attachments").use { cursor ->
                check(cursor.moveToFirst())
                assertEquals(0, cursor.getInt(0))
            }
            close()
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

        helper.runMigrationsAndValidate(DATABASE_2_3, 3, true, AppDatabase.MIGRATION_2_3).apply {
            query("SELECT text, serverSeq FROM messages WHERE messageId = 'old-message'").use { cursor ->
                check(cursor.moveToFirst())
                assertEquals("旧消息", cursor.getString(0))
                assertEquals(true, cursor.isNull(1))
            }
            close()
        }
    }

    private companion object {
        const val DATABASE_NAME = "migration-1-2"
        const val DATABASE_2_3 = "migration-2-3"
    }
}
