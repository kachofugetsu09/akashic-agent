package com.akashic.mobile.data.local

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(
    entities = [
        ServerProfileEntity::class,
        ConversationEntity::class,
        MessageEntity::class,
        TurnBlockEntity::class,
        OutboxCommandEntity::class,
        AttachmentTransferEntity::class,
        RealtimeCursorEntity::class,
        MediaAttachmentEntity::class,
        MessageAttachmentEntity::class,
        ConversationReadStateEntity::class,
        ComposerDraftEntity::class,
        PendingMessageNotificationEntity::class,
    ],
    version = 8,
    exportSchema = true,
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun serverProfiles(): ServerProfileDao

    abstract fun conversations(): ConversationDao

    abstract fun conversationReadStates(): ConversationReadStateDao

    abstract fun composerDrafts(): ComposerDraftDao

    abstract fun messages(): MessageDao

    abstract fun outbox(): OutboxDao

    abstract fun attachmentTransfers(): AttachmentTransferDao

    abstract fun realtimeCursors(): RealtimeCursorDao

    abstract fun mediaAttachments(): MediaAttachmentDao

    abstract fun pendingMessageNotifications(): PendingMessageNotificationDao

    companion object {
        fun create(context: Context): AppDatabase = Room.databaseBuilder(
            context.applicationContext,
            AppDatabase::class.java,
            "akashic-mobile.db",
        ).addMigrations(
            MIGRATION_1_2,
            MIGRATION_2_3,
            MIGRATION_3_4,
            MIGRATION_4_5,
            MIGRATION_5_6,
            MIGRATION_6_7,
            MIGRATION_7_8,
        ).build()

        val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `media_attachments` (
                        `attachmentId` TEXT NOT NULL,
                        `serverId` TEXT NOT NULL,
                        `sessionId` TEXT NOT NULL,
                        `filename` TEXT NOT NULL,
                        `contentType` TEXT NOT NULL,
                        `sizeBytes` INTEGER NOT NULL,
                        `sha256` TEXT NOT NULL,
                        `transferredBytes` INTEGER NOT NULL,
                        `state` TEXT NOT NULL,
                        `cachePath` TEXT NOT NULL,
                        `lastAccessedAt` INTEGER NOT NULL,
                        `updatedAt` INTEGER NOT NULL,
                        PRIMARY KEY(`attachmentId`),
                        FOREIGN KEY(`serverId`) REFERENCES `server_profiles`(`serverId`) ON UPDATE NO ACTION ON DELETE CASCADE,
                        FOREIGN KEY(`sessionId`) REFERENCES `conversations`(`sessionId`) ON UPDATE NO ACTION ON DELETE CASCADE
                    )
                    """.trimIndent(),
                )
                db.execSQL("CREATE INDEX IF NOT EXISTS `index_media_attachments_serverId` ON `media_attachments` (`serverId`)")
                db.execSQL("CREATE INDEX IF NOT EXISTS `index_media_attachments_sessionId` ON `media_attachments` (`sessionId`)")
                db.execSQL("CREATE INDEX IF NOT EXISTS `index_media_attachments_state` ON `media_attachments` (`state`)")
                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `message_attachments` (
                        `messageId` TEXT NOT NULL,
                        `attachmentId` TEXT NOT NULL,
                        `ordinal` INTEGER NOT NULL,
                        PRIMARY KEY(`messageId`, `attachmentId`),
                        FOREIGN KEY(`messageId`) REFERENCES `messages`(`messageId`) ON UPDATE NO ACTION ON DELETE CASCADE,
                        FOREIGN KEY(`attachmentId`) REFERENCES `media_attachments`(`attachmentId`) ON UPDATE NO ACTION ON DELETE CASCADE
                    )
                    """.trimIndent(),
                )
                db.execSQL("CREATE INDEX IF NOT EXISTS `index_message_attachments_attachmentId` ON `message_attachments` (`attachmentId`)")
                db.execSQL("CREATE UNIQUE INDEX IF NOT EXISTS `index_message_attachments_messageId_ordinal` ON `message_attachments` (`messageId`, `ordinal`)")
            }
        }

        val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE `messages` ADD COLUMN `serverSeq` INTEGER")
            }
        }

        val MIGRATION_3_4 = object : Migration(3, 4) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE `messages` ADD COLUMN `replyToMessageId` TEXT")
                db.execSQL("ALTER TABLE `messages` ADD COLUMN `replyRole` TEXT")
                db.execSQL("ALTER TABLE `messages` ADD COLUMN `replyPreview` TEXT")
            }
        }

        val MIGRATION_4_5 = object : Migration(4, 5) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `conversation_read_states` (
                        `sessionId` TEXT NOT NULL,
                        `lastReadAt` INTEGER NOT NULL,
                        `anchorMessageId` TEXT,
                        `anchorOffsetPx` INTEGER NOT NULL,
                        `updatedAt` INTEGER NOT NULL,
                        PRIMARY KEY(`sessionId`),
                        FOREIGN KEY(`sessionId`) REFERENCES `conversations`(`sessionId`) ON UPDATE NO ACTION ON DELETE CASCADE
                    )
                    """.trimIndent(),
                )
            }
        }

        val MIGRATION_5_6 = object : Migration(5, 6) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    "ALTER TABLE `conversations` ADD COLUMN `remoteKnown` INTEGER NOT NULL DEFAULT 0",
                )
                db.execSQL(
                    """
                    UPDATE conversations
                    SET remoteKnown = 1
                    WHERE EXISTS (
                        SELECT 1 FROM messages
                        WHERE messages.sessionId = conversations.sessionId
                          AND (
                              messages.serverSeq IS NOT NULL
                              OR (
                                  messages.role = 'assistant'
                                  AND messages.deliveryState IN ('streaming', 'complete', 'interrupted')
                              )
                              OR (
                                  messages.role = 'user'
                                  AND messages.deliveryState = 'sent'
                              )
                          )
                    )
                    """.trimIndent(),
                )
            }
        }

        val MIGRATION_6_7 = object : Migration(6, 7) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `composer_drafts` (
                        `sessionId` TEXT NOT NULL,
                        `serverId` TEXT NOT NULL,
                        `text` TEXT NOT NULL,
                        `replyToMessageId` TEXT,
                        `updatedAt` INTEGER NOT NULL,
                        PRIMARY KEY(`sessionId`),
                        FOREIGN KEY(`serverId`) REFERENCES `server_profiles`(`serverId`) ON UPDATE NO ACTION ON DELETE CASCADE,
                        FOREIGN KEY(`sessionId`) REFERENCES `conversations`(`sessionId`) ON UPDATE NO ACTION ON DELETE CASCADE
                    )
                    """.trimIndent(),
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS `index_composer_drafts_serverId` ON `composer_drafts` (`serverId`)",
                )
            }
        }

        val MIGRATION_7_8 = object : Migration(7, 8) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `pending_message_notifications` (
                        `messageId` TEXT NOT NULL,
                        `serverId` TEXT NOT NULL,
                        `sessionId` TEXT NOT NULL,
                        `content` TEXT NOT NULL,
                        `hasAttachments` INTEGER NOT NULL,
                        `attention` TEXT NOT NULL,
                        `createdAt` INTEGER NOT NULL,
                        PRIMARY KEY(`messageId`),
                        FOREIGN KEY(`serverId`) REFERENCES `server_profiles`(`serverId`) ON UPDATE NO ACTION ON DELETE CASCADE,
                        FOREIGN KEY(`messageId`) REFERENCES `messages`(`messageId`) ON UPDATE NO ACTION ON DELETE CASCADE
                    )
                    """.trimIndent(),
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS `index_pending_message_notifications_serverId` ON `pending_message_notifications` (`serverId`)",
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS `index_pending_message_notifications_createdAt` ON `pending_message_notifications` (`createdAt`)",
                )
            }
        }
    }
}
