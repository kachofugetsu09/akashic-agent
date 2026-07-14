package com.akashic.mobile.data.local

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(
    entities = [
        ServerProfileEntity::class,
        ConversationEntity::class,
        MessageEntity::class,
        TurnBlockEntity::class,
        OutboxCommandEntity::class,
        AttachmentTransferEntity::class,
        RealtimeCursorEntity::class,
    ],
    version = 1,
    exportSchema = true,
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun serverProfiles(): ServerProfileDao

    abstract fun conversations(): ConversationDao

    abstract fun messages(): MessageDao

    abstract fun outbox(): OutboxDao

    abstract fun attachmentTransfers(): AttachmentTransferDao

    abstract fun realtimeCursors(): RealtimeCursorDao

    companion object {
        fun create(context: Context): AppDatabase = Room.databaseBuilder(
            context.applicationContext,
            AppDatabase::class.java,
            "akashic-mobile.db",
        ).build()
    }
}
