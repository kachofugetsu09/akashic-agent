package com.akashic.mobile.data.local

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Transaction
import androidx.room.Upsert
import kotlinx.coroutines.flow.Flow

@Dao
interface ServerProfileDao {
    @Upsert
    suspend fun upsert(profile: ServerProfileEntity)

    @Query("SELECT * FROM server_profiles WHERE serverId = :serverId")
    suspend fun get(serverId: String): ServerProfileEntity?

    @Query("SELECT * FROM server_profiles ORDER BY createdAt LIMIT 1")
    suspend fun first(): ServerProfileEntity?

    @Query("SELECT * FROM server_profiles ORDER BY createdAt")
    fun observeAll(): Flow<List<ServerProfileEntity>>

    @Query("DELETE FROM server_profiles WHERE serverId = :serverId")
    suspend fun delete(serverId: String): Int
}

@Dao
interface ConversationDao {
    @Upsert
    suspend fun upsert(conversation: ConversationEntity)

    @Query("SELECT * FROM conversations WHERE serverId = :serverId ORDER BY updatedAt DESC")
    fun observeForServer(serverId: String): Flow<List<ConversationEntity>>

    @Query("SELECT * FROM conversations WHERE sessionId = :sessionId")
    suspend fun get(sessionId: String): ConversationEntity?
}

@Dao
interface MessageDao {
    @Upsert
    suspend fun upsert(message: MessageEntity)

    @Upsert
    suspend fun upsertBlocks(blocks: List<TurnBlockEntity>)

    @Query("SELECT * FROM messages WHERE sessionId = :sessionId ORDER BY createdAt, messageId")
    fun observeMessages(sessionId: String): Flow<List<MessageEntity>>

    @Query("SELECT * FROM turn_blocks WHERE messageId = :messageId ORDER BY ordinal")
    fun observeBlocks(messageId: String): Flow<List<TurnBlockEntity>>

    @Query("SELECT * FROM turn_blocks WHERE messageId = :messageId ORDER BY ordinal")
    suspend fun getBlocks(messageId: String): List<TurnBlockEntity>

    @Query("SELECT * FROM messages WHERE messageId = :messageId")
    suspend fun get(messageId: String): MessageEntity?

    @Query("SELECT * FROM turn_blocks WHERE blockId = :blockId")
    suspend fun getBlock(blockId: String): TurnBlockEntity?

    @Query("UPDATE messages SET deliveryState = :state, updatedAt = :updatedAt WHERE clientMessageId = :clientMessageId")
    suspend fun updateDelivery(clientMessageId: String, state: String, updatedAt: Long): Int

    @Query("UPDATE turn_blocks SET status = 'completed', updatedAt = :updatedAt WHERE messageId = :messageId AND status = 'running'")
    suspend fun completeRunningBlocks(messageId: String, updatedAt: Long): Int

    @Query(
        "UPDATE turn_blocks SET status = 'completed', updatedAt = :updatedAt " +
            "WHERE messageId = :messageId AND kind = 'thinking' AND status = 'running'",
    )
    suspend fun completeRunningThinking(messageId: String, updatedAt: Long): Int

    @Transaction
    @Query("SELECT * FROM messages WHERE sessionId = :sessionId ORDER BY createdAt, messageId")
    fun observeMessageGraph(sessionId: String): Flow<List<MessageWithBlocks>>
}

@Dao
interface OutboxDao {
    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun enqueue(command: OutboxCommandEntity)

    @Query(
        """
        SELECT * FROM outbox_commands
        WHERE serverId = :serverId AND state IN ('pending', 'retry')
        ORDER BY createdAt, commandId
        """,
    )
    suspend fun pending(serverId: String): List<OutboxCommandEntity>

    @Query(
        """
        UPDATE outbox_commands
        SET state = 'in_flight', attemptCount = attemptCount + 1, lastAttemptAt = :attemptedAt
        WHERE commandId = :commandId AND state IN ('pending', 'retry')
        """,
    )
    suspend fun markInFlight(commandId: String, attemptedAt: Long): Int

    @Query("UPDATE outbox_commands SET state = 'retry' WHERE commandId = :commandId AND state = 'in_flight'")
    suspend fun markForRetry(commandId: String): Int

    @Query("DELETE FROM outbox_commands WHERE commandId = :commandId")
    suspend fun deleteAcknowledged(commandId: String): Int

    @Query("SELECT * FROM outbox_commands WHERE commandId = :commandId")
    suspend fun get(commandId: String): OutboxCommandEntity?

    @Query("UPDATE outbox_commands SET state = 'retry' WHERE serverId = :serverId AND state = 'in_flight'")
    suspend fun resetInFlight(serverId: String): Int

    @Query("SELECT COUNT(*) FROM outbox_commands WHERE serverId = :serverId")
    fun observeCount(serverId: String): Flow<Int>
}

@Dao
interface AttachmentTransferDao {
    @Upsert
    suspend fun upsert(transfer: AttachmentTransferEntity)

    @Query("SELECT * FROM attachment_transfers WHERE attachmentId = :attachmentId")
    suspend fun get(attachmentId: String): AttachmentTransferEntity?
}

@Dao
interface RealtimeCursorDao {
    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun insert(cursor: RealtimeCursorEntity)

    @Query("SELECT * FROM realtime_cursors WHERE deviceId = :deviceId")
    suspend fun get(deviceId: String): RealtimeCursorEntity?

    @Upsert
    suspend fun upsert(cursor: RealtimeCursorEntity)

    @Query(
        """
        UPDATE realtime_cursors
        SET lastAcknowledgedEventSeq = :throughEventSeq,
            connectionEpoch = :connectionEpoch,
            updatedAt = :updatedAt
        WHERE deviceId = :deviceId
          AND lastAcknowledgedEventSeq <= :throughEventSeq
          AND connectionEpoch <= :connectionEpoch
        """,
    )
    suspend fun advance(
        deviceId: String,
        throughEventSeq: Long,
        connectionEpoch: Long,
        updatedAt: Long,
    ): Int
}
