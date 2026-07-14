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

    @Query("SELECT COUNT(*) FROM messages WHERE sessionId = :sessionId")
    suspend fun countForSession(sessionId: String): Int

    @Query("SELECT * FROM turn_blocks WHERE blockId = :blockId")
    suspend fun getBlock(blockId: String): TurnBlockEntity?

    @Query("DELETE FROM turn_blocks WHERE messageId = :messageId")
    suspend fun deleteBlocks(messageId: String): Int

    @Query("UPDATE turn_blocks SET messageId = :targetId WHERE messageId = :sourceId")
    suspend fun moveBlocks(sourceId: String, targetId: String): Int

    @Query("DELETE FROM messages WHERE messageId = :messageId")
    suspend fun delete(messageId: String): Int

    @Query("UPDATE messages SET clientMessageId = NULL WHERE messageId = :messageId")
    suspend fun clearClientMessageId(messageId: String): Int

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

    @Upsert
    suspend fun upsertAll(transfers: List<AttachmentTransferEntity>)

    @Query("SELECT * FROM attachment_transfers WHERE attachmentId = :attachmentId")
    suspend fun get(attachmentId: String): AttachmentTransferEntity?

    @Query("SELECT * FROM attachment_transfers")
    suspend fun all(): List<AttachmentTransferEntity>

    @Query(
        """
        SELECT * FROM attachment_transfers
        WHERE serverId = :serverId AND sessionId = :sessionId
          AND state IN ('pending', 'uploading', 'finishing', 'ready', 'failed')
        ORDER BY updatedAt ASC
        """,
    )
    fun observeDrafts(serverId: String, sessionId: String): Flow<List<AttachmentTransferEntity>>

    @Query(
        """
        SELECT * FROM attachment_transfers
        WHERE serverId = :serverId AND sessionId = :sessionId
          AND state IN ('pending', 'uploading', 'finishing', 'ready', 'failed')
        ORDER BY updatedAt ASC
        """,
    )
    suspend fun drafts(serverId: String, sessionId: String): List<AttachmentTransferEntity>

    @Query(
        """
        SELECT * FROM attachment_transfers
        WHERE serverId = :serverId AND state IN ('pending', 'uploading', 'finishing')
        ORDER BY updatedAt ASC
        """,
    )
    suspend fun pendingUploads(serverId: String): List<AttachmentTransferEntity>

    @Query(
        """
        UPDATE attachment_transfers
        SET state = 'uploading', updatedAt = :updatedAt
        WHERE attachmentId = :attachmentId AND state IN ('pending', 'uploading', 'finishing')
        """,
    )
    suspend fun claimUploading(attachmentId: String, updatedAt: Long): Int

    @Query(
        """
        UPDATE attachment_transfers
        SET transferredBytes = :transferredBytes, state = :state, updatedAt = :updatedAt
        WHERE attachmentId = :attachmentId
        """,
    )
    suspend fun updateState(
        attachmentId: String,
        transferredBytes: Long,
        state: String,
        updatedAt: Long,
    ): Int

    @Query("UPDATE attachment_transfers SET state = 'sent', updatedAt = :updatedAt WHERE attachmentId IN (:ids)")
    suspend fun markSent(ids: List<String>, updatedAt: Long): Int

    @Query("UPDATE attachment_transfers SET state = 'sending', updatedAt = :updatedAt WHERE attachmentId IN (:ids) AND state = 'ready'")
    suspend fun markSending(ids: List<String>, updatedAt: Long): Int

    @Query("UPDATE attachment_transfers SET state = 'ready', updatedAt = :updatedAt WHERE attachmentId IN (:ids) AND state = 'sending'")
    suspend fun restoreReady(ids: List<String>, updatedAt: Long): Int

    @Query(
        """
        DELETE FROM attachment_transfers
        WHERE attachmentId = :attachmentId AND state IN ('pending', 'ready', 'failed')
        """,
    )
    suspend fun deleteDraft(attachmentId: String): Int

    @Query("DELETE FROM attachment_transfers WHERE state = 'sent'")
    suspend fun deleteSent(): Int
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

@Dao
interface MediaAttachmentDao {
    @Upsert
    suspend fun upsert(attachment: MediaAttachmentEntity)

    @Upsert
    suspend fun upsertAll(attachments: List<MediaAttachmentEntity>)

    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun linkAll(links: List<MessageAttachmentEntity>): List<Long>

    @Query("DELETE FROM message_attachments WHERE messageId = :messageId")
    suspend fun deleteLinks(messageId: String): Int

    @Query("UPDATE message_attachments SET messageId = :targetId WHERE messageId = :sourceId")
    suspend fun moveLinks(sourceId: String, targetId: String): Int

    @Query("SELECT * FROM media_attachments WHERE attachmentId = :attachmentId")
    suspend fun get(attachmentId: String): MediaAttachmentEntity?

    @Query(
        """
        SELECT * FROM media_attachments
        WHERE serverId = :serverId AND state IN ('pending', 'downloading')
        ORDER BY updatedAt, attachmentId
        """,
    )
    suspend fun pendingDownloads(serverId: String): List<MediaAttachmentEntity>

    @Query(
        """
        UPDATE media_attachments
        SET transferredBytes = :transferredBytes, state = :state, updatedAt = :updatedAt
        WHERE attachmentId = :attachmentId
        """,
    )
    suspend fun updateDownload(
        attachmentId: String,
        transferredBytes: Long,
        state: String,
        updatedAt: Long,
    ): Int

    @Query(
        """
        UPDATE media_attachments
        SET state = 'pending', updatedAt = :updatedAt
        WHERE attachmentId = :attachmentId AND state IN ('failed', 'evicted')
        """,
    )
    suspend fun requestDownload(attachmentId: String, updatedAt: Long): Int

    @Query("UPDATE media_attachments SET lastAccessedAt = :accessedAt WHERE attachmentId = :attachmentId AND state = 'cached'")
    suspend fun touch(attachmentId: String, accessedAt: Long): Int

    @Query(
        """
        UPDATE media_attachments
        SET transferredBytes = sizeBytes, state = 'cached',
            lastAccessedAt = :updatedAt, updatedAt = :updatedAt
        WHERE attachmentId = :attachmentId
          AND state IN ('pending', 'downloading', 'failed', 'cached')
        """,
    )
    suspend fun markCached(attachmentId: String, updatedAt: Long): Int

    @Query(
        """
        UPDATE media_attachments
        SET transferredBytes = 0, state = 'evicted', updatedAt = :updatedAt
        WHERE attachmentId = :attachmentId
          AND state IN ('cached', 'failed', 'pending', 'downloading', 'evicted')
        """,
    )
    suspend fun markEvicted(attachmentId: String, updatedAt: Long): Int

    @Query("SELECT * FROM media_attachments WHERE attachmentId IN (:ids)")
    suspend fun getAll(ids: List<String>): List<MediaAttachmentEntity>

    @Query("SELECT * FROM media_attachments")
    suspend fun all(): List<MediaAttachmentEntity>

    @Query(
        """
        SELECT media_attachments.* FROM media_attachments
        LEFT JOIN message_attachments
          ON message_attachments.attachmentId = media_attachments.attachmentId
        WHERE message_attachments.attachmentId IS NULL
        """,
    )
    suspend fun unreferenced(): List<MediaAttachmentEntity>

    @Query("DELETE FROM media_attachments WHERE attachmentId = :attachmentId")
    suspend fun delete(attachmentId: String): Int

    @Query(
        """
        SELECT media_attachments.* FROM media_attachments
        INNER JOIN message_attachments
          ON message_attachments.attachmentId = media_attachments.attachmentId
        WHERE message_attachments.messageId = :messageId
        ORDER BY message_attachments.ordinal
        """,
    )
    suspend fun forMessage(messageId: String): List<MediaAttachmentEntity>
}
