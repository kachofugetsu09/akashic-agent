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

    @Query(
        """
        SELECT
          conversation.sessionId AS sessionId,
          conversation.title AS title,
          (
            SELECT CASE
              WHEN TRIM(message.text) != '' THEN REPLACE(REPLACE(TRIM(message.text), CHAR(10), ' '), CHAR(13), ' ')
              WHEN EXISTS (
                SELECT 1 FROM message_attachments AS relation
                WHERE relation.messageId = message.messageId
              ) THEN '[附件]'
              ELSE NULL
            END
            FROM messages AS message
            WHERE message.sessionId = conversation.sessionId
            ORDER BY message.createdAt DESC, message.messageId DESC
            LIMIT 1
          ) AS lastMessagePreview,
          (
            SELECT message.createdAt
            FROM messages AS message
            WHERE message.sessionId = conversation.sessionId
            ORDER BY message.createdAt DESC, message.messageId DESC
            LIMIT 1
          ) AS lastMessageAt,
          CASE WHEN read_state.sessionId IS NULL THEN 0 ELSE (
            SELECT COUNT(*)
            FROM messages AS message
            WHERE message.sessionId = conversation.sessionId
              AND message.role = 'assistant'
              AND message.deliveryState = 'complete'
              AND message.createdAt > read_state.lastReadAt
          ) END AS unreadCount,
          EXISTS (
            SELECT 1 FROM messages AS message
            WHERE message.sessionId = conversation.sessionId
              AND message.role = 'assistant'
              AND message.deliveryState = 'streaming'
          ) AS isRunning,
          read_state.anchorMessageId AS anchorMessageId,
          COALESCE(read_state.anchorOffsetPx, 0) AS anchorOffsetPx,
          conversation.remoteKnown AS remoteKnown,
          (
            EXISTS (
              SELECT 1 FROM messages AS message
              WHERE message.sessionId = conversation.sessionId
                AND message.clientMessageId IS NOT NULL
                AND message.deliveryState IN (
                  'pending', 'sent', 'failed', 'failed_retryable', 'outcome_unknown'
                )
            ) OR EXISTS (
              SELECT 1 FROM attachment_transfers AS transfer
              WHERE transfer.serverId = conversation.serverId
                AND transfer.sessionId = conversation.sessionId
                AND transfer.state IN ('pending', 'uploading', 'finishing', 'ready', 'failed')
            ) OR EXISTS (
              SELECT 1 FROM composer_drafts AS draft
              WHERE draft.serverId = conversation.serverId
                AND draft.sessionId = conversation.sessionId
            )
          ) AS hasLocalWork
        FROM conversations AS conversation
        LEFT JOIN conversation_read_states AS read_state
          ON read_state.sessionId = conversation.sessionId
        WHERE conversation.serverId = :serverId
        ORDER BY COALESCE(lastMessageAt, conversation.updatedAt) DESC, conversation.sessionId
        """,
    )
    fun observeSummaries(serverId: String): Flow<List<ConversationSummary>>

    @Query("SELECT * FROM conversations WHERE sessionId = :sessionId")
    suspend fun get(sessionId: String): ConversationEntity?

    @Query("UPDATE conversations SET remoteKnown = 1 WHERE sessionId = :sessionId")
    suspend fun markRemoteKnown(sessionId: String): Int

    @Query("DELETE FROM conversations WHERE serverId = :serverId AND sessionId = :sessionId")
    suspend fun delete(serverId: String, sessionId: String): Int

    @Query(
        """
        DELETE FROM conversations
        WHERE serverId = :serverId
          AND (:preservedSessionId IS NULL OR sessionId != :preservedSessionId)
          AND NOT EXISTS (SELECT 1 FROM messages WHERE messages.sessionId = conversations.sessionId)
          AND NOT EXISTS (
            SELECT 1 FROM attachment_transfers
            WHERE attachment_transfers.sessionId = conversations.sessionId
              AND attachment_transfers.serverId = :serverId
          )
          AND NOT EXISTS (
            SELECT 1 FROM composer_drafts
            WHERE composer_drafts.sessionId = conversations.sessionId
              AND composer_drafts.serverId = :serverId
          )
        """,
    )
    suspend fun deleteEmptyProjection(serverId: String, preservedSessionId: String?): Int
}

@Dao
interface ConversationReadStateDao {
    @Query(
        """
        UPDATE conversation_read_states
        SET anchorMessageId = :targetMessageId
        WHERE sessionId = :sessionId AND anchorMessageId = :sourceMessageId
        """,
    )
    suspend fun moveAnchor(sessionId: String, sourceMessageId: String, targetMessageId: String): Int

    @Query(
        """
        INSERT INTO conversation_read_states (
          sessionId, lastReadAt, anchorMessageId, anchorOffsetPx, updatedAt
        ) VALUES (
          :sessionId,
          COALESCE((
            SELECT MAX(createdAt) FROM messages
            WHERE sessionId = :sessionId AND role = 'assistant'
          ), 0),
          :messageId,
          :offsetPx,
          :updatedAt
        )
        ON CONFLICT(sessionId) DO UPDATE SET
          anchorMessageId = excluded.anchorMessageId,
          anchorOffsetPx = excluded.anchorOffsetPx,
          updatedAt = excluded.updatedAt
        """,
    )
    suspend fun savePosition(sessionId: String, messageId: String, offsetPx: Int, updatedAt: Long)

    @Query(
        """
        UPDATE conversation_read_states
        SET anchorMessageId = NULL,
            anchorOffsetPx = 0,
            updatedAt = :updatedAt
        WHERE sessionId = :sessionId
        """,
    )
    suspend fun clearPosition(sessionId: String, updatedAt: Long): Int

    @Query(
        """
        INSERT INTO conversation_read_states (
          sessionId, lastReadAt, anchorMessageId, anchorOffsetPx, updatedAt
        ) VALUES (:sessionId, :readAt, NULL, 0, :updatedAt)
        ON CONFLICT(sessionId) DO UPDATE SET
          lastReadAt = MAX(lastReadAt, excluded.lastReadAt),
          anchorMessageId = NULL,
          anchorOffsetPx = 0,
          updatedAt = excluded.updatedAt
        """,
    )
    suspend fun markReadThrough(sessionId: String, readAt: Long, updatedAt: Long)
}

@Dao
interface ComposerDraftDao {
    @Query(
        """
        UPDATE composer_drafts
        SET replyToMessageId = :targetMessageId
        WHERE sessionId = :sessionId AND replyToMessageId = :sourceMessageId
        """,
    )
    suspend fun moveReplyTarget(
        sessionId: String,
        sourceMessageId: String,
        targetMessageId: String,
    ): Int

    @Query(
        "SELECT * FROM composer_drafts WHERE serverId = :serverId AND sessionId = :sessionId",
    )
    fun observe(serverId: String, sessionId: String): Flow<ComposerDraftEntity?>

    @Query(
        "SELECT * FROM composer_drafts WHERE serverId = :serverId AND sessionId = :sessionId",
    )
    suspend fun get(serverId: String, sessionId: String): ComposerDraftEntity?

    @Upsert
    suspend fun upsert(draft: ComposerDraftEntity)

    @Query(
        "DELETE FROM composer_drafts WHERE serverId = :serverId AND sessionId = :sessionId",
    )
    suspend fun delete(serverId: String, sessionId: String): Int
}

@Dao
interface MessageDao {
    @Upsert
    suspend fun upsert(message: MessageEntity)

    @Upsert
    suspend fun upsertBlocks(blocks: List<TurnBlockEntity>)

    @Query(
        """
        SELECT * FROM messages AS local
        WHERE local.sessionId = :sessionId
        ORDER BY
          CASE WHEN local.serverSeq IS NOT NULL THEN local.serverSeq ELSE COALESCE(
            (
              SELECT MIN(remote.serverSeq) FROM messages AS remote
              WHERE remote.sessionId = local.sessionId
                AND remote.serverSeq IS NOT NULL
                AND remote.createdAt >= local.createdAt
            ),
            9223372036854775807
          ) END,
          CASE WHEN local.serverSeq IS NULL THEN 0 ELSE 1 END,
          local.createdAt,
          local.messageId
        """,
    )
    fun observeMessages(sessionId: String): Flow<List<MessageEntity>>

    @Query("SELECT * FROM turn_blocks WHERE messageId = :messageId ORDER BY ordinal")
    fun observeBlocks(messageId: String): Flow<List<TurnBlockEntity>>

    @Query("SELECT * FROM turn_blocks WHERE messageId = :messageId ORDER BY ordinal")
    suspend fun getBlocks(messageId: String): List<TurnBlockEntity>

    @Query("SELECT * FROM messages WHERE messageId = :messageId")
    suspend fun get(messageId: String): MessageEntity?

    @Query("SELECT * FROM messages WHERE clientMessageId = :clientMessageId")
    suspend fun getByClientMessageId(clientMessageId: String): MessageEntity?

    @Query(
        """
        SELECT * FROM messages
        WHERE sessionId = :sessionId
          AND role = 'user'
          AND text = :text
          AND messageId LIKE 'user:%'
          AND clientMessageId IS NOT NULL
          AND deliveryState IN ('sent', 'complete')
          AND createdAt BETWEEN :earliestCreatedAt AND :latestCreatedAt
        ORDER BY createdAt, messageId
        """,
    )
    suspend fun findLegacyOptimisticUsers(
        sessionId: String,
        text: String,
        earliestCreatedAt: Long,
        latestCreatedAt: Long,
    ): List<MessageEntity>

    @Query(
        """
        SELECT * FROM messages
        WHERE sessionId = :sessionId
          AND role = 'assistant'
          AND text = :text
          AND messageId LIKE 'ephemeral:%'
          AND deliveryState = 'complete'
          AND updatedAt BETWEEN :earliestUpdatedAt AND :latestUpdatedAt
        ORDER BY createdAt, messageId
        """,
    )
    suspend fun findEphemeralAssistants(
        sessionId: String,
        text: String,
        earliestUpdatedAt: Long,
        latestUpdatedAt: Long,
    ): List<MessageEntity>

    @Query("SELECT COUNT(*) FROM messages WHERE sessionId = :sessionId")
    suspend fun countForSession(sessionId: String): Int

    @Query(
        """
        SELECT COUNT(*) FROM messages
        WHERE sessionId = :sessionId
          AND clientMessageId IS NOT NULL
          AND deliveryState IN ('pending', 'sent', 'failed', 'failed_retryable', 'outcome_unknown')
        """,
    )
    suspend fun countLocalWorkForSession(sessionId: String): Int

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

    @Query(
        """
        DELETE FROM messages
        WHERE sessionId IN (SELECT sessionId FROM conversations WHERE serverId = :serverId)
          AND NOT (
            clientMessageId IS NOT NULL
            AND deliveryState IN ('pending', 'sent', 'failed', 'failed_retryable', 'outcome_unknown')
          )
        """,
    )
    suspend fun deleteServerProjection(serverId: String): Int

    @Query(
        """
        DELETE FROM messages
        WHERE sessionId IN (SELECT sessionId FROM conversations WHERE serverId = :serverId)
          AND NOT (
            clientMessageId IS NOT NULL
            AND deliveryState IN ('pending', 'failed', 'failed_retryable', 'outcome_unknown')
          )
        """,
    )
    suspend fun deleteReloadableServerCache(serverId: String): Int

    @Query("UPDATE messages SET deliveryState = :state, updatedAt = :updatedAt WHERE clientMessageId = :clientMessageId")
    suspend fun updateDelivery(clientMessageId: String, state: String, updatedAt: Long): Int

    @Query(
        """
        UPDATE messages
        SET clientMessageId = :newClientMessageId, deliveryState = 'pending', updatedAt = :updatedAt
        WHERE messageId = :messageId
          AND clientMessageId = :oldClientMessageId
          AND deliveryState = 'failed_retryable'
        """,
    )
    suspend fun replaceRetryIdentity(
        messageId: String,
        oldClientMessageId: String,
        newClientMessageId: String,
        updatedAt: Long,
    ): Int

    @Query("UPDATE turn_blocks SET status = 'completed', updatedAt = :updatedAt WHERE messageId = :messageId AND status = 'running'")
    suspend fun completeRunningBlocks(messageId: String, updatedAt: Long): Int

    @Query(
        "UPDATE turn_blocks SET status = 'completed', updatedAt = :updatedAt " +
            "WHERE messageId = :messageId AND kind = 'thinking' AND status = 'running'",
    )
    suspend fun completeRunningThinking(messageId: String, updatedAt: Long): Int

    @Transaction
    @Query(
        """
        SELECT * FROM messages AS local
        WHERE local.sessionId = :sessionId
        ORDER BY
          CASE WHEN local.serverSeq IS NOT NULL THEN local.serverSeq ELSE COALESCE(
            (
              SELECT MIN(remote.serverSeq) FROM messages AS remote
              WHERE remote.sessionId = local.sessionId
                AND remote.serverSeq IS NOT NULL
                AND remote.createdAt >= local.createdAt
            ),
            9223372036854775807
          ) END,
          CASE WHEN local.serverSeq IS NULL THEN 0 ELSE 1 END,
          local.createdAt,
          local.messageId
        """,
    )
    fun observeMessageGraph(sessionId: String): Flow<List<MessageWithBlocks>>
}

@Dao
interface PendingMessageNotificationDao {
    @Upsert
    suspend fun upsert(notification: PendingMessageNotificationEntity)

    @Query(
        "SELECT * FROM pending_message_notifications WHERE serverId = :serverId ORDER BY createdAt, messageId",
    )
    fun observeForServer(serverId: String): Flow<List<PendingMessageNotificationEntity>>

    @Query("DELETE FROM pending_message_notifications WHERE messageId = :messageId")
    suspend fun delete(messageId: String): Int
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

    @Query("UPDATE outbox_commands SET state = :failureState WHERE commandId = :commandId AND state = 'in_flight'")
    suspend fun markFailed(commandId: String, failureState: String): Int

    @Query(
        """
        UPDATE outbox_commands
        SET state = 'failed_retryable'
        WHERE commandId = :commandId AND state IN ('pending', 'retry')
        """,
    )
    suspend fun markUnsentFailed(commandId: String): Int

    @Query("UPDATE outbox_commands SET state = 'retry' WHERE commandId = :commandId AND state = 'outcome_unknown'")
    suspend fun recheckUnknown(commandId: String): Int

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
        SELECT * FROM attachment_transfers
        WHERE serverId = :serverId AND state IN ('pending', 'uploading', 'finishing')
        ORDER BY updatedAt ASC
        """,
    )
    fun observeActiveUploads(serverId: String): Flow<List<AttachmentTransferEntity>>

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

    @Query(
        """
        UPDATE attachment_transfers
        SET state = 'pending', updatedAt = :updatedAt
        WHERE attachmentId = :attachmentId AND state = 'failed'
        """,
    )
    suspend fun retryFailed(attachmentId: String, updatedAt: Long): Int

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

    @Query("DELETE FROM attachment_transfers WHERE serverId = :serverId AND sessionId = :sessionId AND state = 'sent'")
    suspend fun deleteSentForSession(serverId: String, sessionId: String): Int
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

    @Query(
        """
        UPDATE realtime_cursors
        SET lastAcknowledgedEventSeq = :eventSeq,
            connectionEpoch = :connectionEpoch,
            updatedAt = :updatedAt
        WHERE deviceId = :deviceId
          AND lastAcknowledgedEventSeq <= :eventSeq
          AND connectionEpoch <= :connectionEpoch
        """,
    )
    suspend fun reset(
        deviceId: String,
        eventSeq: Long,
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
        WHERE attachmentId = :attachmentId AND state IN ('remote', 'failed', 'evicted')
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
