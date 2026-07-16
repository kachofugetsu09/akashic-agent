package com.akashic.mobile.data.local

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.Embedded
import androidx.room.PrimaryKey
import androidx.room.Relation

@Entity(tableName = "server_profiles")
data class ServerProfileEntity(
    @PrimaryKey val serverId: String,
    val displayName: String,
    val deviceId: String,
    val keyAlias: String,
    val applicationKeyFingerprint: String,
    val lanEndpointsJson: String,
    val tunnelEndpointsJson: String,
    val tlsSpkiPinsJson: String,
    val createdAt: Long,
)

@Entity(
    tableName = "conversations",
    foreignKeys = [
        ForeignKey(
            entity = ServerProfileEntity::class,
            parentColumns = ["serverId"],
            childColumns = ["serverId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("serverId")],
)
data class ConversationEntity(
    @PrimaryKey val sessionId: String,
    val serverId: String,
    val title: String,
    val updatedAt: Long,
)

@Entity(
    tableName = "conversation_read_states",
    foreignKeys = [
        ForeignKey(
            entity = ConversationEntity::class,
            parentColumns = ["sessionId"],
            childColumns = ["sessionId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
)
data class ConversationReadStateEntity(
    @PrimaryKey val sessionId: String,
    val lastReadAt: Long,
    val anchorMessageId: String?,
    val anchorOffsetPx: Int,
    val updatedAt: Long,
)

data class ConversationSummary(
    val sessionId: String,
    val title: String,
    val lastMessagePreview: String?,
    val lastMessageAt: Long?,
    val unreadCount: Int,
    val isRunning: Boolean,
    val anchorMessageId: String?,
    val anchorOffsetPx: Int,
)

@Entity(
    tableName = "messages",
    foreignKeys = [
        ForeignKey(
            entity = ConversationEntity::class,
            parentColumns = ["sessionId"],
            childColumns = ["sessionId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("sessionId"), Index(value = ["clientMessageId"], unique = true)],
)
data class MessageEntity(
    @PrimaryKey val messageId: String,
    val clientMessageId: String?,
    val sessionId: String,
    val role: String,
    val text: String,
    val deliveryState: String,
    val createdAt: Long,
    val updatedAt: Long,
    val serverSeq: Long? = null,
    val replyToMessageId: String? = null,
    val replyRole: String? = null,
    val replyPreview: String? = null,
)

@Entity(
    tableName = "turn_blocks",
    foreignKeys = [
        ForeignKey(
            entity = MessageEntity::class,
            parentColumns = ["messageId"],
            childColumns = ["messageId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("messageId"), Index(value = ["turnId", "ordinal"], unique = true)],
)
data class TurnBlockEntity(
    @PrimaryKey val blockId: String,
    val messageId: String,
    val turnId: String,
    val ordinal: Int,
    val kind: String,
    val status: String,
    val content: String,
    val updatedAt: Long,
)

@Entity(
    tableName = "outbox_commands",
    foreignKeys = [
        ForeignKey(
            entity = ServerProfileEntity::class,
            parentColumns = ["serverId"],
            childColumns = ["serverId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("serverId"), Index(value = ["state", "createdAt"])],
)
data class OutboxCommandEntity(
    @PrimaryKey val commandId: String,
    val serverId: String,
    val envelopeJson: String,
    val state: String,
    val attemptCount: Int,
    val createdAt: Long,
    val lastAttemptAt: Long?,
)

@Entity(
    tableName = "attachment_transfers",
    foreignKeys = [
        ForeignKey(
            entity = ServerProfileEntity::class,
            parentColumns = ["serverId"],
            childColumns = ["serverId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("serverId"), Index("sessionId")],
)
data class AttachmentTransferEntity(
    @PrimaryKey val attachmentId: String,
    val serverId: String,
    val sessionId: String,
    val filename: String,
    val contentType: String,
    val sizeBytes: Long,
    val sha256: String,
    val transferredBytes: Long,
    val state: String,
    val updatedAt: Long,
)

@Entity(
    tableName = "realtime_cursors",
    foreignKeys = [
        ForeignKey(
            entity = ServerProfileEntity::class,
            parentColumns = ["serverId"],
            childColumns = ["serverId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("serverId", unique = true)],
)
data class RealtimeCursorEntity(
    @PrimaryKey val deviceId: String,
    val serverId: String,
    val lastAcknowledgedEventSeq: Long,
    val connectionEpoch: Long,
    val updatedAt: Long,
)

@Entity(
    tableName = "media_attachments",
    foreignKeys = [
        ForeignKey(
            entity = ServerProfileEntity::class,
            parentColumns = ["serverId"],
            childColumns = ["serverId"],
            onDelete = ForeignKey.CASCADE,
        ),
        ForeignKey(
            entity = ConversationEntity::class,
            parentColumns = ["sessionId"],
            childColumns = ["sessionId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("serverId"), Index("sessionId"), Index("state")],
)
data class MediaAttachmentEntity(
    @PrimaryKey val attachmentId: String,
    val serverId: String,
    val sessionId: String,
    val filename: String,
    val contentType: String,
    val sizeBytes: Long,
    val sha256: String,
    val transferredBytes: Long,
    val state: String,
    val cachePath: String,
    val lastAccessedAt: Long,
    val updatedAt: Long,
)

@Entity(
    tableName = "message_attachments",
    primaryKeys = ["messageId", "attachmentId"],
    foreignKeys = [
        ForeignKey(
            entity = MessageEntity::class,
            parentColumns = ["messageId"],
            childColumns = ["messageId"],
            onDelete = ForeignKey.CASCADE,
        ),
        ForeignKey(
            entity = MediaAttachmentEntity::class,
            parentColumns = ["attachmentId"],
            childColumns = ["attachmentId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("attachmentId"), Index(value = ["messageId", "ordinal"], unique = true)],
)
data class MessageAttachmentEntity(
    val messageId: String,
    val attachmentId: String,
    val ordinal: Int,
)

data class MessageWithBlocks(
    @Embedded val message: MessageEntity,
    @Relation(parentColumn = "messageId", entityColumn = "messageId")
    val blocks: List<TurnBlockEntity>,
    @Relation(
        entity = MessageAttachmentEntity::class,
        parentColumn = "messageId",
        entityColumn = "messageId",
    )
    val attachmentLinks: List<MessageAttachmentWithMedia>,
)

data class MessageAttachmentWithMedia(
    @Embedded val link: MessageAttachmentEntity,
    @Relation(parentColumn = "attachmentId", entityColumn = "attachmentId")
    val attachment: MediaAttachmentEntity,
)
