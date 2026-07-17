package com.akashic.mobile

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.akashic.mobile.data.local.MessageWithBlocks
import com.akashic.mobile.data.local.PreparedComposerDraftResult
import com.akashic.mobile.data.local.AttachmentTransferEntity
import com.akashic.mobile.data.local.ComposerDraftEntity
import com.akashic.mobile.data.local.ConversationSummary
import com.akashic.mobile.data.local.canRemoveFrom
import com.akashic.mobile.data.local.isRemoteMissingIn
import com.akashic.mobile.data.local.MessageAttachmentWithMedia
import com.akashic.mobile.data.local.decodeStoredToolBlock
import com.akashic.mobile.data.realtime.TransferNetworkKind
import com.akashic.mobile.data.realtime.MobileSessionState
import com.akashic.mobile.domain.model.ConnectionPhase
import com.akashic.mobile.domain.model.ConnectionState
import com.akashic.mobile.ui.conversation.ConnectionStatusUi
import com.akashic.mobile.ui.conversation.CommandUi
import com.akashic.mobile.ui.conversation.ComposerAttachmentState
import com.akashic.mobile.ui.conversation.ComposerAttachmentUi
import com.akashic.mobile.ui.conversation.ComposerDraftUi
import com.akashic.mobile.ui.conversation.ConversationUiState
import com.akashic.mobile.ui.conversation.AssistantTurnStatus
import com.akashic.mobile.ui.conversation.MessageUi
import com.akashic.mobile.ui.conversation.MessageDeliveryActionUi
import com.akashic.mobile.ui.conversation.MessageReplyUi
import com.akashic.mobile.ui.conversation.MessageAttachmentState
import com.akashic.mobile.ui.conversation.MessageAttachmentUi
import com.akashic.mobile.ui.conversation.ProcessBlockKind
import com.akashic.mobile.ui.conversation.ProcessBlockState
import com.akashic.mobile.ui.conversation.ProcessBlockUi
import com.akashic.mobile.ui.conversation.ReadingPositionUi
import com.akashic.mobile.ui.conversation.NavigationTargetUi
import com.akashic.mobile.ui.conversation.PendingMessageUi
import com.akashic.mobile.ui.conversation.SessionUi
import com.akashic.mobile.ui.conversation.TransferStatusUi
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlin.math.ceil

private const val LARGE_TRANSFER_BYTES = 10L * 1024 * 1024

data class IncomingShareUi(
    val id: String,
    val text: String?,
    val targetSessionId: String?,
    val hasPendingAttachments: Boolean,
    val hasPreparedText: Boolean,
    val revision: Int,
    val errorMessage: String?,
)

private data class QueuedIncomingShare(
    val content: IncomingShare,
    val targetSessionId: String? = null,
    val preparedText: String? = null,
    val preparedReplyToMessageId: String? = null,
    val preparedBaseText: String? = null,
    val preparedBaseReplyToMessageId: String? = null,
    val preparedBaseUpdatedAt: Long? = null,
    val attachmentInFlight: Boolean = false,
    val textInFlight: Boolean = false,
    val revision: Int = 0,
    val errorMessage: String? = null,
)

private data class ComposerLocalState(
    val attachments: List<AttachmentTransferEntity>,
    val draft: ComposerDraftEntity?,
)

private data class ConversationProjection(
    val session: MobileSessionState,
    val graph: List<MessageWithBlocks>,
    val conversations: List<ConversationSummary>,
    val composer: ComposerLocalState,
)

@OptIn(ExperimentalCoroutinesApi::class)
class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val container = (application as App).container
    val sessionState = container.realtimeSession.state
    val pluginUiCatalog = container.realtimeSession.pluginUi.catalog
    val pluginUiResults = container.realtimeSession.pluginUi.results
    val pluginUiAssetStore = container.pluginUiAssetStore
    private val navigationTarget = MutableStateFlow<NavigationTargetUi?>(null)
    private val incomingShareQueue = MutableStateFlow<List<QueuedIncomingShare>>(emptyList())
    val incomingShare = incomingShareQueue.map { queue ->
        queue.firstOrNull()?.let { share ->
            IncomingShareUi(
                id = share.content.id,
                text = share.content.text,
                targetSessionId = share.targetSessionId,
                hasPendingAttachments = share.content.uris.isNotEmpty(),
                hasPreparedText = share.preparedText != null,
                revision = share.revision,
                errorMessage = share.errorMessage,
            )
        }
    }.stateIn(
        viewModelScope,
        SharingStarted.Eagerly,
        null,
    )

    init {
        viewModelScope.launch {
            val restored = container.incomingShareStore.load().map { persisted ->
                QueuedIncomingShare(
                    content = persisted.content,
                    targetSessionId = persisted.targetSessionId,
                    preparedText = persisted.preparedText,
                    preparedReplyToMessageId = persisted.preparedReplyToMessageId,
                    preparedBaseText = persisted.preparedBaseText,
                    preparedBaseReplyToMessageId = persisted.preparedBaseReplyToMessageId,
                    preparedBaseUpdatedAt = persisted.preparedBaseUpdatedAt,
                )
            }
            val restoredIds = restored.mapTo(mutableSetOf()) { it.content.id }
            incomingShareQueue.value = restored + incomingShareQueue.value.filterNot {
                it.content.id in restoredIds
            }
        }
    }

    private val conversationProjection = sessionState.flatMapLatest { state ->
        val serverId = state.serverId
        val sessionId = state.currentSessionId
        val graph = sessionId?.let(container.database.messages()::observeMessageGraph) ?: flowOf(emptyList())
        val conversations = serverId?.let(container.database.conversations()::observeSummaries) ?: flowOf(emptyList())
        val composer = if (serverId == null || sessionId == null) {
            flowOf(ComposerLocalState(emptyList(), null))
        } else {
            combine(
                container.database.attachmentTransfers().observeDrafts(serverId, sessionId),
                container.database.composerDrafts().observe(serverId, sessionId),
            ) { attachments, draft ->
                ComposerLocalState(attachments, draft)
            }
        }
        combine(graph, conversations, composer) { currentGraph, currentConversations, currentComposer ->
            ConversationProjection(state, currentGraph, currentConversations, currentComposer)
        }
    }

    val conversationState = combine(
        conversationProjection,
        navigationTarget,
    ) { projection, target ->
        val session = projection.session
        val graph = projection.graph
        val conversations = projection.conversations
        val composerLocal = projection.composer
        val attachments = composerLocal.attachments
        val draft = composerLocal.draft
        val scopedGraph = graph.filter { it.message.sessionId == session.currentSessionId }
        val messages = scopedGraph.map(::toMessageUi)
        val connection = connectionPresentation(session.connection, session.errorMessage)
        val composerAttachments = attachments.map { attachment ->
            val waitingForConnection = session.connection.phase != ConnectionPhase.READY &&
                attachment.state in setOf("pending", "uploading", "finishing")
            val waitingForMeteredApproval = attachment.sizeBytes >= LARGE_TRANSFER_BYTES &&
                session.transferNetwork.kind == TransferNetworkKind.METERED &&
                !session.meteredLargeTransferApproved &&
                attachment.state in setOf("pending", "uploading", "finishing")
            ComposerAttachmentUi(
                id = attachment.attachmentId,
                filename = attachment.filename,
                contentType = attachment.contentType,
                sizeBytes = attachment.sizeBytes,
                transferredBytes = attachment.transferredBytes,
                state = when {
                    waitingForConnection -> ComposerAttachmentState.WAITING_FOR_CONNECTION
                    waitingForMeteredApproval -> ComposerAttachmentState.WAITING_FOR_METERED_APPROVAL
                    attachment.state == "ready" -> ComposerAttachmentState.READY
                    attachment.state == "failed" -> ComposerAttachmentState.FAILED
                    else -> ComposerAttachmentState.UPLOADING
                },
                canRemove = attachment.state in setOf("pending", "ready", "failed"),
            )
        }
        val largeTransfer = composerAttachments.firstOrNull {
            it.sizeBytes >= LARGE_TRANSFER_BYTES &&
                it.state !in setOf(ComposerAttachmentState.READY, ComposerAttachmentState.FAILED)
        }
        val transferStatus = largeTransfer?.let { transfer ->
            val requiresApproval = session.transferNetwork.kind == TransferNetworkKind.METERED &&
                !session.meteredLargeTransferApproved
            val progress = (transfer.transferredBytes * 100 / transfer.sizeBytes).toInt().coerceIn(0, 100)
            TransferStatusUi(
                title = if (requiresApproval) "大文件上传已暂停" else "大文件正在后台上传",
                detail = when {
                    requiresApproval && session.transferNetwork.cellular -> "当前为移动网络，确认后会从 $progress% 继续"
                    requiresApproval -> "当前网络按流量计费，确认后会从 $progress% 继续"
                    session.transferNetwork.cellular -> "移动网络 · 离开会话仍会继续"
                    else -> "切换会话或进入后台仍会继续"
                },
                progressPercent = progress,
                requiresMeteredApproval = requiresApproval,
            )
        }
        val mobileConversations = conversations.filter { it.sessionId.startsWith("mobile:") }
        val selectedRemoteMissing = mobileConversations
            .firstOrNull { it.sessionId == session.currentSessionId }
            ?.isRemoteMissingIn(session.remoteSessionIds) == true
        ConversationUiState(
            connectionLabel = connection.label,
            connectionStatus = connection.status,
            connectionNotice = connection.notice,
            errorNotice = session.errorMessage,
            sessions = mobileConversations
                .map {
                    SessionUi(
                        sessionId = it.sessionId,
                        title = it.title,
                        lastMessagePreview = it.lastMessagePreview?.take(160),
                        lastMessageAtMillis = it.lastMessageAt,
                        unreadCount = it.unreadCount,
                        isRunning = it.sessionId in session.activeSessionIds,
                        isAvailable = !it.isRemoteMissingIn(session.remoteSessionIds),
                        canRemove = it.canRemoveFrom(session.remoteSessionIds),
                    )
                },
            selectedSessionId = session.currentSessionId,
            readingPosition = conversations
                .firstOrNull { it.sessionId == session.currentSessionId }
                ?.let { summary ->
                    summary.anchorMessageId?.let { ReadingPositionUi(it, summary.anchorOffsetPx) }
                },
            navigationTarget = target,
            projectionGeneration = session.projectionGeneration,
            messages = messages,
            composerDraft = ComposerDraftUi(
                text = draft?.text.orEmpty(),
                replyToMessageId = draft?.replyToMessageId,
            ),
            pendingMessages = messages.filterIsInstance<MessageUi.User>()
                .filter { it.deliveryLabel == "待发送" }
                .map {
                    PendingMessageUi(
                        messageId = it.id,
                        preview = it.text.trim().replace(Regex("\\s+"), " ").take(120)
                            .ifBlank { "[附件]" },
                        createdAtMillis = it.createdAtMillis,
                    )
                },
            attachments = composerAttachments,
            transferStatus = transferStatus,
            commands = session.commands.map { CommandUi(it.command, it.description) },
            isStreaming = scopedGraph.any { it.message.deliveryState == "streaming" },
            isResyncing = session.connection.phase == ConnectionPhase.SYNCING,
            canResync = session.connection.phase == ConnectionPhase.READY &&
                session.activeTurnId == null &&
                !session.hasActiveAttachmentDownload,
            isStopping = session.isStopping,
            canStop = session.activeTurnId != null &&
                session.connection.phase == ConnectionPhase.READY &&
                !session.isStopping,
            canSend = session.hasProfile &&
                !selectedRemoteMissing &&
                composerAttachments.all { it.state == ComposerAttachmentState.READY },
        )
    }.stateIn(
        viewModelScope,
        SharingStarted.WhileSubscribed(5_000),
        ConversationUiState(
            connectionLabel = "正在连接",
            connectionStatus = ConnectionStatusUi.CONNECTING,
            connectionNotice = null,
            errorNotice = null,
            sessions = emptyList(),
            selectedSessionId = null,
            readingPosition = null,
            navigationTarget = null,
            projectionGeneration = 0,
            messages = emptyList(),
            attachments = emptyList(),
            composerDraft = ComposerDraftUi("", null),
            pendingMessages = emptyList(),
            commands = emptyList(),
            isStreaming = false,
            isResyncing = false,
            canResync = false,
            isStopping = false,
            canStop = false,
            canSend = false,
        ),
    )

    fun onQrCode(value: String) = container.realtimeSession.beginPairing(value)

    fun sendMessage(
        value: String,
        replyToMessageId: String?,
        expectedAttachmentIds: List<String>,
        onPersisted: (Boolean) -> Unit,
    ) = container.realtimeSession.sendMessage(
        value,
        replyToMessageId,
        expectedAttachmentIds,
        onPersisted,
    )

    fun sendCommand(value: String) = container.realtimeSession.sendCommand(value)

    fun queryPluginUi(
        requestId: String,
        ownerId: String,
        slot: String,
        sessionId: String?,
        turnId: String?,
        pluginId: String,
        method: String,
        payloadJson: String,
    ) = container.realtimeSession.queryPluginUi(
        requestId,
        ownerId,
        slot,
        sessionId,
        turnId,
        pluginId,
        method,
        payloadJson,
    )

    fun cancelPluginUiOwner(ownerId: String) =
        container.realtimeSession.cancelPluginUiOwner(ownerId)

    fun disposePluginUiWebView() = container.realtimeSession.pluginUi.onWebViewDisposed()

    fun stopCurrentTurn() = container.realtimeSession.stopCurrentTurn()

    fun dismissError() = container.realtimeSession.dismissError()

    fun addAttachments(uris: List<android.net.Uri>) = container.realtimeSession.addAttachments(uris)

    /** 先持久接收系统分享，再把它交给进程内 UI 队列。 */
    suspend fun acceptIncomingShare(incoming: IncomingShare) {
        val persisted = container.incomingShareStore.enqueue(incoming)
        if (incomingShareQueue.value.none { it.content.id == incoming.id }) {
            incomingShareQueue.value = incomingShareQueue.value + QueuedIncomingShare(
                persisted.content,
                persisted.targetSessionId,
            )
        }
    }

    /** 首次处理时绑定当前会话，文字和文件共用同一个 owner。 */
    fun claimIncomingShareTarget(shareId: String) {
        val sessionId = requireNotNull(sessionState.value.currentSessionId) { "系统分享没有目标会话" }
        viewModelScope.launch {
            container.incomingShareStore.claimTarget(shareId, sessionId)
            incomingShareQueue.value = incomingShareQueue.value.map { share ->
                if (share.content.id != shareId || share.targetSessionId != null) share
                else share.copy(targetSessionId = sessionId)
            }
        }
    }

    /** 成功导入后才消费共享 URI；失败保留原请求供用户重试。 */
    fun dispatchIncomingShareAttachments(shareId: String) {
        val share = incomingShareQueue.value.firstOrNull { it.content.id == shareId } ?: return
        if (share.content.uris.isEmpty() || share.attachmentInFlight || share.errorMessage != null) return
        val sessionId = requireNotNull(share.targetSessionId) { "系统分享尚未绑定目标会话" }
        incomingShareQueue.value = incomingShareQueue.value.map { current ->
            if (current.content.id == shareId) current.copy(attachmentInFlight = true) else current
        }
        container.realtimeSession.addSharedAttachments(
            sessionId,
            share.content.uris,
            requireNotNull(share.content.attachmentIds) { "系统分享缺少稳定附件 ID" },
        ) { errorMessage ->
            viewModelScope.launch {
                completeIncomingShareAttachments(shareId, errorMessage)
            }
        }
    }

    /** Room 提交成功后才消费共享文字。 */
    fun commitIncomingShareText(
        shareId: String,
        sessionId: String,
        text: String,
        replyToMessageId: String?,
    ) {
        val share = incomingShareQueue.value.firstOrNull { it.content.id == shareId } ?: return
        require(share.targetSessionId == sessionId) { "系统分享目标会话已变化" }
        if (share.textInFlight) return
        incomingShareQueue.value = incomingShareQueue.value.map { current ->
            if (current.content.id == shareId) current.copy(textInFlight = true) else current
        }
        viewModelScope.launch(start = CoroutineStart.UNDISPATCHED) {
            try {
                val serverId = requireNotNull(sessionState.value.serverId) { "系统分享没有电脑 owner" }
                val base = container.database.composerDrafts().get(serverId, sessionId)
                container.incomingShareStore.prepareText(
                    shareId,
                    text,
                    replyToMessageId,
                    base?.text,
                    base?.replyToMessageId,
                    base?.updatedAt,
                )
                incomingShareQueue.value = incomingShareQueue.value.map { current ->
                    if (current.content.id != shareId) current
                    else current.copy(
                        content = current.content.copy(text = null),
                        preparedText = text,
                        preparedReplyToMessageId = replyToMessageId,
                        preparedBaseText = base?.text,
                        preparedBaseReplyToMessageId = base?.replyToMessageId,
                        preparedBaseUpdatedAt = base?.updatedAt,
                    )
                }
                persistIncomingShareText(
                    shareId,
                    sessionId,
                    text,
                    replyToMessageId,
                    base?.text,
                    base?.replyToMessageId,
                    base?.updatedAt,
                )
            } catch (error: IllegalArgumentException) {
                failIncomingShare(shareId, error.message ?: "共享文字无法写入当前会话")
            } catch (_: android.database.sqlite.SQLiteException) {
                failIncomingShare(shareId, "本地草稿记录失败，请重试")
            }
        }
    }

    /** 进程恢复后重放已冻结的最终草稿，不再次经过 Web 合并。 */
    fun resumePreparedIncomingShareText(shareId: String) {
        val share = incomingShareQueue.value.firstOrNull { it.content.id == shareId } ?: return
        val text = share.preparedText ?: return
        if (share.textInFlight || share.errorMessage != null) return
        val sessionId = requireNotNull(share.targetSessionId) { "系统分享没有目标会话" }
        incomingShareQueue.value = incomingShareQueue.value.map { current ->
            if (current.content.id == shareId) current.copy(textInFlight = true) else current
        }
        viewModelScope.launch(start = CoroutineStart.UNDISPATCHED) {
            try {
                persistIncomingShareText(
                    shareId,
                    sessionId,
                    text,
                    share.preparedReplyToMessageId,
                    share.preparedBaseText,
                    share.preparedBaseReplyToMessageId,
                    share.preparedBaseUpdatedAt,
                )
            } catch (error: IllegalArgumentException) {
                failIncomingShare(shareId, error.message ?: "共享文字无法写入当前会话")
            } catch (_: android.database.sqlite.SQLiteException) {
                failIncomingShare(shareId, "本地草稿记录失败，请重试")
            }
        }
    }

    fun reportIncomingShareError(shareId: String, message: String) = failIncomingShare(shareId, message)

    fun retryIncomingShare(shareId: String) {
        val current = incomingShareQueue.value.firstOrNull { it.content.id == shareId } ?: return
        val targetSessionId = requireNotNull(current.targetSessionId) { "系统分享没有目标会话" }
        if (conversationState.value.sessions.none { it.sessionId == targetSessionId }) {
            failIncomingShare(shareId, "原会话已不可用，请放弃后重新分享")
            return
        }
        selectSession(targetSessionId)
        incomingShareQueue.value = incomingShareQueue.value.map { share ->
            if (share.content.id == shareId) share.copy(
                attachmentInFlight = false,
                textInFlight = false,
                revision = share.revision + 1,
                errorMessage = null,
            ) else share
        }
    }

    fun discardIncomingShare(shareId: String) {
        incomingShareQueue.value = incomingShareQueue.value.filterNot { it.content.id == shareId }
        viewModelScope.launch {
            container.incomingShareStore.discard(shareId)
        }
    }

    private fun completeIncomingShareAttachments(shareId: String, errorMessage: String?) {
        if (errorMessage != null) {
            failIncomingShare(shareId, errorMessage)
            return
        }
        viewModelScope.launch {
            container.incomingShareStore.consumeFiles(shareId)
            incomingShareQueue.value = incomingShareQueue.value.mapNotNull { share ->
                if (share.content.id != shareId) return@mapNotNull share
                share.copy(
                    content = share.content.copy(uris = emptyList()),
                    attachmentInFlight = false,
                ).takeIf { it.content.text != null || it.preparedText != null }
            }
        }
    }

    private fun consumeIncomingShareText(shareId: String) {
        incomingShareQueue.value = incomingShareQueue.value.mapNotNull { share ->
            if (share.content.id != shareId) return@mapNotNull share
            share.copy(
                content = share.content.copy(text = null),
                preparedText = null,
                preparedReplyToMessageId = null,
                preparedBaseText = null,
                preparedBaseReplyToMessageId = null,
                preparedBaseUpdatedAt = null,
                textInFlight = false,
            ).takeIf { it.content.uris.isNotEmpty() }
        }
    }

    private suspend fun persistIncomingShareText(
        shareId: String,
        sessionId: String,
        text: String,
        replyToMessageId: String?,
        baseText: String?,
        baseReplyToMessageId: String?,
        baseUpdatedAt: Long?,
    ) {
        val serverId = requireNotNull(sessionState.value.serverId) { "系统分享没有电脑 owner" }
        val result = container.deliveryStore.commitPreparedComposerDraft(
            sessionId = sessionId,
            text = text,
            replyToMessageId = replyToMessageId,
            baseText = baseText,
            baseReplyToMessageId = baseReplyToMessageId,
            baseUpdatedAt = baseUpdatedAt,
            expectedServerId = serverId,
            updatedAt = System.currentTimeMillis(),
        )
        require(result != PreparedComposerDraftResult.CONFLICT) {
            "当前草稿已变化，请确认内容后放弃本次分享"
        }
        container.incomingShareStore.consumeText(shareId)
        consumeIncomingShareText(shareId)
    }

    private fun failIncomingShare(shareId: String, message: String) {
        incomingShareQueue.value = incomingShareQueue.value.map { share ->
            if (share.content.id == shareId) share.copy(
                attachmentInFlight = false,
                textInFlight = false,
                errorMessage = message,
            ) else share
        }
    }

    fun removeAttachment(attachmentId: String) = container.realtimeSession.removeAttachment(attachmentId)

    fun retryAttachment(attachmentId: String) = container.realtimeSession.retryAttachment(attachmentId)

    fun continueLargeTransfersOnMeteredNetwork() =
        container.realtimeSession.continueLargeTransfersOnMeteredNetwork()

    fun retryFailedMessage(messageId: String) = container.realtimeSession.retryFailedMessage(messageId)

    /** 按桥接调用顺序保存当前电脑的一份会话草稿。 */
    fun saveComposerDraft(sessionId: String, text: String, replyToMessageId: String?) {
        viewModelScope.launch(start = CoroutineStart.UNDISPATCHED) {
            container.deliveryStore.saveComposerDraft(
                sessionId = sessionId,
                text = text,
                replyToMessageId = replyToMessageId,
                expectedServerId = sessionState.value.serverId,
                updatedAt = System.currentTimeMillis(),
            )
        }
    }

    fun retryDownloadedAttachment(attachmentId: String) =
        container.realtimeSession.retryDownloadedAttachment(attachmentId)

    fun touchDownloadedAttachment(attachmentId: String) =
        container.realtimeSession.touchDownloadedAttachment(attachmentId)

    fun createSession() = container.realtimeSession.createSession()

    fun selectSession(sessionId: String) = container.realtimeSession.selectSession(sessionId)

    fun removeUnavailableSession(sessionId: String) =
        container.realtimeSession.removeUnavailableSession(sessionId)

    /** 按桥接调用顺序校验并保存会话阅读锚点。 */
    fun saveReadingPosition(sessionId: String, messageId: String, offsetPx: Int) {
        viewModelScope.launch(start = CoroutineStart.UNDISPATCHED) {
            container.deliveryStore.saveReadingPosition(
                sessionId = sessionId,
                messageId = messageId,
                offsetPx = offsetPx,
                expectedServerId = sessionState.value.serverId,
                updatedAt = System.currentTimeMillis(),
            )
        }
    }

    /** 按桥接调用顺序推进已读水位并清除旧锚点。 */
    fun markSessionReadThrough(sessionId: String, readAtMillis: Long) {
        viewModelScope.launch(start = CoroutineStart.UNDISPATCHED) {
            container.deliveryStore.markSessionReadThrough(
                sessionId = sessionId,
                readAt = readAtMillis,
                expectedServerId = sessionState.value.serverId,
                updatedAt = System.currentTimeMillis(),
            )
        }
    }

    fun openNotificationTarget(sessionId: String, messageId: String) {
        container.realtimeSession.selectSession(sessionId)
        navigationTarget.value = NavigationTargetUi(sessionId, messageId)
    }

    fun acknowledgeNavigationTarget(messageId: String) {
        if (navigationTarget.value?.messageId == messageId) navigationTarget.value = null
    }

    fun restartPairing() = MobileConnectionService.disconnect(getApplication())

    fun reloadFromServer() = container.realtimeSession.reloadFromServer()

    private fun toMessageUi(graph: MessageWithBlocks): MessageUi {
        val message = graph.message
        if (message.role == "user") {
            return MessageUi.User(
                id = message.messageId,
                sessionId = message.sessionId,
                text = message.text,
                deliveryLabel = when (message.deliveryState) {
                    "pending" -> "待发送"
                    "sent", "complete" -> "已发送"
                    "failed" -> "发送失败"
                    "failed_retryable" -> "发送失败"
                    "outcome_unknown" -> "结果待确认"
                    else -> error("未知用户消息状态: ${message.deliveryState}")
                },
                deliveryAction = when (message.deliveryState) {
                    "failed_retryable" -> MessageDeliveryActionUi.RETRY
                    "outcome_unknown" -> MessageDeliveryActionUi.VERIFY
                    else -> null
                },
                replyable = userMessageCanReply(message.deliveryState),
                createdAtMillis = message.createdAt,
                reply = message.toReplyUi(),
                attachments = graph.attachmentLinks.toMessageAttachmentUi(),
                updatedAtMillis = message.updatedAt,
            )
        }
        return MessageUi.AssistantTurn(
            id = message.messageId,
            sessionId = message.sessionId,
            intro = null,
            blocks = graph.blocks.sortedBy { it.ordinal }.map { block ->
                val storedTool = if (block.kind == "tool") decodeStoredToolBlock(block.content) else null
                ProcessBlockUi(
                    id = block.blockId,
                    kind = if (block.kind == "thinking") ProcessBlockKind.THINKING else ProcessBlockKind.TOOL,
                    title = storedTool?.name ?: "思考",
                    detail = storedTool?.description ?: if (storedTool == null) block.content else "",
                    state = when (block.status) {
                        "running" -> ProcessBlockState.RUNNING
                        "failed" -> ProcessBlockState.FAILED
                        else -> ProcessBlockState.COMPLETED
                    },
                    arguments = storedTool?.arguments,
                    resultPreview = storedTool?.resultPreview,
                    durationMillis = storedTool?.durationMillis,
                )
            },
            answer = message.text,
            status = when (message.deliveryState) {
                "streaming" -> AssistantTurnStatus.STREAMING
                "complete" -> AssistantTurnStatus.COMPLETE
                "interrupted" -> AssistantTurnStatus.INTERRUPTED
                else -> error("未知助手消息状态: ${message.deliveryState}")
            },
            durationSeconds = turnDurationSeconds(
                startedAt = message.createdAt,
                updatedAt = message.updatedAt,
                isTerminal = message.deliveryState in setOf("complete", "interrupted"),
            ),
            createdAtMillis = message.createdAt,
            reply = message.toReplyUi(),
            attachments = graph.attachmentLinks.toMessageAttachmentUi(),
            updatedAtMillis = message.updatedAt,
        )
    }

    private fun com.akashic.mobile.data.local.MessageEntity.toReplyUi(): MessageReplyUi? {
        val target = replyToMessageId ?: return null
        return MessageReplyUi(
            messageId = target,
            role = requireNotNull(replyRole) { "引用消息缺少角色: $messageId" },
            preview = requireNotNull(replyPreview) { "引用消息缺少预览: $messageId" },
        )
    }
}

internal fun List<MessageAttachmentWithMedia>.toMessageAttachmentUi(): List<MessageAttachmentUi> =
    sortedBy { it.link.ordinal }.map { relation ->
        val attachment = relation.attachment
        MessageAttachmentUi(
            id = attachment.attachmentId,
            filename = attachment.filename,
            contentType = attachment.contentType,
            sizeBytes = attachment.sizeBytes,
            transferredBytes = attachment.transferredBytes,
            state = when (attachment.state) {
                "remote" -> MessageAttachmentState.REMOTE
                "pending" -> MessageAttachmentState.PENDING
                "downloading" -> MessageAttachmentState.DOWNLOADING
                "cached" -> MessageAttachmentState.CACHED
                "failed" -> MessageAttachmentState.FAILED
                "evicted" -> MessageAttachmentState.EVICTED
                else -> error("未知附件下载状态: ${attachment.state}")
            },
            cachePath = attachment.cachePath,
        )
    }

internal fun userMessageCanReply(deliveryState: String): Boolean = deliveryState == "complete"

internal data class ConnectionPresentation(
    val label: String,
    val status: ConnectionStatusUi,
    val notice: String?,
)

/** 把实时链路状态映射为用户可理解的连接语义。 */
internal fun connectionPresentation(
    connection: ConnectionState,
    errorMessage: String? = null,
): ConnectionPresentation {
    val reconnecting = connection.retryCount > 0 &&
        connection.phase in setOf(ConnectionPhase.CONNECTING, ConnectionPhase.DEGRADED)
    if (reconnecting) {
        return ConnectionPresentation(
            label = "正在重连",
            status = ConnectionStatusUi.RECONNECTING,
            notice = "正在重连 · 消息已缓存",
        )
    }
    return when (connection.phase) {
        ConnectionPhase.READY -> ConnectionPresentation("连接正常", ConnectionStatusUi.READY, null)
        ConnectionPhase.SYNCING -> ConnectionPresentation(
            "正在同步消息",
            ConnectionStatusUi.CONNECTING,
            "正在从电脑更新本地消息",
        )
        ConnectionPhase.DEGRADED -> ConnectionPresentation(
            "网络不稳 · 正在续传",
            ConnectionStatusUi.DEGRADED,
            "网络不稳 · 消息已缓存，正在续传",
        )
        ConnectionPhase.CLOSED -> ConnectionPresentation(
            "连接已断开",
            ConnectionStatusUi.DISCONNECTED,
            "连接已断开 · 消息已缓存",
        )
        ConnectionPhase.FAILED -> ConnectionPresentation(
            "启动失败",
            ConnectionStatusUi.DISCONNECTED,
            errorMessage ?: "启动检查失败，请重新打开应用",
        )
        else -> ConnectionPresentation("正在连接", ConnectionStatusUi.CONNECTING, null)
    }
}

internal fun turnDurationSeconds(startedAt: Long, updatedAt: Long, isTerminal: Boolean): Int? {
    if (!isTerminal) return null
    return ceil((updatedAt - startedAt).coerceAtLeast(1) / 1_000.0).toInt()
}
