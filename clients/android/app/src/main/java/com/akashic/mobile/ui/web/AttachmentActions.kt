package com.akashic.mobile.ui.web

import android.content.ActivityNotFoundException
import android.content.ClipData
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.widget.Toast
import androidx.core.content.FileProvider
import com.akashic.mobile.BuildConfig
import com.akashic.mobile.ui.conversation.ConversationUiState
import com.akashic.mobile.ui.conversation.MessageAttachmentState
import com.akashic.mobile.ui.conversation.MessageAttachmentUi
import com.akashic.mobile.ui.conversation.MessageUi
import java.io.File
import java.io.IOException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

internal fun ConversationUiState.findCachedAttachment(attachmentId: String): MessageAttachmentUi? {
    val attachment = messages.asSequence()
        .flatMap { message ->
            when (message) {
                is MessageUi.User -> message.attachments.asSequence()
                is MessageUi.AssistantTurn -> message.attachments.asSequence()
            }
        }
        .firstOrNull { it.id == attachmentId }
        ?: return null
    return attachment.takeIf {
        it.state == MessageAttachmentState.CACHED && File(it.cachePath).isFile
    }
}

internal fun withCachedAttachment(
    context: Context,
    state: ConversationUiState,
    attachmentId: String,
    action: (MessageAttachmentUi) -> Unit,
) {
    val attachment = state.findCachedAttachment(attachmentId)
    if (attachment == null) {
        Toast.makeText(context, "文件已失效，请重新下载", Toast.LENGTH_SHORT).show()
        return
    }
    action(attachment)
}

internal fun openCachedAttachment(context: Context, attachment: MessageAttachmentUi) {
    val intent = Intent(Intent.ACTION_VIEW).apply {
        setDataAndType(attachment.contentUri(context), attachment.contentType)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    try {
        context.startActivity(intent)
    } catch (_: ActivityNotFoundException) {
        Toast.makeText(context, "没有可打开此文件的应用", Toast.LENGTH_SHORT).show()
    }
}

internal fun shareCachedAttachment(context: Context, attachment: MessageAttachmentUi) {
    val uri = attachment.contentUri(context)
    val send = Intent(Intent.ACTION_SEND).apply {
        type = attachment.contentType
        putExtra(Intent.EXTRA_STREAM, uri)
        clipData = ClipData.newUri(context.contentResolver, attachment.filename, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    try {
        context.startActivity(Intent.createChooser(send, "分享 ${attachment.filename}"))
    } catch (_: ActivityNotFoundException) {
        Toast.makeText(context, "没有可分享此文件的应用", Toast.LENGTH_SHORT).show()
    }
}

/** 把已校验的私有缓存复制到用户通过系统文件选择器确定的位置。 */
internal suspend fun saveCachedAttachment(
    context: Context,
    attachment: MessageAttachmentUi,
    destination: Uri,
) = withContext(Dispatchers.IO) {
    // 1. 在系统 URI 边界打开目标，不把私有路径暴露给外部应用
    val output = context.contentResolver.openOutputStream(destination, "w")
        ?: throw IOException("无法写入所选位置")

    // 2. 完整复制并确认写入结束
    File(attachment.cachePath).inputStream().buffered().use { input ->
        output.buffered().use(input::copyTo)
    }
}

private fun MessageAttachmentUi.contentUri(context: Context) = FileProvider.getUriForFile(
    context,
    "${BuildConfig.APPLICATION_ID}.files",
    File(cachePath),
)
