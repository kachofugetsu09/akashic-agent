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

internal data class MobileTextSharePayload(
    val mimeType: String,
    val text: String,
    val chooserTitle: String,
)

internal enum class MobileTextShareMode {
    INLINE,
    CACHE_FILE,
}

internal const val MOBILE_INLINE_SHARE_MAX_UTF8_BYTES = 64 * 1024
internal const val MOBILE_TEXT_SHARE_CACHE_TTL_MILLIS = 24 * 60 * 60 * 1_000L
internal const val MOBILE_TEXT_SHARE_FILE_PREFIX = "akashic-message-"
internal const val MOBILE_TEXT_SHARE_CHOOSER_TITLE = "分享消息"

internal fun mobileTextShareMode(text: String) = if (
    text.toByteArray(Charsets.UTF_8).size <= MOBILE_INLINE_SHARE_MAX_UTF8_BYTES
) {
    MobileTextShareMode.INLINE
} else {
    MobileTextShareMode.CACHE_FILE
}

internal fun mobileTextSharePayload(text: String) = MobileTextSharePayload(
    mimeType = "text/plain",
    text = text,
    chooserTitle = MOBILE_TEXT_SHARE_CHOOSER_TITLE,
)

internal data class PreparedMobileTextShare(
    val intent: Intent,
    val cacheFile: File?,
)

internal fun mobileTextShareDirectory(context: Context) = File(context.cacheDir, "shared-text")

/** 只清理分享专用目录中过期且属于本功能的缓存文件。 */
internal fun pruneMobileTextShareCache(
    directory: File,
    nowMillis: Long = System.currentTimeMillis(),
) {
    directory.listFiles()?.forEach { file ->
        if (
            file.isFile &&
            file.name.startsWith(MOBILE_TEXT_SHARE_FILE_PREFIX) &&
            nowMillis - file.lastModified() >= MOBILE_TEXT_SHARE_CACHE_TTL_MILLIS
        ) {
            file.delete()
        }
    }
}

/** 把短文本放进 Intent，长文本改用 FileProvider 缓存文件。 */
internal fun preparePlainTextShare(context: Context, text: String): PreparedMobileTextShare {
    val payload = mobileTextSharePayload(text)
    val directory = mobileTextShareDirectory(context).apply {
        mkdirs()
        pruneMobileTextShareCache(this)
    }
    val send = Intent(Intent.ACTION_SEND).apply {
        type = payload.mimeType
    }
    if (mobileTextShareMode(text) == MobileTextShareMode.INLINE) {
        return PreparedMobileTextShare(
            intent = send.apply { putExtra(Intent.EXTRA_TEXT, payload.text) },
            cacheFile = null,
        )
    }

    val file = File.createTempFile(MOBILE_TEXT_SHARE_FILE_PREFIX, ".txt", directory)
    try {
        file.writeText(payload.text, Charsets.UTF_8)
    } catch (error: IOException) {
        file.delete()
        throw error
    }
    val uri = FileProvider.getUriForFile(context, "${BuildConfig.APPLICATION_ID}.files", file)
    return PreparedMobileTextShare(
        intent = send.apply {
            putExtra(Intent.EXTRA_STREAM, uri)
            clipData = ClipData.newUri(context.contentResolver, file.name, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        },
        cacheFile = file,
    )
}

internal fun launchPlainTextShare(context: Context, prepared: PreparedMobileTextShare): Boolean {
    try {
        context.startActivity(Intent.createChooser(prepared.intent, MOBILE_TEXT_SHARE_CHOOSER_TITLE))
        return true
    } catch (_: ActivityNotFoundException) {
        prepared.cacheFile?.delete()
        Toast.makeText(context, "没有可分享消息的应用", Toast.LENGTH_SHORT).show()
        return false
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
