package com.akashic.mobile

import android.content.Intent
import android.net.Uri
import androidx.core.content.IntentCompat
import java.security.MessageDigest
import java.util.UUID

data class IncomingShare(
    val id: String,
    val text: String?,
    val uris: List<Uri>,
    val attachmentIds: List<String>? = null,
    val sourceFingerprint: String = incomingShareFingerprint(text, uris),
)

/** 解析系统分享边界，只接受文字和已授权的 content URI。 */
fun parseIncomingShare(intent: Intent): IncomingShare? {
    if (intent.action !in setOf(Intent.ACTION_SEND, Intent.ACTION_SEND_MULTIPLE)) return null
    if (intent.flags and Intent.FLAG_ACTIVITY_LAUNCHED_FROM_HISTORY != 0) return null

    // 1. 规范化分享文字
    val text = normalizeSharedText(
        text = intent.getCharSequenceExtra(Intent.EXTRA_TEXT)?.toString(),
        subject = intent.getCharSequenceExtra(Intent.EXTRA_SUBJECT)?.toString(),
    )

    // 2. 合并 ClipData 与 EXTRA_STREAM，并拒绝可伪造的文件路径
    val uris = buildList {
        intent.clipData?.let { clipData ->
            repeat(clipData.itemCount) { index -> clipData.getItemAt(index).uri?.let(::add) }
        }
        when (intent.action) {
            Intent.ACTION_SEND -> IntentCompat.getParcelableExtra(
                intent,
                Intent.EXTRA_STREAM,
                Uri::class.java,
            )?.let(::add)
            Intent.ACTION_SEND_MULTIPLE -> addAll(
                IntentCompat.getParcelableArrayListExtra(
                    intent,
                    Intent.EXTRA_STREAM,
                    Uri::class.java,
                ).orEmpty(),
            )
        }
    }.distinctBy(Uri::toString)
    require(uris.all { it.scheme == "content" }) { "只能分享系统授权的文件" }
    require(text != null || uris.isNotEmpty()) { "分享内容为空" }
    val shareId = intent.getStringExtra(EXTRA_INCOMING_SHARE_ID)?.let { rawId ->
        require(UUID.fromString(rawId).toString() == rawId) { "系统分享接收 ID 无效" }
        rawId
    } ?: UUID.randomUUID().toString().also { intent.putExtra(EXTRA_INCOMING_SHARE_ID, it) }
    return IncomingShare(shareId, text, uris)
}

internal fun incomingShareFingerprint(text: String?, uris: List<Uri>): String {
    val digest = MessageDigest.getInstance("SHA-256")
    digest.update(text.orEmpty().toByteArray(Charsets.UTF_8))
    uris.forEach { uri ->
        digest.update(0)
        digest.update(uri.toString().toByteArray(Charsets.UTF_8))
    }
    return digest.digest().joinToString("") { "%02x".format(it) }
}

internal fun normalizeSharedText(text: String?, subject: String?): String? {
    val normalizedText = text?.trim()?.takeIf(String::isNotEmpty)
    val normalizedSubject = subject?.trim()?.takeIf(String::isNotEmpty)
    val result = when {
        normalizedText == null -> normalizedSubject
        normalizedSubject != null && (
            normalizedText.startsWith("https://") || normalizedText.startsWith("http://")
        ) -> "$normalizedSubject\n$normalizedText"
        else -> normalizedText
    }
    require(result == null || result.length <= MAX_SHARED_TEXT_CHARS) {
        "分享文字不能超过 $MAX_SHARED_TEXT_CHARS 个字符"
    }
    return result
}

private const val MAX_SHARED_TEXT_CHARS = 65_536
private const val EXTRA_INCOMING_SHARE_ID = "com.akashic.mobile.extra.INCOMING_SHARE_ID"
