package com.akashic.mobile.data.local

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import android.webkit.MimeTypeMap
import androidx.core.content.FileProvider
import com.akashic.mobile.BuildConfig
import com.akashic.mobile.IncomingShare
import com.akashic.mobile.incomingShareFingerprint
import com.akashic.mobile.data.realtime.Ulid
import java.io.File
import java.util.UUID
import org.json.JSONArray
import org.json.JSONObject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

data class PersistedIncomingShare(
    val content: IncomingShare,
    val targetSessionId: String?,
    val preparedText: String?,
    val preparedReplyToMessageId: String?,
    val preparedBaseText: String?,
    val preparedBaseReplyToMessageId: String?,
    val preparedBaseUpdatedAt: Long?,
)

/** 在外部 URI 授权期内持久接收系统分享，并按确认结果清理私有 staging。 */
class IncomingShareStore(
    private val context: Context,
    private val root: File,
) {
    private val mutex = Mutex()
    private val preferences = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)

    suspend fun enqueue(incoming: IncomingShare): PersistedIncomingShare = locked {
        // 1. 在外部边界复制 URI，避免授权随来源 Activity 或进程消失
        require(UUID.fromString(incoming.id).toString() == incoming.id) { "系统分享接收 ID 无效" }
        val records = readRecords()
        val now = System.currentTimeMillis()
        val replayReceipt = records.lastOrNull()?.takeIf {
            it.targetSessionId == null &&
                it.preparedText == null &&
                it.receivedAt != null && now - it.receivedAt in 0..REPLAY_WINDOW_MS &&
                it.sourceFingerprint == incoming.sourceFingerprint
        }
        (records.firstOrNull { it.id == incoming.id } ?: replayReceipt)?.let { existing ->
            val normalized = existing.withStableAttachmentIds()
            if (normalized != existing) {
                writeRecords(readRecords().map { if (it.id == existing.id) normalized else it })
            }
            return@locked normalized.toPublic()
        }
        require(incoming.uris.size <= MAX_FILES) { "单次最多分享 $MAX_FILES 个附件" }
        val staging = root.resolve(".${incoming.id}")
        check(!staging.exists() && staging.mkdirs()) { "无法创建系统分享暂存目录" }
        val files = mutableListOf<String>()
        var totalBytes = 0L
        try {
            incoming.uris.forEachIndexed { index, uri ->
                val filename = uniqueFilename(staging, safeFilename(uri, index))
                val target = staging.resolve(filename)
                val source = requireNotNull(context.contentResolver.openInputStream(uri)) {
                    "无法读取共享附件"
                }
                source.buffered().use { input ->
                    target.outputStream().buffered().use { output ->
                        val buffer = ByteArray(COPY_BUFFER_BYTES)
                        while (true) {
                            val count = input.read(buffer)
                            if (count < 0) break
                            totalBytes += count
                            require(totalBytes <= MAX_BYTES) { "单次分享附件总量不能超过 50 MiB" }
                            output.write(buffer, 0, count)
                        }
                    }
                }
                require(target.length() > 0L) { "不能分享空文件" }
                files += "${incoming.id}/$filename"
            }
            val destination = root.resolve(incoming.id)
            check(staging.renameTo(destination)) { "无法提交系统分享暂存文件" }
            val attachmentIds = files.indices.map { index -> Ulid.next(System.currentTimeMillis() + index) }
            val record = StoredShare(
                id = incoming.id,
                text = incoming.text,
                targetSessionId = null,
                files = files,
                attachmentIds = attachmentIds,
                preparedText = null,
                preparedReplyToMessageId = null,
                preparedBaseText = null,
                preparedBaseReplyToMessageId = null,
                preparedBaseUpdatedAt = null,
                sourceFingerprint = incoming.sourceFingerprint,
                receivedAt = now,
            )
            try {
                writeRecords(readRecords() + record)
            } catch (error: Throwable) {
                destination.deleteRecursively()
                throw error
            }
            record.toPublic()
        } catch (error: Throwable) {
            staging.deleteRecursively()
            throw error
        }
    }

    suspend fun load(): List<PersistedIncomingShare> = locked {
        // 1. 为旧队列补齐稳定附件 ID，再向 UI 暴露可恢复记录
        val original = readRecords()
        val records = original.map { it.withStableAttachmentIds() }
        if (records != original) writeRecords(records)
        cleanupUnownedDirectories(records)
        records.map { it.toPublic() }
    }

    suspend fun claimTarget(id: String, sessionId: String) = locked {
        writeRecords(readRecords().map { record ->
            if (record.id == id) record.copy(targetSessionId = record.targetSessionId ?: sessionId)
            else record
        })
    }

    suspend fun prepareText(
        id: String,
        text: String,
        replyToMessageId: String?,
        baseText: String?,
        baseReplyToMessageId: String?,
        baseUpdatedAt: Long?,
    ) = locked {
        updateContent(id) { record ->
            check(record.preparedText == null || record.preparedText == text) {
                "系统分享草稿已被不同内容认领"
            }
            record.copy(
                preparedText = text,
                preparedReplyToMessageId = replyToMessageId,
                preparedBaseText = baseText,
                preparedBaseReplyToMessageId = baseReplyToMessageId,
                preparedBaseUpdatedAt = baseUpdatedAt,
            )
        }
    }

    suspend fun consumeText(id: String) = locked {
        updateContent(id) {
            it.copy(
                text = null,
                preparedText = null,
                preparedReplyToMessageId = null,
                preparedBaseText = null,
                preparedBaseReplyToMessageId = null,
                preparedBaseUpdatedAt = null,
            )
        }
    }

    suspend fun consumeFiles(id: String) = locked {
        val records = readRecords()
        val consumed = records.firstOrNull { it.id == id } ?: return@locked
        val updated = records.mapNotNull { record ->
            if (record.id != id) return@mapNotNull record
            record.copy(files = emptyList(), attachmentIds = emptyList()).takeIf {
                it.text != null || it.preparedText != null
            }
        }
        writeRecords(updated)
        consumed.files.forEach { relative -> check(root.resolve(relative).delete()) }
        if (updated.none { it.id == id }) root.resolve(id).deleteRecursively()
    }

    suspend fun discard(id: String) = locked {
        val records = readRecords()
        val discarded = records.firstOrNull { it.id == id }
        writeRecords(records.filterNot { it.id == id })
        discarded?.files?.forEach { relative ->
            check(!root.resolve(relative).exists() || root.resolve(relative).delete())
        }
        root.resolve(id).deleteRecursively()
    }

    private fun updateContent(id: String, transform: (StoredShare) -> StoredShare) {
        val updated = readRecords().mapNotNull { record ->
            if (record.id != id) return@mapNotNull record
            transform(record).takeIf {
                it.text != null || it.preparedText != null || it.files.isNotEmpty()
            }
        }
        writeRecords(updated)
        if (updated.none { it.id == id }) root.resolve(id).deleteRecursively()
    }

    private suspend fun <T> locked(block: () -> T): T = withContext(Dispatchers.IO) {
        mutex.withLock { block() }
    }

    private fun StoredShare.toPublic(): PersistedIncomingShare {
        val publicUris = files.map { relative ->
            FileProvider.getUriForFile(
                context,
                "${BuildConfig.APPLICATION_ID}.files",
                root.resolve(relative),
            )
        }
        return PersistedIncomingShare(
            content = IncomingShare(
                id = id,
                text = text.takeIf { preparedText == null },
                uris = publicUris,
                attachmentIds = requireNotNull(attachmentIds).also {
                    check(it.size == files.size) { "系统分享附件 ID 数量不匹配" }
                },
                sourceFingerprint = sourceFingerprint ?: incomingShareFingerprint(text, publicUris),
            ),
            targetSessionId = targetSessionId,
            preparedText = preparedText,
            preparedReplyToMessageId = preparedReplyToMessageId,
            preparedBaseText = preparedBaseText,
            preparedBaseReplyToMessageId = preparedBaseReplyToMessageId,
            preparedBaseUpdatedAt = preparedBaseUpdatedAt,
        )
    }

    private fun StoredShare.withStableAttachmentIds(): StoredShare {
        if (attachmentIds?.size == files.size) return this
        return copy(
            attachmentIds = files.indices.map { index ->
                Ulid.next(System.currentTimeMillis() + index)
            },
        )
    }

    private fun safeFilename(uri: Uri, index: Int): String {
        val displayName = context.contentResolver.query(
            uri,
            arrayOf(OpenableColumns.DISPLAY_NAME),
            null,
            null,
            null,
        )?.use { cursor ->
            if (!cursor.moveToFirst()) null
            else cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                .takeIf { it >= 0 }
                ?.let(cursor::getString)
        }
        val fallbackExtension = context.contentResolver.getType(uri)
            ?.let(MimeTypeMap.getSingleton()::getExtensionFromMimeType)
            ?.let { ".$it" }
            .orEmpty()
        val candidate = displayName?.takeIf(String::isNotBlank) ?: "shared-$index$fallbackExtension"
        require(candidate.length <= 255 && '/' !in candidate && '\\' !in candidate) {
            "共享附件文件名无效"
        }
        require(candidate.none { it.code < 32 || it.code == 127 }) { "共享附件文件名无效" }
        return candidate
    }

    private fun uniqueFilename(directory: File, filename: String): String {
        if (!directory.resolve(filename).exists()) return filename
        val extension = filename.substringAfterLast('.', missingDelimiterValue = "")
        val stem = if (extension.isEmpty()) filename else filename.removeSuffix(".$extension")
        var suffix = 2
        while (true) {
            val candidate = "$stem ($suffix)" + extension.takeIf(String::isNotEmpty)?.let { ".$it" }.orEmpty()
            if (!directory.resolve(candidate).exists()) return candidate
            suffix += 1
        }
    }

    private fun readRecords(): List<StoredShare> {
        val array = JSONArray(preferences.getString(KEY_QUEUE, "[]"))
        return buildList {
            repeat(array.length()) { index ->
                val item = array.getJSONObject(index)
                val fileArray = item.getJSONArray("files")
                add(
                    StoredShare(
                        id = item.getString("id"),
                        text = item.optString("text").takeIf { !item.isNull("text") },
                        targetSessionId = item.optString("targetSessionId")
                            .takeIf { !item.isNull("targetSessionId") },
                        files = buildList {
                            repeat(fileArray.length()) { fileIndex -> add(fileArray.getString(fileIndex)) }
                        },
                        attachmentIds = item.optJSONArray("attachmentIds")?.let { attachmentIdArray ->
                            buildList {
                                repeat(attachmentIdArray.length()) { attachmentIndex ->
                                    add(attachmentIdArray.getString(attachmentIndex))
                                }
                            }
                        },
                        preparedText = item.optString("preparedText")
                            .takeIf { !item.isNull("preparedText") },
                        preparedReplyToMessageId = item.optString("preparedReplyToMessageId")
                            .takeIf { !item.isNull("preparedReplyToMessageId") },
                        preparedBaseText = item.optString("preparedBaseText")
                            .takeIf { !item.isNull("preparedBaseText") },
                        preparedBaseReplyToMessageId = item.optString("preparedBaseReplyToMessageId")
                            .takeIf { !item.isNull("preparedBaseReplyToMessageId") },
                        preparedBaseUpdatedAt = item.optLong("preparedBaseUpdatedAt")
                            .takeIf { !item.isNull("preparedBaseUpdatedAt") },
                        sourceFingerprint = item.optString("sourceFingerprint")
                            .takeIf { !item.isNull("sourceFingerprint") },
                        receivedAt = item.optLong("receivedAt").takeIf { !item.isNull("receivedAt") },
                    ),
                )
            }
        }
    }

    private fun writeRecords(records: List<StoredShare>) {
        val array = JSONArray()
        records.forEach { record ->
            array.put(
                JSONObject()
                    .put("id", record.id)
                    .put("text", record.text ?: JSONObject.NULL)
                    .put("targetSessionId", record.targetSessionId ?: JSONObject.NULL)
                    .put("files", JSONArray(record.files))
                    .put(
                        "attachmentIds",
                        record.attachmentIds?.let(::JSONArray) ?: JSONObject.NULL,
                    )
                    .put("preparedText", record.preparedText ?: JSONObject.NULL)
                    .put(
                        "preparedReplyToMessageId",
                        record.preparedReplyToMessageId ?: JSONObject.NULL,
                    )
                    .put("preparedBaseText", record.preparedBaseText ?: JSONObject.NULL)
                    .put(
                        "preparedBaseReplyToMessageId",
                        record.preparedBaseReplyToMessageId ?: JSONObject.NULL,
                    )
                    .put("preparedBaseUpdatedAt", record.preparedBaseUpdatedAt ?: JSONObject.NULL)
                    .put("sourceFingerprint", record.sourceFingerprint ?: JSONObject.NULL)
                    .put("receivedAt", record.receivedAt ?: JSONObject.NULL),
            )
        }
        check(preferences.edit().putString(KEY_QUEUE, array.toString()).commit()) {
            "无法持久化系统分享队列"
        }
    }

    private fun cleanupUnownedDirectories(records: List<StoredShare>) {
        root.mkdirs()
        check(root.isDirectory) { "系统分享暂存目录不可用" }
        val owned = records.mapTo(mutableSetOf(), StoredShare::id)
        requireNotNull(root.listFiles()) { "无法扫描系统分享暂存目录" }
            .filter { it.isDirectory && it.name.removePrefix(".") !in owned }
            .forEach { check(it.deleteRecursively()) }
    }

    private data class StoredShare(
        val id: String,
        val text: String?,
        val targetSessionId: String?,
        val files: List<String>,
        val attachmentIds: List<String>?,
        val preparedText: String?,
        val preparedReplyToMessageId: String?,
        val preparedBaseText: String?,
        val preparedBaseReplyToMessageId: String?,
        val preparedBaseUpdatedAt: Long?,
        val sourceFingerprint: String?,
        val receivedAt: Long?,
    )

    private companion object {
        const val PREFERENCES = "incoming_shares"
        const val KEY_QUEUE = "queue"
        const val MAX_FILES = 10
        const val MAX_BYTES = 50L * 1024 * 1024
        const val COPY_BUFFER_BYTES = 64 * 1024
        const val REPLAY_WINDOW_MS = 10_000L
    }
}
