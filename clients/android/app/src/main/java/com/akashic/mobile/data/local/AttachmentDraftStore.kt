package com.akashic.mobile.data.local

import android.content.ContentResolver
import android.net.Uri
import android.provider.OpenableColumns
import com.akashic.mobile.data.realtime.Ulid
import java.io.File
import java.security.MessageDigest
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

class AttachmentDraftStore(
    private val contentResolver: ContentResolver,
    private val root: File,
    private val dao: AttachmentTransferDao,
) {
    private val mutex = Mutex()

    /** 复制系统文档到私有目录，校验并持久化可跨重启上传的草稿。 */
    suspend fun import(
        serverId: String,
        sessionId: String,
        uris: List<Uri>,
        now: Long,
    ): List<AttachmentTransferEntity> = mutex.withLock {
        // 1. 在文档边界限制数量和整条消息大小
        require(uris.isNotEmpty()) { "没有选择文件" }
        val existing = dao.drafts(serverId, sessionId)
        require(existing.size + uris.size <= MAX_ATTACHMENTS_PER_MESSAGE) {
            "单条消息最多添加 $MAX_ATTACHMENTS_PER_MESSAGE 个附件"
        }

        // 2. 先完整复制整批文件，避免上传协调器观察到半批草稿
        val staged = mutableListOf<AttachmentTransferEntity>()
        val createdFiles = mutableListOf<File>()
        var totalBytes = existing.sumOf { it.sizeBytes }
        try {
            uris.forEachIndexed { index, uri ->
                val attachmentId = Ulid.next(now + index)
                val filename = queryFilename(uri)
                validateFilename(filename)
                val contentType = contentResolver.getType(uri) ?: "application/octet-stream"
                require(contentType.length <= 255 && MIME_TYPE.matches(contentType)) {
                    "文件 MIME type 无效：$contentType"
                }
                val file = fileFor(attachmentId)
                val copied = copyAndDigest(uri, file)
                createdFiles += file
                totalBytes += copied.sizeBytes
                if (totalBytes > MAX_ATTACHMENTS_BYTES) {
                    deleteFile(file)
                    throw IllegalArgumentException("单条消息附件总量不能超过 50 MiB")
                }
                staged += AttachmentTransferEntity(
                    attachmentId = attachmentId,
                    serverId = serverId,
                    sessionId = sessionId,
                    filename = filename,
                    contentType = contentType,
                    sizeBytes = copied.sizeBytes,
                    sha256 = copied.sha256,
                    transferredBytes = 0,
                    state = "pending",
                    updatedAt = now,
                )
            }
            dao.upsertAll(staged)
        } catch (error: Throwable) {
            createdFiles.forEach(::deleteFile)
            throw error
        }
        staged
    }

    suspend fun remove(attachmentId: String) = mutex.withLock {
        val transfer = requireNotNull(dao.get(attachmentId)) { "附件草稿不存在：$attachmentId" }
        check(dao.deleteDraft(attachmentId) == 1) { "上传中的附件不能移除" }
        deleteFile(fileFor(transfer.attachmentId))
    }

    suspend fun retry(attachmentId: String, now: Long) = mutex.withLock {
        val transfer = requireNotNull(dao.get(attachmentId)) { "附件草稿不存在：$attachmentId" }
        check(transfer.state == "failed") { "只有失败附件可以重试" }
        check(
            dao.updateState(
                attachmentId = attachmentId,
                transferredBytes = transfer.transferredBytes,
                state = "pending",
                updatedAt = now,
            ) == 1,
        )
    }

    suspend fun deleteSentFiles(attachmentIds: List<String>) = mutex.withLock {
        attachmentIds.forEach { deleteFile(fileFor(it)) }
        dao.deleteSent()
    }

    /** 启动时清理已发送记录和没有数据库所有者的私有文件。 */
    suspend fun reconcile() = mutex.withLock {
        // 1. 先完成已提交消息遗留的文件删除，再移除其记录
        root.mkdirs()
        check(root.isDirectory) { "附件私有目录不可用" }
        val transfers = dao.all()
        transfers.filter { it.state == "sent" }.forEach { deleteFile(fileFor(it.attachmentId)) }
        dao.deleteSent()

        // 2. 删除复制落盘后、Room 提交前退出留下的孤儿文件
        val ownedNames = transfers.filter { it.state != "sent" }
            .mapTo(mutableSetOf()) { "${it.attachmentId}.upload" }
        val files = requireNotNull(root.listFiles()) { "无法扫描附件私有目录" }
        files.filter { it.isFile && it.extension == "upload" && it.name !in ownedNames }
            .forEach(::deleteFile)
    }

    private fun queryFilename(uri: Uri): String {
        val projection = arrayOf(OpenableColumns.DISPLAY_NAME)
        contentResolver.query(uri, projection, null, null, null).use { cursor ->
            requireNotNull(cursor) { "无法读取文件信息" }
            require(cursor.moveToFirst()) { "文件信息为空" }
            val index = cursor.getColumnIndexOrThrow(OpenableColumns.DISPLAY_NAME)
            return requireNotNull(cursor.getString(index)) { "文件名为空" }
        }
    }

    private fun copyAndDigest(uri: Uri, target: File): CopiedAttachment {
        root.mkdirs()
        check(root.isDirectory) { "附件私有目录不可用" }
        val digest = MessageDigest.getInstance("SHA-256")
        var size = 0L
        try {
            val source = requireNotNull(contentResolver.openInputStream(uri)) { "无法打开所选文件" }
            source.buffered().use { input ->
                target.outputStream().buffered().use { output ->
                    val buffer = ByteArray(COPY_BUFFER_BYTES)
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        size += count
                        if (size > MAX_ATTACHMENTS_BYTES) {
                            throw IllegalArgumentException("单个附件不能超过 50 MiB")
                        }
                        digest.update(buffer, 0, count)
                        output.write(buffer, 0, count)
                    }
                }
            }
        } catch (error: Throwable) {
            deleteFile(target)
            throw error
        }
        if (size == 0L) {
            deleteFile(target)
            throw IllegalArgumentException("不能上传空文件")
        }
        return CopiedAttachment(size, digest.digest().toHex())
    }

    private fun validateFilename(filename: String) {
        require(filename.isNotBlank() && filename.length <= 255) { "文件名必须为 1..255 字符" }
        require('/' !in filename && '\\' !in filename) { "文件名不能包含路径分隔符" }
        require(filename.none { it.code < 32 || it.code == 127 }) { "文件名不能包含控制字符" }
    }

    private data class CopiedAttachment(val sizeBytes: Long, val sha256: String)

    fun fileFor(attachmentId: String): File = File(root, "$attachmentId.upload")

    private fun deleteFile(file: File) {
        check(!file.exists() || file.delete()) { "无法删除附件私有文件：${file.name}" }
    }

    private fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it) }

    private companion object {
        const val MAX_ATTACHMENTS_PER_MESSAGE = 10
        const val MAX_ATTACHMENTS_BYTES = 50L * 1024 * 1024
        const val COPY_BUFFER_BYTES = 64 * 1024
        val MIME_TYPE = Regex("^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$")
    }
}
