package com.akashic.mobile.data.local

import java.io.File
import java.io.FileInputStream
import java.io.RandomAccessFile
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

class MediaCacheStore(
    root: File,
    private val dao: MediaAttachmentDao,
    private val quotaBytes: Long = DEFAULT_QUOTA_BYTES,
) {
    private val root = root.canonicalFile

    init {
        require(quotaBytes > 0) { "附件缓存配额必须为正数" }
        check(this.root.isDirectory || this.root.mkdirs()) { "无法创建附件缓存目录: ${this.root}" }
    }

    fun cachePath(attachmentId: String): String = root.resolve("${cacheKey(attachmentId)}.bin").absolutePath

    fun finalFile(transfer: MediaAttachmentEntity): File = requireRegularPath(rawFile(transfer, ".bin"))

    fun partialFile(transfer: MediaAttachmentEntity): File = requireRegularPath(rawFile(transfer, ".bin.part"))

    /** 把已上传草稿原子复制到 received cache，供本地消息长期引用。 */
    suspend fun importOutbound(
        transfer: AttachmentTransferEntity,
        source: File,
        updatedAt: Long,
    ): MediaAttachmentEntity {
        require(source.isFile && !Files.isSymbolicLink(source.toPath())) { "附件草稿文件无效" }
        require(source.length() == transfer.sizeBytes && sha256(source) == transfer.sha256.lowercase()) {
            "附件草稿内容与上传元数据不一致"
        }
        val cached = MediaAttachmentEntity(
            attachmentId = transfer.attachmentId,
            serverId = transfer.serverId,
            sessionId = transfer.sessionId,
            filename = transfer.filename,
            contentType = transfer.contentType,
            sizeBytes = transfer.sizeBytes,
            sha256 = transfer.sha256.lowercase(),
            transferredBytes = transfer.sizeBytes,
            state = "cached",
            cachePath = cachePath(transfer.attachmentId),
            lastAccessedAt = updatedAt,
            updatedAt = updatedAt,
        )
        val target = finalFile(cached)
        if (target.exists() && isComplete(target, cached)) return cached
        reserve(cached)
        val staged = root.resolve("${cacheKey(transfer.attachmentId)}.bin.import")
        require(!Files.isSymbolicLink(staged.toPath())) { "附件缓存暂存路径不能是符号链接" }
        try {
            FileInputStream(source).use { input ->
                RandomAccessFile(staged, "rw").use { output ->
                    output.setLength(0)
                    val buffer = ByteArray(1024 * 1024)
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        output.write(buffer, 0, count)
                    }
                    output.fd.sync()
                }
            }
            require(isComplete(staged, cached)) { "附件缓存复制校验失败" }
            Files.move(
                staged.toPath(),
                target.toPath(),
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING,
            )
            return cached
        } finally {
            Files.deleteIfExists(staged.toPath())
        }
    }

    /** 启动时校准 DB/文件状态、删除孤儿，并执行引用感知配额回收。 */
    suspend fun reconcile() {
        // 1. 先删除已无消息引用的记录及其文件
        dao.unreferenced().forEach { transfer ->
            deleteIfExists(rawFile(transfer, ".bin"))
            deleteIfExists(rawFile(transfer, ".bin.part"))
            check(dao.delete(transfer.attachmentId) == 1) { "未引用附件记录已消失" }
        }

        // 2. 以持久文件为准修复下载状态
        val records = dao.all()
        records.forEach { reconcileRecord(it) }
        val owned = records.flatMapTo(mutableSetOf()) {
            listOf(rawFile(it, ".bin").name, rawFile(it, ".bin.part").name)
        }
        requireNotNull(root.listFiles()) { "无法扫描附件缓存目录" }
            .filter { (it.isFile || Files.isSymbolicLink(it.toPath())) && it.name !in owned }
            .forEach(::deleteIfExists)

        // 3. 最后按 LRU 驱逐内容，保留描述符和消息引用以便重新下载
        trimToQuota(reserveBytes = 0, activeAttachmentId = null)
    }

    suspend fun enforceQuota() {
        trimToQuota(reserveBytes = 0, activeAttachmentId = null)
    }

    suspend fun reserve(transfer: MediaAttachmentEntity) {
        val remaining = Math.subtractExact(transfer.sizeBytes, partialFile(transfer).length())
        trimToQuota(remaining, transfer.attachmentId)
        require(root.usableSpace >= remaining) { "附件缓存磁盘空间不足" }
    }

    fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { stream ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private suspend fun reconcileRecord(transfer: MediaAttachmentEntity) {
        val final = rawFile(transfer, ".bin")
        val partial = rawFile(transfer, ".bin.part")
        if (Files.isSymbolicLink(final.toPath())) deleteIfExists(final)
        if (Files.isSymbolicLink(partial.toPath())) deleteIfExists(partial)
        if (transfer.state == "remote") {
            check(transfer.transferredBytes == 0L && !final.exists() && !partial.exists()) {
                "未请求附件不能持有本地内容: ${transfer.attachmentId}"
            }
            return
        }
        if (transfer.state == "evicted") {
            deleteIfExists(final)
            deleteIfExists(partial)
            if (transfer.transferredBytes != 0L) {
                check(dao.markEvicted(transfer.attachmentId, System.currentTimeMillis()) == 1)
            }
            return
        }
        if (final.exists()) {
            if (isComplete(final, transfer)) {
                deleteIfExists(partial)
                if (transfer.state != "cached" || transfer.transferredBytes != transfer.sizeBytes) {
                    check(dao.markCached(transfer.attachmentId, System.currentTimeMillis()) == 1)
                }
                return
            }
            deleteIfExists(final)
        }

        val actual = if (partial.exists()) partial.length() else 0L
        if (actual > transfer.sizeBytes) {
            deleteIfExists(partial)
            check(dao.updateDownload(transfer.attachmentId, 0, "failed", System.currentTimeMillis()) == 1)
            return
        }
        if (actual == transfer.sizeBytes && actual > 0) {
            if (isComplete(partial, transfer)) {
                Files.move(
                    partial.toPath(),
                    final.toPath(),
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING,
                )
                check(dao.markCached(transfer.attachmentId, System.currentTimeMillis()) == 1)
            } else {
                deleteIfExists(partial)
                check(dao.updateDownload(transfer.attachmentId, 0, "failed", System.currentTimeMillis()) == 1)
            }
            return
        }
        val state = if (transfer.state == "failed") "failed" else if (actual == 0L) "pending" else "downloading"
        if (transfer.transferredBytes != actual || transfer.state != state) {
            check(dao.updateDownload(transfer.attachmentId, actual, state, System.currentTimeMillis()) == 1)
        }
    }

    private suspend fun trimToQuota(reserveBytes: Long, activeAttachmentId: String?) {
        require(reserveBytes >= 0) { "附件缓存预留不能为负数" }
        val records = dao.all()
        var occupied = requireNotNull(root.listFiles()) { "无法扫描附件缓存目录" }
            .filter { it.isFile }
            .sumOf { it.length() }
        val candidates = records
            .filter { it.attachmentId != activeAttachmentId && it.state in EVICTABLE_STATES }
            .sortedWith(compareBy<MediaAttachmentEntity> { it.lastAccessedAt }.thenBy { it.attachmentId })
        for (transfer in candidates) {
            if (Math.addExact(occupied, reserveBytes) <= quotaBytes) break
            val final = rawFile(transfer, ".bin")
            val partial = rawFile(transfer, ".bin.part")
            val released = Math.addExact(final.length(), partial.length())
            deleteIfExists(final)
            deleteIfExists(partial)
            check(dao.markEvicted(transfer.attachmentId, System.currentTimeMillis()) == 1)
            occupied = Math.subtractExact(occupied, released)
        }
        require(Math.addExact(occupied, reserveBytes) <= quotaBytes) { "附件缓存配额不足" }
    }

    private fun isComplete(file: File, transfer: MediaAttachmentEntity): Boolean =
        !Files.isSymbolicLink(file.toPath()) &&
            file.isFile &&
            file.length() == transfer.sizeBytes &&
            sha256(file) == transfer.sha256.lowercase()

    private fun rawFile(transfer: MediaAttachmentEntity, suffix: String): File {
        val expected = root.resolve("${cacheKey(transfer.attachmentId)}$suffix")
        val configured = if (suffix == ".bin") File(transfer.cachePath) else File("${transfer.cachePath}.part")
        require(configured.absoluteFile == expected.absoluteFile) { "附件缓存路径不属于 cache owner" }
        require(configured.parentFile?.canonicalFile == root) { "附件缓存父目录越界" }
        return configured
    }

    private fun requireRegularPath(file: File): File {
        require(!Files.isSymbolicLink(file.toPath())) { "附件缓存不能是符号链接" }
        return file
    }

    private fun deleteIfExists(file: File) {
        val exists = Files.exists(file.toPath(), LinkOption.NOFOLLOW_LINKS)
        check(!exists || Files.deleteIfExists(file.toPath())) { "无法删除附件缓存文件: ${file.name}" }
    }

    private fun cacheKey(attachmentId: String): String =
        MessageDigest.getInstance("SHA-256")
            .digest(attachmentId.toByteArray(Charsets.UTF_8))
            .joinToString("") { "%02x".format(it) }

    private companion object {
        const val DEFAULT_QUOTA_BYTES = 512L * 1024 * 1024
        val EVICTABLE_STATES = setOf("cached", "failed", "pending", "downloading")
    }
}
