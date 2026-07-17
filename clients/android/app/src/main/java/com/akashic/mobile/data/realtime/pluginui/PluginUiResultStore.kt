package com.akashic.mobile.data.realtime.pluginui

import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest

class PluginUiResultStore(private val root: File) {
    init {
        check(root.exists() || root.mkdirs()) { "无法创建插件 UI 结果缓存目录: $root" }
        require(root.isDirectory) { "插件 UI 结果缓存根目录不是目录: $root" }
    }

    fun identity(vararg parts: String): String = parts
        .joinToString("\u0000")
        .toByteArray(Charsets.UTF_8)
        .sha256()

    fun read(identity: String): String? {
        validateIdentity(identity)
        val file = root.resolve("$identity.json")
        if (!file.isFile) return null
        if (file.length() > MAX_RESULT_BYTES) {
            check(file.delete()) { "无法删除超限插件 UI 结果: $identity" }
            return null
        }
        file.setLastModified(System.currentTimeMillis())
        return file.readText()
    }

    fun discard(identity: String) {
        validateIdentity(identity)
        val file = root.resolve("$identity.json")
        if (file.exists()) check(file.delete()) { "无法删除损坏插件 UI 结果: $identity" }
    }

    /** 原子提交一个按完整查询身份隔离的不可变结果。 */
    fun store(identity: String, content: String) {

        // 1. 在持久化边界限制单条缓存体积
        validateIdentity(identity)
        val bytes = content.toByteArray(Charsets.UTF_8)
        require(bytes.size <= MAX_RESULT_BYTES) { "插件 UI 结果超过缓存预算" }

        // 2. 同文件系统提交并按最近使用时间裁剪
        val destination = root.resolve("$identity.json")
        val temporary = root.resolve(".$identity.${System.nanoTime()}.tmp")
        FileOutputStream(temporary).use { output ->
            output.write(bytes)
            output.fd.sync()
        }
        if (destination.exists()) check(destination.delete()) { "无法替换插件 UI 结果: $identity" }
        check(temporary.renameTo(destination)) { "插件 UI 结果原子提交失败: $identity" }
        trim()
    }

    private fun trim() {
        val files = root.listFiles { file -> file.isFile && file.extension == "json" }
            ?.sortedByDescending(File::lastModified)
            ?: return
        files.drop(MAX_RESULTS).forEach { file ->
            check(file.delete()) { "无法裁剪插件 UI 结果: ${file.name}" }
        }
    }

    private fun validateIdentity(identity: String) {
        require(SHA256.matches(identity)) { "插件 UI 结果身份无效" }
    }

    private fun ByteArray.sha256(): String = MessageDigest.getInstance("SHA-256")
        .digest(this)
        .joinToString("") { "%02x".format(it) }

    private companion object {
        const val MAX_RESULT_BYTES = 256 * 1024L
        const val MAX_RESULTS = 2_048
        val SHA256 = Regex("^[0-9a-f]{64}$")
    }
}
