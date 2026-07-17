package com.akashic.mobile.data.realtime.pluginui

import android.util.Log
import com.akashic.mobile.data.realtime.MobileUiCatalogPayload
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest
import kotlinx.serialization.SerializationException
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

class PluginUiCatalogStore(private val root: File) {
    private val json = Json { ignoreUnknownKeys = false; explicitNulls = false }

    init {
        check(root.exists() || root.mkdirs()) { "无法创建插件目录缓存: $root" }
        require(root.isDirectory) { "插件目录缓存根路径不是目录: $root" }
    }

    fun read(scope: String): MobileUiCatalogPayload? {
        val file = target(scope)
        if (!file.isFile) return null
        return try {
            json.decodeFromString<MobileUiCatalogPayload>(file.readText())
        } catch (error: SerializationException) {
            discardCorrupt(file, error)
        } catch (error: IllegalArgumentException) {
            discardCorrupt(file, error)
        }
    }

    /** 按服务端身份原子保存最后一次完整目录。 */
    fun store(scope: String, catalog: MobileUiCatalogPayload) {

        // 1. 序列化已经通过协议校验的完整目录
        val destination = target(scope)
        val bytes = json.encodeToString(catalog).toByteArray(Charsets.UTF_8)
        require(bytes.size <= MAX_CATALOG_BYTES) { "插件目录缓存超过预算" }

        // 2. 同文件系统原子替换旧快照
        val temporary = root.resolve(".${destination.name}.${System.nanoTime()}.tmp")
        FileOutputStream(temporary).use { output ->
            output.write(bytes)
            output.fd.sync()
        }
        if (destination.exists()) check(destination.delete()) { "无法替换插件目录缓存" }
        check(temporary.renameTo(destination)) { "插件目录缓存原子提交失败" }
    }

    fun discard(scope: String, error: IllegalArgumentException) {
        val file = target(scope)
        if (!file.exists()) return
        Log.w(TAG, "丢弃语义无效的插件目录缓存: ${file.name}", error)
        check(file.delete()) { "无法删除语义无效的插件目录缓存: $file" }
    }

    private fun target(scope: String): File {
        require(scope.isNotBlank()) { "插件目录缓存 scope 为空" }
        val identity = MessageDigest.getInstance("SHA-256")
            .digest(scope.toByteArray(Charsets.UTF_8))
            .joinToString("") { "%02x".format(it) }
        return root.resolve("$identity.json")
    }

    private fun discardCorrupt(file: File, error: RuntimeException): MobileUiCatalogPayload? {
        Log.w(TAG, "丢弃损坏的插件目录缓存: ${file.name}", error)
        check(file.delete()) { "无法删除损坏插件目录缓存: $file" }
        return null
    }

    private companion object {
        const val TAG = "PluginUiCatalogStore"
        const val MAX_CATALOG_BYTES = 512 * 1024
    }
}
