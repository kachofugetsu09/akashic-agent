package com.akashic.mobile.data.realtime.pluginui

import android.webkit.WebResourceResponse
import androidx.webkit.WebViewAssetLoader
import java.io.File
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.security.MessageDigest

class PluginUiAssetStore(private val root: File) : WebViewAssetLoader.PathHandler {
    init {
        check(root.exists() || root.mkdirs()) { "无法创建插件 UI 缓存目录: $root" }
        require(root.isDirectory) { "插件 UI 缓存根目录不是目录: $root" }
    }

    fun contains(sha256: String, kind: String, expectedBytes: Int): Boolean {
        validateIdentity(sha256, kind)
        val file = target(sha256, kind)
        if (!file.isFile) return false
        val valid = file.length() == expectedBytes.toLong() && file.readBytes().sha256() == sha256
        if (!valid) check(file.delete()) { "无法删除损坏插件资源: $sha256" }
        return valid
    }

    /** 校验摘要并原子提交一个不可变插件资源。 */
    fun store(sha256: String, kind: String, content: String, expectedBytes: Int) {

        // 1. 在网络边界验证长度和内容身份
        validateIdentity(sha256, kind)
        val bytes = content.toByteArray(Charsets.UTF_8)
        require(bytes.size == expectedBytes) { "插件 UI 资源长度不一致: $sha256" }
        require(bytes.sha256() == sha256) { "插件 UI 资源摘要不一致: $sha256" }
        if (contains(sha256, kind, expectedBytes)) return

        // 2. 同一文件系统内写临时文件并原子改名
        val directory = root.resolve(sha256)
        check(directory.exists() || directory.mkdirs()) { "无法创建插件资源目录: $directory" }
        val destination = target(sha256, kind)
        val temporary = directory.resolve(".${destination.name}.${System.nanoTime()}.tmp")
        FileOutputStream(temporary).use { output ->
            output.write(bytes)
            output.fd.sync()
        }
        if (destination.exists()) check(destination.delete()) { "无法删除损坏插件资源: $sha256" }
        check(temporary.renameTo(destination)) { "插件 UI 资源原子提交失败: $sha256" }
    }

    fun contentUrl(sha256: String, kind: String): String {
        validateIdentity(sha256, kind)
        return "https://appassets.androidplatform.net/plugin-ui/$sha256/${filename(kind)}"
    }

    override fun handle(path: String): WebResourceResponse? {
        val parts = path.split('/')
        if (parts.size != 2) return null
        val sha256 = parts[0]
        val kind = when (parts[1]) {
            "module.js" -> "module"
            "style.css" -> "stylesheet"
            else -> return null
        }
        if (!SHA256.matches(sha256)) return null
        val file = target(sha256, kind)
        if (!file.isFile) return null
        val stream = try {
            file.inputStream()
        } catch (_: FileNotFoundException) {
            return null
        }
        val mime = if (kind == "module") "text/javascript" else "text/css"
        return WebResourceResponse(mime, "utf-8", stream)
    }

    private fun target(sha256: String, kind: String): File =
        root.resolve(sha256).resolve(filename(kind))

    private fun validateIdentity(sha256: String, kind: String) {
        require(SHA256.matches(sha256)) { "插件 UI sha256 无效" }
        require(kind in setOf("module", "stylesheet")) { "插件 UI 资源类型无效" }
    }

    private fun filename(kind: String): String =
        if (kind == "module") "module.js" else "style.css"

    private fun ByteArray.sha256(): String = MessageDigest.getInstance("SHA-256")
        .digest(this)
        .joinToString("") { "%02x".format(it) }

    private companion object {
        val SHA256 = Regex("^[0-9a-f]{64}$")
    }
}
