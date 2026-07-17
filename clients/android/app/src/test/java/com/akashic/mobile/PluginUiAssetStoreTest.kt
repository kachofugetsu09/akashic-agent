package com.akashic.mobile.data.realtime.pluginui

import java.nio.file.Files
import java.security.MessageDigest
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class PluginUiAssetStoreTest {
    @Test
    fun storeCommitsVerifiedContentAndRejectsLaterCorruption() {
        val root = Files.createTempDirectory("plugin-ui-store").toFile()
        try {
            val store = PluginUiAssetStore(root)
            val content = "export const value = 1;"
            val bytes = content.toByteArray(Charsets.UTF_8)
            val sha256 = bytes.sha256()

            store.store(sha256, "module", content, bytes.size)
            assertTrue(store.contains(sha256, "module", bytes.size))

            root.resolve("$sha256/module.js").writeText("corrupt")
            assertFalse(store.contains(sha256, "module", bytes.size))
            assertFalse(root.resolve("$sha256/module.js").exists())
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun storeRejectsContentWhoseDigestDoesNotMatchCatalog() {
        val root = Files.createTempDirectory("plugin-ui-store").toFile()
        try {
            val content = "export const value = 1;"
            assertThrows(IllegalArgumentException::class.java) {
                PluginUiAssetStore(root).store("0".repeat(64), "module", content, content.length)
            }
        } finally {
            root.deleteRecursively()
        }
    }

    private fun ByteArray.sha256(): String = MessageDigest.getInstance("SHA-256")
        .digest(this)
        .joinToString("") { "%02x".format(it) }
}
