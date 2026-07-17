package com.akashic.mobile.ui.web

import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Test

class MobileTextShareTest {
    @Test
    fun `text selection uses the Android plain-text chooser contract`() {
        assertEquals(
            MobileTextSharePayload(
                mimeType = "text/plain",
                text = "你 · 今天 10:21\n问题\n\nAkashic · 今天 10:22\n回答",
                chooserTitle = "分享消息",
            ),
            mobileTextSharePayload("你 · 今天 10:21\n问题\n\nAkashic · 今天 10:22\n回答"),
        )
    }

    @Test
    fun `inline text stops at the UTF-8 binder budget`() {
        val boundary = "a".repeat(MOBILE_INLINE_SHARE_MAX_UTF8_BYTES)

        assertEquals(MobileTextShareMode.INLINE, mobileTextShareMode(boundary))
        assertEquals(MobileTextShareMode.CACHE_FILE, mobileTextShareMode("${boundary}a"))
    }

    @Test
    fun `multibyte text uses its encoded byte size`() {
        val chinese = "你".repeat(MOBILE_INLINE_SHARE_MAX_UTF8_BYTES / 3)

        assertEquals(MobileTextShareMode.INLINE, mobileTextShareMode(chinese))
        assertEquals(MobileTextShareMode.CACHE_FILE, mobileTextShareMode("${chinese}你"))
    }

    @Test
    fun `cache pruning stays inside its narrow directory and prefix`() {
        val directory = Files.createTempDirectory("akashic-share-prune").toFile()
        val now = 2 * MOBILE_TEXT_SHARE_CACHE_TTL_MILLIS
        val expiredOwn = directory.resolve("${MOBILE_TEXT_SHARE_FILE_PREFIX}expired.txt").apply {
            writeText("expired")
            setLastModified(now - MOBILE_TEXT_SHARE_CACHE_TTL_MILLIS)
        }
        val freshOwn = directory.resolve("${MOBILE_TEXT_SHARE_FILE_PREFIX}fresh.txt").apply {
            writeText("fresh")
            setLastModified(now - MOBILE_TEXT_SHARE_CACHE_TTL_MILLIS + 1)
        }
        val unrelated = directory.resolve("another-feature.txt").apply {
            writeText("keep")
            setLastModified(0)
        }
        val nested = directory.resolve("nested").apply { mkdir() }
        val nestedOwn = nested.resolve("${MOBILE_TEXT_SHARE_FILE_PREFIX}nested.txt").apply {
            writeText("keep")
            setLastModified(0)
        }

        pruneMobileTextShareCache(directory, now)

        assertEquals(false, expiredOwn.exists())
        assertEquals(true, freshOwn.exists())
        assertEquals(true, unrelated.exists())
        assertEquals(true, nestedOwn.exists())
        directory.deleteRecursively()
    }
}
