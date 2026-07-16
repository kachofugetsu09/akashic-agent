package com.akashic.mobile.data.local

import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import org.junit.Assert.assertEquals
import org.junit.Test

class StoredToolBlockTest {
    @Test
    fun decodesToolNameDescriptionAndResult() {
        val block = decodeStoredToolBlock(
            """tool.v1:{"name":"shell","description":"读取运行日志","resultPreview":"完成"}""",
        )

        assertEquals(StoredToolBlock("shell", "读取运行日志", "完成"), block)
    }

    @Test
    fun marksPreV1RowsAsLegacyToolDetails() {
        val block = decodeStoredToolBlock("旧版结果摘要")

        assertEquals(StoredToolBlock("工具调用", "旧版结果摘要"), block)
    }

    @Test
    fun decodesSafeArgumentsAndDuration() {
        val block = decodeStoredToolBlock(
            """tool.v1:{"name":"read_file","arguments":{"path":"/sandbox/a.md","secret":"[已隐藏]"},"durationMillis":840}""",
        )

        assertEquals("read_file", block.name)
        assertEquals(
            buildJsonObject {
                put("path", "/sandbox/a.md")
                put("secret", "[已隐藏]")
            },
            block.arguments,
        )
        assertEquals(840L, block.durationMillis)
    }
}
