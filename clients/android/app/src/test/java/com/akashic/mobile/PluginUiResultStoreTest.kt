package com.akashic.mobile.data.realtime.pluginui

import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class PluginUiResultStoreTest {
    @Test
    fun immutableResultSurvivesStoreRecreationAndScopesDoNotCollide() {
        val root = Files.createTempDirectory("plugin-ui-results").toFile()
        try {
            val first = PluginUiResultStore(root)
            val labKey = first.identity("mobile-lab", "observe", "1.2.0", "usage", "{}")
            val mainKey = first.identity("main", "observe", "1.2.0", "usage", "{}")

            first.store(labKey, "{\"usage\":{\"output_tokens\":321}}")

            val restored = PluginUiResultStore(root)
            assertEquals("{\"usage\":{\"output_tokens\":321}}", restored.read(labKey))
            assertNull(restored.read(mainKey))
        } finally {
            root.deleteRecursively()
        }
    }
}
