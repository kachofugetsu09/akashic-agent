package com.akashic.mobile.data.realtime.pluginui

import com.akashic.mobile.data.realtime.MobileUiCatalogItem
import com.akashic.mobile.data.realtime.MobileUiCatalogPayload
import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class PluginUiCatalogStoreTest {
    @Test
    fun lastCompleteCatalogIsPersistedPerServerScope() {
        val root = Files.createTempDirectory("plugin-ui-catalog").toFile()
        try {
            val store = PluginUiCatalogStore(root)
            val catalog = MobileUiCatalogPayload(
                catalogRevision = "a".repeat(64),
                items = listOf(
                    MobileUiCatalogItem(
                        id = "observe",
                        revision = "1.2.0",
                        moduleSha256 = "b".repeat(64),
                        moduleBytes = 128,
                        slots = listOf("turn.after_answer"),
                    ),
                ),
            )

            store.store("mobile-lab", catalog)

            assertEquals(catalog, PluginUiCatalogStore(root).read("mobile-lab"))
            assertNull(store.read("main"))
        } finally {
            root.deleteRecursively()
        }
    }
}
