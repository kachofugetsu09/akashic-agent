package com.akashic.mobile.data.realtime.pluginui

import com.akashic.mobile.data.realtime.MobileUiCatalogItem
import com.akashic.mobile.data.realtime.MobileUiCatalogPayload
import com.akashic.mobile.data.realtime.WIRE_PROTOCOL_VERSION
import com.akashic.mobile.data.realtime.WireEnvelope
import com.akashic.mobile.data.realtime.WireKind
import java.nio.file.Files
import java.security.MessageDigest
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.encodeToJsonElement
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.put
import org.junit.Assert.assertEquals
import org.junit.Test

class PluginUiCoordinatorTest {
    @Test
    fun immutableQueriesUseSingleflightAndPersistAcrossCoordinatorRecreation() = runBlocking {
        val root = Files.createTempDirectory("plugin-ui-coordinator").toFile()
        try {
            val assets = PluginUiAssetStore(root.resolve("assets"))
            val catalogs = PluginUiCatalogStore(root.resolve("catalogs"))
            val results = PluginUiResultStore(root.resolve("results"))
            val module = "export default { slots: {} };"
            val moduleBytes = module.toByteArray(Charsets.UTF_8)
            val moduleSha = moduleBytes.sha256()
            assets.store(moduleSha, "module", module, moduleBytes.size)
            val catalog = MobileUiCatalogPayload(
                catalogRevision = "a".repeat(64),
                items = listOf(
                    MobileUiCatalogItem(
                        id = "observe",
                        revision = "1.2.0",
                        moduleSha256 = moduleSha,
                        moduleBytes = moduleBytes.size,
                        slots = listOf("turn.after_answer"),
                    ),
                ),
            )
            val sent = mutableListOf<Pair<String, String>>()
            val coordinator = PluginUiCoordinator(assets, catalogs, results) { type, _, _, _ ->
                "command-${sent.size + 1}".also { sent += type to it }
            }
            coordinator.onConnectionReady("mobile-lab")
            coordinator.onReply(catalogReply(sent.single().second, catalog))

            coordinator.query(
                "request-1", "owner-1", "turn.after_answer", "mobile:one", null,
                "observe", "kvcache.message_usage", "{\"message_id\":\"message-1\"}",
                "immutable", "mobile-lab",
            )
            coordinator.query(
                "request-2", "owner-2", "turn.after_answer", "mobile:one", null,
                "observe", "kvcache.message_usage", "{\"message_id\":\"message-1\"}",
                "immutable", "mobile-lab",
            )

            val queryCommands = sent.filter { it.first == "plugin.ui.query" }
            assertEquals(1, queryCommands.size)
            coordinator.onReply(queryReply(queryCommands.single().second))
            assertEquals(
                setOf("request-1", "request-2"),
                coordinator.results.take(2).toList().map { it.requestId }.toSet(),
            )

            val restoredSent = mutableListOf<String>()
            val restored = PluginUiCoordinator(assets, catalogs, results) { type, _, _, _ ->
                "restored-${restoredSent.size + 1}".also { restoredSent += type }
            }
            restored.onConnectionReady("mobile-lab")
            restored.query(
                "request-3", "owner-3", "turn.after_answer", "mobile:one", null,
                "observe", "kvcache.message_usage", "{\"message_id\":\"message-1\"}",
                "immutable", "mobile-lab",
            )
            assertEquals(listOf("plugin.ui.catalog"), restoredSent)
            assertEquals("request-3", restored.results.take(1).toList().single().requestId)
        } finally {
            root.deleteRecursively()
        }
    }

    private fun catalogReply(id: String, catalog: MobileUiCatalogPayload) = WireEnvelope(
        v = WIRE_PROTOCOL_VERSION,
        kind = WireKind.REPLY,
        type = "plugin.ui.catalog.ok",
        id = id,
        payload = Json.encodeToJsonElement(catalog).jsonObject,
    )

    private fun queryReply(id: String) = WireEnvelope(
        v = WIRE_PROTOCOL_VERSION,
        kind = WireKind.REPLY,
        type = "plugin.ui.query.ok",
        id = id,
        payload = buildJsonObject {
            put("result", buildJsonObject {
                put("usage", buildJsonObject { put("output_tokens", 321) })
            })
        },
    )

    private fun ByteArray.sha256(): String = MessageDigest.getInstance("SHA-256")
        .digest(this)
        .joinToString("") { "%02x".format(it) }
}
