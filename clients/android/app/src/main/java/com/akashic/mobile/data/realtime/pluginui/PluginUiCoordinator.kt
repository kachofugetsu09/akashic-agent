package com.akashic.mobile.data.realtime.pluginui

import com.akashic.mobile.data.realtime.MobileUiAssetPayload
import com.akashic.mobile.data.realtime.MobileUiCatalogItem
import com.akashic.mobile.data.realtime.MobileUiCatalogPayload
import com.akashic.mobile.data.realtime.ProtocolCodec
import com.akashic.mobile.data.realtime.WireEnvelope
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put

class PluginUiCoordinator(
    val assetStore: PluginUiAssetStore,
    private val catalogStore: PluginUiCatalogStore,
    private val resultStore: PluginUiResultStore,
    private val send: (String, JsonObject, String?, String?) -> String?,
) {
    private data class PendingAsset(
        val item: MobileUiCatalogItem,
        val kind: String,
        val sha256: String,
        val bytes: Int,
    )

    private data class QuerySubscriber(
        val requestId: String,
        val ownerId: String,
    )

    private data class PendingQuery(
        val wireOwnerId: String,
        val cacheKey: String?,
        val subscribers: MutableList<QuerySubscriber>,
    )

    private val json = Json { ignoreUnknownKeys = false; explicitNulls = false }
    private val mutableCatalog = MutableStateFlow(
        PluginUiWebCatalog(catalogRevision = "", updating = true, plugins = emptyList()),
    )
    val catalog: StateFlow<PluginUiWebCatalog> = mutableCatalog.asStateFlow()
    private val resultChannel = Channel<PluginUiWebResult>(capacity = 128)
    val results: Flow<PluginUiWebResult> = resultChannel.receiveAsFlow()
    private var pendingCatalogId: String? = null
    private var stagedCatalog: MobileUiCatalogPayload? = null
    private var refreshQueued = false
    private var acceptsQueries = false
    private var activeScope: String? = null
    private val pendingAssets = mutableMapOf<String, PendingAsset>()
    private val pendingQueries = mutableMapOf<String, PendingQuery>()
    private val pendingByCacheKey = mutableMapOf<String, String>()
    private val ignoredQueries = linkedSetOf<String>()
    private val pendingCancels = mutableSetOf<String>()

    /** 先激活当前服务端的最后成功快照，再增量检查远端目录。 */
    fun onConnectionReady(scope: String) {

        // 1. 切换服务端时绝不沿用上一台电脑的目录
        if (activeScope != scope) {
            activeScope = scope
            acceptsQueries = false
            mutableCatalog.value = PluginUiWebCatalog(
                catalogRevision = "",
                updating = true,
                plugins = emptyList(),
            )
            val cached = catalogStore.read(scope)
            if (cached != null) {
                try {
                    validateCatalog(cached)
                    if (cachedAssetsAvailable(cached)) {
                        stagedCatalog = cached
                        activateStagedCatalog()
                    }
                } catch (error: IllegalArgumentException) {
                    catalogStore.discard(scope, error)
                }
            }
        } else if (mutableCatalog.value.catalogRevision.isNotBlank()) {
            acceptsQueries = true
        }

        // 2. 快照只负责首帧，连接恢复后仍检查最新 revision
        requestCatalog()
    }

    /** 停止旧 generation 查询并启动完整 catalog 切换。 */
    fun onCatalogChanged() {

        acceptsQueries = false
        mutableCatalog.value = mutableCatalog.value.copy(updating = true, error = null)
        val owners = pendingQueries.values.map { it.wireOwnerId }.toSet()
        for (pending in pendingQueries.values) {
            pending.subscribers.forEach {
                publishResult(PluginUiWebResult(it.requestId, error = "插件界面正在更新"))
            }
        }
        rememberIgnoredQueries(pendingQueries.keys)
        pendingQueries.clear()
        pendingByCacheKey.clear()
        owners.forEach(::sendCancel)
        requestCatalog()
    }

    fun onDisconnected(reason: String) {
        pendingQueries.values.forEach { pending ->
            pending.subscribers.forEach {
                publishResult(PluginUiWebResult(it.requestId, error = reason))
            }
        }
        pendingCatalogId = null
        stagedCatalog = null
        pendingAssets.clear()
        pendingQueries.clear()
        pendingByCacheKey.clear()
        ignoredQueries.clear()
        pendingCancels.clear()
        refreshQueued = false
        acceptsQueries = false
        mutableCatalog.value = mutableCatalog.value.copy(updating = true)
    }

    /** 校验 WebView 请求并映射到当前插件 revision。 */
    fun query(
        requestId: String,
        ownerId: String,
        slot: String,
        sessionId: String?,
        turnId: String?,
        pluginId: String,
        method: String,
        payloadJson: String,
        cacheMode: String,
        cacheScope: String,
    ) {
        // 1. WebView 是外部输入边界
        if (requestId.length !in 1..128 || ownerId.length !in 1..128) return
        if (slot !in SLOT_NAMES) {
            publishResult(PluginUiWebResult(requestId, error = "插件 slot 无效"))
            return
        }
        if (cacheMode !in CACHE_MODES) {
            publishResult(PluginUiWebResult(requestId, error = "插件缓存模式无效"))
            return
        }
        if (cacheMode == "immutable" && !slot.startsWith("turn.")) {
            publishResult(PluginUiWebResult(requestId, error = "不可变缓存只能用于轮次投影"))
            return
        }
        val plugin = mutableCatalog.value.plugins.firstOrNull { it.id == pluginId }
        if (plugin == null) {
            publishResult(PluginUiWebResult(requestId, error = "插件界面尚未就绪"))
            return
        }
        val payload = try {
            json.parseToJsonElement(payloadJson).jsonObject
        } catch (error: SerializationException) {
            publishResult(PluginUiWebResult(requestId, error = "插件参数不是有效 JSON：${error.message}"))
            return
        } catch (_: IllegalArgumentException) {
            publishResult(PluginUiWebResult(requestId, error = "插件参数必须是 JSON 对象"))
            return
        }
        if (payloadJson.toByteArray(Charsets.UTF_8).size > 64 * 1024) {
            publishResult(PluginUiWebResult(requestId, error = "插件参数超过 64 KiB"))
            return
        }

        // 2. 不可变轮次投影优先从设备缓存读取
        val cacheKey = if (cacheMode == "immutable") {
            resultStore.identity(
                cacheScope,
                plugin.id,
                plugin.revision,
                method,
                payloadJson,
                sessionId.orEmpty(),
                turnId.orEmpty(),
            )
        } else {
            null
        }
        val cached = cacheKey?.let(resultStore::read)
        if (cached != null) {
            try {
                json.parseToJsonElement(cached).jsonObject
                publishResult(PluginUiWebResult(requestId, resultJson = cached))
                return
            } catch (_: SerializationException) {
                resultStore.discard(cacheKey)
            } catch (_: IllegalArgumentException) {
                resultStore.discard(cacheKey)
            }
        }
        if (!acceptsQueries) {
            publishResult(PluginUiWebResult(requestId, error = "插件界面尚未就绪"))
            return
        }
        val sharedCommandId = cacheKey?.let(pendingByCacheKey::get)
        if (sharedCommandId != null) {
            pendingQueries.getValue(sharedCommandId).subscribers += QuerySubscriber(requestId, ownerId)
            return
        }

        // 3. command id 只用于 wire，WebView request id 直接映射回结果
        val commandId = send(
            "plugin.ui.query",
            buildJsonObject {
                put("owner_id", ownerId)
                put("plugin_id", plugin.id)
                put("plugin_revision", plugin.revision)
                put("method", method)
                put("payload", payload)
                put("slot", slot)
            },
            sessionId,
            turnId,
        )
        if (commandId == null) {
            publishResult(PluginUiWebResult(requestId, error = "连接不可用，插件请求未发送"))
            return
        }
        pendingQueries[commandId] = PendingQuery(
            wireOwnerId = ownerId,
            cacheKey = cacheKey,
            subscribers = mutableListOf(QuerySubscriber(requestId, ownerId)),
        )
        if (cacheKey != null) pendingByCacheKey[cacheKey] = commandId
    }

    fun cancelOwner(ownerId: String) {
        if (ownerId.length !in 1..128) return
        val abandoned = mutableListOf<Pair<String, PendingQuery>>()
        for ((commandId, pending) in pendingQueries) {
            pending.subscribers.removeAll { it.ownerId == ownerId }
            if (pending.subscribers.isEmpty()) abandoned += commandId to pending
        }
        if (abandoned.isEmpty()) return
        rememberIgnoredQueries(abandoned.map { it.first })
        abandoned.forEach { (commandId, pending) ->
            pendingQueries.remove(commandId)
            pending.cacheKey?.let(pendingByCacheKey::remove)
            sendCancel(pending.wireOwnerId)
        }
    }

    /** WebView 销毁时取消全部 owner，并丢弃只属于旧页面的缓冲结果。 */
    fun onWebViewDisposed() {
        val owners = pendingQueries.values.map { it.wireOwnerId }.toSet()
        rememberIgnoredQueries(pendingQueries.keys)
        pendingQueries.clear()
        pendingByCacheKey.clear()
        owners.forEach(::sendCancel)
        while (resultChannel.tryReceive().isSuccess) {
            // resultChannel 只承载瞬时 WebView 结果，旧页面销毁后没有恢复语义。
        }
    }

    fun reject(requestId: String, message: String) {
        publishResult(PluginUiWebResult(requestId, error = message))
    }

    fun onReply(envelope: WireEnvelope): Boolean {
        val id = envelope.id ?: return false
        if (id == pendingCatalogId) {
            applyCatalogReply(envelope)
            return true
        }
        val pendingAsset = pendingAssets.remove(id)
        if (pendingAsset != null) {
            applyAssetReply(envelope, pendingAsset)
            return true
        }
        val pendingQuery = pendingQueries.remove(id)
        if (pendingQuery != null) {
            pendingQuery.cacheKey?.let(pendingByCacheKey::remove)
            applyQueryReply(envelope, pendingQuery)
            return true
        }
        if (ignoredQueries.remove(id)) return true
        if (pendingCancels.remove(id)) return true
        return false
    }

    private fun requestCatalog() {
        if (pendingCatalogId != null || pendingAssets.isNotEmpty()) {
            refreshQueued = true
            return
        }
        refreshQueued = false
        pendingCatalogId = send(
            "plugin.ui.catalog",
            buildJsonObject {
                put("subscribe", true)
                val revision = mutableCatalog.value.catalogRevision
                if (revision.isNotBlank()) put("if_revision", revision)
            },
            null,
            null,
        )
        if (pendingCatalogId == null) failCatalog("插件目录请求未发送")
    }

    private fun applyCatalogReply(envelope: WireEnvelope) {
        pendingCatalogId = null
        if (envelope.type.endsWith(".error")) {
            failCatalog(envelope.payload["message"]?.jsonPrimitive?.content ?: "插件目录加载失败")
            return
        }
        if (envelope.type == "plugin.ui.catalog.not_modified") {
            val revision = envelope.payload["catalog_revision"]?.jsonPrimitive?.content
            require(revision == mutableCatalog.value.catalogRevision) {
                "插件目录 not_modified revision 不匹配"
            }
            acceptsQueries = true
            mutableCatalog.value = mutableCatalog.value.copy(updating = false, error = null)
            if (refreshQueued) requestCatalog()
            return
        }
        require(envelope.type == "plugin.ui.catalog.ok") { "插件目录 reply 类型不匹配" }
        val next = ProtocolCodec.decodePayload<MobileUiCatalogPayload>(envelope.payload)
        validateCatalog(next)
        stagedCatalog = next
        pendingAssets.clear()
        for (item in next.items) {
            if (!requestAssetIfMissing(item, "module", item.moduleSha256, item.moduleBytes)) return
            val styleSha = item.stylesheetSha256
            if (styleSha != null) {
                if (!requestAssetIfMissing(item, "stylesheet", styleSha, item.stylesheetBytes)) return
            }
        }
        if (pendingAssets.isEmpty()) activateStagedCatalog()
    }

    private fun requestAssetIfMissing(
        item: MobileUiCatalogItem,
        kind: String,
        sha256: String,
        bytes: Int,
    ): Boolean {
        if (assetStore.contains(sha256, kind, bytes)) return true
        val commandId = send(
            "plugin.ui.asset.get",
            buildJsonObject {
                put("plugin_id", item.id)
                put("plugin_revision", item.revision)
                put("kind", kind)
                put("sha256", sha256)
            },
            null,
            null,
        ) ?: run {
            failCatalog("插件资源请求未发送: ${item.id}.$kind")
            return false
        }
        pendingAssets[commandId] = PendingAsset(item, kind, sha256, bytes)
        return true
    }

    private fun applyAssetReply(envelope: WireEnvelope, expected: PendingAsset) {
        if (envelope.type.endsWith(".error")) {
            failCatalog(envelope.payload["message"]?.jsonPrimitive?.content ?: "插件资源加载失败")
            return
        }
        require(envelope.type == "plugin.ui.asset.get.ok") { "插件资源 reply 类型不匹配" }
        val asset = ProtocolCodec.decodePayload<MobileUiAssetPayload>(envelope.payload)
        require(asset.pluginId == expected.item.id) { "插件资源 ID 不匹配" }
        require(asset.pluginRevision == expected.item.revision) { "插件资源 revision 不匹配" }
        require(asset.kind == expected.kind && asset.sha256 == expected.sha256) {
            "插件资源身份不匹配"
        }
        assetStore.store(asset.sha256, asset.kind, asset.content, expected.bytes)
        if (pendingAssets.isEmpty()) {
            if (refreshQueued) {
                stagedCatalog = null
                requestCatalog()
            } else {
                activateStagedCatalog()
            }
        }
    }

    private fun activateStagedCatalog() {
        val next = requireNotNull(stagedCatalog) { "插件 catalog 激活缺少 staging" }
        val plugins = next.items.map { item ->
            PluginUiWebPlugin(
                id = item.id,
                revision = item.revision,
                moduleUrl = assetStore.contentUrl(item.moduleSha256, "module"),
                stylesheetUrl = item.stylesheetSha256?.let {
                    assetStore.contentUrl(it, "stylesheet")
                },
                navigation = item.navigation?.let {
                    PluginUiWebNavigation(it.label, it.description)
                },
                slots = item.slots,
            )
        }
        stagedCatalog = null
        acceptsQueries = true
        mutableCatalog.value = PluginUiWebCatalog(
            catalogRevision = next.catalogRevision,
            updating = false,
            error = null,
            plugins = plugins,
        )
        catalogStore.store(requireNotNull(activeScope) { "插件目录激活缺少服务端 scope" }, next)
    }

    private fun cachedAssetsAvailable(catalog: MobileUiCatalogPayload): Boolean = catalog.items.all { item ->
        assetStore.contains(item.moduleSha256, "module", item.moduleBytes) &&
            (
                item.stylesheetSha256 == null ||
                    assetStore.contains(item.stylesheetSha256, "stylesheet", item.stylesheetBytes)
            )
    }

    private fun applyQueryReply(envelope: WireEnvelope, pending: PendingQuery) {
        if (envelope.type == "plugin.ui.query.ok") {
            val result = requireNotNull(envelope.payload["result"]?.jsonObject) {
                "插件查询成功 reply 缺少 result"
            }
            val resultJson = result.toString()
            if (pending.cacheKey != null && result.values.any { it !== JsonNull }) {
                resultStore.store(pending.cacheKey, resultJson)
            }
            pending.subscribers.forEach {
                publishResult(PluginUiWebResult(it.requestId, resultJson = resultJson))
            }
            return
        }
        require(envelope.type == "plugin.ui.query.error") { "插件查询 reply 类型不匹配" }
        val message = envelope.payload["message"]?.jsonPrimitive?.content ?: "插件请求失败"
        pending.subscribers.forEach {
            publishResult(PluginUiWebResult(it.requestId, error = message))
        }
    }

    private fun sendCancel(ownerId: String) {
        val commandId = send(
            "plugin.ui.cancel",
            buildJsonObject { put("owner_id", ownerId) },
            null,
            null,
        )
        if (commandId != null) pendingCancels += commandId
    }

    private fun failCatalog(message: String) {
        stagedCatalog = null
        pendingAssets.clear()
        val hasSnapshot = mutableCatalog.value.catalogRevision.isNotBlank()
        acceptsQueries = hasSnapshot
        mutableCatalog.value = mutableCatalog.value.copy(
            updating = false,
            error = if (hasSnapshot) null else message,
        )
    }

    private fun validateCatalog(catalog: MobileUiCatalogPayload) {
        require(SHA256.matches(catalog.catalogRevision)) { "插件 catalog revision 无效" }
        require(catalog.items.map { it.id }.distinct().size == catalog.items.size) {
            "插件 catalog ID 重复"
        }
        for (item in catalog.items) {
            require(PLUGIN_ID.matches(item.id)) { "插件 catalog ID 无效: ${item.id}" }
            require(item.revision.length in 1..128) { "插件 revision 无效: ${item.id}" }
            require(SHA256.matches(item.moduleSha256) && item.moduleBytes in 1..240 * 1024) {
                "插件 module 元数据无效: ${item.id}"
            }
            require(
                (item.stylesheetSha256 == null && item.stylesheetBytes == 0) ||
                    (
                        item.stylesheetSha256?.matches(SHA256) == true &&
                            item.stylesheetBytes in 1..240 * 1024
                    ),
            ) { "插件 stylesheet 元数据无效: ${item.id}" }
            require(item.moduleBytes + item.stylesheetBytes <= 240 * 1024) {
                "插件资源总量超过预算: ${item.id}"
            }
            require(item.slots.distinct().size == item.slots.size && item.slots.all(SLOT_NAMES::contains)) {
                "插件 slots 无效: ${item.id}"
            }
            item.navigation?.let {
                require(it.label.isNotBlank() && it.label.length <= 64) { "插件导航标题无效" }
                require(it.description.isNotBlank() && it.description.length <= 160) { "插件导航说明无效" }
            }
        }
    }

    private fun publishResult(result: PluginUiWebResult) {
        check(resultChannel.trySend(result).isSuccess) { "插件 UI 结果队列已满" }
    }

    private fun rememberIgnoredQueries(commandIds: Collection<String>) {
        ignoredQueries += commandIds
        while (ignoredQueries.size > MAX_IGNORED_QUERIES) {
            ignoredQueries.remove(ignoredQueries.first())
        }
    }

    private companion object {
        const val MAX_IGNORED_QUERIES = 128
        val SHA256 = Regex("^[0-9a-f]{64}$")
        val PLUGIN_ID = Regex("^[a-zA-Z0-9][a-zA-Z0-9_.@-]{0,128}$")
        val SLOT_NAMES = setOf(
            "dashboard.main",
            "turn.before_reasoning",
            "turn.before_tool",
            "turn.after_answer",
            "drawer.panel",
        )
        val CACHE_MODES = setOf("none", "immutable")
    }
}
