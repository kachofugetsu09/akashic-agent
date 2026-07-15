package com.akashic.mobile.ui.web

import android.annotation.SuppressLint
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.webkit.JavascriptInterface
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import android.widget.Toast
import androidx.annotation.Keep
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import androidx.webkit.WebViewAssetLoader
import androidx.webkit.WebViewClientCompat
import com.akashic.mobile.ui.conversation.ConversationUiState
import com.akashic.mobile.ui.conversation.MessageUi
import java.io.File
import java.io.FileNotFoundException
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.encodeToString
import kotlinx.serialization.SerializationException
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json

private const val MOBILE_WEB_URL = "https://appassets.androidplatform.net/assets/mobile.html"

@SuppressLint("SetJavaScriptEnabled")
@Composable
fun MobileWebChat(
    state: ConversationUiState,
    onSelectSession: (String) -> Unit,
    onNewSession: () -> Unit,
    onRestartPairing: () -> Unit,
    onReloadFromServer: () -> Unit,
    onAttach: () -> Unit,
    onRemoveAttachment: (String) -> Unit,
    onRetryAttachment: (String) -> Unit,
    onRetryDownloadedAttachment: (String) -> Unit,
    onTouchDownloadedAttachment: (String) -> Unit,
    onOpenDownloadedAttachment: (String) -> Unit,
    onShareDownloadedAttachment: (String) -> Unit,
    onDismissError: () -> Unit,
    onSend: (String) -> Unit,
    onSendCommand: (String) -> Unit,
    onPluginUiCall: (String, String, String, String) -> Unit,
    onPluginUiResponsesAcknowledged: (Set<String>) -> Unit,
    onStop: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val latestState by rememberUpdatedState(state)
    val mediaRegistry = remember { MobileMediaRegistry() }
    var webView by remember { mutableStateOf<WebView?>(null) }
    var snapshotPump by remember { mutableStateOf<MobileSnapshotPump?>(null) }

    val callbacks by rememberUpdatedState(
        MobileWebCallbacks(
            onSelectSession = onSelectSession,
            onNewSession = onNewSession,
            onRestartPairing = onRestartPairing,
            onReloadFromServer = onReloadFromServer,
            onAttach = onAttach,
            onRemoveAttachment = onRemoveAttachment,
            onRetryAttachment = onRetryAttachment,
            onRetryDownloadedAttachment = onRetryDownloadedAttachment,
            onTouchDownloadedAttachment = onTouchDownloadedAttachment,
            onOpenDownloadedAttachment = onOpenDownloadedAttachment,
            onShareDownloadedAttachment = onShareDownloadedAttachment,
            onDismissError = onDismissError,
            onSend = onSend,
            onSendCommand = onSendCommand,
            onPluginUiCall = onPluginUiCall,
            onPluginUiResponsesAcknowledged = onPluginUiResponsesAcknowledged,
            onStop = onStop,
        ),
    )

    AndroidView(
        modifier = modifier,
        factory = { context ->
            val assetLoader = WebViewAssetLoader.Builder()
                .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(context))
                .addPathHandler("/media/", mediaRegistry)
                .build()
            WebView(context).apply {
                settings.javaScriptEnabled = true
                settings.domStorageEnabled = false
                settings.allowFileAccess = false
                settings.allowContentAccess = false
                settings.mixedContentMode = WebSettings.MIXED_CONTENT_NEVER_ALLOW
                settings.mediaPlaybackRequiresUserGesture = true
                settings.blockNetworkLoads = true
                setBackgroundColor(android.graphics.Color.TRANSPARENT)
                webViewClient = MobileWebClient(context, assetLoader)
                addJavascriptInterface(
                    MobileWebBridge(
                        dispatch = { work -> post { work(callbacks) } },
                        requestSnapshot = { post { snapshotPump?.request(latestState) } },
                    ),
                    "AkashicNative",
                )
                loadUrl(MOBILE_WEB_URL)
                webView = this
                snapshotPump = MobileSnapshotPump(this, mediaRegistry)
            }
        },
    )

    SideEffect { snapshotPump?.submit(state) }
    DisposableEffect(Unit) {
        onDispose {
            snapshotPump?.cancel()
            webView?.removeJavascriptInterface("AkashicNative")
            webView?.destroy()
        }
    }
}

private data class MobileWebCallbacks(
    val onSelectSession: (String) -> Unit,
    val onNewSession: () -> Unit,
    val onRestartPairing: () -> Unit,
    val onReloadFromServer: () -> Unit,
    val onAttach: () -> Unit,
    val onRemoveAttachment: (String) -> Unit,
    val onRetryAttachment: (String) -> Unit,
    val onRetryDownloadedAttachment: (String) -> Unit,
    val onTouchDownloadedAttachment: (String) -> Unit,
    val onOpenDownloadedAttachment: (String) -> Unit,
    val onShareDownloadedAttachment: (String) -> Unit,
    val onDismissError: () -> Unit,
    val onSend: (String) -> Unit,
    val onSendCommand: (String) -> Unit,
    val onPluginUiCall: (String, String, String, String) -> Unit,
    val onPluginUiResponsesAcknowledged: (Set<String>) -> Unit,
    val onStop: () -> Unit,
)

@Keep
private class MobileWebBridge(
    private val dispatch: ((MobileWebCallbacks) -> Unit) -> Unit,
    private val requestSnapshot: () -> Unit,
) {
    @JavascriptInterface
    fun requestSnapshot() = requestSnapshot.invoke()

    @JavascriptInterface
    fun selectSession(sessionId: String) = dispatch { it.onSelectSession(sessionId) }

    @JavascriptInterface
    fun createSession() = dispatch { it.onNewSession() }

    @JavascriptInterface
    fun restartPairing() = dispatch { it.onRestartPairing() }

    @JavascriptInterface
    fun reloadFromServer() = dispatch { it.onReloadFromServer() }

    @JavascriptInterface
    fun chooseAttachments() = dispatch { it.onAttach() }

    @JavascriptInterface
    fun removeAttachment(attachmentId: String) = dispatch { it.onRemoveAttachment(attachmentId) }

    @JavascriptInterface
    fun retryAttachment(attachmentId: String) = dispatch { it.onRetryAttachment(attachmentId) }

    @JavascriptInterface
    fun retryDownloadedAttachment(attachmentId: String) = dispatch {
        it.onRetryDownloadedAttachment(attachmentId)
    }

    @JavascriptInterface
    fun touchDownloadedAttachment(attachmentId: String) = dispatch {
        it.onTouchDownloadedAttachment(attachmentId)
    }

    @JavascriptInterface
    fun openDownloadedAttachment(attachmentId: String) = dispatch {
        it.onOpenDownloadedAttachment(attachmentId)
    }

    @JavascriptInterface
    fun shareDownloadedAttachment(attachmentId: String) = dispatch {
        it.onShareDownloadedAttachment(attachmentId)
    }

    @JavascriptInterface
    fun dismissError() = dispatch { it.onDismissError() }

    @JavascriptInterface
    fun sendMessage(text: String) = dispatch { it.onSend(text) }

    @JavascriptInterface
    fun sendCommand(command: String) = dispatch { it.onSendCommand(command) }

    @JavascriptInterface
    fun callPluginUi(requestId: String, pluginId: String, method: String, payloadJson: String) =
        dispatch { it.onPluginUiCall(requestId, pluginId, method, payloadJson) }

    @JavascriptInterface
    fun acknowledgePluginUiResponses(requestIdsJson: String) {
        if (requestIdsJson.toByteArray(Charsets.UTF_8).size > 64 * 1024) return
        val requestIds = try {
            Json.decodeFromString<List<String>>(requestIdsJson).toSet()
        } catch (_: SerializationException) {
            return
        } catch (_: IllegalArgumentException) {
            return
        }
        if (requestIds.size > 512 || requestIds.any { it.length !in 1..128 }) return
        dispatch { it.onPluginUiResponsesAcknowledged(requestIds) }
    }

    @JavascriptInterface
    fun stopTurn() = dispatch { it.onStop() }
}

private class MobileWebClient(
    private val context: Context,
    private val assetLoader: WebViewAssetLoader,
) : WebViewClientCompat() {
    override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? =
        assetLoader.shouldInterceptRequest(request.url)

    override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
        return when (mobileNavigationAction(request.url.toString(), request.isForMainFrame)) {
            MobileNavigationAction.ALLOW_INTERNAL -> false
            MobileNavigationAction.BLOCK -> true
            MobileNavigationAction.OPEN_EXTERNAL -> {
                try {
                    context.startActivity(Intent(Intent.ACTION_VIEW, request.url))
                } catch (_: ActivityNotFoundException) {
                    Toast.makeText(context, "没有可打开此链接的应用", Toast.LENGTH_SHORT).show()
                }
                true
            }
        }
    }
}

internal enum class MobileNavigationAction { ALLOW_INTERNAL, OPEN_EXTERNAL, BLOCK }

/** 只允许应用主页面留在 WebView，普通网页交给系统浏览器。 */
internal fun mobileNavigationAction(url: String, isMainFrame: Boolean): MobileNavigationAction {
    if (!isMainFrame) return MobileNavigationAction.BLOCK
    if (url == MOBILE_WEB_URL) return MobileNavigationAction.ALLOW_INTERNAL
    if (url.startsWith("https://appassets.androidplatform.net/")) {
        return MobileNavigationAction.BLOCK
    }
    val scheme = url.substringBefore(':', missingDelimiterValue = "").lowercase()
    return if (scheme in setOf("https", "http")) {
        MobileNavigationAction.OPEN_EXTERNAL
    } else {
        MobileNavigationAction.BLOCK
    }
}

private data class MobileMediaResource(val path: String, val contentType: String)

private class MobileMediaRegistry : WebViewAssetLoader.PathHandler {
    private val resources = AtomicReference<Map<String, MobileMediaResource>>(emptyMap())

    fun replace(next: Map<String, MobileMediaResource>) {
        resources.set(next.toMap())
    }

    override fun handle(path: String): WebResourceResponse? {
        val resource = resources.get()[path] ?: return null
        val file = File(resource.path)
        if (!file.isFile) return null
        val stream = try {
            file.inputStream()
        } catch (_: FileNotFoundException) {
            return null
        }
        return WebResourceResponse(resource.contentType, null, stream)
    }
}

private fun ConversationUiState.mediaResources(): Map<String, MobileMediaResource> =
    messages.asSequence()
        .flatMap { message ->
            when (message) {
                is MessageUi.User -> message.attachments.asSequence()
                is MessageUi.AssistantTurn -> message.attachments.asSequence()
            }
        }
        .filter { it.cachePath.isNotBlank() }
        .associate { it.id to MobileMediaResource(it.cachePath, it.contentType) }

private fun WebView.pushSnapshot(snapshotJson: String) {
    evaluateJavascript("window.AkashicMobile?.receiveSnapshot($snapshotJson)", null)
}

private fun WebView.pushPluginAssets(assetsJson: String) {
    evaluateJavascript("window.AkashicMobile?.receivePluginAssets($assetsJson)", null)
}

private class MobileSnapshotPump(
    private val webView: WebView,
    private val mediaRegistry: MobileMediaRegistry,
) {
    private val json = Json { explicitNulls = false }
    private val states = Channel<ConversationUiState>(Channel.CONFLATED)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var pluginSignature: String? = null
    private val forcePluginAssets = AtomicBoolean(false)

    init {
        scope.launch {
            for (first in states) {
                delay(16)
                var latest = first
                while (true) {
                    latest = states.tryReceive().getOrNull() ?: break
                }
                val snapshotJson = json.encodeToString(latest.toMobileWebSnapshot())
                val nextPluginSignature = latest.pluginUiAssets.joinToString("|") { "${it.id}:${it.sha256}" }
                val pluginAssetsJson = if (
                    forcePluginAssets.getAndSet(false) || nextPluginSignature != pluginSignature
                ) {
                    json.encodeToString(latest.toMobileWebPluginAssets())
                } else {
                    null
                }
                mediaRegistry.replace(latest.mediaResources())
                withContext(Dispatchers.Main.immediate) {
                    if (pluginAssetsJson != null) {
                        webView.pushPluginAssets(pluginAssetsJson)
                        pluginSignature = nextPluginSignature
                    }
                    webView.pushSnapshot(snapshotJson)
                }
            }
        }
    }

    fun submit(state: ConversationUiState) {
        states.trySend(state).getOrThrow()
    }

    fun request(state: ConversationUiState) {
        forcePluginAssets.set(true)
        submit(state)
    }

    fun cancel() {
        states.close()
        scope.cancel()
    }
}
