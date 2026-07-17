package com.akashic.mobile.ui.web

import android.annotation.SuppressLint
import android.content.ClipData
import android.content.ClipboardManager
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.webkit.JavascriptInterface
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.ConsoleMessage
import android.util.Log
import android.view.HapticFeedbackConstants
import android.widget.Toast
import androidx.activity.compose.BackHandler
import androidx.annotation.Keep
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.Alignment
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.webkit.WebResourceErrorCompat
import androidx.webkit.WebViewAssetLoader
import androidx.webkit.WebViewClientCompat
import com.akashic.mobile.BuildConfig
import com.akashic.mobile.ui.conversation.ConversationUiState
import com.akashic.mobile.ui.conversation.MessageUi
import java.io.ByteArrayInputStream
import java.io.File
import java.io.FileNotFoundException
import java.io.IOException
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
import org.json.JSONObject

private const val MOBILE_WEB_URL = "https://appassets.androidplatform.net/assets/mobile.html"
private const val MOBILE_WEB_LOG_TAG = "AkashicMobileWeb"
private const val MOBILE_WEB_RENDER_DEADLINE_MILLIS = 10_000L

data class MobileSharedTextDraft(
    val id: String,
    val sessionId: String,
    val text: String,
    val revision: Int,
)

@SuppressLint("SetJavaScriptEnabled")
@Composable
fun MobileWebChat(
    state: ConversationUiState,
    sharedTextDraft: MobileSharedTextDraft?,
    onCommitSharedText: (String, String, String, String?) -> Unit,
    onSharedTextRejected: (String, String) -> Unit,
    onSelectSession: (String) -> Unit,
    onRemoveUnavailableSession: (String) -> Unit,
    onNewSession: () -> Unit,
    onRestartPairing: () -> Unit,
    onReloadFromServer: () -> Unit,
    onAttach: () -> Unit,
    onRemoveAttachment: (String) -> Unit,
    onRetryAttachment: (String) -> Unit,
    onContinueMeteredTransfer: () -> Unit,
    onRetryFailedMessage: (String) -> Unit,
    onSaveComposerDraft: (String, String, String?) -> Unit,
    onSaveReadingPosition: (String, String, Int) -> Unit,
    onMarkSessionReadThrough: (String, Long) -> Unit,
    onNavigationTargetHandled: (String) -> Unit,
    onRetryDownloadedAttachment: (String) -> Unit,
    onTouchDownloadedAttachment: (String) -> Unit,
    onOpenDownloadedAttachment: (String) -> Unit,
    onShareDownloadedAttachment: (String) -> Unit,
    onSaveDownloadedAttachment: (String) -> Unit,
    onDismissError: () -> Unit,
    onSend: (String, String?, List<String>, (Boolean) -> Unit) -> Unit,
    onSendCommand: (String) -> Unit,
    onPluginUiCall: (String, String?, String?, String, String, String) -> Unit,
    onPluginUiResponsesAcknowledged: (Set<String>) -> Unit,
    onStop: () -> Unit,
    onBackAtRoot: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val latestState by rememberUpdatedState(state)
    val latestSharedTextDraft by rememberUpdatedState(sharedTextDraft)
    val mediaRegistry = remember { MobileMediaRegistry() }
    val shareScope = remember { CoroutineScope(SupervisorJob() + Dispatchers.IO) }
    var webView by remember { mutableStateOf<WebView?>(null) }
    var snapshotPump by remember { mutableStateOf<MobileSnapshotPump?>(null) }
    var webLoadError by remember { mutableStateOf<String?>(null) }
    var webHistoryActive by remember { mutableStateOf(false) }
    var webReady by remember { mutableStateOf(false) }

    val callbacks by rememberUpdatedState(
        MobileWebCallbacks(
            onCommitSharedText = onCommitSharedText,
            onSharedTextRejected = onSharedTextRejected,
            onSelectSession = onSelectSession,
            onRemoveUnavailableSession = onRemoveUnavailableSession,
            onNewSession = onNewSession,
            onRestartPairing = onRestartPairing,
            onReloadFromServer = onReloadFromServer,
            onAttach = onAttach,
            onRemoveAttachment = onRemoveAttachment,
            onRetryAttachment = onRetryAttachment,
            onContinueMeteredTransfer = onContinueMeteredTransfer,
            onRetryFailedMessage = onRetryFailedMessage,
            onSaveComposerDraft = onSaveComposerDraft,
            onSaveReadingPosition = onSaveReadingPosition,
            onMarkSessionReadThrough = onMarkSessionReadThrough,
            onNavigationTargetHandled = onNavigationTargetHandled,
            onRetryDownloadedAttachment = onRetryDownloadedAttachment,
            onTouchDownloadedAttachment = onTouchDownloadedAttachment,
            onOpenDownloadedAttachment = onOpenDownloadedAttachment,
            onShareDownloadedAttachment = onShareDownloadedAttachment,
            onSaveDownloadedAttachment = onSaveDownloadedAttachment,
            onDismissError = onDismissError,
            onSend = onSend,
            onSendCommand = onSendCommand,
            onPluginUiCall = onPluginUiCall,
            onPluginUiResponsesAcknowledged = onPluginUiResponsesAcknowledged,
            onStop = onStop,
        ),
    )

    Box(modifier = modifier) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { context ->
            shareScope.launch { pruneMobileTextShareCache(mobileTextShareDirectory(context)) }
            val assetLoader = WebViewAssetLoader.Builder()
                .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(context))
                .addPathHandler("/media/", mediaRegistry)
                .build()
            WebView(context).apply {
                settings.javaScriptEnabled = true
                settings.domStorageEnabled = false
                settings.cacheMode = WebSettings.LOAD_NO_CACHE
                settings.allowFileAccess = false
                settings.allowContentAccess = false
                settings.mixedContentMode = WebSettings.MIXED_CONTENT_NEVER_ALLOW
                settings.mediaPlaybackRequiresUserGesture = true
                // appassets 使用受控 HTTPS origin；外部请求由 MobileWebClient 拒绝。
                settings.blockNetworkLoads = false
                // WebView 默认走硬件合成；强制软件层会让动态页面退回整层位图重绘。
                setLayerType(android.view.View.LAYER_TYPE_NONE, null)
                setBackgroundColor(android.graphics.Color.rgb(243, 247, 252))
                webViewClient = MobileWebClient(
                    context,
                    assetLoader,
                    onMainFrameStarted = { post {
                        webLoadError = null
                        webReady = false
                    } },
                    onMainFrameError = { message -> post { webLoadError = message } },
                )
                addJavascriptInterface(
                    MobileWebBridge(
                        dispatch = { work -> post { work(callbacks) } },
                        reportReady = { post {
                            webLoadError = null
                            webReady = true
                        } },
                        requestSnapshot = {
                            post {
                                val pump = snapshotPump
                                Log.i(MOBILE_WEB_LOG_TAG, "bridge requestSnapshot: pumpReady=${pump != null}")
                                pump?.request(latestState)
                            }
                        },
                        copyText = { text ->
                            context.getSystemService(ClipboardManager::class.java).setPrimaryClip(
                                ClipData.newPlainText("Akashic message", text),
                            )
                        },
                        shareText = { requestId, text ->
                            shareScope.launch {
                                val prepared = try {
                                    preparePlainTextShare(context, text)
                                } catch (_: IOException) {
                                    null
                                }
                                val launched = withContext(Dispatchers.Main) {
                                    if (prepared === null) {
                                        Toast.makeText(context, "分享文件准备失败，请重试", Toast.LENGTH_SHORT).show()
                                        false
                                    } else {
                                        launchPlainTextShare(context, prepared)
                                    }
                                }
                                post {
                                    evaluateJavascript(
                                        "window.AkashicMobile?.receiveShareResult(" +
                                            "${JSONObject.quote(requestId)},$launched)",
                                        null,
                                    )
                                }
                            }
                        },
                        performActionHaptic = {
                            post {
                                performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
                            }
                        },
                        setWebHistoryActive = { active -> post { webHistoryActive = active } },
                        reportSendResult = { requestId, accepted ->
                            post {
                                evaluateJavascript(
                                    "window.AkashicMobile?.receiveSendResult(" +
                                        "${JSONObject.quote(requestId)},$accepted)",
                                    null,
                                )
                            }
                        },
                    ),
                    "AkashicNative",
                )
                webChromeClient = object : WebChromeClient() {
                    override fun onConsoleMessage(message: ConsoleMessage): Boolean {
                        Log.e(
                            MOBILE_WEB_LOG_TAG,
                            "console ${message.messageLevel()} ${message.sourceId()}:${message.lineNumber()} ${message.message()}",
                        )
                        return true
                    }
                }
                loadUrl(mobileWebUrl(BuildConfig.VERSION_CODE))
                webView = this
                snapshotPump = MobileSnapshotPump(this, mediaRegistry)
            }
            },
        )
        if (webLoadError != null) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(32.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text(
                    text = "会话界面加载失败",
                    style = MaterialTheme.typography.titleMedium,
                    modifier = Modifier.padding(top = 96.dp),
                )
                Text(
                    text = requireNotNull(webLoadError),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(top = 8.dp),
                )
                TextButton(onClick = { webView?.reload() }) { Text("重新加载") }
            }
        }
    }

    BackHandler {
        val current = webView
        if (current == null) {
            onBackAtRoot()
        } else {
            current.evaluateJavascript(
                "window.AkashicMobile?.navigateBack?.() ?? false",
            ) { result ->
                if (!mobileWebBackHandled(result)) current.post(onBackAtRoot)
            }
        }
    }

    SideEffect { snapshotPump?.submit(state) }
    LaunchedEffect(webReady, sharedTextDraft?.id, sharedTextDraft?.revision, webView) {
        val current = webView
        val draft = latestSharedTextDraft
        if (webReady && current != null && draft != null) current.pushSharedTextDraft(draft)
    }
    BackHandler(enabled = webHistoryActive) {
        val current = requireNotNull(webView) { "Web history owner is unavailable" }
        check(current.canGoBack()) { "Web history active without a back entry" }
        webHistoryActive = false
        current.goBack()
    }
    DisposableEffect(Unit) {
        onDispose {
            shareScope.cancel()
            snapshotPump?.cancel()
            webView?.removeJavascriptInterface("AkashicNative")
            webView?.destroy()
        }
    }
}

private data class MobileWebCallbacks(
    val onCommitSharedText: (String, String, String, String?) -> Unit,
    val onSharedTextRejected: (String, String) -> Unit,
    val onSelectSession: (String) -> Unit,
    val onRemoveUnavailableSession: (String) -> Unit,
    val onNewSession: () -> Unit,
    val onRestartPairing: () -> Unit,
    val onReloadFromServer: () -> Unit,
    val onAttach: () -> Unit,
    val onRemoveAttachment: (String) -> Unit,
    val onRetryAttachment: (String) -> Unit,
    val onContinueMeteredTransfer: () -> Unit,
    val onRetryFailedMessage: (String) -> Unit,
    val onSaveComposerDraft: (String, String, String?) -> Unit,
    val onSaveReadingPosition: (String, String, Int) -> Unit,
    val onMarkSessionReadThrough: (String, Long) -> Unit,
    val onNavigationTargetHandled: (String) -> Unit,
    val onRetryDownloadedAttachment: (String) -> Unit,
    val onTouchDownloadedAttachment: (String) -> Unit,
    val onOpenDownloadedAttachment: (String) -> Unit,
    val onShareDownloadedAttachment: (String) -> Unit,
    val onSaveDownloadedAttachment: (String) -> Unit,
    val onDismissError: () -> Unit,
    val onSend: (String, String?, List<String>, (Boolean) -> Unit) -> Unit,
    val onSendCommand: (String) -> Unit,
    val onPluginUiCall: (String, String?, String?, String, String, String) -> Unit,
    val onPluginUiResponsesAcknowledged: (Set<String>) -> Unit,
    val onStop: () -> Unit,
)

@Keep
private class MobileWebBridge(
    private val dispatch: ((MobileWebCallbacks) -> Unit) -> Unit,
    private val reportReady: () -> Unit,
    private val requestSnapshot: () -> Unit,
    private val copyText: (String) -> Unit,
    private val shareText: (String, String) -> Unit,
    private val performActionHaptic: () -> Unit,
    private val setWebHistoryActive: (Boolean) -> Unit,
    private val reportSendResult: (String, Boolean) -> Unit,
) {
    @JavascriptInterface
    fun reportReady() = reportReady.invoke()

    @JavascriptInterface
    fun requestSnapshot() = requestSnapshot.invoke()

    @JavascriptInterface
    fun commitSharedText(
        draftId: String,
        sessionId: String,
        text: String,
        replyToMessageId: String,
    ) = dispatch {
        it.onCommitSharedText(draftId, sessionId, text, replyToMessageId.ifBlank { null })
    }

    @JavascriptInterface
    fun rejectSharedText(draftId: String, message: String) = dispatch {
        it.onSharedTextRejected(draftId, message)
    }

    @JavascriptInterface
    fun selectSession(sessionId: String) = dispatch { it.onSelectSession(sessionId) }

    @JavascriptInterface
    fun removeUnavailableSession(sessionId: String) = dispatch {
        it.onRemoveUnavailableSession(sessionId)
    }

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
    fun continueMeteredTransfer() = dispatch { it.onContinueMeteredTransfer() }

    @JavascriptInterface
    fun retryFailedMessage(messageId: String) = dispatch { it.onRetryFailedMessage(messageId) }

    @JavascriptInterface
    fun saveComposerDraft(sessionId: String, text: String, replyToMessageId: String) = dispatch {
        it.onSaveComposerDraft(sessionId, text, replyToMessageId.ifBlank { null })
    }

    @JavascriptInterface
    fun saveReadingPosition(sessionId: String, messageId: String, offsetPx: Int) = dispatch {
        it.onSaveReadingPosition(sessionId, messageId, offsetPx)
    }

    @JavascriptInterface
    fun markSessionReadThrough(sessionId: String, readAtMillis: Long) = dispatch {
        it.onMarkSessionReadThrough(sessionId, readAtMillis)
    }

    @JavascriptInterface
    fun navigationTargetHandled(messageId: String) = dispatch {
        it.onNavigationTargetHandled(messageId)
    }

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
    fun saveDownloadedAttachment(attachmentId: String) = dispatch {
        it.onSaveDownloadedAttachment(attachmentId)
    }

    @JavascriptInterface
    fun setWebHistoryActive(active: Boolean) = setWebHistoryActive.invoke(active)

    @JavascriptInterface
    fun dismissError() = dispatch { it.onDismissError() }

    @JavascriptInterface
    fun sendMessage(
        requestId: String,
        text: String,
        replyToMessageId: String,
        attachmentIdsJson: String,
    ) = dispatch {
        val attachmentIds = try {
            Json.decodeFromString<List<String>>(attachmentIdsJson)
        } catch (_: SerializationException) {
            reportSendResult(requestId, false)
            return@dispatch
        }
        if (attachmentIds.distinct().size != attachmentIds.size) {
            reportSendResult(requestId, false)
            return@dispatch
        }
        it.onSend(text, replyToMessageId.ifBlank { null }, attachmentIds) { accepted ->
            reportSendResult(requestId, accepted)
        }
    }

    @JavascriptInterface
    fun copyText(text: String) = copyText.invoke(text)

    @JavascriptInterface
    fun shareText(requestId: String, text: String) = shareText.invoke(requestId, text)

    @JavascriptInterface
    fun performActionHaptic() = performActionHaptic.invoke()

    @JavascriptInterface
    fun sendCommand(command: String) = dispatch { it.onSendCommand(command) }

    @JavascriptInterface
    fun callPluginUi(
        requestId: String,
        sessionId: String?,
        turnId: String?,
        pluginId: String,
        method: String,
        payloadJson: String,
    ) = dispatch { it.onPluginUiCall(requestId, sessionId, turnId, pluginId, method, payloadJson) }

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
    private val onMainFrameStarted: WebView.() -> Unit,
    private val onMainFrameError: WebView.(String) -> Unit,
) : WebViewClientCompat() {
    private var pageGeneration = 0L

    override fun onPageStarted(view: WebView, url: String, favicon: android.graphics.Bitmap?) {
        Log.i(MOBILE_WEB_LOG_TAG, "page started: $url")
        pageGeneration += 1
        view.onMainFrameStarted()
    }

    override fun onPageFinished(view: WebView, url: String) {
        Log.i(MOBILE_WEB_LOG_TAG, "page finished: $url")
        if (!isMobileWebUrl(url)) return
        val generation = pageGeneration
        view.postDelayed({
            if (generation != pageGeneration) return@postDelayed
            view.evaluateJavascript(
                """document.getElementById('root')?.childElementCount > 0""",
            ) { rendered ->
                Log.i(MOBILE_WEB_LOG_TAG, "page deadline rendered: $rendered")
                if (rendered != "true") view.onMainFrameError("会话脚本没有生成界面")
            }
        }, MOBILE_WEB_RENDER_DEADLINE_MILLIS)
    }

    override fun onReceivedError(
        view: WebView,
        request: WebResourceRequest,
        error: WebResourceErrorCompat,
    ) {
        Log.e(MOBILE_WEB_LOG_TAG, "resource error: ${request.url} ${error.errorCode} ${error.description}")
        if (request.isForMainFrame || request.isCriticalAppAsset()) {
            view.onMainFrameError("资源加载失败: ${request.url.path} (${error.description})")
        }
    }

    override fun onReceivedHttpError(
        view: WebView,
        request: WebResourceRequest,
        errorResponse: WebResourceResponse,
    ) {
        if (request.isForMainFrame || request.isCriticalAppAsset()) {
            Log.e(
                MOBILE_WEB_LOG_TAG,
                "critical HTTP error: ${request.url} ${errorResponse.statusCode} ${errorResponse.reasonPhrase}",
            )
            view.onMainFrameError(
                "资源加载失败: ${request.url.path} (HTTP ${errorResponse.statusCode})",
            )
        }
    }

    override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? {
        if (request.url.host == "appassets.androidplatform.net") {
            return assetLoader.shouldInterceptRequest(request.url) ?: blockedResponse(404, "Not Found")
        }
        return blockedResponse(403, "Blocked")
    }

    private fun blockedResponse(statusCode: Int, reason: String) =
        WebResourceResponse(
            "text/plain",
            "utf-8",
            statusCode,
            reason,
            emptyMap(),
            ByteArrayInputStream(ByteArray(0)),
        )

    private fun WebResourceRequest.isCriticalAppAsset(): Boolean =
        url.host == "appassets.androidplatform.net" && url.lastPathSegment.orEmpty().startsWith("mobile-")

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

internal fun mobileWebBackHandled(javascriptResult: String?): Boolean = javascriptResult == "true"

internal fun mobileWebUrl(versionCode: Int): String {
    require(versionCode > 0) { "应用版本号必须为正数" }
    return "$MOBILE_WEB_URL?appVersion=$versionCode"
}

private fun isMobileWebUrl(url: String): Boolean = url.substringBefore('?') == MOBILE_WEB_URL

/** 只允许应用主页面留在 WebView，普通网页交给系统浏览器。 */
internal fun mobileNavigationAction(url: String, isMainFrame: Boolean): MobileNavigationAction {
    if (!isMainFrame) return MobileNavigationAction.BLOCK
    if (isMobileWebUrl(url)) return MobileNavigationAction.ALLOW_INTERNAL
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
    evaluateJavascript(
        """(() => { if (!window.AkashicMobile?.receiveSnapshot) return 'missing'; window.AkashicMobile.receiveSnapshot($snapshotJson); return 'delivered'; })()""",
    ) { result -> Log.d(MOBILE_WEB_LOG_TAG, "snapshot push: $result") }
}

private fun WebView.pushPluginAssets(assetsJson: String) {
    evaluateJavascript("window.AkashicMobile?.receivePluginAssets($assetsJson)", null)
}

private fun WebView.pushSharedTextDraft(draft: MobileSharedTextDraft) {
    evaluateJavascript(
        "window.AkashicMobile?.receiveSharedText(" +
            "${JSONObject.quote(draft.id)},${JSONObject.quote(draft.sessionId)}," +
            "${JSONObject.quote(draft.text)})",
        null,
    )
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
