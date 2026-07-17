package com.akashic.mobile.data.realtime.pluginui

import android.webkit.WebView
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

class PluginUiWebBridge(private val webView: WebView) {
    private val json = Json { explicitNulls = false }

    fun publishCatalog(catalog: PluginUiWebCatalog) {
        val encoded = json.encodeToString(catalog)
        webView.evaluateJavascript(
            "window.AkashicMobile?.receivePluginCatalog($encoded)",
            null,
        )
    }

    fun publishResult(result: PluginUiWebResult) {
        val encoded = json.encodeToString(result)
        webView.evaluateJavascript(
            "window.AkashicMobile?.receivePluginUiResult($encoded)",
            null,
        )
    }
}
