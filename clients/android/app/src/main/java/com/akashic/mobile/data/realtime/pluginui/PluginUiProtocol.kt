package com.akashic.mobile.data.realtime.pluginui

import kotlinx.serialization.Serializable

@Serializable
data class PluginUiWebCatalog(
    val catalogRevision: String,
    val updating: Boolean,
    val error: String? = null,
    val plugins: List<PluginUiWebPlugin>,
)

@Serializable
data class PluginUiWebPlugin(
    val id: String,
    val revision: String,
    val moduleUrl: String,
    val stylesheetUrl: String? = null,
    val navigation: PluginUiWebNavigation? = null,
    val slots: List<String>,
)

@Serializable
data class PluginUiWebNavigation(
    val label: String,
    val description: String,
)

@Serializable
data class PluginUiWebResult(
    val requestId: String,
    val resultJson: String? = null,
    val error: String? = null,
)
