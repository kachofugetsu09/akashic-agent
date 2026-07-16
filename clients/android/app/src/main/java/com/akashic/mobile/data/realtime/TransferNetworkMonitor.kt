package com.akashic.mobile.data.realtime

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

enum class TransferNetworkKind {
    UNAVAILABLE,
    UNMETERED,
    METERED,
}

data class TransferNetworkState(
    val kind: TransferNetworkKind,
    val cellular: Boolean,
)

/** 把系统默认网络转换为附件策略所需的计费语义。 */
class TransferNetworkMonitor(context: Context) {
    private val connectivity = context.getSystemService(ConnectivityManager::class.java)
    private val mutableState = MutableStateFlow(readCurrent())
    val state: StateFlow<TransferNetworkState> = mutableState.asStateFlow()

    init {
        connectivity.registerDefaultNetworkCallback(
            object : ConnectivityManager.NetworkCallback() {
                override fun onAvailable(network: Network) = publish(network)
                override fun onCapabilitiesChanged(network: Network, capabilities: NetworkCapabilities) {
                    mutableState.value = capabilities.toTransferNetworkState()
                }
                override fun onLost(network: Network) {
                    if (connectivity.activeNetwork == null) {
                        mutableState.value = TransferNetworkState(TransferNetworkKind.UNAVAILABLE, false)
                    } else {
                        mutableState.value = readCurrent()
                    }
                }
            },
        )
    }

    private fun publish(network: Network) {
        mutableState.value = connectivity.getNetworkCapabilities(network)
            ?.toTransferNetworkState()
            ?: TransferNetworkState(TransferNetworkKind.UNAVAILABLE, false)
    }

    private fun readCurrent(): TransferNetworkState {
        val network = connectivity.activeNetwork
            ?: return TransferNetworkState(TransferNetworkKind.UNAVAILABLE, false)
        return connectivity.getNetworkCapabilities(network)
            ?.toTransferNetworkState()
            ?: TransferNetworkState(TransferNetworkKind.UNAVAILABLE, false)
    }
}

private fun NetworkCapabilities.toTransferNetworkState() = TransferNetworkState(
    kind = if (hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED)) {
        TransferNetworkKind.UNMETERED
    } else {
        TransferNetworkKind.METERED
    },
    cellular = hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR),
)
