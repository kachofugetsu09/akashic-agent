package com.akashic.mobile.data.realtime

import android.annotation.SuppressLint
import com.akashic.mobile.domain.model.EndpointRoute
import com.akashic.mobile.domain.model.ServerEndpoint
import java.security.cert.CertificateException
import java.security.cert.X509Certificate
import javax.net.ssl.X509TrustManager

internal fun validateEndpointSecurity(endpoint: ServerEndpoint) {
    when (endpoint.route) {
        EndpointRoute.LAN -> require(endpoint.tlsSpkiPins.isNotEmpty()) {
            "LAN endpoint requires a QR-bound SPKI pin"
        }
        EndpointRoute.TUNNEL -> require(endpoint.tlsSpkiPins.isEmpty()) {
            "Tunnel endpoint must use system trust without LAN pins"
        }
    }
}

/** 只信任 QR 已绑定的 LAN leaf SPKI；主机名仍由 OkHttp 默认 verifier 校验。 */
@SuppressLint("CustomX509TrustManager")
internal class LanPinnedTrustManager(private val acceptedPins: Set<String>) : X509TrustManager {
    init {
        require(acceptedPins.isNotEmpty()) { "At least one LAN SPKI pin is required" }
    }

    override fun checkClientTrusted(chain: Array<out X509Certificate>?, authType: String?) {
        throw CertificateException("Client certificates are not accepted")
    }

    override fun checkServerTrusted(chain: Array<out X509Certificate>?, authType: String?) {
        val leaf = chain?.firstOrNull() ?: throw CertificateException("Server certificate chain is empty")
        leaf.checkValidity()
        if (spkiSha256Pin(leaf.publicKey) !in acceptedPins) {
            throw CertificateException("LAN leaf SPKI does not match the QR pin")
        }
    }

    override fun getAcceptedIssuers(): Array<X509Certificate> = emptyArray()
}
