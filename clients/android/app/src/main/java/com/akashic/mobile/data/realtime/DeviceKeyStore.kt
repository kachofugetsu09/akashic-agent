package com.akashic.mobile.data.realtime

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.security.KeyPairGenerator
import java.security.KeyStore
import java.security.MessageDigest
import java.security.PublicKey
import java.security.Signature
import java.security.spec.ECGenParameterSpec

class DeviceKeyStore {
    private val keyStore = KeyStore.getInstance(ANDROID_KEY_STORE).apply { load(null) }

    fun aliasForServer(serverId: String): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(serverId.toByteArray(Charsets.UTF_8))
        val encoded = Base64.encodeToString(digest, Base64.NO_WRAP or Base64.NO_PADDING or Base64.URL_SAFE)
        return "akashic-mobile-$encoded"
    }

    /** 创建不可导出的 P-256 设备签名密钥。 */
    fun create(alias: String): PublicKey {
        // 1. 防止无提示覆盖已配对设备身份
        check(!keyStore.containsAlias(alias)) { "Device key already exists: $alias" }

        // 2. 把签名用途与摘要约束交给 Android Keystore
        val specification = KeyGenParameterSpec.Builder(alias, KeyProperties.PURPOSE_SIGN)
            .setAlgorithmParameterSpec(ECGenParameterSpec("secp256r1"))
            .setDigests(KeyProperties.DIGEST_SHA256)
            .setUserAuthenticationRequired(false)
            .build()
        val generator = KeyPairGenerator.getInstance(KeyProperties.KEY_ALGORITHM_EC, ANDROID_KEY_STORE)
        generator.initialize(specification)
        return generator.generateKeyPair().public
    }

    fun contains(alias: String): Boolean = keyStore.containsAlias(alias)

    fun publicKey(alias: String): PublicKey = keyStore.getCertificate(alias).publicKey

    fun publicKeyDer(alias: String): ByteArray = publicKey(alias).encoded

    fun sign(alias: String, transcript: ByteArray): ByteArray {
        val privateKey = keyStore.getKey(alias, null) as java.security.PrivateKey
        return Signature.getInstance("SHA256withECDSA").run {
            initSign(privateKey)
            update(transcript)
            sign()
        }
    }

    fun privateKeyIsExportable(alias: String): Boolean {
        val privateKey = keyStore.getKey(alias, null) as java.security.PrivateKey
        return privateKey.encoded != null
    }

    fun delete(alias: String) {
        keyStore.deleteEntry(alias)
    }

    private companion object {
        const val ANDROID_KEY_STORE = "AndroidKeyStore"
    }
}
