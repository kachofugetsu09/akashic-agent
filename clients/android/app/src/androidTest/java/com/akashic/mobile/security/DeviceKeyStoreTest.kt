package com.akashic.mobile.security

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.akashic.mobile.data.realtime.DeviceKeyStore
import java.security.Signature
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class DeviceKeyStoreTest {
    @Test
    fun deviceKeySignsAndCannotBeExported() {
        val store = DeviceKeyStore()
        val alias = "instrumented-${System.nanoTime()}"
        val transcript = "server-challenge|device-id|epoch".toByteArray()

        val publicKey = store.create(alias)
        try {
            val signature = store.sign(alias, transcript)
            val verifier = Signature.getInstance("SHA256withECDSA").apply {
                initVerify(publicKey)
                update(transcript)
            }

            assertTrue(verifier.verify(signature))
            assertFalse(store.privateKeyIsExportable(alias))
        } finally {
            store.delete(alias)
        }
    }
}
