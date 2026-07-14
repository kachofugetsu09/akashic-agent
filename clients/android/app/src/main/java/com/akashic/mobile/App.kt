package com.akashic.mobile

import android.app.Application
import com.akashic.mobile.data.local.AppDatabase
import com.akashic.mobile.data.local.AppPreferences
import com.akashic.mobile.data.local.AttachmentDraftStore
import com.akashic.mobile.data.local.LocalDeliveryStore
import com.akashic.mobile.data.realtime.DeviceKeyStore
import com.akashic.mobile.data.realtime.RealtimeSession
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob

class App : Application() {
    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer(this)
    }
}

class AppContainer(application: Application) {
    private val applicationScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    val database = AppDatabase.create(application)
    val deliveryStore = LocalDeliveryStore(database)
    val preferences = AppPreferences(application)
    val attachmentDraftStore = AttachmentDraftStore(
        contentResolver = application.contentResolver,
        root = application.filesDir.resolve("pending-attachments"),
        dao = database.attachmentTransfers(),
    )
    val deviceKeyStore = DeviceKeyStore()
    val realtimeSession = RealtimeSession(
        database = database,
        deliveryStore = deliveryStore,
        attachmentDrafts = attachmentDraftStore,
        preferences = preferences,
        deviceKeys = deviceKeyStore,
        scope = applicationScope,
        allowInsecureTransport = BuildConfig.ALLOW_INSECURE_WS,
    )
}
