package com.akashic.mobile

import android.app.Application
import com.akashic.mobile.data.local.AppDatabase
import com.akashic.mobile.data.local.AppPreferences
import com.akashic.mobile.data.local.AttachmentDraftStore
import com.akashic.mobile.data.local.LocalDeliveryStore
import com.akashic.mobile.data.local.MediaCacheStore
import com.akashic.mobile.data.local.IncomingShareStore
import com.akashic.mobile.data.realtime.DeviceKeyStore
import com.akashic.mobile.data.realtime.RealtimeSession
import com.akashic.mobile.data.realtime.TransferNetworkMonitor
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob

class App : Application() {
    lateinit var container: AppContainer
        private set
    val visibility = AppVisibilityTracker()

    override fun onCreate() {
        super.onCreate()
        registerActivityLifecycleCallbacks(visibility)
        container = AppContainer(this)
    }
}

class AppContainer(application: Application) {
    private val applicationScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    val database = AppDatabase.create(application)
    private val mediaCacheRoot = application.filesDir.resolve("received-attachments")
    val mediaCacheStore = MediaCacheStore(mediaCacheRoot, database.mediaAttachments())
    val deliveryStore = LocalDeliveryStore(database, mediaCacheStore)
    val preferences = AppPreferences(application)
    val attachmentDraftStore = AttachmentDraftStore(
        contentResolver = application.contentResolver,
        root = application.filesDir.resolve("pending-attachments"),
        dao = database.attachmentTransfers(),
    )
    val incomingShareStore = IncomingShareStore(
        context = application,
        root = application.filesDir.resolve("incoming-shares"),
    )
    val deviceKeyStore = DeviceKeyStore()
    val transferNetwork = TransferNetworkMonitor(application)
    val realtimeSession = RealtimeSession(
        database = database,
        deliveryStore = deliveryStore,
        attachmentDrafts = attachmentDraftStore,
        mediaCache = mediaCacheStore,
        preferences = preferences,
        deviceKeys = deviceKeyStore,
        transferNetwork = transferNetwork.state,
        scope = applicationScope,
        allowInsecureTransport = BuildConfig.ALLOW_INSECURE_WS,
    )
}
