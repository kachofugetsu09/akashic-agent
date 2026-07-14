package com.akashic.mobile

import android.app.Application
import com.akashic.mobile.data.local.AppDatabase
import com.akashic.mobile.data.local.AppPreferences
import com.akashic.mobile.data.local.LocalDeliveryStore
import com.akashic.mobile.data.realtime.DeviceKeyStore

class App : Application() {
    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer(this)
    }
}

class AppContainer(application: Application) {
    val database = AppDatabase.create(application)
    val deliveryStore = LocalDeliveryStore(database)
    val preferences = AppPreferences(application)
    val deviceKeyStore = DeviceKeyStore()
}
