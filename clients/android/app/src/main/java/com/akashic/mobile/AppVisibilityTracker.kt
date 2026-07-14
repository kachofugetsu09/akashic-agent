package com.akashic.mobile

import android.app.Activity
import android.app.Application
import android.os.Bundle

class AppVisibilityTracker : Application.ActivityLifecycleCallbacks {
    private var startedActivityCount = 0

    @Volatile
    var isVisible: Boolean = false
        private set

    override fun onActivityStarted(activity: Activity) {
        startedActivityCount += 1
        isVisible = true
    }

    override fun onActivityStopped(activity: Activity) {
        startedActivityCount -= 1
        check(startedActivityCount >= 0) { "Activity lifecycle start/stop is unbalanced" }
        isVisible = startedActivityCount > 0
    }

    override fun onActivityCreated(activity: Activity, savedInstanceState: Bundle?) = Unit

    override fun onActivityResumed(activity: Activity) = Unit

    override fun onActivityPaused(activity: Activity) = Unit

    override fun onActivitySaveInstanceState(activity: Activity, outState: Bundle) = Unit

    override fun onActivityDestroyed(activity: Activity) = Unit
}
