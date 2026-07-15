package com.akashic.mobile.ui.web

import org.junit.Assert.assertEquals
import org.junit.Test

class MobileNavigationPolicyTest {
    @Test
    fun onlyAllowsBundledEntryPageInsideWebView() {
        assertEquals(
            MobileNavigationAction.ALLOW_INTERNAL,
            mobileNavigationAction(
                "https://appassets.androidplatform.net/assets/mobile.html",
                isMainFrame = true,
            ),
        )
        assertEquals(
            MobileNavigationAction.BLOCK,
            mobileNavigationAction(
                "https://appassets.androidplatform.net/media/untrusted.html",
                isMainFrame = true,
            ),
        )
        assertEquals(
            MobileNavigationAction.BLOCK,
            mobileNavigationAction("https://example.com/frame", isMainFrame = false),
        )
    }

    @Test
    fun onlyHandsHttpLinksToExternalApps() {
        assertEquals(
            MobileNavigationAction.OPEN_EXTERNAL,
            mobileNavigationAction("https://example.com/page", isMainFrame = true),
        )
        assertEquals(
            MobileNavigationAction.BLOCK,
            mobileNavigationAction("intent://unsafe", isMainFrame = true),
        )
        assertEquals(
            MobileNavigationAction.BLOCK,
            mobileNavigationAction("file:///tmp/unsafe", isMainFrame = true),
        )
    }
}
