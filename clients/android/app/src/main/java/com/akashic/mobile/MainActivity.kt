package com.akashic.mobile

import android.graphics.Color
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.akashic.mobile.ui.conversation.ConversationScreen
import com.akashic.mobile.ui.conversation.EmptyConversationState
import com.akashic.mobile.ui.design.AkashicTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.auto(Color.TRANSPARENT, Color.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.auto(Color.TRANSPARENT, Color.TRANSPARENT),
        )
        setContent {
            AkashicTheme {
                ConversationScreen(
                    state = EmptyConversationState,
                    onOpenNavigation = null,
                    onOpenMenu = null,
                    onAttach = {},
                    onSend = {},
                    onStop = {},
                )
            }
        }
    }
}
