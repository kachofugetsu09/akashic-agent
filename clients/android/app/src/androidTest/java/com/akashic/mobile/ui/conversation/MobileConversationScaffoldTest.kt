package com.akashic.mobile.ui.conversation

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import com.akashic.mobile.ui.design.AkashicTheme
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test

class MobileConversationScaffoldTest {
    @get:Rule
    val compose = createComposeRule()

    @Test
    fun reloadConfirmsAndPreservesConnectionSemantics() {
        var reloads = 0
        compose.setContent {
            AkashicTheme {
                MobileConversationScaffold(
                    state = EmptyConversationState.copy(canResync = true, canSend = true),
                    onSelectSession = {},
                    onNewSession = {},
                    onRestartPairing = {},
                    onReloadFromServer = { reloads += 1 },
                    onAttach = {},
                    onRemoveAttachment = {},
                    onRetryAttachment = {},
                    onSend = {},
                    onStop = {},
                    onRetryDownloadedAttachment = {},
                    onOpenDownloadedAttachment = {},
                    onDismissError = {},
                )
            }
        }

        compose.onNodeWithContentDescription("打开对话列表").performClick()
        compose.onNodeWithText("重新同步消息").performClick()
        compose.onNodeWithText("连接密钥、当前配对和待发送内容都会保留。", substring = true)
            .assertIsDisplayed()
        compose.onNodeWithText("清理并重新同步").performClick()
        assertEquals(1, reloads)
    }
}
