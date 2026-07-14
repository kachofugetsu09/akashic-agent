package com.akashic.mobile.ui.conversation

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTouchInput
import androidx.compose.ui.test.swipeDown
import com.akashic.mobile.ui.design.AkashicTheme
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test

class ConversationInteractionsTest {
    @get:Rule
    val compose = createComposeRule()

    @Test
    fun readyErrorNoticeIsVisibleAndDismissible() {
        var dismissals = 0
        show(
            EmptyConversationState.copy(
                connectionStatus = ConnectionStatusUi.READY,
                errorNotice = "附件读取失败",
                canSend = true,
            ),
            onDismissError = { dismissals += 1 },
        )

        compose.onNodeWithTag("conversation-error-notice").assertIsDisplayed()
        compose.onNodeWithText("附件读取失败").assertIsDisplayed()
        compose.onNodeWithContentDescription("关闭错误提示").performClick()
        assertEquals(1, dismissals)
    }

    @Test
    fun activeTurnCanStopOnlyOnceWhileRequestIsPending() {
        var stops = 0
        var state by mutableStateOf(
            EmptyConversationState.copy(
                isStreaming = true,
                canStop = true,
                canSend = true,
            ),
        )
        compose.setContent {
            AkashicTheme {
                Screen(state = state, onStop = { stops += 1 })
            }
        }

        compose.onNodeWithTag("composer-send-stop").assertIsEnabled().performClick()
        compose.runOnIdle { state = state.copy(isStopping = true, canStop = false) }
        compose.onNodeWithTag("composer-send-stop").assertIsNotEnabled()
        assertEquals(1, stops)
    }

    @Test
    fun contentGrowthFollowsBottomUntilUserScrollsAway() {
        var state by mutableStateOf(messageState("bottom-before"))
        compose.setContent {
            AkashicTheme { Screen(state = state) }
        }

        compose.waitForIdle()
        compose.onNodeWithText("bottom-before").assertIsDisplayed()
        compose.runOnIdle { state = messageState("bottom-after") }
        compose.onNodeWithText("bottom-after").assertIsDisplayed()

        repeat(3) {
            compose.onNodeWithTag("conversation-message-list").performTouchInput { swipeDown() }
        }
        compose.onNodeWithText("bottom-after").assertDoesNotExist()
        compose.runOnIdle { state = messageState("bottom-final") }
        compose.onNodeWithText("bottom-final").assertDoesNotExist()
    }

    private fun show(
        state: ConversationUiState,
        onDismissError: () -> Unit = {},
    ) {
        compose.setContent {
            AkashicTheme { Screen(state = state, onDismissError = onDismissError) }
        }
    }

    private fun messageState(lastAnswer: String) = EmptyConversationState.copy(
        messages = buildList {
            repeat(12) { index ->
                add(
                    MessageUi.User(
                        id = "user-$index",
                        text = "question-$index",
                        deliveryLabel = "已发送",
                    ),
                )
            }
            add(
                MessageUi.AssistantTurn(
                    id = "assistant-last",
                    intro = null,
                    blocks = emptyList(),
                    answer = lastAnswer,
                    isStreaming = true,
                    durationSeconds = null,
                ),
            )
        },
        canSend = true,
    )

    @androidx.compose.runtime.Composable
    private fun Screen(
        state: ConversationUiState,
        onStop: () -> Unit = {},
        onDismissError: () -> Unit = {},
    ) {
        ConversationScreen(
            state = state,
            onAttach = {},
            onRemoveAttachment = {},
            onRetryAttachment = {},
            onRetryDownloadedAttachment = {},
            onOpenDownloadedAttachment = {},
            onDismissError = onDismissError,
            onSend = {},
            onStop = onStop,
        )
    }
}
