package com.akashic.mobile.ui.conversation

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.assertTextEquals
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollToIndex
import androidx.compose.ui.test.performTouchInput
import androidx.compose.ui.test.swipeDown
import androidx.compose.ui.test.longClick
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
        compose.onNodeWithTag("turn-stop-pending").assertIsDisplayed()
        assertEquals(1, stops)
    }

    @Test
    fun sendStopButtonExposesOnlyCurrentAction() {
        var state by mutableStateOf(EmptyConversationState.copy(canSend = true))
        compose.setContent { AkashicTheme { Screen(state) } }

        compose.onNodeWithContentDescription("发送消息").assertIsDisplayed()
        compose.onNodeWithContentDescription("停止生成").assertDoesNotExist()

        compose.runOnIdle {
            state = state.copy(
                isStreaming = true,
                canStop = true,
            )
        }
        compose.onNodeWithContentDescription("停止生成").assertIsDisplayed()
        compose.onNodeWithContentDescription("发送消息").assertDoesNotExist()
    }

    @Test
    fun interruptedTurnKeepsTerminalFeedbackInConversation() {
        show(
            EmptyConversationState.copy(
                messages = listOf(
                    MessageUi.AssistantTurn(
                        id = "interrupted",
                        intro = null,
                        blocks = listOf(
                            ProcessBlockUi(
                                id = "thinking",
                                kind = ProcessBlockKind.THINKING,
                                title = "思考",
                                detail = "正在检查链路",
                                state = ProcessBlockState.COMPLETED,
                            ),
                        ),
                        answer = "已经完成的部分会保留。",
                        status = AssistantTurnStatus.INTERRUPTED,
                        durationSeconds = 2,
                    ),
                ),
                canSend = true,
            ),
        )

        compose.onNodeWithText("已中止 · 2s").assertIsDisplayed()
        compose.onNodeWithTag("turn-interrupted-interrupted").assertIsDisplayed()
        compose.onNodeWithText("生成已中止，可继续补充").assertIsDisplayed()
    }

    @Test
    fun commandMenuFillsComposerWithoutSending() {
        var sends = 0
        compose.setContent {
            AkashicTheme {
                Screen(
                    state = EmptyConversationState.copy(
                        commands = listOf(CommandUi("undo", "撤销上一轮对话")),
                        canSend = true,
                    ),
                    onSend = { sends += 1 },
                )
            }
        }

        compose.onNodeWithContentDescription("打开快捷命令").performClick()
        compose.onNodeWithText("快捷命令").assertIsDisplayed()
        compose.onNodeWithTag("command-undo").performClick()
        compose.onNodeWithTag("composer-input").assertTextEquals("/undo ")
        assertEquals(0, sends)
    }

    @Test
    fun longCommandCatalogRemainsScrollable() {
        val commands = (1..20).map { index ->
            CommandUi("command$index", "第 $index 个服务端命令")
        }
        show(EmptyConversationState.copy(commands = commands, canSend = true))

        compose.onNodeWithContentDescription("打开快捷命令").performClick()
        compose.onNodeWithTag("command-sheet-list").performScrollToIndex(19)
        compose.onNodeWithTag("command-command20").performClick()

        compose.onNodeWithTag("composer-input").assertTextEquals("/command20 ")
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

    @Test
    fun longPressingMessageTextKeepsTheSelectionSurfaceStable() {
        show(
            EmptyConversationState.copy(
                messages = listOf(
                    MessageUi.AssistantTurn(
                        id = "selectable",
                        intro = null,
                        blocks = emptyList(),
                        answer = "这段正文可以局部选择并复制。",
                        status = AssistantTurnStatus.COMPLETE,
                        durationSeconds = 1,
                    ),
                ),
                canSend = true,
            ),
        )

        compose.onNodeWithText("这段正文可以局部选择并复制。")
            .performTouchInput { longClick() }
        compose.onNodeWithText("这段正文可以局部选择并复制。").assertIsDisplayed()
    }

    @Test
    fun markdownUsesCompactHeadingsAndNativeDisplayMath() {
        show(
            EmptyConversationState.copy(
                messages = listOf(
                    MessageUi.AssistantTurn(
                        id = "rich-markdown",
                        intro = null,
                        blocks = emptyList(),
                        answer = """
                            ## 核心思路：两步握手

                            ### 第一步：RID — 共鸣兴趣提炼

                            评分器使用余弦相似度：
                            \[ s_\phi(u, h_t) = \cos(\mathbf{u}, \mathbf{z}_t) \]

                            - **u** 是用户特征向量
                            - **z_t** 是会话摘要向量

                            ### 第二步：ISG — 互动式开场生成
                        """.trimIndent(),
                        status = AssistantTurnStatus.COMPLETE,
                        durationSeconds = 9,
                    ),
                ),
                canSend = true,
            ),
        )

        compose.onNodeWithText("核心思路：两步握手").assertIsDisplayed()
        compose.onNodeWithText("第一步：RID — 共鸣兴趣提炼").assertIsDisplayed()
        compose.onNodeWithTag("message-math-1").assertIsDisplayed()
        compose.onNodeWithText("第二步：ISG — 互动式开场生成").assertIsDisplayed()
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
                    status = AssistantTurnStatus.STREAMING,
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
        onSend: (String) -> Unit = {},
    ) {
        ConversationScreen(
            state = state,
            onAttach = {},
            onRemoveAttachment = {},
            onRetryAttachment = {},
            onRetryDownloadedAttachment = {},
            onOpenDownloadedAttachment = {},
            onDismissError = onDismissError,
            onSend = onSend,
            onStop = onStop,
        )
    }
}
