package com.akashic.mobile.ui.conversation

import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ConversationScrollTest {
    @Test
    fun bottomDetectionAllowsSmallLayoutGrowthButRejectsEarlierItem() {
        assertTrue(isConversationAtBottom(4, 3, 1040, 1000, 48))
        assertFalse(isConversationAtBottom(4, 2, 800, 1000, 48))
    }

    @Test
    fun revisionChangesForStreamingAnswerAndProcessDetail() {
        val initial = assistant(answer = "a", detail = "读取中")
        val answerGrowth = assistant(answer = "ab", detail = "读取中")
        val processGrowth = assistant(answer = "a", detail = "读取完成")

        assertNotEquals(messageContentRevision(initial), messageContentRevision(answerGrowth))
        assertNotEquals(messageContentRevision(initial), messageContentRevision(processGrowth))
    }

    private fun assistant(answer: String, detail: String) = listOf(
        MessageUi.AssistantTurn(
            id = "assistant",
            intro = "先检查",
            blocks = listOf(
                ProcessBlockUi(
                    id = "tool",
                    kind = ProcessBlockKind.TOOL,
                    title = "读取文件",
                    detail = detail,
                    state = ProcessBlockState.RUNNING,
                ),
            ),
            answer = answer,
            status = AssistantTurnStatus.STREAMING,
            durationSeconds = null,
        ),
    )
}
