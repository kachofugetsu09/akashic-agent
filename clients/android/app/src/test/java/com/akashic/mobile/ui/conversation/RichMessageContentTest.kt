package com.akashic.mobile.ui.conversation

import org.junit.Assert.assertEquals
import org.junit.Test

class RichMessageContentTest {
    @Test
    fun splitsDisplayMathWithoutChangingSurroundingMarkdown() {
        val segments = richMessageSegments(
            """
            评分器：
            \[
            s_\phi(u, h_t) = \cos(\mathbf{u}, \mathbf{z}_t)
            \]
            - **u** 是用户特征
            """.trimIndent(),
        )

        assertEquals(
            listOf(
                RichMessageSegment.Markdown("评分器："),
                RichMessageSegment.BlockMath("s_\\phi(u, h_t) = \\cos(\\mathbf{u}, \\mathbf{z}_t)"),
                RichMessageSegment.Markdown("- **u** 是用户特征"),
            ),
            segments,
        )
    }

    @Test
    fun preservesFormulaMarkersInsideCodeFence() {
        val content = """
            ```tex
            \[
            x^2
            \]
            ```
        """.trimIndent()

        assertEquals(listOf(RichMessageSegment.Markdown(content)), richMessageSegments(content))
    }

    @Test
    fun splitsTheSingleLineDisplayMathUsedByRealHistory() {
        val content = """
            评分器是一个简单的余弦相似度模型：
            \[ s_\phi(u, h_t) = \cos(\mathbf{u}, \mathbf{z}_t) \]

            - **u** 是用户特征向量
        """.trimIndent()

        assertEquals(
            listOf(
                RichMessageSegment.Markdown("评分器是一个简单的余弦相似度模型："),
                RichMessageSegment.BlockMath("s_\\phi(u, h_t) = \\cos(\\mathbf{u}, \\mathbf{z}_t)"),
                RichMessageSegment.Markdown("- **u** 是用户特征向量"),
            ),
            richMessageSegments(content),
        )
    }

    @Test
    fun preservesUnclosedStreamingFormula() {
        val content = "说明\n\\[\nx^2"

        assertEquals(listOf(RichMessageSegment.Markdown(content)), richMessageSegments(content))
    }
}
