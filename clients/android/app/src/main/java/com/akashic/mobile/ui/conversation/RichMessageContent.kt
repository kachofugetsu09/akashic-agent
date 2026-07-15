package com.akashic.mobile.ui.conversation

internal sealed interface RichMessageSegment {
    data class Markdown(val content: String) : RichMessageSegment

    data class BlockMath(val content: String) : RichMessageSegment
}

/** 将 Markdown 中的块级公式切成独立原生渲染段。 */
internal fun richMessageSegments(content: String): List<RichMessageSegment> {
    val segments = mutableListOf<RichMessageSegment>()
    val markdown = StringBuilder()
    val math = StringBuilder()
    var mathClosing: String? = null
    var codeFence = false

    // 1. 只识别代码块之外、独占一行的标准块级公式边界
    content.lineSequence().forEach { line ->
        val trimmed = line.trim()
        if (mathClosing == null && trimmed.startsWith("```")) codeFence = !codeFence
        val singleLineMath = when {
            !codeFence && mathClosing == null &&
                trimmed.startsWith("\\[") && trimmed.endsWith("\\]") && trimmed.length > 4 -> {
                trimmed.removePrefix("\\[").removeSuffix("\\]").trim()
            }
            !codeFence && mathClosing == null &&
                trimmed.startsWith("$$") && trimmed.endsWith("$$") && trimmed.length > 4 -> {
                trimmed.removePrefix("$$").removeSuffix("$$").trim()
            }
            else -> null
        }
        if (singleLineMath != null) {
            markdown.takeIf { it.isNotEmpty() }?.let {
                segments += RichMessageSegment.Markdown(it.toString().trim('\n', '\r'))
                it.clear()
            }
            segments += RichMessageSegment.BlockMath(singleLineMath)
        } else if (!codeFence && mathClosing == null && trimmed in setOf("\\[", "$$")) {
            markdown.takeIf { it.isNotEmpty() }?.let {
                segments += RichMessageSegment.Markdown(it.toString().trim('\n', '\r'))
                it.clear()
            }
            mathClosing = if (trimmed == "\\[") "\\]" else "$$"
        } else if (mathClosing != null && trimmed == mathClosing) {
            segments += RichMessageSegment.BlockMath(math.toString().trim())
            math.clear()
            mathClosing = null
        } else if (mathClosing != null) {
            math.appendLine(line)
        } else {
            markdown.appendLine(line)
        }
    }

    // 2. 不完整公式按原文展示，避免流式阶段闪烁或丢字
    if (mathClosing != null) {
        val incomplete = buildString {
            appendLine(if (mathClosing == "\\]") "\\[" else "$$")
            append(math)
        }.trimEnd()
        val previous = segments.lastOrNull() as? RichMessageSegment.Markdown
        if (previous != null) {
            segments[segments.lastIndex] = RichMessageSegment.Markdown("${previous.content}\n$incomplete")
        } else {
            markdown.append(incomplete)
        }
    }
    markdown.takeIf { it.isNotEmpty() }?.let {
        segments += RichMessageSegment.Markdown(it.toString().trim('\n', '\r'))
    }
    return segments
}
