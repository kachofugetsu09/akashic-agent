package com.akashic.mobile.data.realtime

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/** 串行化附件草稿 mutation 与消息发送准备。 */
internal class AttachmentOperationOwner {
    private val mutex = Mutex()

    suspend fun <T> perform(operation: suspend () -> T): T = mutex.withLock { operation() }
}

internal fun attachmentDraftMatchesExpected(actualIds: List<String>, expectedIds: List<String>): Boolean =
    actualIds.size == expectedIds.size &&
        expectedIds.distinct().size == expectedIds.size &&
        actualIds.toSet() == expectedIds.toSet()

/** 保证发送持久化结果恰好回调一次，同时让内部异常继续暴露。 */
internal suspend fun withSendResult(
    callback: (Boolean) -> Unit,
    operation: suspend ((Boolean) -> Unit) -> Unit,
) {
    var reported = false
    fun report(result: Boolean) {
        if (reported) return
        reported = true
        callback(result)
    }
    try {
        operation(::report)
    } finally {
        report(false)
    }
}
